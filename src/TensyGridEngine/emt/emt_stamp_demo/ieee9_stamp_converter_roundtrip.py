#!/usr/bin/env python3
"""Build and verify the local STAMP IEEE9 EMT grids across .veragrid reloads."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if not __package__:
    # Direct execution from the repository root does not add src to sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.optimize import linear_sum_assignment

from VeraGridEngine.api import open_file, save_file
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import (
    SmallSignalStabilityEmtDriver,
)
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import (
    SmallSignalStabilityEmtOptions,
)
from VeraGridEngine.enumerations import (
    DynamicIntegrationMethod,
    EmtInitializationMethod,
    EmtProblemTypes,
    EmtSolverTypes,
)
from veragrid_stamp.emt_case import build_stamp_wscc_emt_grid

OUTPUT_DIR = Path(__file__).resolve().parent / "saved_grids"
CASE_NAMES = {"stamp": "ieee9_stamp_gfor_gfol.veragrid",
              "multilinear": "ieee9_stamp_gfor_gfol_multilinear.veragrid"}


def build_case(case: str):
    """Build the case from the editable STAMP sources in this repository."""
    if case == "stamp":
        return build_stamp_wscc_emt_grid()
    if case == "multilinear":
        from TensyGridEngine.emt.emt_stamp_demo.ieee9_stamp_multilinear_ssa import build_grid
        return build_grid(multilinear_converters=True)
    raise ValueError(f"Unknown converter case: {case}")


def _model_parameter_values(block) -> dict[str, float]:
    """Collect numeric model constants, including runtime/event parameters."""
    values = {}
    for variable, expression in (*block.parameters.items(), *block.event_dict.items()):
        if hasattr(expression, "value") and expression.value is not None:
            values[f"{block.name}:{variable.name}"] = float(expression.value)
    for child in block.children:
        values.update(_model_parameter_values(child))
    return values


def verify_saved_grid(source, reloaded) -> None:
    """Check frequency, device allocation, and saved EMT model constants."""
    if source.fBase != reloaded.fBase:
        raise AssertionError(f"fBase changed: {source.fBase} -> {reloaded.fBase}")
    before = {device.name: device for device in source.generators}
    after = {device.name: device for device in reloaded.generators}
    if before.keys() != after.keys():
        raise AssertionError("The saved generator allocation changed")
    for name in ("STAMP SG1", "STAMP GFOR1", "STAMP GFOL2"):
        lhs = before[name].emt_model
        rhs = after[name].emt_model
        if lhs.empty() or rhs.empty():
            raise AssertionError(f"{name} lost its EMT model")
        if _model_parameter_values(lhs) != _model_parameter_values(rhs):
            raise AssertionError(f"{name} lost an EMT model parameter")


def floquet_multipliers(grid, steps_per_period: int, settling_periods: int,
                        modes: int) -> np.ndarray:
    """Run the same STAMP Floquet settings before and after reloading."""
    period = 1.0 / grid.fBase
    pf_options = PowerFlowOptions(retry_with_other_methods=True)
    pf = PowerFlowDriver(grid, pf_options)
    pf.run()
    pf3 = PowerFlowDriver3Ph(grid, pf_options)
    pf3.run()
    if not bool(pf.results.converged) or not bool(pf3.results.converged):
        raise RuntimeError("STAMP IEEE9 power flow did not converge")
    emt_options = EmtOptions(
        time_step=period / steps_per_period,
        simulation_time=period,
        tolerance=1e-8,
        solver_type=EmtSolverTypes.StructuralCompiled,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
        problem_type=EmtProblemTypes.Multilinear,
        verbose=0,
    )
    ssa_options = SmallSignalStabilityEmtOptions(
        k=modes, target_period=period,
        ss_assessment_time=settling_periods * period, verbose=0,
    )
    driver = SmallSignalStabilityEmtDriver(
        grid=grid, emt_options=emt_options, sss_options=ssa_options,
        pf_results=pf.results, pf_results_3ph=pf3.results,
    )
    driver.run()
    if driver.results is None:
        raise RuntimeError("STAMP Floquet analysis returned no results")
    return np.asarray(driver.results.multipliers)


def run_case(case: str, output_dir: Path, verify_floquet: bool,
             steps_per_period: int, settling_periods: int, modes: int) -> Path:
    source = build_case(case)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / CASE_NAMES[case]
    save_file(source, str(destination))
    reloaded = open_file(str(destination))
    verify_saved_grid(source, reloaded)
    print(f"{case}: saved={destination} fBase={reloaded.fBase:g} Hz; "
          "SG/GFOR/GFOL parameters preserved", flush=True)

    if verify_floquet:
        before = floquet_multipliers(source, steps_per_period, settling_periods, modes)
        after = floquet_multipliers(reloaded, steps_per_period, settling_periods, modes)
        rows, columns = linear_sum_assignment(abs(before[:, None] - after[None, :]))
        max_difference = float(np.max(abs(before[rows] - after[columns])))
        print(f"{case}: rho_before={np.max(abs(before)):.12g} "
              f"rho_after={np.max(abs(after)):.12g} "
              f"max_matched_multiplier_difference={max_difference:.3e}", flush=True)
        if max_difference > 1e-6:
            raise AssertionError("Floquet spectrum changed after .veragrid round-trip")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("stamp", "multilinear", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-verify-floquet", dest="verify_floquet", action="store_false",
                        help="Save and inspect the grids without running Floquet analysis")
    parser.set_defaults(verify_floquet=True)
    parser.add_argument("--steps-per-period", type=int, default=1000)
    parser.add_argument("--settling-periods", type=int, default=3)
    parser.add_argument("--modes", type=int, default=30)
    args = parser.parse_args()
    if min(args.steps_per_period, args.settling_periods, args.modes) <= 0:
        parser.error("steps, settling periods, and modes must be positive")
    cases = CASE_NAMES if args.case == "both" else (args.case,)
    for case in cases:
        run_case(case, args.output_dir, args.verify_floquet,
                 args.steps_per_period, args.settling_periods, args.modes)


if __name__ == "__main__":
    main()
