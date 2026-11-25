"""Evaluate the participant tuner on intro operating points and basic grids."""
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

# Support direct execution without requiring PYTHONPATH to be configured.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_REPOSITORY_ROOT / "src"), str(_REPOSITORY_ROOT)]

import numpy as np
from VeraGridEngine.api import open_file
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import SmallSignalStabilityRmsDriver
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_options import RmsSmallSignalStabilityOptions
from hackathon_files import TunableGrid, additional_grid_evaluator
from hackathon_files.intro.rms_tunner import FinalTuner
from hackathon_files.basic import basic_evaluator, case_files, initialize_grid as initialize_basic_grid


def make_evaluator():
    """Extend the shared evaluator without changing other hackathon examples.

    These inclusive search ranges are teaching defaults, not stability guarantees.
    Keep physical filter parameters and current limits outside the tuning catalog.
    """
    evaluator = additional_grid_evaluator()
    evaluator.control_ranges = {
        **evaluator.control_ranges,
        "Kdp": (0.001, 0.2),
        "Kdq": (0.001, 0.2),
        "Kp_icl": (0.001, 1.0),
        "Ki_icl": (1.0, 200.0),
        "tau_P": (0.001, 0.1),
        "tau_Q": (0.001, 0.1),
    }
    return evaluator


def initialize_grid(grid, *, enable_trials=True):
    """Solve the operating point and assemble an initialized RMS problem."""
    power_flow = PowerFlowDriver(grid)
    power_flow.run()
    if not bool(power_flow.results.converged):
        raise RuntimeError("AC power flow did not converge")

    # Initialization uses the solved voltages and generator injections.
    driver = SmallSignalStabilityRmsDriver(
        grid=grid, pf_results=power_flow.results,
        sss_options=RmsSmallSignalStabilityOptions(k=0, ss_assessment_time=0.0),
    )
    evaluator = make_evaluator()  # No Kundur-specific line corridors.
    tunable = TunableGrid.from_problem(
        driver.problem, evaluator.control_ranges,
        shunt_bus_names=(), line_corridors=(),
        max_test_tunings=evaluator.max_test_tunings,
        evaluator=evaluator if enable_trials else None,
    )
    return tunable



@dataclass(frozen=True)
class Scenario:
    name: str
    load_scale: float
    reactive_scale: float


# Sample operating points for the single evaluation workflow.
SCENARIOS = (
    Scenario("nominal", 1.00, 1.00),
    Scenario("light", 0.95, 0.95),
    Scenario("heavy", 1.05, 1.05),
    Scenario("low-reactive", 1.00, 0.90),
    Scenario("high-reactive", 1.00, 1.10),
    Scenario("light-high-q", 0.95, 1.10),
    Scenario("heavy-low-q", 1.05, 0.90),
    Scenario("light-low-q", 0.95, 0.90),
    Scenario("heavy-high-q", 1.05, 1.10),
    Scenario("mid-light", 0.975, 1.00),
    Scenario("mid-heavy", 1.025, 1.00),
)


def load_benchmark():
    return open_file(str(Path(__file__).with_name("testgrids") / "benchmark_grid.veragrid"))


def make_scenario(benchmark, scenario):
    """Copy before editing so cases and repeated runs cannot contaminate others.

    Scale active generation with active demand; AC power flow lets the slack
    supply the remaining mismatch and losses. Reactive demand varies separately.
    RMS initialization then derives model injections from this operating point.
    """
    grid = benchmark.copy()
    for load in grid.loads:
        load.P *= scenario.load_scale
        load.Q *= scenario.reactive_scale
    for generator in grid.generators:
        generator.P *= scenario.load_scale
    return grid


def evaluate_basic(path):
    """Use basic-case permissions and keep final scoring outside the trial budget."""
    baseline = initialize_basic_grid(open_file(str(path)))
    evaluator = basic_evaluator()
    participant = evaluator.start_session(baseline)
    proposal = FinalTuner().tune(participant)
    return evaluator.evaluate_proposal(baseline, proposal)


def main() -> int:
    """Evaluate every sample with the same electrical and load-step checks."""
    start = perf_counter()
    benchmark = load_benchmark()
    results = []
    failures = 0
    cases = [
        ("intro", scenario.name,
         lambda scenario=scenario: FinalTuner().test_tuning(
             make_scenario(benchmark, scenario), make_evaluator()))
        for scenario in SCENARIOS
    ]
    cases.extend(("basic", path.stem, lambda path=path: evaluate_basic(path))
                 for path in case_files())
    totals = {"intro": [0, 0], "basic": [0, 0]}
    for collection, name, evaluate in cases:
        label = f"{collection}/{name}"
        totals[collection][1] += 1
        try:
            # Fresh tuner per case: budgets and algorithm state do not leak.
            # This public helper validates the proposal, recomputes PF and SSA,
            # and simulates the standard 2% load increase over three seconds.
            metrics = evaluate()
            results.append(metrics)
            passed = (metrics.stable and metrics.voltage_violations == 0
                      and metrics.thermal_violations == 0)
            failures += not passed
            totals[collection][0] += int(passed)
            damping = metrics.min_oscillatory_damping
            damping_text = "n/a" if damping is None else f"{100 * damping:.2f}%"
            print(f"{label:32s} losses={metrics.active_losses_mw:8.3f} MW  "
                  f"damping={damping_text:>8s}  max Re={metrics.max_eigenvalue_real:+.5f}  "
                  f"voltage/thermal violations={metrics.voltage_violations}/{metrics.thermal_violations}  "
                  f"peak df={metrics.peak_frequency_deviation_hz:.4f} Hz  "
                  f"changes={metrics.structural_changes}/{metrics.max_structural_changes}  "
                  f"S_changes={metrics.changes_score:.3f}  "
                  f"{'PASS' if passed else 'FAIL'}")
        except Exception as error:
            failures += 1
            print(f"{label:32s} ERROR: {type(error).__name__}: {error}")
    for collection, (passed, total) in totals.items():
        print(f"\n{collection}: {passed}/{total} passed")
    print(f"\nEvaluation: {len(cases) - failures}/{len(cases)} passed; "
          f"{perf_counter() - start:.3f} s.")
    if results:
        print(f"Mean losses over {len(results)} successfully evaluated cases: "
              f"{sum(m.active_losses_mw for m in results) / len(results):.3f} MW")
        print(f"Mean change score over {len(results)} successfully evaluated cases: "
              f"{sum(m.changes_score for m in results) / len(results):.3f}")
        damping_values = [m.min_oscillatory_damping for m in results
                          if m.min_oscillatory_damping is not None]
        if damping_values:
            print(f"Worst oscillatory damping: {100 * min(damping_values):.2f}%")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
