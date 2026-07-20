# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def ensure_repo_import_paths() -> None:
    """Add repository roots used by trunk scripts to ``sys.path``."""
    repo_root: Path = Path(__file__).resolve().parents[3]
    src_root: Path = repo_root / "src"
    trunk_root: Path = repo_root / "trunk"

    repo_root_text = str(repo_root)
    src_root_text = str(src_root)
    trunk_root_text = str(trunk_root)

    if src_root_text not in sys.path:
        sys.path.insert(0, src_root_text)
    if trunk_root_text not in sys.path:
        sys.path.insert(0, trunk_root_text)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)


ensure_repo_import_paths()

import VeraGridEngine.api as gce
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Templates.Emt.converter_emt_multilinear_template import get_emt_ideal_converter_multilinear
from VeraGridEngine.Templates.Emt.converter_emt_template import get_emt_ideal_converter
from VeraGridEngine.Templates.Emt.dc_load_emt_template import get_dc_load_emt_template
from VeraGridEngine.Templates.Emt.transformer_emt_template import get_transformer_emt_template
from VeraGridEngine.Templates.Emt.thevenin_equivalent_emt_generator_template import get_generator_thevenin_rl_emt_template_with_ref
from VeraGridEngine.Templates.templates_common_functions import set_emt_model
from VeraGridEngine.Utils.Symbolic.block import Var, find_name_in_block
from VeraGridEngine.Utils.procedural_logic import build_boundary_updater_from_block
from VeraGridEngine.enumerations import DynamicIntegrationMethod, EmtInitializationMethod, EmtSolverTypes, SolverType


@dataclass
class CaseResults:
    label: str
    time_arr: np.ndarray
    i_a: np.ndarray
    i_b: np.ndarray
    i_c: np.ndarray
    i_dc: np.ndarray
    p: np.ndarray
    q: np.ndarray


def build_emt_options() -> EmtOptions:
    """Return one compact EMT options setup for converter comparison."""
    options = EmtOptions(
        time_step=1.0e-5,
        simulation_time=2.0e-3,
        tolerance=1.0e-6,
        solver_type=EmtSolverTypes.Symbolic,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
        verbose=0,
    )
    options.newton_max_iter = 20
    return options


def build_pf_options() -> PowerFlowOptions:
    """Return PF options used to seed the EMT initial condition."""
    return PowerFlowOptions(
        solver_type=SolverType.NR,
        retry_with_other_methods=False,
        verbose=0,
        max_iter=30,
        tolerance=1.0e-6,
        control_q=False,
        control_taps_modules=False,
        control_taps_phase=False,
        orthogonalize_controls=False,
    )


def _get_signal(problem: EmtProblemDae, state_traj: np.ndarray, variable_name: str) -> np.ndarray:
    """Extract one variable trajectory by symbolic name from one EMT run."""
    variable: Var | None = find_name_in_block(variable_name, problem.sys_block)
    if variable is None:
        raise KeyError(f"Variable '{variable_name}' not found in EMT problem")
    idx = int(problem.get_var_idx(variable))
    return state_traj[:, idx]


def simulate_case(label: str, converter_builder: callable) -> CaseResults:
    """Build one AC/DC grid, run PF+EMT and return converter time series."""
    grid = gce.MultiCircuit(name=f"Converter compare {label}", Sbase=100.0, fbase=50.0)
    bus_slack_ac = gce.Bus(name="Bus_Slack_AC", Vnom=230.0, is_slack=True)
    bus_ac = gce.Bus(name="Bus_AC", Vnom=230.0)
    bus_dc = gce.Bus(name="Bus_DC", Vnom=320.0, is_dc=True)
    gen = gce.Generator(name="Gen_Thev", vset=1.0, Snom=100.0, freq=50.0, r1=0.001, x1=0.04)
    trafo = gce.Transformer2W(
        name="Trafo_AC",
        bus_from=bus_slack_ac,
        bus_to=bus_ac,
        rate=100.0,
        r=0.01,
        x=0.1,
        tap_module=1.0,
        tap_phase=0.0,
    )
    vsc = gce.VSC(
        name="vsc_cmp",
        bus_from=bus_dc,
        bus_to=bus_ac,
        rate=100.0,
        control1=gce.ConverterControlType.Qac,
        control2=gce.ConverterControlType.Vm_dc,
        control1_val=0.0,
        control2_val=1.0,
    )
    dc_load = gce.Load(name="DC_Load", P=30.0, Q=0.0)

    grid.add_bus(bus_slack_ac)
    grid.add_bus(bus_ac)
    grid.add_bus(bus_dc)
    grid.add_generator(bus_slack_ac, gen)
    grid.add_transformer2w(trafo)
    grid.add_vsc(vsc)
    grid.add_load(bus_dc, dc_load)
    grid.add_emt_events_group(gce.EmtEventsGroup(name=f"cmp_group_{label}"))

    for bus in grid.buses:
        get_bus_emt_template(grid, bus)

    pf_results = gce.power_flow(grid=grid, options=build_pf_options())

    gen_mdl = get_generator_thevenin_rl_emt_template_with_ref(vf=grid.var_factory).block
    trafo_mdl = get_transformer_emt_template(vf=grid.var_factory, name=trafo.name).block
    vsc_mdl = converter_builder(vf=grid.var_factory, name=vsc.name).block
    dc_load_mdl = get_dc_load_emt_template(vf=grid.var_factory, name="dc_load_emt_cmp").block

    set_emt_model(device=gen, model=gen_mdl, var_factory=grid.var_factory)
    set_emt_model(device=trafo, model=trafo_mdl, var_factory=grid.var_factory)
    set_emt_model(device=vsc, model=vsc_mdl, var_factory=grid.var_factory)
    set_emt_model(device=dc_load, model=dc_load_mdl, var_factory=grid.var_factory)

    problem = EmtProblemDae(
        grid=grid,
        options=build_emt_options(),
        pf_results=pf_results,
        pf_results_3ph=None,
    )
    solver = build_emt_solver(
        options=problem.options,
        problem=problem,
        t0=0.0,
        t_end=float(problem.options.simulation_time),
        h=float(problem.options.time_step),
        method=problem.options.integration_method,
    )
    boundary_updater = build_boundary_updater_from_block(problem)
    time_arr, state_traj, _diff_traj, _ok_init, _ok_conv = solver.simulate(boundary_updater=boundary_updater)

    return CaseResults(
        label=label,
        time_arr=time_arr,
        i_a=_get_signal(problem, state_traj, f"i_A_{vsc.name}"),
        i_b=_get_signal(problem, state_traj, f"i_B_{vsc.name}"),
        i_c=_get_signal(problem, state_traj, f"i_C_{vsc.name}"),
        i_dc=_get_signal(problem, state_traj, f"i_dc_{vsc.name}"),
        p=_get_signal(problem, state_traj, f"P_{vsc.name}"),
        q=_get_signal(problem, state_traj, f"Q_{vsc.name}"),
    )


def print_time_series(std_case: CaseResults, ml_case: CaseResults) -> None:
    """Print sampled time-series rows for both converter variants."""
    n = min(len(std_case.time_arr), len(ml_case.time_arr))
    sample_idx = np.linspace(0, n - 1, 12, dtype=int)

    print("time_s,iA_std,iA_ml,iB_std,iB_ml,iC_std,iC_ml,idc_std,idc_ml,P_std,P_ml,Q_std,Q_ml")
    for idx in sample_idx:
        print(
            f"{std_case.time_arr[idx]:.6e},"
            f"{std_case.i_a[idx]:.8e},{ml_case.i_a[idx]:.8e},"
            f"{std_case.i_b[idx]:.8e},{ml_case.i_b[idx]:.8e},"
            f"{std_case.i_c[idx]:.8e},{ml_case.i_c[idx]:.8e},"
            f"{std_case.i_dc[idx]:.8e},{ml_case.i_dc[idx]:.8e},"
            f"{std_case.p[idx]:.8e},{ml_case.p[idx]:.8e},"
            f"{std_case.q[idx]:.8e},{ml_case.q[idx]:.8e}"
        )


def plot_results(std_case: CaseResults, ml_case: CaseResults) -> None:
    """Save and show a standard-vs-ML trace comparison plot."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)

    axes[0, 0].plot(std_case.time_arr, std_case.i_a, label="iA std", linewidth=2.0)
    axes[0, 0].plot(ml_case.time_arr, ml_case.i_a, label="iA ml", linestyle="--", linewidth=2.0)
    axes[0, 0].set_title("Phase A current")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(std_case.time_arr, std_case.i_b, label="iB std", linewidth=2.0)
    axes[0, 1].plot(ml_case.time_arr, ml_case.i_b, label="iB ml", linestyle="--", linewidth=2.0)
    axes[0, 1].set_title("Phase B current")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(std_case.time_arr, std_case.i_c, label="iC std", linewidth=2.0)
    axes[1, 0].plot(ml_case.time_arr, ml_case.i_c, label="iC ml", linestyle="--", linewidth=2.0)
    axes[1, 0].set_title("Phase C current")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(std_case.time_arr, std_case.i_dc, label="idc std", linewidth=2.0)
    axes[1, 1].plot(ml_case.time_arr, ml_case.i_dc, label="idc ml", linestyle="--", linewidth=2.0)
    axes[1, 1].set_title("DC current")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[2, 0].plot(std_case.time_arr, std_case.p, label="P std", linewidth=2.0)
    axes[2, 0].plot(ml_case.time_arr, ml_case.p, label="P ml", linestyle="--", linewidth=2.0)
    axes[2, 0].set_title("Active power")
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()

    axes[2, 1].plot(std_case.time_arr, std_case.q, label="Q std", linewidth=2.0)
    axes[2, 1].plot(ml_case.time_arr, ml_case.q, label="Q ml", linestyle="--", linewidth=2.0)
    axes[2, 1].set_title("Reactive power")
    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()

    fig.suptitle("EMT-driver comparison: standard vs multilinear-trig averaged converter", fontsize=12)
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "converter_emt_multilinear_compare.png"
    fig.savefig(output_path, dpi=170)
    print(f"Plot saved to: {output_path}")
    plt.show()


def main() -> None:
    """Run both converter variants with PF-seeded EMT and compare traces."""
    std_case = simulate_case(label="standard", converter_builder=get_emt_ideal_converter)
    ml_case = simulate_case(label="ml_trig", converter_builder=get_emt_ideal_converter_multilinear)
    print_time_series(std_case=std_case, ml_case=ml_case)
    plot_results(std_case=std_case, ml_case=ml_case)


if __name__ == "__main__":
    main()
