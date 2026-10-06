# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat


def ensure_repo_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    src_root = repo_root / "src"
    trunk_root = repo_root / "trunk"
    for path in (str(src_root), str(trunk_root), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


ensure_repo_import_paths()

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_template import EmtProblemTemplate
from VeraGridEngine.Templates.Emt.converter_emt_template import (
    _converter_control_type_code,
    get_full_pseudo_emt_converter,
)
from VeraGridEngine.Utils.Symbolic.block import Block, find_name_in_block
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Expr, Var
import VeraGridEngine.Utils.Symbolic.symbolic as sym
from VeraGridEngine.enumerations import ConverterControlType, DynamicIntegrationMethod, EmtSolverTypes


class StandaloneEmtProblem(EmtProblemTemplate):
    """Minimal EMT problem wrapper for one symbolic converter block."""

    __slots__ = []


def set_event_value(block: Block, variable_name: str, value: float) -> None:
    for var in block.event_dict:
        if var.name == variable_name:
            block.event_dict[var] = Const(float(value))
            return
    raise KeyError(f"Runtime parameter '{variable_name}' not found")


def evaluate_expr(expr: Expr | Var | Const | float, bindings: dict[int, float]) -> float:
    if isinstance(expr, Const):
        return 0.0 if expr.value is None else float(expr.value)
    if isinstance(expr, Var):
        return float(bindings.get(expr.uid, 0.0))
    if isinstance(expr, Expr):
        for var in expr.get_vars():
            bindings.setdefault(var.uid, 0.0)
        return float(expr.eval_uid(bindings))
    return float(expr)


def seed_initial_guess(problem: StandaloneEmtProblem) -> None:
    """Evaluate explicit template init equations once at t=0."""
    runtime_values = problem.event_params_values.copy()
    bindings: dict[int, float] = {problem.glob_time.uid: 0.0}

    for var, value in zip(problem.get_variable_parameters(), runtime_values):
        bindings[var.uid] = float(value)
    for var, value in zip(problem.get_constant_parameters(), problem.get_parameters_values()):
        bindings[var.uid] = 0.0 if value.value is None else float(value.value)

    pending = dict(problem.sys_block.init_eqs)
    for _ in range(max(1, len(pending))):
        changed = False
        for var, expr in list(pending.items()):
            value = evaluate_expr(expr, bindings)
            problem.init_guess[var.uid] = value
            bindings[var.uid] = value
            pending.pop(var)
            changed = True
        if not pending or not changed:
            break

    for var in problem.get_state_vars() + problem.get_algebraic_vars():
        if var.uid not in problem.init_guess:
            problem.init_guess[var.uid] = 0.0

    for var, expr in problem.sys_block.diff_init_eqs.items():
        problem.diff_init_guess[var.uid] = evaluate_expr(expr, bindings)
    for var in problem.get_diff_vars():
        if var.uid not in problem.diff_init_guess:
            problem.diff_init_guess[var.uid] = 0.0


def build_emt_options() -> EmtOptions:
    options = EmtOptions(
        time_step=2.0e-5,
        simulation_time=2.0e-2,
        tolerance=1.0e-7,
        solver_type=EmtSolverTypes.Symbolic,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        verbose=0,
    )
    options.newton_max_iter = 30
    options.init_newton_max_iter = 30
    return options


def build_converter_problem() -> tuple[StandaloneEmtProblem, dict[str, Var], tuple[Var, Var, Var, Var]]:
    vf = VarFactory()
    t = vf.add_var("t_pseudo_converter_standalone")
    name = "pseudo_converter_standalone"
    converter = get_full_pseudo_emt_converter(vf=vf, name=name).block

    f_grid = 50.0
    omega = 2.0 * np.pi * f_grid
    vpk = np.sqrt(2.0)
    vdc_bus = 1.0

    v_a, v_b, v_c, v_dc = converter.in_vars
    converter.event_dict[v_a] = Const(vpk) * sym.sin(Const(omega) * t) + sym.rand(t)
    converter.event_dict[v_b] = Const(vpk) * sym.sin(Const(omega) * t - Const(2.0 * np.pi / 3.0)) + sym.rand(t)
    converter.event_dict[v_c] = Const(vpk) * sym.sin(Const(omega) * t + Const(2.0 * np.pi / 3.0)) + sym.rand(t)
    converter.event_dict[v_dc] = Const(vdc_bus) + sym.rand(t)

    set_event_value(converter, "sbase", 1.0)
    set_event_value(converter, "P0", 0.40)
    set_event_value(converter, "control1", _converter_control_type_code(ConverterControlType.Pac))
    set_event_value(converter, "control2", _converter_control_type_code(ConverterControlType.Qac))
    set_event_value(converter, "control1_val", 0.40)
    set_event_value(converter, "control2_val", 0.05)
    set_event_value(converter, "omega_base", omega)
    set_event_value(converter, "phi_v", 0.0)
    set_event_value(converter, "Vpk", vpk)
    set_event_value(converter, "Vdc_nom", vdc_bus)

    root = Block(name="StandalonePseudoEmtConverter", children=[converter])
    root.unify_blocks()

    problem = StandaloneEmtProblem(
        sys_block=root,
        static_parameter_values_mapping={},
        glob_time=t,
    )
    seed_initial_guess(problem)

    tracked = {
        key: find_name_in_block(key, problem.sys_block)
        for key in ("i_A", "i_B", "i_C", "i_dc", "P", "Q", "v_dc", "theta_pll", "v_cmd_d", "v_cmd_q")
    }
    missing = [key for key, var in tracked.items() if var is None]
    if missing:
        raise KeyError(f"Missing expected converter variables: {missing}")

    return problem, tracked, (v_a, v_b, v_c, v_dc)


def save_dataforid_mat(
    path: str,
    xmsim: np.ndarray,
    u: np.ndarray,
    tstep: float = 1e-5,
) -> None:
    xmsim_arr = np.asarray(xmsim, dtype=np.float64)
    u_arr = np.asarray(u, dtype=np.float64)

    if xmsim_arr.ndim == 1:
        xmsim_arr = xmsim_arr.reshape(-1, 1)

    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(-1, 1)

    if xmsim_arr.shape[0] != u_arr.shape[0]:
        raise ValueError(
            f"xmsim and u must have the same number of samples. "
            f"Got xmsim={xmsim_arr.shape}, u={u_arr.shape}."
        )

    n_samples = xmsim_arr.shape[0]
    t = np.arange(n_samples, dtype=np.float64).reshape(-1, 1) * tstep

    savemat(
        path,
        {
            "xmsim": xmsim_arr,
            "u": u_arr,
            "t": t,
        },
    )


def get_signal(problem: StandaloneEmtProblem, trajectory: np.ndarray, variable: Var) -> np.ndarray:
    return trajectory[:, int(problem.get_var_idx(variable))]


def collect_xmsim(problem: StandaloneEmtProblem, trajectory: np.ndarray) -> np.ndarray:
    x_vars: list[Var] = []
    seen: set[int] = set()

    for var in problem.get_state_vars():
        x_vars.append(var)
        seen.add(var.uid)

    for diff_var in problem.get_diff_vars():
        base_var = diff_var.base_var
        if base_var is not None and base_var.uid not in seen:
            x_vars.append(base_var)
            seen.add(base_var.uid)

    return np.column_stack([get_signal(problem, trajectory, var) for var in x_vars])


def collect_inputs(problem: StandaloneEmtProblem, input_vars: tuple[Var, ...], time_arr: np.ndarray) -> np.ndarray:
    runtime_values = problem.event_params_values.copy()
    input_indices = [int(problem.uid2idx_event_params[var.uid]) for var in input_vars]
    u = np.zeros((len(time_arr), len(input_vars)), dtype=np.float64)

    for i, t in enumerate(time_arr):
        runtime_values = problem.def_event_params_fn(runtime_values, float(t))
        u[i, :] = runtime_values[input_indices]

    return u


def plot_time_series(time_arr: np.ndarray, signals: dict[str, np.ndarray], inputs: np.ndarray) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(time_arr, inputs[:, 0], label="v_A")
    axes[0].plot(time_arr, inputs[:, 1], label="v_B")
    axes[0].plot(time_arr, inputs[:, 2], label="v_C")
    axes[0].plot(time_arr, inputs[:, 3], label="v_dc_bus", linestyle="--")
    axes[0].set_title("Preset converter inputs")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(time_arr, signals["i_A"], label="i_A")
    axes[1].plot(time_arr, signals["i_B"], label="i_B")
    axes[1].plot(time_arr, signals["i_C"], label="i_C")
    axes[1].set_title("AC currents")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(time_arr, signals["P"], label="P")
    axes[2].plot(time_arr, signals["Q"], label="Q")
    axes[2].plot(time_arr, signals["i_dc"], label="i_dc")
    axes[2].set_title("Power and DC current")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].plot(time_arr, signals["v_dc"], label="v_dc")
    axes[3].plot(time_arr, signals["theta_pll"], label="theta_pll")
    axes[3].plot(time_arr, signals["v_cmd_d"], label="v_cmd_d")
    axes[3].plot(time_arr, signals["v_cmd_q"], label="v_cmd_q")
    axes[3].set_title("Internal states and commands")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    fig.suptitle("Standalone pseudo-EMT converter simulation", fontsize=12)
    fig.tight_layout()
    plot_path = Path(__file__).resolve().with_name("pseudo_emt_converter_standalone_timeseries.png")
    fig.savefig(plot_path, dpi=170)
    plt.show()
    return plot_path


def main() -> None:
    problem, tracked, input_vars = build_converter_problem()
    options = build_emt_options()
    solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=float(options.simulation_time),
        h=float(options.time_step),
        method=options.integration_method,
    )
    time_arr, state_traj, _diff_traj, ok_init, ok_conv = solver.simulate()

    signals = {key: get_signal(problem, state_traj, var) for key, var in tracked.items()}
    xmsim = collect_xmsim(problem=problem, trajectory=state_traj)
    u = collect_inputs(problem=problem, input_vars=input_vars, time_arr=time_arr)
    plot_path = plot_time_series(time_arr=time_arr, signals=signals, inputs=u)
    mat_path = Path(__file__).resolve().with_name("dataforid_emt_converter.mat")
    save_dataforid_mat(str(mat_path), xmsim=xmsim, u=u, tstep=float(options.time_step))

    print(f"initialized={ok_init}, converged={ok_conv}")
    print(f"dataforid_mat={mat_path}")
    print(f"timeseries_plot={plot_path}")
    header = "time_s," + ",".join(tracked.keys())
    print(header)
    for idx in np.linspace(0, len(time_arr) - 1, 12, dtype=int):
        row = [time_arr[idx]] + [signals[key][idx] for key in tracked]
        print(",".join(f"{value:.8e}" for value in row))


if __name__ == "__main__":
    main()
