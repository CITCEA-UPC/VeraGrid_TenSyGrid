"""Audit IEEE9 EMT state initialization without advancing time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.EMT.initialization_emt import (
    _build_constant_param_array,
    _build_runtime_param_array,
)
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Utils.Symbolic.block import find_name_in_block
from VeraGridEngine.Utils.Symbolic.explicit_initialization_symbolic import (
    evaluate_explicit_init_equation,
)

from ieee9_emt_from_scratch import (
    attach_baseline_emt_models,
    build_emt_options,
    build_ieee9_multicircuit,
    build_power_flow_options,
)


def three_phase_metrics(values: np.ndarray, derivatives: np.ndarray) -> dict[str, float]:
    norm_sq = float(values @ values)
    return {
        "state_norm": float(np.sqrt(norm_sq)),
        "derivative_norm": float(np.linalg.norm(derivatives)),
        "phase_sum": float(np.sum(values)),
        "derivative_sum": float(np.sum(derivatives)),
        "radial_growth_rate": float(values @ derivatives / norm_sq) if norm_sq > 1e-20 else np.nan,
    }


def main() -> None:
    grid = build_ieee9_multicircuit()
    attach_baseline_emt_models(
        grid,
        frozen_generator_excitation=True,
        generator_mechanical_damping=0.0,
    )
    pf = PowerFlowDriver(grid=grid, options=build_power_flow_options())
    pf.run()
    pf3 = PowerFlowDriver3Ph(grid=grid, options=build_power_flow_options())
    pf3.run()
    if not bool(pf.results.converged) or not bool(pf3.results.converged):
        raise RuntimeError("Power flow failed")

    problem = EmtProblemDae(
        grid=grid,
        options=build_emt_options(),
        pf_results_3ph=pf3.results,
        pf_results=pf.results,
    )
    x0 = problem.get_x0()
    dx0 = problem.get_dx0()
    runtime_params = _build_runtime_param_array(problem)
    constant_params = _build_constant_param_array(problem)
    state_rhs_by_uid: dict[int, float] = {}
    for state, rhs in zip(problem.get_state_vars(), problem.get_state_eqs()):
        state_rhs_by_uid[state.uid] = float(evaluate_explicit_init_equation(
            eq=rhs,
            event_params_array=runtime_params,
            x=x0,
            params_array=constant_params,
            dx=dx0,
            uid2idx_event_params=problem.uid2idx_event_params,
            uid2idx_vars=problem.uid2idx_vars,
            uid2idx_params=problem.uid2idx_params,
            uid2idx_diff=problem.uid2idx_diff,
        ))
    rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []

    devices = list(grid.generators) + list(grid.transformers2w) + list(grid.lines)
    for device in devices:
        model = device.emt_model
        device_rows = []
        for state in model.state_vars:
            state_idx = problem.uid2idx_vars.get(state.uid)
            diff_var = state.diff_var
            diff_idx = None if diff_var is None else problem.uid2idx_diff.get(diff_var.uid)
            if state_idx is None or diff_idx is None:
                continue
            row = {
                "device": device.name,
                "device_type": type(device).__name__,
                "state": state.name,
                "differential": diff_var.name,
                "value": float(x0[state_idx]),
                "derivative": float(dx0[diff_idx]),
                "runtime_rhs": state_rhs_by_uid[state.uid],
                "rhs_minus_derivative": (
                    state_rhs_by_uid[state.uid] - float(dx0[diff_idx])
                ),
            }
            rows.append(row)
            device_rows.append(row)

        # EMT branch templates order their phase states in contiguous triples:
        # transformer i_f/i_t; PI line i_series/q_from/q_to.
        if type(device).__name__ in {"Transformer2W", "Line"}:
            for start in range(0, len(device_rows), 3):
                group = device_rows[start:start + 3]
                if len(group) != 3:
                    continue
                values = np.asarray([float(row["value"]) for row in group])
                derivatives = np.asarray([float(row["derivative"]) for row in group])
                group_rows.append({
                    "device": device.name,
                    "device_type": type(device).__name__,
                    "group": ",".join(str(row["state"]) for row in group),
                    **three_phase_metrics(values, derivatives),
                })

    output_dir = Path(__file__).parent
    state_path = output_dir / "ieee9_emt_init_state_derivatives.csv"
    group_path = output_dir / "ieee9_emt_init_three_phase_metrics.csv"
    state_frame = pd.DataFrame(rows)
    group_frame = pd.DataFrame(group_rows)
    state_frame.to_csv(state_path, index=False)
    group_frame.to_csv(group_path, index=False)

    machine = next(gen for gen in grid.generators if not gen.bus.is_slack)
    voltage_vars = machine.bus.emt_model.out_vars[:3]
    current_vars = [find_name_in_block(f"i_{phase}", machine.emt_model) for phase in ("A", "B", "C")]
    pe_var = find_name_in_block("p_e", machine.emt_model)
    if pe_var is None or any(var is None for var in current_vars):
        raise RuntimeError("Could not locate machine power/current variables")
    voltages = np.asarray([x0[int(problem.get_var_idx(var))] for var in voltage_vars])
    currents = np.asarray([x0[int(problem.get_var_idx(var))] for var in current_vars])
    stored_pe = float(x0[int(problem.get_var_idx(pe_var))])
    direct_pe = float(voltages @ currents / 3.0)

    print("MACHINE INITIAL ALGEBRAIC POWER CHECK")
    print(f"voltages={voltages}")
    print(f"currents={currents}")
    print(
        f"stored_p_e={stored_pe:.12e}, direct_sum_vi_over_3={direct_pe:.12e}, "
        f"residual={stored_pe - direct_pe:.12e}"
    )

    def machine_value(name: str) -> float:
        variable = find_name_in_block(name, machine.emt_model)
        if variable is None:
            raise RuntimeError(f"Missing machine variable {name}")
        return float(x0[int(problem.get_var_idx(variable))])

    iq0 = machine_value("i_q_")
    psiq0 = machine_value("psi_q_")
    edp0 = machine_value("e_dp_")
    psippq0 = machine_value("psi_pp_q_")
    xq, xqp, xqpp, xl = 1.70, 0.55, 0.25, 0.15
    tq0p, tq0pp = 0.8, 0.05
    gamma_q1 = (xqpp - xl) / (xqp - xl)
    gamma_q2 = (xqp - xqpp) / ((xqp - xl) ** 2)
    rhs_edp = (
        -edp0
        + (xq - xqp) * (gamma_q1 * iq0 - gamma_q2 * psippq0 - gamma_q2 * edp0)
    ) / tq0p
    rhs_psippq = (-psippq0 - edp0 - (xqp - xl) * iq0) / tq0pp
    q_alg_residual = (
        psiq0 + xqpp * iq0 + gamma_q1 * edp0 - (1.0 - gamma_q1) * psippq0
    )
    print(
        f"q_axis_runtime_rhs: d_e_dp={rhs_edp:.12e}, "
        f"d_psi_pp_q={rhs_psippq:.12e}, algebraic_residual={q_alg_residual:.12e}"
    )

    print("THREE-PHASE INITIALIZATION METRICS")
    print(group_frame.to_string(index=False))
    print("\nLARGEST ABSOLUTE INITIAL DERIVATIVES")
    print(
        state_frame.assign(abs_derivative=state_frame["derivative"].abs())
        .sort_values("abs_derivative", ascending=False)
        .head(30)
        .to_string(index=False)
    )
    print("\nLARGEST INITIAL STATE-EQUATION INCONSISTENCIES")
    print(
        state_frame.assign(
            abs_rhs_mismatch=state_frame["rhs_minus_derivative"].abs()
        )
        .sort_values("abs_rhs_mismatch", ascending=False)
        .head(30)
        .to_string(index=False)
    )
    max_rhs_mismatch = float(state_frame["rhs_minus_derivative"].abs().max())
    print(f"max_abs_state_rhs_mismatch={max_rhs_mismatch:.12e}")
    print(f"state_csv={state_path}")
    print(f"group_csv={group_path}")


if __name__ == "__main__":
    main()
