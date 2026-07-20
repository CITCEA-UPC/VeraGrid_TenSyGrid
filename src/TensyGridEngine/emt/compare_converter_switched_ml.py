# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def ensure_repo_import_paths() -> None:
    repo_root: Path = Path(__file__).resolve().parents[3]
    src_root: Path = repo_root / "src"
    trunk_root: Path = repo_root / "trunk"
    for path in (str(src_root), str(trunk_root), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


ensure_repo_import_paths()

from VeraGridEngine.Templates.Emt.converter_switched_emt_multilinear_template import get_switched_emt_converter_multilinear
from VeraGridEngine.Utils.Symbolic.diagnostic import NewtonTraceCollector
from dynamics_emt.support.switched_converter_handover_case import (
    SwitchedConverterHandoverTrace,
    build_emt_options,
    simulate_switched_converter_handover,
)


def print_sampled_comparison(ref_trace: SwitchedConverterHandoverTrace, ml_trace: SwitchedConverterHandoverTrace) -> None:
    n = min(len(ref_trace.t_arr), len(ml_trace.t_arr))
    sample_idx = np.linspace(0, n - 1, 12, dtype=int)
    v_d_cap_ref_arr = np.clip(ref_trace.v_cmd_d_u_arr, -ref_trace.v_lim_arr, ref_trace.v_lim_arr)
    v_d_cap_ml_arr = ml_trace.v_d_cap_arr
    print("time_s,v_d_cap_ref,v_d_cap_ml,v_cmd_d_ref,v_cmd_d_ml,v_cmd_d_u_ref,v_cmd_d_u_ml,vlim_ref,vlim_ml,vlim_diff")
    for idx in sample_idx:
        print(
            f"{ref_trace.t_arr[idx]:.6e},"
            f"{v_d_cap_ref_arr[idx]:.8e},{v_d_cap_ml_arr[idx]:.8e},"
            f"{ref_trace.v_cmd_d_arr[idx]:.8e},{ml_trace.v_cmd_d_arr[idx]:.8e},"
            f"{ref_trace.v_cmd_d_u_arr[idx]:.8e},{ml_trace.v_cmd_d_u_arr[idx]:.8e},"
            f"{ref_trace.v_lim_arr[idx]:.8e},{ml_trace.v_lim_arr[idx]:.8e},"
            f"{(ml_trace.v_lim_arr[idx] - ref_trace.v_lim_arr[idx]):.8e}"
        )


def print_ml_equation_diagnostics(ml_trace: SwitchedConverterHandoverTrace) -> None:
    u = ml_trace.v_cmd_d_u_arr
    vlim = ml_trace.v_lim_arr

    plus1_max = ml_trace.ml_sat_u_plus1_max_arr
    plus2_max = ml_trace.ml_sat_u_plus2_max_arr
    minus1_max = ml_trace.ml_sat_u_minus1_max_arr
    minus2_max = ml_trace.ml_sat_u_minus2_max_arr
    plus1_min = ml_trace.ml_sat_u_plus1_min_arr
    plus2_min = ml_trace.ml_sat_u_plus2_min_arr
    minus1_min = ml_trace.ml_sat_u_minus1_min_arr
    minus2_min = ml_trace.ml_sat_u_minus2_min_arr

    plus_max_prod = plus1_max * plus2_max
    minus_max_prod = minus1_max * minus2_max
    plus_min_prod = plus1_min * plus2_min
    minus_min_prod = minus1_min * minus2_min

    sat_max_balance = (u - vlim) - (plus_max_prod - minus_max_prod)
    sat_max_complementarity = plus1_max * minus1_max
    sat_max_eq_plus = plus1_max - plus2_max
    sat_max_eq_minus = minus1_max - minus2_max
    sat_max_plus_part_error = plus_max_prod - np.maximum(u - vlim, 0.0)

    sat_min_balance = (u + vlim) - (plus_min_prod - minus_min_prod)
    sat_min_complementarity = plus1_min * minus1_min
    sat_min_eq_plus = plus1_min - plus2_min
    sat_min_eq_minus = minus1_min - minus2_min
    sat_min_plus_part_error = plus_min_prod - np.maximum(u + vlim, 0.0)

    v_d_cap_reconstructed = -vlim + plus_min_prod - plus_max_prod
    sat_reconstruction_error = v_d_cap_reconstructed - ml_trace.v_d_cap_arr
    upper_limit_violation = np.maximum(ml_trace.v_d_cap_arr - vlim, 0.0)
    lower_limit_violation = np.maximum(-vlim - ml_trace.v_d_cap_arr, 0.0)

    print("\nML equation diagnostics (max abs / rmse):")
    diag = [
        ("v_lim_aux - v_lim", ml_trace.v_lim_aux_arr - ml_trace.v_lim_arr),
        ("v_cmd_norm - v_cmd_norm_aux", ml_trace.v_cmd_norm_arr - ml_trace.v_cmd_norm_aux_arr),
        ("v_cmd_d_aux - v_cmd_d", ml_trace.v_cmd_d_aux_arr - ml_trace.v_cmd_d_arr),
        ("v_cmd_q_aux - v_cmd_q", ml_trace.v_cmd_q_aux_arr - ml_trace.v_cmd_q_arr),
        ("sat_max balance", sat_max_balance),
        ("sat_max complementarity", sat_max_complementarity),
        ("sat_max plus1-plus2", sat_max_eq_plus),
        ("sat_max minus1-minus2", sat_max_eq_minus),
        ("sat_max plus-part error", sat_max_plus_part_error),
        ("sat_min balance", sat_min_balance),
        ("sat_min complementarity", sat_min_complementarity),
        ("sat_min plus1-plus2", sat_min_eq_plus),
        ("sat_min minus1-minus2", sat_min_eq_minus),
        ("sat_min plus-part error", sat_min_plus_part_error),
        ("sat v_d_cap reconstruction", sat_reconstruction_error),
        ("v_d_cap upper-limit violation", upper_limit_violation),
        ("v_d_cap lower-limit violation", lower_limit_violation),
    ]
    for label, arr in diag:
        max_abs = float(np.max(np.abs(arr)))
        rmse = float(np.sqrt(np.mean(arr * arr)))
        print(f"  {label}: max_abs={max_abs:.6e}, rmse={rmse:.6e}")


def print_newton_residual_diagnostics(collector: NewtonTraceCollector, limit: int = 8) -> None:
    if len(collector.residual_records) == 0:
        print("\nNewton residual diagnostics: no records captured.")
        return
    ranked = sorted(collector.residual_records, key=lambda r: abs(float(r["res_norm_inf"])), reverse=True)
    print("\nNewton residual diagnostics (worst residual snapshots):")
    for rec in ranked[: max(1, int(limit))]:
        print(
            f"  t={rec['t']:.6e}, step={rec['step']}, iter={rec['newton_iter']}, "
            f"res_inf={rec['res_norm_inf']:.6e}"
        )
        for top_row in rec["top"]:
            print(
                "    "
                f"eq={top_row['eq_idx']}, abs={top_row['abs_res']:.6e}, "
                f"res={top_row['res']:.6e}, kind={top_row.get('kind', '')}, "
                f"var={top_row.get('var_name', '')}, label={top_row.get('label', '')}"
            )


def fail_if_residual_too_large(
    collector: NewtonTraceCollector,
    max_residual_inf: float = 1.0e9,
    max_state_residual_inf: float = 1.0e2,
) -> None:
    if len(collector.residual_records) == 0:
        return
    worst = max(collector.residual_records, key=lambda r: abs(float(r["res_norm_inf"])))
    worst_res = float(worst["res_norm_inf"])
    if worst_res > float(max_residual_inf):
        raise RuntimeError(
            "Simulation rejected due to large nonlinear residual: "
            f"res_inf={worst_res:.6e} at t={worst['t']:.6e}, "
            f"step={worst['step']}, iter={worst['newton_iter']} "
            f"(limit={max_residual_inf:.6e})."
        )

    worst_state = 0.0
    worst_state_meta = None
    for rec in collector.residual_records:
        for row in rec.get("top", []):
            if row.get("kind", "") == "STATE":
                val = abs(float(row.get("res", 0.0)))
                if val > worst_state:
                    worst_state = val
                    worst_state_meta = (rec, row)
    if worst_state > float(max_state_residual_inf) and worst_state_meta is not None:
        rec, row = worst_state_meta
        raise RuntimeError(
            "Simulation rejected due to large state-equation residual: "
            f"abs_res={worst_state:.6e} at t={rec['t']:.6e}, "
            f"step={rec['step']}, iter={rec['newton_iter']}, "
            f"eq={row.get('eq_idx')}, var={row.get('var_name', '')} "
            f"(limit={max_state_residual_inf:.6e})."
        )


def plot_comparison(ref_trace: SwitchedConverterHandoverTrace, ml_trace: SwitchedConverterHandoverTrace) -> None:
    v_d_cap_ref_arr = np.clip(ref_trace.v_cmd_d_u_arr, -ref_trace.v_lim_arr, ref_trace.v_lim_arr)
    v_d_cap_ml_arr = ml_trace.v_d_cap_arr

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)

    axes[0, 0].plot(ref_trace.t_arr, ref_trace.i_a_arr, label="iA ref", linewidth=1.8)
    axes[0, 0].plot(ml_trace.t_arr, ml_trace.i_a_arr, "--", label="iA ml", linewidth=1.5)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title("Phase-A current")
    axes[0, 0].legend()

    axes[0, 1].plot(ref_trace.t_arr, ref_trace.v_dc_arr, label="vdc ref", linewidth=1.8)
    axes[0, 1].plot(ml_trace.t_arr, ml_trace.v_dc_arr, "--", label="vdc ml", linewidth=1.5)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title("DC voltage")
    axes[0, 1].legend()

    axes[1, 0].plot(ref_trace.t_arr, ref_trace.v_conv_a_arr, label="vconv_a ref", linewidth=1.8)
    axes[1, 0].plot(ml_trace.t_arr, ml_trace.v_conv_a_arr, "--", label="vconv_a ml", linewidth=1.5)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title("Converter phase-A voltage")
    axes[1, 0].legend()

    axes[1, 1].plot(ref_trace.t_arr, ref_trace.gate_a_arr, label="gate_a ref", linewidth=1.4)
    axes[1, 1].plot(ml_trace.t_arr, ml_trace.gate_a_arr, "--", label="gate_a ml", linewidth=1.2)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title("Gate-A mode")
    axes[1, 1].legend()

    axes[2, 0].plot(ref_trace.t_arr, ref_trace.P_f_arr, label="P_f ref", linewidth=1.8)
    axes[2, 0].plot(ml_trace.t_arr, ml_trace.P_f_arr, "--", label="P_f ml", linewidth=1.5)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_title("Filtered active power")
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 0].legend()

    axes[2, 1].plot(ref_trace.t_arr, ref_trace.Q_f_arr, label="Q_f ref", linewidth=1.8)
    axes[2, 1].plot(ml_trace.t_arr, ml_trace.Q_f_arr, "--", label="Q_f ml", linewidth=1.5)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_title("Filtered reactive power")
    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 1].legend()

    fig.suptitle("Switched converter handover: reference vs multilinear bridge", fontsize=12)
    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / "converter_switched_ml_compare.png"
    fig.savefig(out_path, dpi=170)
    print(f"Plot saved to: {out_path}")

    fig_aux, axes_aux = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes_aux[0, 0].plot(ref_trace.t_arr, v_d_cap_ref_arr, label="v_d_cap ref (derived)", linewidth=1.8)
    axes_aux[0, 0].plot(ml_trace.t_arr, v_d_cap_ml_arr, "--", label="v_d_cap ml", linewidth=1.5)
    axes_aux[0, 0].grid(True, alpha=0.3)
    axes_aux[0, 0].set_title("v_d_cap (applied d command)")
    axes_aux[0, 0].legend()

    axes_aux[0, 1].plot(ref_trace.t_arr, ref_trace.v_cmd_d_u_arr, label="v_cmd_d_u ref", linewidth=1.8)
    axes_aux[0, 1].plot(ref_trace.t_arr, ref_trace.v_lim_arr, "--", label="+v_lim ref", linewidth=1.4)
    axes_aux[0, 1].plot(ref_trace.t_arr, -ref_trace.v_lim_arr, "--", label="-v_lim ref", linewidth=1.4)
    axes_aux[0, 1].plot(ml_trace.t_arr, ml_trace.v_lim_arr, ":", label="+v_lim ml", linewidth=1.4)
    axes_aux[0, 1].plot(ml_trace.t_arr, -ml_trace.v_lim_arr, ":", label="-v_lim ml", linewidth=1.4)
    axes_aux[0, 1].grid(True, alpha=0.3)
    axes_aux[0, 1].set_title("Reference command with ref/ml v_lim")
    axes_aux[0, 1].legend()

    axes_aux[1, 0].plot(ml_trace.t_arr, ml_trace.v_cmd_d_u_arr, label="v_cmd_d_u ml", linewidth=1.8)
    axes_aux[1, 0].plot(ml_trace.t_arr, ml_trace.v_lim_arr, "--", label="+v_lim ml", linewidth=1.4)
    axes_aux[1, 0].plot(ml_trace.t_arr, -ml_trace.v_lim_arr, "--", label="-v_lim ml", linewidth=1.4)
    axes_aux[1, 0].grid(True, alpha=0.3)
    axes_aux[1, 0].set_title("ML: v_cmd_d_u vs +/-v_lim")
    axes_aux[1, 0].legend()

    axes_aux[1, 1].plot(ref_trace.t_arr, v_d_cap_ref_arr, label="v_d_cap ref (derived)", linewidth=1.7)
    axes_aux[1, 1].plot(ml_trace.t_arr, v_d_cap_ml_arr, label="v_d_cap ml", linewidth=1.7)
    axes_aux[1, 1].plot(ref_trace.t_arr, ref_trace.v_cmd_d_u_arr, "--", label="v_cmd_d_u ref", linewidth=1.3)
    axes_aux[1, 1].plot(ml_trace.t_arr, ml_trace.v_cmd_d_u_arr, "--", label="v_cmd_d_u ml", linewidth=1.3)
    axes_aux[1, 1].grid(True, alpha=0.3)
    axes_aux[1, 1].set_title("v_d_cap and v_cmd_d_u")
    axes_aux[1, 1].legend()

    plus_max_prod = ml_trace.ml_sat_u_plus1_max_arr * ml_trace.ml_sat_u_plus2_max_arr
    plus_min_prod = ml_trace.ml_sat_u_plus1_min_arr * ml_trace.ml_sat_u_plus2_min_arr
    v_d_cap_reconstructed = -ml_trace.v_lim_arr + plus_min_prod - plus_max_prod
    error_max = plus_max_prod - np.maximum(ml_trace.v_cmd_d_u_arr - ml_trace.v_lim_arr, 0.0)

    axes_aux[2, 0].plot(ml_trace.t_arr, -ml_trace.v_lim_arr, label="-v_lim", linewidth=1.3)
    axes_aux[2, 0].plot(ml_trace.t_arr, plus_min_prod, label="plus1m*plus2m", linewidth=1.3)
    axes_aux[2, 0].plot(ml_trace.t_arr, -plus_max_prod, label="-(plus1*plus2)", linewidth=1.3)
    axes_aux[2, 0].plot(ml_trace.t_arr, v_d_cap_reconstructed, label="reconstructed v_d_cap", linewidth=1.8)
    axes_aux[2, 0].plot(ml_trace.t_arr, v_d_cap_ml_arr, "--", label="reported v_d_cap", linewidth=1.4)
    axes_aux[2, 0].plot(ml_trace.t_arr, error_max , "--", label="reported error", linewidth=0.8)
    axes_aux[2, 0].plot(ml_trace.t_arr, ml_trace.v_lim_arr, "--", label="+v_lim ml", linewidth=1.4)
    axes_aux[2, 0].set_title("v_d_cap decomposition (ml_hard_sat)")
    axes_aux[2, 0].set_xlabel("Time [s]")
    axes_aux[2, 0].legend()

    minus_max_prod = ml_trace.ml_sat_u_minus1_max_arr * ml_trace.ml_sat_u_minus2_max_arr
    minus_min_prod = ml_trace.ml_sat_u_minus1_min_arr * ml_trace.ml_sat_u_minus2_min_arr
    sat_max_balance = (ml_trace.v_cmd_d_u_arr - ml_trace.v_lim_arr) - (plus_max_prod - minus_max_prod)
    sat_min_balance = (ml_trace.v_cmd_d_u_arr + ml_trace.v_lim_arr) - (plus_min_prod - minus_min_prod)
    sat_min_plus_part_error = plus_min_prod - np.maximum(ml_trace.v_cmd_d_u_arr + ml_trace.v_lim_arr, 0.0)

    axes_aux[2, 1].plot(ml_trace.t_arr, sat_max_balance, label="max balance residual", linewidth=1.5)
    axes_aux[2, 1].plot(ml_trace.t_arr, sat_min_balance, label="min balance residual", linewidth=1.5)
    axes_aux[2, 1].plot(ml_trace.t_arr, error_max, "--", label="max plus-part error", linewidth=1.3)
    axes_aux[2, 1].plot(ml_trace.t_arr, sat_min_plus_part_error, "--", label="min plus-part error", linewidth=1.3)
    axes_aux[2, 1].grid(True, alpha=0.3)
    axes_aux[2, 1].set_title("ml_hard_sat residuals")
    axes_aux[2, 1].set_xlabel("Time [s]")
    axes_aux[2, 1].legend()

    fig_aux.suptitle("ML auxiliary variable diagnostics", fontsize=12)
    fig_aux.tight_layout()
    out_aux_path = Path(__file__).resolve().parent / "converter_switched_ml_aux_compare.png"
    fig_aux.savefig(out_aux_path, dpi=170)
    print(f"Plot saved to: {out_aux_path}")
    plt.show()


def main() -> None:
    emt_options = build_emt_options()
    emt_options.tolerance = 1.0e-10
    emt_options.newton_max_iter = 60
    print(
        "Using tightened EMT nonlinear settings: "
        f"tolerance={emt_options.tolerance:.1e}, "
        f"newton_max_iter={emt_options.newton_max_iter}"
    )
    print("Running reference switched converter case...")
    ref_trace = simulate_switched_converter_handover(emt_options=emt_options)
    print("Running multilinear switched converter case...")
    ml_newton_trace = NewtonTraceCollector()
    ml_trace = simulate_switched_converter_handover(
        emt_options=emt_options,
        converter_builder=get_switched_emt_converter_multilinear,
        newton_trace_collector=ml_newton_trace,
        max_residual_inf_fail=float("inf"),
        max_state_residual_inf_fail=float("inf"),
    )
    print_sampled_comparison(ref_trace=ref_trace, ml_trace=ml_trace)
    print_ml_equation_diagnostics(ml_trace=ml_trace)
    print_newton_residual_diagnostics(collector=ml_newton_trace)
    plot_comparison(ref_trace=ref_trace, ml_trace=ml_trace)


if __name__ == "__main__":
    main()
