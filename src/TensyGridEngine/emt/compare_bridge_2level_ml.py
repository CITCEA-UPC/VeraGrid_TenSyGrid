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

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Templates.Emt.bridge_2level_3ph_emt_template import get_bridge_2level_3ph_emt_template
from VeraGridEngine.Templates.Emt.bridge_2level_3ph_emt_multilinear_template import get_bridge_2level_3ph_emt_multilinear_template


def compute_raw_pwm_refs_direct(theta_pwm: np.ndarray, v_cmd_d: float, v_cmd_q: float, v_cmd_0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_b = theta_pwm - 2.0 * np.pi / 3.0
    theta_c = theta_pwm + 2.0 * np.pi / 3.0
    a = v_cmd_d * np.sin(theta_pwm) - v_cmd_q * np.cos(theta_pwm) + v_cmd_0
    b = v_cmd_d * np.sin(theta_b) - v_cmd_q * np.cos(theta_b) + v_cmd_0
    c = v_cmd_d * np.sin(theta_c) - v_cmd_q * np.cos(theta_c) + v_cmd_0
    return a, b, c


def compute_raw_pwm_refs_ml(theta_pwm: np.ndarray, v_cmd_d: float, v_cmd_q: float, v_cmd_0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_cos = np.cos(theta_pwm)
    u_sin = np.sin(theta_pwm)
    c120 = np.cos(2.0 * np.pi / 3.0)
    s120 = np.sin(2.0 * np.pi / 3.0)
    sin_m120 = u_sin * c120 - u_cos * s120
    cos_m120 = u_cos * c120 + u_sin * s120
    sin_p120 = u_sin * c120 + u_cos * s120
    cos_p120 = u_cos * c120 - u_sin * s120
    a = v_cmd_d * u_sin - v_cmd_q * u_cos + v_cmd_0
    b = v_cmd_d * sin_m120 - v_cmd_q * cos_m120 + v_cmd_0
    c = v_cmd_d * sin_p120 - v_cmd_q * cos_p120 + v_cmd_0
    return a, b, c


def main() -> None:
    vf_ref = VarFactory()
    vf_ml = VarFactory()
    bridge_ref = get_bridge_2level_3ph_emt_template(vf=vf_ref, name="bridge_ref").block
    bridge_ml = get_bridge_2level_3ph_emt_multilinear_template(vf=vf_ml, name="bridge_ml").block
    print("Model summary")
    print(f"reference: state_vars={len(bridge_ref.state_vars)}, algebraic_vars={len(bridge_ref.algebraic_vars)}, procedural_logic={len(bridge_ref.procedural_logic)}")
    print(f"ml trig  : state_vars={len(bridge_ml.state_vars)}, algebraic_vars={len(bridge_ml.algebraic_vars)}, procedural_logic={len(bridge_ml.procedural_logic)}")

    omega_base = 2.0 * np.pi * 50.0
    omega_sw = 2.0 * np.pi * 1000.0
    t_arr = np.linspace(0.0, 0.02, 400)
    theta_pll = omega_base * t_arr
    theta_pwm = theta_pll + omega_base * np.pi / (2.0 * omega_sw)

    v_cmd_d = 0.15
    v_cmd_q = 0.82
    v_cmd_0 = 0.0

    a_ref, b_ref, c_ref = compute_raw_pwm_refs_direct(theta_pwm, v_cmd_d, v_cmd_q, v_cmd_0)
    a_ml, b_ml, c_ml = compute_raw_pwm_refs_ml(theta_pwm, v_cmd_d, v_cmd_q, v_cmd_0)

    print("max|a_ref-a_ml|:", float(np.max(np.abs(a_ref - a_ml))))
    print("max|b_ref-b_ml|:", float(np.max(np.abs(b_ref - b_ml))))
    print("max|c_ref-c_ml|:", float(np.max(np.abs(c_ref - c_ml))))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t_arr, a_ref, label="a direct", linewidth=2.0)
    axes[0].plot(t_arr, a_ml, "--", label="a ml", linewidth=1.6)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(t_arr, b_ref, label="b direct", linewidth=2.0)
    axes[1].plot(t_arr, b_ml, "--", label="b ml", linewidth=1.6)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[2].plot(t_arr, c_ref, label="c direct", linewidth=2.0)
    axes[2].plot(t_arr, c_ml, "--", label="c ml", linewidth=1.6)
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.suptitle("Bridge PWM raw-reference direct vs ML trig identities")
    fig.tight_layout()

    out_path = Path(__file__).resolve().parent / "bridge_2level_pwm_ref_compare.png"
    fig.savefig(out_path, dpi=170)
    print(f"Plot saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
