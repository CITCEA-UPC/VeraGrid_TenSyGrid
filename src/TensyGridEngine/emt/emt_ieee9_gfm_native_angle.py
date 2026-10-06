"""Isolated IEEE9 GFM test using the model's native positive angle convention."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from TensyGridEngine.emt import ieee9_emt_ibr_simulation as base


def _install_positive_angle_seed(validation, original_seed):
    def seed(problem, grid, pf_results, block, bus):
        # Populate every ancillary variable first, then replace the electrical
        # operating point with one derived for theta_dot = +omega_base*omega.
        original_seed(problem, grid, pf_results, block, bus)
        bus_index = grid.buses.index(bus)
        voltage = complex(pf_results.voltage[bus_index])
        peak = float(np.sqrt(2.0) * abs(voltage))
        angle = float(np.angle(voltage))
        p0 = float(np.real(pf_results.Sbus[bus_index]) / grid.Sbase)
        q0 = float(np.imag(pf_results.Sbus[bus_index]) / grid.Sbase)
        va = peak * np.sin(angle)
        vb = peak * np.sin(angle - 2.0 * np.pi / 3.0)
        vc = peak * np.sin(angle + 2.0 * np.pi / 3.0)

        def abc_from_dq(d, q, theta):
            c, s, root3 = np.cos(theta), np.sin(theta), np.sqrt(3.0)
            return np.asarray((
                d * c + q * s,
                d * (-0.5 * c - 0.5 * root3 * s) + q * (-0.5 * s + 0.5 * root3 * c),
                d * (-0.5 * c + 0.5 * root3 * s) + q * (-0.5 * s - 0.5 * root3 * c),
            ))

        def powers(d, q, theta):
            ia, ib, ic = abc_from_dq(d, q, theta)
            return np.asarray((
                (va * ia + vb * ib + vc * ic) / 3.0,
                ((vb - vc) * ia + (vc - va) * ib + (va - vb) * ic) / (3.0 * np.sqrt(3.0)),
            ))

        theta = angle
        transform = np.column_stack((powers(1.0, 0.0, theta), powers(0.0, 1.0, theta)))
        id_g, iq_g = np.linalg.solve(transform, np.asarray((p0, q0)))
        rc, lc, rf, lf, cf, rcap = 0.01, 0.10, 0.02, 0.15, 0.05, 1.0e6
        vd_g, vq_g = 0.0, peak
        vd_f = vd_g + rc * id_g + lc * iq_g
        vq_f = vq_g + rc * iq_g - lc * id_g
        id_c = id_g + cf * vq_f + vd_f / rcap
        iq_c = iq_g - cf * vd_f + vq_f / rcap
        vd_c = vd_f + rf * id_c + lf * iq_c
        vq_c = vq_f + rf * iq_c - lf * id_c

        # Align the q axis with the capacitor voltage, as required by vd_ref=0.
        delta = float(np.arctan2(vd_f, vq_f))
        theta += delta

        def rotate(d, q):
            c, s = np.cos(delta), np.sin(delta)
            return float(d * c - q * s), float(d * s + q * c)

        vd_g, vq_g = rotate(vd_g, vq_g)
        id_g, iq_g = rotate(id_g, iq_g)
        vd_f, vq_f = rotate(vd_f, vq_f)
        id_c, iq_c = rotate(id_c, iq_c)
        vd_c, vq_c = rotate(vd_c, vq_c)
        i_g_abc = abc_from_dq(id_g, iq_g, theta)
        i_c_abc = abc_from_dq(id_c, iq_c, theta)
        v_f_abc = abc_from_dq(vd_f, vq_f, theta)
        v_c_abc = abc_from_dq(vd_c, vq_c, theta)
        kp, ki = 0.00075, 0.2
        z_vd = (id_c - id_g - cf * vq_f - kp * (0.0 - vd_f)) / ki
        z_vq = (iq_c - iq_g + cf * vd_f - kp * (vq_f - vq_f)) / ki
        vd_ctrl = vd_c - vd_f - lf * iq_c
        vq_ctrl = vq_c - vq_f + lf * id_c
        p_conv = 0.5 * (vq_c * iq_c + vd_c * id_c)
        values = {
            "theta": theta, "omega": 1.0, "P": p0, "Q": q0,
            "P_ref": p0, "Q_ref": q0, "y_p_lp": p0, "y_q_lp": q0,
            "V_ref": vq_f, "V": vq_f, "vd_ref": 0.0, "vq_ref": vq_f,
            "vd_g": vd_g, "vq_g": vq_g, "id_g": id_g, "iq_g": iq_g,
            "vd_f": vd_f, "vq_f": vq_f, "id_c": id_c, "iq_c": iq_c,
            "vd_c": vd_c, "vq_c": vq_c, "vd_c_ref": vd_c, "vq_c_ref": vq_c,
            "id_ref": id_c, "iq_ref": iq_c, "id_ref_sat": id_c, "iq_ref_sat": iq_c,
            "z_vd_loop": z_vd, "z_vq_loop": z_vq,
            "z_id_loop": vd_ctrl / ki, "z_iq_loop": vq_ctrl / ki,
            "vd_ctrl_out": vd_ctrl, "vq_ctrl_out": vq_ctrl,
            "Pt_vsc": -p0, "Qt_vsc": -q0, "Pf_vsc": -p_conv, "Qf_vsc": 0.0,
        }
        for phase, value in zip("ABC", i_g_abc):
            values[f"i_{phase}"] = value
            values[f"i_g_{phase}"] = value
        for phase, value in zip("ABC", i_c_abc):
            values[f"i_c_{phase}"] = value
        for phase, value in zip("ABC", v_f_abc):
            values[f"v_f_{phase}"] = value
        for phase, value in zip("ABC", v_c_abc):
            values[f"{phase}_v_c"] = value
        for name, value in values.items():
            validation.set_runtime_value(problem, block, name, float(value))

    return seed


def main() -> None:
    original_loader = base._load_deliverable_models

    def loader():
        gfl, gfm, validation = original_loader()
        validation.seed_gfm_from_power_flow = _install_positive_angle_seed(
            validation, validation.seed_gfm_from_power_flow
        )
        return gfl, gfm, validation

    base._load_deliverable_models = loader
    base._adapt_gfm_rotation_convention = lambda _block: None
    problem, time, values, _, _, gfm_buses = base.run_case(0, 1, multilinear_inverters=False)
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for generator in problem.grid.generators:
        if generator.bus.name not in gfm_buses:
            continue
        for name, axis in zip(("omega", "P", "Q"), axes):
            signal = base._get_signal(problem, values, generator.emt_model, name)
            if signal is not None:
                axis.plot(1e3 * time, signal)
    for axis, label in zip(axes, ("omega [p.u.]", "P [p.u.]", "Q [p.u.]")):
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("Time [ms]")
    axes[0].set_title("IEEE9 GFM: native positive-angle convention")
    output = Path(__file__).with_name("ieee9_emt_gfm_native_angle.png")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"plot={output}")


if __name__ == "__main__":
    main()
