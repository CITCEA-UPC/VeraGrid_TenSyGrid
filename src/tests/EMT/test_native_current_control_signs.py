"""Verify the unsaturated current controller against the independent abc plant."""
import numpy as np

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Templates.Emt.bridge_filter_control_2level_3ph_emt_template import _build_bridge_filter_current_control_block


def test_current_control_decouples_phase_plant_with_negative_feedback():
    block = _build_bridge_filter_current_control_block(VarFactory(), 'check', current_from_power_scale=2.)
    equations = {v.name: e for v, e in zip(block.algebraic_vars, block.algebraic_eqs)}
    rng = np.random.default_rng(7)
    for _ in range(100):
        theta, omega = rng.uniform(-np.pi, np.pi), rng.uniform(300, 330)
        omega_base, resistance, reactance = 2*np.pi*50, .02, .08
        id_, iq, vd, vq, pi_d, pi_q = rng.normal(size=6)
        command_d = vd-resistance*id_-(omega/omega_base)*reactance*iq-pi_d
        command_q = vq-resistance*iq+(omega/omega_base)*reactance*id_-pi_q
        bindings = dict(v_d_inner_in=vd, v_q_inner_in=vq, i_d_inner_in=id_, i_q_inner_in=iq,
                        R_eq_inner_in=resistance, L_eq_inner_in=reactance,
                        omega_pll_inner_in=omega, omega_base_inner_in=omega_base,
                        v_pi_d_u=pi_d, v_pi_q_u=pi_q,
                        v_cmd_d_u=command_d, v_cmd_q_u=command_q)
        for name in ('v_cmd_d_u', 'v_cmd_q_u'):
            assert abs(equations[name].eval(**bindings)) < 1e-9
        angles = theta+np.array([0., -2*np.pi/3, 2*np.pi/3])
        sine, cosine = np.sin(angles), np.cos(angles)
        phase_current = sine*id_-cosine*iq
        phase_voltage = sine*vd-cosine*vq
        bridge_voltage = sine*command_d-cosine*command_q
        phase_derivative = omega_base/reactance*(phase_voltage-bridge_voltage-resistance*phase_current)
        # Differentiate the rotating Park basis independently of the controller.
        did = 2/3*np.dot(sine, phase_derivative)+omega*2/3*np.dot(cosine, phase_current)
        diq = -2/3*np.dot(cosine, phase_derivative)+omega*2/3*np.dot(sine, phase_current)
        np.testing.assert_allclose([did, diq], omega_base/reactance*np.array([pi_d, pi_q]), atol=1e-9)
        # Phase voltages/currents are peaks; system-base total power uses 1/3.
        np.testing.assert_allclose(np.dot(phase_voltage, phase_current)/3, .5*(vd*id_+vq*iq), atol=1e-12)
