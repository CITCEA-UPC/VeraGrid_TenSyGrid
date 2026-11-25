"""Check angle tracking, phase advance and the shared soft-sign electrical plant."""
import numpy as np
import pytest

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Templates.Emt.bridge_2level_3ph_emt_multilinear_template import get_bridge_2level_3ph_emt_multilinear_template
from VeraGridEngine.Templates.Emt.bridge_filter_control_2level_3ph_emt_multilinear_template import get_bridge_filter_control_2level_3ph_emt_multilinear_template
from VeraGridEngine.Templates.Emt.bridge_2level_3ph_emt_soft_sign_template import get_bridge_2level_3ph_emt_soft_sign_template, get_switched_emt_converter_soft_sign
from VeraGridEngine.Utils.Symbolic.block import build_name_to_var_lookup


def test_trig_states_follow_actual_pll_frequency():
    block = get_bridge_filter_control_2level_3ph_emt_multilinear_template(VarFactory()).block
    angle = .37
    bindings = dict(omega_pll=327., omega_base_in=2*np.pi*50,
                    u_sin_pll=np.sin(angle), u_cos_pll=np.cos(angle))
    rates = {state.name: equation for state, equation in zip(block.state_vars, block.state_eqs)}
    assert rates['u_cos_pll'].eval(**bindings) == pytest.approx(-327*np.sin(angle))
    assert rates['u_sin_pll'].eval(**bindings) == pytest.approx(327*np.cos(angle))


@pytest.mark.parametrize('frequency', [0., 1000., 5000.])
def test_pwm_trig_rotation_tracks_enable_and_frequency(frequency):
    block = get_bridge_2level_3ph_emt_multilinear_template(VarFactory()).block
    angle = .37
    advance = np.pi*50/(2*frequency) if frequency else 0.
    bindings = dict(omega_base_in=2*np.pi*50, omega_sw_in=2*np.pi*frequency,
                    u_sin_pll=np.sin(angle), u_cos_pll=np.cos(angle),
                    u_sin_pwm=np.sin(angle+advance), u_cos_pwm=np.cos(angle+advance))
    for state, equation in zip(block.algebraic_vars, block.algebraic_eqs):
        if state.name in ('u_cos_pwm', 'u_sin_pwm'):
            assert abs(equation.eval(**bindings)) < 1e-12


def test_soft_sign_gates_drive_the_same_floating_neutral_bridge():
    block = get_bridge_2level_3ph_emt_soft_sign_template(VarFactory()).block
    variables = build_name_to_var_lookup(block)
    gates = np.array([.05, .4, .95])
    bindings = dict(gate_a=gates[0], gate_b=gates[1], gate_c=gates[2],
                    k_v_conv_in=.7, v_dc_in=1.1, vdc_floor_in=.05)
    for phase, gate in zip('abc', gates):
        voltage = 2*.7*1.1*(gate - np.mean(gates))
        bindings['v_conv_' + phase] = voltage
        index = block.algebraic_vars.index(variables['v_conv_' + phase])
        assert abs(block.algebraic_eqs[index].eval(**bindings)) < 1e-12
    assert len(block.in_vars) == 11
    assert [v.name for v in block.out_vars[:3]] == ['gate_a', 'gate_b', 'gate_c']


def test_soft_sign_is_symmetric_and_approaches_hard_switching():
    block = get_bridge_2level_3ph_emt_soft_sign_template(VarFactory()).block
    variables = build_name_to_var_lookup(block)
    row = block.algebraic_vars.index(variables['gate_a'])
    equation = block.algebraic_eqs[row]
    # With gate_a=0, minus the residual is the soft comparator's gate output.
    def gate(margin, width):
        return -equation.eval(gate_a=0., m_a_sample=margin, carrier_pwm=0., soft_sign_lambda=width)
    assert gate(0., 1e-6) == pytest.approx(.5)
    assert gate(.1, 1e-6) + gate(-.1, 1e-6) == pytest.approx(1.)
    assert 0 < gate(.1, 1e-6) < 1
    assert 1-gate(.1, 1e-8) < 1-gate(.1, 1e-6) < 1-gate(.1, 1e-4)


def test_soft_sign_full_converter_carrier_uses_absolute_clock():
    block = get_switched_emt_converter_soft_sign(VarFactory()).block
    expressions = [expr for owner in block.get_all_blocks()
                   for var, expr in owner.event_dict.items() if var.name == 'carrier_pwm']
    assert len(expressions) == 1
    expression = expressions[0]
    assert 'omega_sw_eff' not in {var.name for var in expression.get_vars()}
    for time, expected in [(0., -1.), (50e-6, 0.), (100e-6, 1.)]:
        assert expression.eval(omega_sw=2*np.pi*5000, glob_time=time,
                               carrier_phase=-np.pi/2) == pytest.approx(expected)
