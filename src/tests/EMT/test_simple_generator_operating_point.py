"""Physical PF operating-point checks for the simple generator used by EMT demos."""

import numpy as np
import pytest

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Templates.Emt.generator_emt_type_template import get_simple_generator_emt_template
from VeraGridEngine.Utils.Symbolic.symbolic import Const
from VeraGridEngine.enumerations import VarPowerFlowReferenceType


@pytest.mark.parametrize('angle', [0.3, 2.2])
@pytest.mark.parametrize('reactive_power', [-0.2, 0.2])
def test_simple_generator_pf_seed_satisfies_flux_power_and_steady_state(angle, reactive_power):
    block = get_simple_generator_emt_template(VarFactory()).block
    all_vars = (block.state_vars + block.algebraic_vars + block.in_vars
                + list(block.event_dict) + list(block.parameters))
    names = {var.name: var for var in all_vars}
    bindings = {var.uid: float(expr.value)
                for var, expr in list(block.event_dict.items()) + list(block.parameters.items())
                if isinstance(expr, Const) and expr.value is not None}
    omega_base, resistance = 2 * np.pi * 50, 0.002
    for name, value in {'omega_base': omega_base, 'Ra': resistance, 'Ld': 1.8,
                        'L0': 0.15, 'R0': 0.001}.items():
        bindings[names[name].uid] = value
    total_power = 0.8 + 1j * reactive_power
    rotation = np.exp(-2j * np.pi * np.arange(3) / 3)
    voltage = 1.04 * np.exp(1j * angle) * rotation
    current = np.conj(total_power / voltage[0]) * rotation
    for phase, v, i in zip('ABC', voltage, current):
        voltage_var = block.in_vars['ABC'.index(phase)]
        bindings[voltage_var.uid] = np.sqrt(2) * v.imag
        for label, value in [(f'i_{phase}', np.sqrt(2) * i.imag),
                             (f'd_v_{phase}', omega_base * np.sqrt(2) * v.real),
                             (f'P_{phase}', total_power.real / 3),
                             (f'Q_{phase}', total_power.imag / 3)]:
            ref = getattr(VarPowerFlowReferenceType, label)
            bindings[block.external_mapping[ref].uid] = value

    assignments = list(block.init_eqs.items()) + list(block.event_dict.items())
    for _ in range(len(assignments)):
        progress = False
        for var, expr in assignments:
            if var.uid in bindings or isinstance(expr, Const) and expr.value is None:
                continue
            if all(dependency.uid in bindings for dependency in expr.get_vars()):
                bindings[var.uid] = float(expr.eval_uid(bindings))
                progress = True
        if not progress:
            break
    assert all(var.uid in bindings for var in block.state_vars + block.algebraic_vars + block.in_vars)
    residual = np.array([eq.eval_uid(bindings) for eq in block.algebraic_eqs])
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)
    assert bindings[names['Pe'].uid] == pytest.approx(total_power.real)
    assert bindings[names['Qe'].uid] == pytest.approx(total_power.imag)
    assert bindings[names['Te'].uid] == pytest.approx(total_power.real + resistance * abs(current[0])**2)
    rhs = np.array([eq.eval_uid(bindings) for eq in block.state_eqs])
    expected = np.zeros(len(rhs))
    expected[4] = omega_base
    np.testing.assert_allclose(rhs, expected, atol=1e-10)
    bindings[block.in_vars[4].uid] += 0.01
    assert block.state_eqs[3].eval_uid(bindings) == pytest.approx(omega_base * 0.01)


def test_simple_generator_stator_resistance_dissipates_flux_at_standstill():
    block = get_simple_generator_emt_template(VarFactory()).block
    names = {var.name: var for var in block.state_vars + block.algebraic_vars
             + list(block.event_dict) + list(block.parameters)}
    omega_base, resistance, q_inductance, zero_inductance = 2*np.pi*50, 0.02, 1.7, 0.15
    values = {'omega_base': omega_base, 'Ra': resistance, 'R0': resistance,
              'omega': 0.0, 'psi_d': 0.0, 'psi_q': 0.4,
              'i_q': -0.4/q_inductance, 'i_0': -0.2/zero_inductance,
              'v_q': 0.0, 'v_0': 0.0}
    bindings = {names[name].uid: value for name, value in values.items()}
    q_slope = block.state_eqs[1].eval_uid(bindings)
    zero_slope = block.state_eqs[2].eval_uid(bindings)
    assert q_slope == pytest.approx(-omega_base*resistance*0.4/q_inductance)
    assert zero_slope == pytest.approx(-omega_base*resistance*0.2/zero_inductance)
    assert 0.4*q_slope < 0 and 0.2*zero_slope < 0
