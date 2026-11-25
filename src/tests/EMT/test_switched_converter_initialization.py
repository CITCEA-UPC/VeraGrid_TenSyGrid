"""Verify the full converter starts at a consistent power-flow operating point."""
import numpy as np
import pytest

from TensyGridEngine.emt.validate_converter_emt_multilinear import build_converter_case, build_emt_options
from VeraGridEngine.Templates.Emt.converter_switched_emt_template import get_switched_emt_converter
from VeraGridEngine.Templates.Emt.converter_switched_emt_multilinear_template import get_switched_emt_converter_multilinear
from VeraGridEngine.Templates.Emt.converter_switched_emt_mti_template import get_switched_emt_converter_mti
from VeraGridEngine.Templates.Emt.bridge_2level_3ph_emt_soft_sign_template import get_switched_emt_converter_soft_sign
from VeraGridEngine.Utils.Symbolic.block import find_name_in_block
from VeraGridEngine.enumerations import VarPowerFlowReferenceType


@pytest.mark.parametrize('builder', [get_switched_emt_converter, get_switched_emt_converter_multilinear, get_switched_emt_converter_mti, get_switched_emt_converter_soft_sign])
def test_switched_converter_pf_seed_is_consistent(builder):
    """Check actual grid DAE residuals, power signs and abc/dq current consistency."""
    problem, model = build_converter_case('initialization_regression', builder, build_emt_options(),
                                          use_mti=builder is get_switched_emt_converter_mti)
    x = problem.get_x0()
    dx = problem.get_dx0()
    runtime = problem.event_params_values
    constants = problem.get_parameters_values()
    bindings = {uid: float(x[idx]) for uid, idx in problem.uid2idx_vars.items()}
    bindings.update({uid: float(dx[idx]) for uid, idx in problem.uid2idx_diff.items()})
    bindings.update({uid: float(runtime[idx]) for uid, idx in problem.uid2idx_event_params.items()})
    bindings.update({uid: float(constants[idx].value) for uid, idx in problem.uid2idx_params.items()})
    bindings[problem.glob_time.uid] = 0.
    residual = np.array([eq.eval_uid(bindings) for eq in problem.get_algebraic_eqs()])
    assert np.isfinite(residual).all()
    worst = int(np.argmax(np.abs(residual)))
    assert abs(residual[worst]) < 1e-7, str(problem.get_algebraic_eqs()[worst])
    assert problem.initialization_report.final_residual_inf < 1e-7
    p_idx = problem.get_var_idx(model.external_mapping[VarPowerFlowReferenceType.P])
    pf_idx = problem.get_var_idx(find_name_in_block('P_f', model))
    assert x[p_idx] > 0.
    assert x[pf_idx] == pytest.approx(x[p_idx], abs=1e-9)
    phase_currents = np.array([x[problem.get_var_idx(model.external_mapping[key])] for key in (
        VarPowerFlowReferenceType.i_A, VarPowerFlowReferenceType.i_B, VarPowerFlowReferenceType.i_C)])
    theta = x[problem.get_var_idx(find_name_in_block('theta_pll', model))]
    shifts = theta + np.array([0., -2*np.pi/3, 2*np.pi/3])
    id_expected = 2/3 * np.dot(np.sin(shifts), phase_currents)
    iq_expected = -2/3 * np.dot(np.cos(shifts), phase_currents)
    assert x[problem.get_var_idx(find_name_in_block('i_d', model))] == pytest.approx(id_expected, abs=1e-8)
    assert x[problem.get_var_idx(find_name_in_block('i_q', model))] == pytest.approx(iq_expected, abs=1e-8)
    if builder is get_switched_emt_converter_soft_sign:
        carrier = next(var for var in problem.get_runtime_continuous_parameters()
                       if var.name == 'carrier_pwm')
        carrier_index = problem.uid2idx_event_params[carrier.uid]
        for time, expected in [(0., -1.), (50e-6, 0.), (100e-6, 1.)]:
            problem.update_variable_params(time)
            assert problem.event_params_values[carrier_index] == pytest.approx(expected)
        problem.enable_compiled_runtime_parameter_evaluator()
        for time, expected in [(0., -1.), (50e-6, 0.), (100e-6, 1.)]:
            problem.update_variable_params(time)
            assert problem.event_params_values[carrier_index] == pytest.approx(expected)
