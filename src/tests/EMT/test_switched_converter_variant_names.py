"""Check the local symbolic names used when composing switched converter variants."""
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Templates.Emt.bridge_2level_3ph_emt_mti_template import get_bridge_2level_3ph_emt_mti_template
from VeraGridEngine.Templates.Emt.bridge_filter_2level_3ph_emt_multilinear_template import get_bridge_filter_2level_3ph_emt_multilinear_template
from VeraGridEngine.Utils.Symbolic.block import find_name_in_block


def test_mti_bridge_exposes_parent_lookup_names():
    block = get_bridge_2level_3ph_emt_mti_template(VarFactory(), name='named_bridge').block
    for name in ('gate_a', 'gate_b', 'gate_c', 'v_conv_a', 'm_a', 'omega_sw_in'):
        assert find_name_in_block(name, block) is not None, name


def test_multilinear_filter_resolves_child_trig_states():
    block = get_bridge_filter_2level_3ph_emt_multilinear_template(VarFactory(), name='named_filter').block
    assert find_name_in_block('u_cos_pll', block) is not None
    assert find_name_in_block('u_sin_pll', block) is not None


def test_rms_compiler_preserves_supplied_time_step_symbol():
    from VeraGridEngine.Utils.Symbolic.jit_compiler import RMSCompiler
    from VeraGridEngine.Utils.Symbolic.symbolic import Var
    step = Var(name='custom_step')
    names = {}
    compiler = RMSCompiler([], [], [], [], names, dt_var=step)
    assert compiler.dt_var is step
    assert names[step.uid] == 'h'
    assert RMSCompiler([], [], [], [], {}).dt_var is not step


def test_mti_control_keeps_switching_and_pll_frequency_inputs_separate():
    from VeraGridEngine.Templates.Emt.bridge_filter_control_2level_3ph_emt_mti_template import (
        get_bridge_filter_control_2level_3ph_emt_mti_template,
    )
    block = get_bridge_filter_control_2level_3ph_emt_mti_template(VarFactory(), name='control').block
    inputs = {v.name: v for v in block.in_vars}
    assert inputs['omega_sw_in'].uid != inputs['omega_meas_in'].uid
