# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""A soft-sign comparator using the reference bridge's electrical equations."""
from VeraGridEngine.Devices.Dynamic.emt_template import EmtModelTemplate
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Simulations.EMT.problems.emt_problem_template import EmtProblemTemplate
from VeraGridEngine.Templates.Emt.bridge_2level_3ph_emt_template import get_bridge_2level_3ph_emt_template
from VeraGridEngine.Templates.Emt.converter_switched_emt_template import get_switched_emt_converter
from VeraGridEngine.Utils.Symbolic.block import build_name_to_var_lookup
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Var
import VeraGridEngine.Utils.Symbolic.symbolic as sym
from VeraGridEngine.Utils.procedural_logic import ThreePhaseCarrierSampledModulationLogic


def get_bridge_2level_3ph_emt_soft_sign_template(
        vf: VarFactory, name: str = 'bridge_soft_sign', regularization: float = 1e-6,
) -> EmtModelTemplate:
    """Replace only the hard comparator, preserving plant equations and held inputs.

    :param vf: Shared variable factory.
    :param name: Bridge name.
    :param regularization: Positive squared-width parameter in the soft-sign law.
    :return: A bridge with the same input/output ports as procedural PWM.
    """
    if regularization <= 0:
        raise ValueError('soft-sign regularization must be positive')
    template = get_bridge_2level_3ph_emt_template(vf, name)
    block = template.block
    variables = build_name_to_var_lookup(block)
    gates = [variables['gate_' + phase] for phase in 'abc']
    modes = [variables['gate_' + phase + '_mode'] for phase in 'abc']
    references = [variables['m_' + phase] for phase in 'abc']
    samples = [vf.add_var('m_' + phase + '_sample') for phase in 'abc']
    rows = [block.algebraic_vars.index(gate) for gate in gates]
    for mode in modes:
        del block.mode_dict[mode]
    block.procedural_logic = []
    block.update_model_bulk(dict(zip(modes, gates)))
    omega = variables['omega_sw_in']
    phase = variables['carrier_phase_in']
    carrier = vf.add_var('carrier_pwm')
    width = vf.add_var('soft_sign_lambda')
    time = Var(EmtProblemTemplate.TIME_NAME)
    block.event_dict[carrier] = Const(2 / 3.141592653589793) * sym.asin(sym.sin(omega * time + phase))
    block.event_dict[width] = Const(regularization)
    for row, gate, sample, reference in zip(rows, gates, samples, references):
        margin = sample - carrier
        expression = Const(.5) * (Const(1.) + margin / sym.sqrt(margin * margin + width))
        block.algebraic_eqs[row] = gate - expression
        block.init_eqs[gate] = expression
        block.mode_dict[sample] = reference
    block.procedural_logic = [ThreePhaseCarrierSampledModulationLogic(
        references[0].name, references[1].name, references[2].name,
        samples[0].name, samples[1].name, samples[2].name,
        omega.name, phase.name, name='soft_sign_modulation_sampler')]
    return template


def get_switched_emt_converter_soft_sign(
        vf: VarFactory, name: str = 'switched_converter_soft_sign',
) -> EmtModelTemplate:
    """Build the reference full converter with only its comparator smoothed.

    :param vf: Shared variable factory.
    :param name: Converter name.
    :return: Full converter with the shared control/filter/DC-link stack.
    """
    template = get_switched_emt_converter(vf, name, bridge_builder=get_bridge_2level_3ph_emt_soft_sign_template)
    variables = build_name_to_var_lookup(template.block)
    # Use the absolute carrier clock through startup, as the hard scheduler
    # does. The plant's enable/blend controls the handover independently.
    for owner in template.block.get_all_blocks():
        for parameter, expression in owner.event_dict.items():
            if parameter.name == 'carrier_pwm':
                owner.event_dict[parameter] = expression.subs({variables['omega_sw_eff']: variables['omega_sw']})
    return template
