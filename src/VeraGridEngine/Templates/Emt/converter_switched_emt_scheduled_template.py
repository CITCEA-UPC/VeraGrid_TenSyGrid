# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Shared switched converter with scheduled, multilinear PWM gate updates."""
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Devices.Dynamic.emt_template import EmtModelTemplate
from VeraGridEngine.Templates.Emt.bridge_2level_3ph_emt_template import get_bridge_2level_3ph_emt_template
from VeraGridEngine.Templates.Emt.converter_switched_emt_template import get_switched_emt_converter
from VeraGridEngine.Utils.procedural_logic import ScheduledMultilinearPwmLogic


def get_bridge_2level_3ph_emt_scheduled_template(
        vf: VarFactory, name: str = 'scheduled_pwm_bridge') -> EmtModelTemplate:
    template = get_bridge_2level_3ph_emt_template(vf, name)
    # Remapping reconstructs the same declarative ports; no shared builders or
    # global functions are patched. Runtime cloning preserves the new type.
    template.block.procedural_logic = [ScheduledMultilinearPwmLogic(
        mod_a_var_name=entry.mod_a_var_name, mod_b_var_name=entry.mod_b_var_name,
        mod_c_var_name=entry.mod_c_var_name, gate_a_mode_var_name=entry.gate_a_mode_var_name,
        gate_b_mode_var_name=entry.gate_b_mode_var_name, gate_c_mode_var_name=entry.gate_c_mode_var_name,
        omega_sw_var_name=entry.omega_sw_var_name, carrier_phase_var_name=entry.carrier_phase_var_name,
        name='scheduled_multilinear_pwm') for entry in template.block.procedural_logic]
    return template


def get_switched_emt_converter_scheduled(
        vf: VarFactory, name: str = 'switched_converter_scheduled') -> EmtModelTemplate:
    return get_switched_emt_converter(vf, name, bridge_builder=get_bridge_2level_3ph_emt_scheduled_template)
