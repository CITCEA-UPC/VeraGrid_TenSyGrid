# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from VeraGridEngine.Devices.Dynamic.emt_template import EmtModelTemplate
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Templates.Emt.bridge_filter_2level_3ph_emt_mti_template import (
    get_bridge_filter_2level_3ph_emt_mti_template,
)
from VeraGridEngine.Utils.Symbolic.block import find_name_in_block
import VeraGridEngine.Templates.Emt.bridge_filter_control_2level_3ph_emt_template as control_ref


def get_bridge_filter_control_2level_3ph_emt_mti_template(
        vf: VarFactory,
        name: str = "bridge_filter_control_2level_3ph_emt_mti",
        current_from_power_scale: float = 2.0 / 3.0,
) -> EmtModelTemplate:
    """Build the standard bridge/filter/control stack with an MTI bridge plant."""
    original_builder = control_ref.get_bridge_filter_2level_3ph_emt_template
    control_ref.get_bridge_filter_2level_3ph_emt_template = get_bridge_filter_2level_3ph_emt_mti_template
    try:
        templ = control_ref.get_bridge_filter_control_2level_3ph_emt_template(
            vf=vf, name=name, current_from_power_scale=current_from_power_scale,
        )
    finally:
        control_ref.get_bridge_filter_2level_3ph_emt_template = original_builder

    # omega_meas_in is the real PWM clock used by the measurement filters,
    # not the PLL's measured electrical frequency. The carrier must keep that
    # clock before the parent enables the switched plant; otherwise MTI starts
    # a fresh ramp at handover while procedural PWM uses absolute carrier time.
    omega_enabled = find_name_in_block("omega_sw_in", templ.block)
    omega_clock = find_name_in_block("omega_meas_in", templ.block)
    carrier = find_name_in_block("carrier_pwm", templ.block)
    if omega_enabled is None or omega_clock is None or carrier is None:
        raise KeyError("MTI bridge is missing its carrier clock inputs")
    for owner in templ.block.get_all_blocks():
        for index, state in enumerate(owner.state_vars):
            if state.uid == carrier.uid:
                owner.state_eqs[index] = owner.state_eqs[index].subs({omega_enabled: omega_clock})
    return templ
