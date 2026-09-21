# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import Dict

from VeraGridEngine.Compilers.Gslv.activation import (
    build_status_dict,
    converter_control_type_dict,
    pg,
)
from VeraGridEngine.Compilers.Gslv.common import fill_profile
from VeraGridEngine.Devices.Branches.vsc import VSC
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Devices.Parents.branch_parent import BranchParent
from VeraGridEngine.Devices.Substation.bus import Bus
from VeraGridEngine.basic_structures import IntVec
from VeraGridEngine.enumerations import ConverterFaultControlType


def convert_vsc_bus_reference(elm: Bus | BranchParent | None, bus_dict: Dict[str, "pg.Bus"]) -> "pg.Bus | None":
    """
    Convert an optional VeraGrid bus reference to a GSLV bus reference.

    :param elm: VeraGrid bus reference.
    :param bus_dict: GSLV bus lookup by VeraGrid idtag.
    :return: Matching GSLV bus or ``None``.
    """
    if isinstance(elm, Bus):
        return bus_dict[elm.idtag]
    else:
        return None


def convert_vsc(elm: VSC,
                bus_dict: Dict[str, "pg.Bus"],
                n_time: int,
                use_time_series: bool,
                time_indices: IntVec | None) -> "pg.Vsc":
    """
    Convert one VeraGrid VSC into one GSLV VSC.

    :param elm: VeraGrid VSC.
    :param bus_dict: Bus lookup by VeraGrid id tag.
    :param n_time: Number of exported time steps.
    :param use_time_series: Whether the export is time-series based.
    :param time_indices: Optional time-series selection.
    :return: GSLV VSC.
    """
    vsc = pg.Vsc(
        nt=n_time,
        bus_from=bus_dict[elm.bus_from.idtag],
        bus_to=bus_dict[elm.bus_to.idtag],
        name=elm.name,
        idtag=elm.idtag,
        code=str(elm.code),
        active=elm.active,
        rate=elm.rate,
        kdp=elm.control1_val_droop,
        alpha1=elm.alpha1,
        alpha2=elm.alpha2,
        alpha3=elm.alpha3,
        mttf=elm.mttf,
        mttr=elm.mttr,
        overload_cost=elm.Cost,
        contingency_factor=elm.contingency_factor,
        protection_rating_factor=elm.protection_rating_factor,
        contingency_enabled=elm.contingency_enabled,
        monitor_loading=elm.monitor_loading,
        capex=elm.capex,
        opex=elm.opex,
        build_status=build_status_dict[elm.build_status],
        control1=converter_control_type_dict[elm.control1],
        control2=converter_control_type_dict[elm.control2],
        control1_val=elm.control1_val,
        control2_val=elm.control2_val,
        control1_dev=convert_vsc_bus_reference(elm.control1_dev, bus_dict),
        control2_dev=convert_vsc_bus_reference(elm.control2_dev, bus_dict),
        bus_dc_n=convert_vsc_bus_reference(elm.bus_dc_n, bus_dict),
        control1_val_min=elm.control1_val_min,
        control1_val_max=elm.control1_val_max,
        control1_val_droop=elm.control1_val_droop,
        control1_droop_val=elm.control1_droop_val,
        control1_droop_val_min=elm.control1_droop_val_min,
        control1_droop_val_max=elm.control1_droop_val_max,
        control2_val_min=elm.control2_val_min,
        control2_val_max=elm.control2_val_max,
        control2_val_droop=elm.control2_val_droop,
        control2_droop_val=elm.control2_droop_val,
        control2_droop_val_min=elm.control2_droop_val_min,
        control2_droop_val_max=elm.control2_droop_val_max,
        min_ac_voltage=elm.min_ac_voltage,
        ysvs=elm.ysvs,
        x=elm.x,
        y=elm.y,
        fault_control=elm.fault_control != ConverterFaultControlType.Standard,
        cost=elm.Cost,
        design_rate=elm.design_rate,
    )

    fill_profile(vsc.active, elm.active_prof, use_time_series, time_indices, n_time, elm.active)
    fill_profile(vsc.rate, elm.rate_prof, use_time_series, time_indices, n_time, elm.rate)
    fill_profile(
        vsc.contingency_factor,
        elm.contingency_factor_prof,
        use_time_series,
        time_indices,
        n_time,
        elm.contingency_factor,
    )
    fill_profile(vsc.cost, elm.Cost_prof, use_time_series, time_indices, n_time, elm.Cost)

    return vsc


def add_vscs(circuit: MultiCircuit,
             gslv_grid: "pg.MultiCircuit",
             bus_dict: Dict[str, "pg.Bus"],
             time_series: bool,
             n_time: int = 1,
             time_indices: IntVec | None = None) -> None:
    """
    Add every VeraGrid VSC to the target GSLV grid.

    :param circuit: VeraGrid circuit.
    :param gslv_grid: GSLV circuit.
    :param bus_dict: Bus lookup by VeraGrid id tag.
    :param time_series: Whether the export is time-series based.
    :param n_time: Number of exported time steps.
    :param time_indices: Optional time-series selection.
    :return: None.
    """
    for elm in circuit.vsc_devices:
        vsc = convert_vsc(
            elm=elm,
            bus_dict=bus_dict,
            n_time=n_time,
            use_time_series=time_series,
            time_indices=time_indices,
        )
        gslv_grid.add_vsc(vsc)
