# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations
import chardet
from typing import List, Type, Dict
import VeraGridEngine.Devices as dev
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.basic_structures import Logger


def _parse_fixed(line: str,
                 start: int,
                 end: int,
                 dtype: Type[str] | Type[int] | Type[float] = str,
                 implicit_decimals: int = 0):
    """
    Extract substring by fixed columns and convert to the desired dtype.
    :param line:
    :param start:
    :param end:
    :param dtype:
    :param implicit_decimals:
    :return:
    """

    raw = line[start - 1:end].strip()
    if raw == "":
        if dtype in (int, float):
            return 0.0 if dtype == float else 0
        return ""
    try:
        if dtype == float:
            val = float(raw)
            if implicit_decimals and "." not in raw:
                val /= 10 ** implicit_decimals
            return val
        elif dtype == int:
            return int(raw)
        else:
            return raw
    except Exception:
        return 0.0 if dtype == float else 0 if dtype == int else raw


# -------------------------------------------------------------------------
#  DBGT — VoltageGroup
# -------------------------------------------------------------------------

class PwfVoltageGroup:
    """
    PwfBus
    """
    __slots__ = (
        "char",
        "voltage",
    )

    def __init__(self):
        self.char: str = ""
        self.voltage: float = 1.0  # Voltage in kV

    def parse(self, line: str) -> None:
        """

        :param line:
        :return:
        """
        self.char = _parse_fixed(line, 1, 2, str)  # int --> str
        self.voltage = _parse_fixed(line, 4, 8, float, 2)

    def __repr__(self):
        return f"<VoltageGroup {self.char}: {self.voltage:.3f} kV>"


# -------------------------------------------------------------------------
#  DBAR — Bus
# -------------------------------------------------------------------------

class PwfBus:
    """
    PwfBus
    """
    __slots__ = (
        "number",
        "operation",
        "type",
        "base_voltage_group",
        "voltage_limit_group",
        "name",
        "voltage",
        "angle",
        "pg",
        "qg",
        "qmin",
        "qmax",
        "controlled_bus",
        "pl",
        "ql",
        "area",
        "v_charge",
        "zone",
        "aggregators",
    )

    def __init__(self) -> None:
        """
        Constructor
        """
        self.number: int = 0
        self.operation: str = "A"
        self.type: int = 1
        self.base_voltage_group: str = ""
        self.voltage_limit_group: str = ""
        self.name: str = ""
        self.voltage: float = 1.0
        self.angle: float = 0.0
        self.pg: float = 0.0
        self.qg: float = 0.0
        self.qmin: float = -9999.0
        self.qmax: float = 9999.0
        self.controlled_bus: int = 0
        self.pl: float = 0.0
        self.ql: float = 0.0
        self.area: int = 1
        self.v_charge: float = 1.0
        self.zone: int = 1
        self.aggregators: List[int] = [0] * 10

    def parse(self, line: str) -> None:
        """

        :param line:
        :return:
        """
        self.number = _parse_fixed(line, 1, 2, int)
        self.operation = _parse_fixed(line, 6, 7, str)
        self.type = _parse_fixed(line, 8, 8, int)
        self.base_voltage_group = _parse_fixed(line, 9, 10, str)
        self.voltage_limit_group = _parse_fixed(line, 23, 24, str)
        self.name = _parse_fixed(line, 11, 22, str)
        self.voltage = _parse_fixed(line, 25, 29, float, 2)
        self.angle = _parse_fixed(line, 30, 34, float, 2)
        self.pg = _parse_fixed(line, 35, 40, float, 1)
        self.qg = _parse_fixed(line, 41, 46, float, 1)
        self.qmin = _parse_fixed(line, 47, 52, float, 1)
        self.qmax = _parse_fixed(line, 53, 58, float, 1)
        self.controlled_bus = _parse_fixed(line, 59, 63, int)
        self.pl = _parse_fixed(line, 64, 69, float, 1)
        self.ql = _parse_fixed(line, 70, 75, float, 1)
        self.area = _parse_fixed(line, 76, 78, int)
        self.v_charge = _parse_fixed(line, 79, 83, float, 2)
        self.zone = _parse_fixed(line, 84, 86, int)

        # Aggregators (10x)
        agg_cols = [
            (87, 91), (92, 96), (97, 101), (102, 106), (107, 111),
            (112, 116), (117, 121), (122, 126), (127, 131), (132, 136)
        ]
        self.aggregators = [
            _parse_fixed(line, s, e, int) for (s, e) in agg_cols
        ]

        if self.controlled_bus == 0:
            self.controlled_bus = self.number

    def to_veragrid(self, vg_dict: Dict[str, "PwfVoltageGroup"]) -> dev.Bus:
        """

        :param vg_dict:
        :return:
        """
        vg = vg_dict.get(self.base_voltage_group, None)
        Vnom = vg.voltage if vg else 1.0

        area_obj = dev.Area(name=f"Area_{self.area}") if isinstance(self.area, int) else self.area
        zone_obj = dev.Zone(name=f"Zone_{self.zone}") if isinstance(self.zone, int) else self.zone

        bus = dev.Bus(
            name=self.name.strip() or f"Bus_{self.number}",
            Vnom=Vnom,
            Vm0=self.voltage if self.voltage != 0.0 else 1.0,
            Va0=self.angle,
            area=area_obj,
            zone=zone_obj
        )
        return bus

    def __repr__(self):
        return f"<Bus {self.number} '{self.name}' Vm={self.voltage:.3f} Va={self.angle:.2f}°>"


# -------------------------------------------------------------------------
#  DLIN — Line
# -------------------------------------------------------------------------

class PwfLine:
    """
    PwfLine
    """
    __slots__ = (
        "from_bus",
        "to_bus",
        "circuit",
        "status",
        "owner",
        "r",
        "x",
        "b",
        "tap",
        "tap_min",
        "tap_max",
        "tap_lag",
        "controlled_bus",
        "normal_capacity",
        "emergency_capacity",
        "ntaps",
        "equipment_capacity",
        "aggregators",
    )

    def __init__(self) -> None:
        self.from_bus: int = 0
        self.to_bus: int = 0
        self.circuit: str = "1"
        self.status: str = "A"
        self.owner: str = ""
        self.r: float = 0.0
        self.x: float = 0.0
        self.b: float = 0.0
        self.tap: float = 1.0
        self.tap_min: float = 0.9
        self.tap_max: float = 1.1
        self.tap_lag: float = 0.0
        self.controlled_bus: int = 0
        self.normal_capacity: float = 9999.0
        self.emergency_capacity: float = 9999.0
        self.ntaps: int = 0
        self.equipment_capacity: float = 9999.0
        self.aggregators: List[int] = [0] * 10

    def parse(self, line: str) -> None:
        """

        :param line:
        :return:
        """
        self.from_bus = _parse_fixed(line, 1, 5, int)
        self.to_bus = _parse_fixed(line, 5, 12, int)
        self.circuit = _parse_fixed(line, 11, 12, str)
        self.status = _parse_fixed(line, 13, 13, str)
        self.owner = _parse_fixed(line, 14, 15, str)
        self.r = _parse_fixed(line, 16, 22, float, 5)
        self.x = _parse_fixed(line, 23, 29, float, 5)
        self.b = _parse_fixed(line, 30, 36, float, 5)
        self.tap = _parse_fixed(line, 37, 41, float, 3)
        self.tap_min = _parse_fixed(line, 42, 46, float, 3)
        self.tap_max = _parse_fixed(line, 47, 51, float, 3)
        self.tap_lag = _parse_fixed(line, 52, 56, float, 2)
        self.controlled_bus = _parse_fixed(line, 57, 61, int)
        self.normal_capacity = _parse_fixed(line, 62, 67, float, 1)
        self.emergency_capacity = _parse_fixed(line, 68, 73, float, 1)
        self.ntaps = _parse_fixed(line, 74, 77, int)
        self.equipment_capacity = _parse_fixed(line, 78, 83, float, 1)

        agg_cols = [
            (84, 88), (89, 93), (94, 98), (99, 103), (104, 108),
            (109, 113), (114, 118), (119, 123), (124, 128), (129, 133)
        ]
        self.aggregators = [
            _parse_fixed(line, s, e, int) for (s, e) in agg_cols
        ]

        if self.controlled_bus == 0:
            self.controlled_bus = self.to_bus

    def to_veragrid(self, bus_dict: Dict[int, dev.Bus]) -> dev.Line:
        """

        :param bus_dict:
        :return:
        """
        from_bus = bus_dict.get(self.from_bus)
        to_bus = bus_dict.get(self.to_bus)

        name = f"L{self.from_bus}-{self.to_bus}_{self.circuit}"

        elm = dev.Line(
            name=name,
            bus_from=from_bus,
            bus_to=to_bus,
            r=self.r,
            x=self.x,
            b=self.b,
            rate=self.normal_capacity,
            active=self.status == 'A'
        )

        return elm

    def __repr__(self):
        return f"<Line {self.from_bus}-{self.to_bus} R={self.r:.4f} X={self.x:.4f}>"


# -------------------------------------------------------------------------
#  DGER — Generator
# -------------------------------------------------------------------------

class PwfGenerator:
    """
    PwfGenerator
    """
    __slots__ = (
        "number",
        "operation",
        "min_active_gen",
        "max_active_gen",
        "participation_factor",
        "remote_participation_factor",
        "nominal_power_factor",
        "armature_service_factor",
        "rotor_service_factor",
        "charge_angle",
        "machine_reactance",
        "nominal_apparent_power",
    )

    def __init__(self) -> None:
        self.number: int = 0
        self.operation: str = "A"
        self.min_active_gen: float = 0.0
        self.max_active_gen: float = 9999.0
        self.participation_factor: float = 0.0
        self.remote_participation_factor: float = 100.0
        self.nominal_power_factor: float = 1.0
        self.armature_service_factor: float = 1.0
        self.rotor_service_factor: float = 1.0
        self.charge_angle: float = 0.0
        self.machine_reactance: float = 0.0
        self.nominal_apparent_power: float = 9999.0

    def parse(self, line: str) -> None:
        """

        :param line:
        :return:
        """
        self.number = _parse_fixed(line, 1, 2, int)
        self.operation = _parse_fixed(line, 6, 6, str)
        self.min_active_gen = _parse_fixed(line, 9, 14, float, 1)
        self.max_active_gen = _parse_fixed(line, 16, 21, float, 1)
        self.participation_factor = _parse_fixed(line, 23, 27, float, 2)
        self.remote_participation_factor = _parse_fixed(line, 29, 33, float, 2)
        self.nominal_power_factor = _parse_fixed(line, 35, 39, float, 2)
        self.armature_service_factor = _parse_fixed(line, 41, 44, float, 2)
        self.rotor_service_factor = _parse_fixed(line, 46, 49, float, 2)
        self.charge_angle = _parse_fixed(line, 51, 54, float, 2)
        self.machine_reactance = _parse_fixed(line, 56, 60, float, 2)
        self.nominal_apparent_power = _parse_fixed(line, 62, 66, float, 2)

    def to_veragrid(self, bus_dict: Dict[int, dev.Bus]) -> dev.Generator:
        """

        :param bus_dict:
        :return:
        """
        gen = dev.Generator()
        gen.name = f"G{self.number}"
        gen.Snom = float(self.nominal_apparent_power)
        gen.Pmin = float(self.min_active_gen)
        gen.Pmax = float(self.max_active_gen)
        gen.P = 0.0  #
        # gen.bus =
        gen.Pf = float(self.nominal_power_factor)
        gen.active = (self.operation == 'A')
        return gen

    def __repr__(self):
        return f"<Generator {self.number} Operation={self.operation} Min/Max={self.min_active_gen}/{self.max_active_gen}>"


# -------------------------------------------------------------------------
#  DELO — Load
# -------------------------------------------------------------------------

class PwfLoad:
    """
    PwfLoad
    """
    __slots__ = (
        "number",
        "operation",
        "bus",
        "active_power",
        "reactive_power",
        "status",
    )

    def __init__(self):
        # Attributes
        self.number: int = 0
        self.operation: str = "A"  # Active by default
        self.bus: int = 0
        self.active_power: float = 0.0
        self.reactive_power: float = 0.0
        self.status: str = "L"  # Default load status: 'L' (Low)

    def parse(self, line: str) -> None:
        """

        :param line:
        :return:
        """
        self.number = _parse_fixed(line, 1, 4, int)
        self.operation = _parse_fixed(line, 5, 5, str)
        self.bus = _parse_fixed(line, 7, 10, int)
        self.active_power = _parse_fixed(line, 11, 16, float, 1)
        self.reactive_power = _parse_fixed(line, 17, 22, float, 1)
        self.status = _parse_fixed(line, 23, 23, str)

    def to_veragrid(self, bus_dict: Dict[int, dev.Bus]) -> dev.Load:
        """

        :param bus_dict:
        :return:
        """
        load = dev.Load()
        load.name = f"Load_{self.number}"
        load.P = float(self.active_power)
        load.Q = float(self.reactive_power)
        load.active = (self.operation == 'A')
        return load

    def __repr__(self):
        return f"<Load {self.number} Bus={self.bus} P={self.active_power} Q={self.reactive_power}>"


# -------------------------------------------------------------------------
#  DTRA — Transformer
# -------------------------------------------------------------------------

class PwfTransformer:
    """
    PwfTransformer
    """
    __slots__ = (
        "number",
        "from_bus",
        "to_bus",
        "r",
        "x",
        "tap",
        "shift",
    )

    def __init__(self):
        self.number: int = 0
        self.from_bus: int = 0
        self.to_bus: int = 0
        self.r: float = 0.0
        self.x: float = 0.0
        self.tap: float = 1.0
        self.shift: float = 0.0

    def parse(self, line: str) -> None:
        """

        :param line:
        :return:
        """
        self.number = _parse_fixed(line, 1, 5, int)
        self.from_bus = _parse_fixed(line, 6, 10, int)
        self.to_bus = _parse_fixed(line, 11, 15, int)
        self.r = _parse_fixed(line, 16, 20, float, 5)
        self.x = _parse_fixed(line, 21, 25, float, 5)
        self.tap = _parse_fixed(line, 26, 30, float, 3)
        self.shift = _parse_fixed(line, 31, 35, float, 3)

    def to_veragrid(self, bus_dict: Dict[int, dev.Bus]) -> dev.Transformer2W:
        """

        :param bus_dict:
        :return:
        """
        from_bus = bus_dict.get(self.from_bus)
        to_bus = bus_dict.get(self.to_bus)
        elm = dev.Transformer2W(
            name=f"T{self.number}_{self.from_bus}-{self.to_bus}",
            bus_from=from_bus,
            bus_to=to_bus,
            r=self.r,
            x=self.x,
            tap_module=self.tap,
            tap_phase=self.shift
        )
        return elm

    def __repr__(self):
        return f"<Transformer {self.number} {self.from_bus} -> {self.to_bus} Tap={self.tap:.3f}>"


# -------------------------------------------------------------------------
#  DSHL — Shunt
# -------------------------------------------------------------------------

class PwfShunt:
    """
    PwfShunt
    """
    __slots__ = (
        "number",
        "from_bus",
        "to_bus",
        "status_from",
        "status_to",
        "shunt_from",
        "shunt_to",
    )

    def __init__(self):
        self.number: int = 0
        self.from_bus: int = 0
        self.to_bus: int = 0
        self.status_from: str = 'L'
        self.status_to: str = 'L'
        self.shunt_from: float = 0.0
        self.shunt_to: float = 0.0

    def parse(self, line: str) -> None:
        """

        :param line:
        :return:
        """
        self.number = _parse_fixed(line, 1, 5, int)
        self.from_bus = _parse_fixed(line, 6, 10, int)
        self.to_bus = _parse_fixed(line, 11, 15, int)
        self.status_from = _parse_fixed(line, 16, 17, str)
        self.status_to = _parse_fixed(line, 18, 19, str)
        self.shunt_from = _parse_fixed(line, 20, 24, float, 2)
        self.shunt_to = _parse_fixed(line, 25, 29, float, 2)

    def to_veragrid(self, bus_dict: Dict[int, dev.Bus]) -> list[tuple[int, dev.Shunt]]:
        """

        :param bus_dict:
        :return:
        """
        out = []
        if self.shunt_from != 0:
            sh = dev.Shunt()
            sh.name = f"Sh_{self.from_bus}"
            sh.B = float(self.shunt_from)
            sh.active = (self.status_from == "L")
            out.append((self.from_bus, sh))
        if self.shunt_to != 0:
            sh = dev.Shunt()
            sh.name = f"Sh_{self.to_bus}"
            sh.B = float(self.shunt_to)
            sh.active = (self.status_to == "L")
            out.append((self.to_bus, sh))
        return out

    def __repr__(self):
        return f"<Shunt {self.number} {self.from_bus} -> {self.to_bus} Shunt={self.shunt_from}/{self.shunt_to}>"


# -------------------------------------------------------------------------
#  StaticCompensator (DCSC)
# -------------------------------------------------------------------------

class PwfStaticCompensator:
    """
    StaticCompensator
    """
    __slots__ = (
        "number",
        "from_bus",
        "to_bus",
        "status",
        "initial_value",
        "specified_value",
    )

    def __init__(self):
        self.number: int = 0
        self.from_bus: int = 0
        self.to_bus: int = 0
        self.status: str = 'L'
        self.initial_value: float = 0.0
        self.specified_value: float = 0.0

    def parse(self, line: str) -> None:
        """

        :param line:
        :return:
        """
        self.number = _parse_fixed(line, 1, 5, int)
        self.from_bus = _parse_fixed(line, 6, 10, int)
        self.to_bus = _parse_fixed(line, 11, 15, int)
        self.status = _parse_fixed(line, 16, 16, str)
        self.initial_value = _parse_fixed(line, 17, 21, float, 2)
        self.specified_value = _parse_fixed(line, 22, 26, float, 2)

    def to_veragrid(self, bus_dict: Dict[int, dev.Bus]) -> tuple[int, dev.ControllableShunt]:
        """

        :param bus_dict:
        :return:
        """
        cs = dev.ControllableShunt()
        cs.name = f"SC{self.number}"
        cs.Bmax = float(self.specified_value)
        cs.Bmin = -float(self.specified_value)
        cs.Vset = 1.0
        cs.active = (self.status == 'L')
        return self.from_bus, cs

    def __repr__(self):
        return f"<StaticCompensator {self.number} {self.from_bus} -> {self.to_bus} Status={self.status}>"


# -------------------------------------------------------------------------
#  DCLine (DCLI)
# -------------------------------------------------------------------------

class PwfDCLine:
    """
    PwfDCLine
    """
    __slots__ = (
        "number",
        "from_bus",
        "to_bus",
        "vdc",
    )

    def __init__(self):
        self.number: int = 0
        self.from_bus: int = 0
        self.to_bus: int = 0
        self.vdc: float = 0.0

    def parse(self, line: str) -> None:
        """

        :param line:
        :return:
        """
        self.number = _parse_fixed(line, 1, 5, int)
        self.from_bus = _parse_fixed(line, 6, 10, int)
        self.to_bus = _parse_fixed(line, 11, 15, int)
        self.vdc = _parse_fixed(line, 16, 20, float, 2)

    def to_veragrid(self, bus_dict: Dict[int, dev.Bus]) -> dev.HvdcLine:
        """
        
        :param bus_dict: 
        :return: 
        """
        from_bus = bus_dict.get(self.from_bus, None)
        to_bus = bus_dict.get(self.to_bus, None)
        elm = dev.HvdcLine(
            name=f"HVDC{self.number}_{self.from_bus}-{self.to_bus}",
            bus_from=from_bus,
            bus_to=to_bus,
            r=0.0,
            # rate=self.vdc,
            active=True
        )
        return elm

    def __repr__(self):
        return f"<DCLine {self.number} {self.from_bus} -> {self.to_bus} Vdc={self.vdc:.2f}>"


# -------------------------------------------------------------------------
#  DGBR — Generator Reactance
# -------------------------------------------------------------------------

class PwfGeneratorReactance:
    """
    PwfGeneratorReactance
    """
    __slots__ = (
        "number",
        "group",
        "reactance",
    )

    def __init__(self):
        self.number: int = 0
        self.group: int = 0
        self.reactance: float = 0.0

    def parse(self, line: str) -> None:
        """
        
        :param line: 
        :return: 
        """
        self.number = _parse_fixed(line, 1, 4, int)
        self.group = _parse_fixed(line, 5, 6, int)
        self.reactance = _parse_fixed(line, 7, 12, float, 4)

    def __repr__(self):
        return f"<GeneratorReactance {self.number} Group={self.group} Reactance={self.reactance:.4f}>"


# -------------------------------------------------------------------------
#  DGLT — Voltage Limit Group
# -------------------------------------------------------------------------

class PwfVoltageLimitGroup:
    """
    PwfVoltageLimitGroup
    """
    __slots__ = (
        "group",
        "lower_bound",
        "upper_bound",
        "lower_emergency_bound",
        "upper_emergency_bound",
    )

    def __init__(self):
        self.group: int = 0
        self.lower_bound: float = 0.0
        self.upper_bound: float = 1.2
        self.lower_emergency_bound: float = 0.8
        self.upper_emergency_bound: float = 1.2

    def parse(self, line: str) -> None:
        """
        
        :param line: 
        :return: 
        """
        self.group = _parse_fixed(line, 1, 2, int)
        self.lower_bound = _parse_fixed(line, 4, 8, float, 2)
        self.upper_bound = _parse_fixed(line, 10, 14, float, 2)
        self.lower_emergency_bound = _parse_fixed(line, 16, 20, float, 2)
        self.upper_emergency_bound = _parse_fixed(line, 22, 26, float, 2)

    def __repr__(self):
        return f"<VoltageLimitGroup {self.group} Bounds=({self.lower_bound}-{self.upper_bound})>"


# -------------------------------------------------------------------------
#  DCSC — Static Compensator
# -------------------------------------------------------------------------
"""
class PwfStaticCompensator:

    def __init__(self):
        self.number: int = 0
        self.from_bus: int = 0
        self.to_bus: int = 0
        self.status: str = 'L'
        self.initial_value: float = 0.0
        self.specified_value: float = 0.0

    def parse(self, line: str) -> None:
        self.number = _parse_fixed(line, 1, 5, int)
        self.from_bus = _parse_fixed(line, 6, 10, int)
        self.to_bus = _parse_fixed(line, 11, 15, int)
        self.status = _parse_fixed(line, 16, 16, str)
        self.initial_value = _parse_fixed(line, 17, 21, float, 2)
        self.specified_value = _parse_fixed(line, 22, 26, float, 2)

    def __repr__(self):
        return f"<StaticCompensator {self.number} {self.from_bus} -> {self.to_bus} Status={self.status}>"

"""


# -------------------------------------------------------------------------
#  DCAR — Equipment Connection
# -------------------------------------------------------------------------

class PwfEquipmentConnection:
    """
    PwfEquipmentConnection
    """
    __slots__ = (
        "number",
        "equipment_type_1",
        "equipment_id_1",
        "condition_1",
        "equipment_type_2",
        "equipment_id_2",
        "condition_2",
        "operation",
        "parameter_a",
        "parameter_b",
        "parameter_c",
        "parameter_d",
        "voltage",
    )

    def __init__(self):
        self.number: int = 0
        self.equipment_type_1: str = ""
        self.equipment_id_1: int = 0
        self.condition_1: str = ""
        self.equipment_type_2: str = ""
        self.equipment_id_2: int = 0
        self.condition_2: str = ""
        self.operation: str = "A"
        self.parameter_a: float = 0.0
        self.parameter_b: float = 0.0
        self.parameter_c: float = 0.0
        self.parameter_d: float = 0.0
        self.voltage: float = 0.0

    def parse(self, line: str) -> None:
        """
        
        :param line: 
        :return: 
        """
        self.number = _parse_fixed(line, 1, 4, int)
        self.equipment_type_1 = _parse_fixed(line, 5, 8, str)
        self.equipment_id_1 = _parse_fixed(line, 9, 13, int)
        self.condition_1 = _parse_fixed(line, 14, 14, str)
        self.equipment_type_2 = _parse_fixed(line, 15, 18, str)
        self.equipment_id_2 = _parse_fixed(line, 19, 23, int)
        self.condition_2 = _parse_fixed(line, 24, 24, str)
        self.operation = _parse_fixed(line, 25, 25, str)
        self.parameter_a = _parse_fixed(line, 26, 28, float, 2)
        self.parameter_b = _parse_fixed(line, 29, 31, float, 2)
        self.parameter_c = _parse_fixed(line, 32, 34, float, 2)
        self.parameter_d = _parse_fixed(line, 35, 37, float, 2)
        self.voltage = _parse_fixed(line, 38, 42, float, 2)

    def __repr__(self):
        return f"<EquipmentConnection {self.number} {self.equipment_type_1} -> {self.equipment_type_2}>"


# -------------------------------------------------------------------------
#  DCTR — Transformer Settings
# -------------------------------------------------------------------------

class PwfTransformerSettings:
    """
    PwfTransformerSettings
    """
    __slots__ = (
        "from_bus",
        "to_bus",
        "circuit",
        "minimum_voltage",
        "maximum_voltage",
        "bounds_control_type",
        "control_mode",
        "minimum_phase",
        "maximum_phase",
        "specified_value",
    )

    def __init__(self):
        self.from_bus: int = 0
        self.to_bus: int = 0
        self.circuit: str = ""
        self.minimum_voltage: float = 0.0
        self.maximum_voltage: float = 1.0
        self.bounds_control_type: str = 'C'
        self.control_mode: str = 'A'
        self.minimum_phase: float = 0.0
        self.maximum_phase: float = 1.0
        self.specified_value: float = 0.0

    def parse(self, line: str) -> None:
        """
        
        :param line: 
        :return: 
        """
        self.from_bus = _parse_fixed(line, 1, 5, int)
        self.to_bus = _parse_fixed(line, 6, 10, int)
        self.circuit = _parse_fixed(line, 11, 12, str)
        self.minimum_voltage = _parse_fixed(line, 13, 17, float, 2)
        self.maximum_voltage = _parse_fixed(line, 18, 22, float, 2)
        self.bounds_control_type = _parse_fixed(line, 23, 23, str)
        self.control_mode = _parse_fixed(line, 24, 24, str)
        self.minimum_phase = _parse_fixed(line, 25, 29, float, 2)
        self.maximum_phase = _parse_fixed(line, 30, 34, float, 2)
        self.specified_value = _parse_fixed(line, 35, 39, float, 2)

    def __repr__(self):
        return f"<TransformerSettings {self.from_bus} -> {self.to_bus} Control={self.control_mode}>"


# -------------------------------------------------------------------------
#  DGEI — Generator Identifiers
# -------------------------------------------------------------------------

class PwfGeneratorIdentification:
    """
    PwfGeneratorIdentification
    """
    __slots__ = (
        "number",
        "operation",
        "automatic_mode",
        "group",
        "status",
        "units",
        "operating_units",
        "active_generation",
        "reactive_generation",
    )

    def __init__(self):
        self.number: int = 0
        self.operation: str = "A"
        self.automatic_mode: str = "N"
        self.group: int = 0
        self.status: str = "L"
        self.units: int = 1
        self.operating_units: int = 1
        self.active_generation: float = 0.0
        self.reactive_generation: float = 0.0

    def parse(self, line: str) -> None:
        """
        
        :param line: 
        :return: 
        """
        self.number = _parse_fixed(line, 1, 5, int)
        self.operation = _parse_fixed(line, 6, 6, str)
        self.automatic_mode = _parse_fixed(line, 7, 7, str)
        self.group = _parse_fixed(line, 8, 9, int)
        self.status = _parse_fixed(line, 10, 10, str)
        self.units = _parse_fixed(line, 11, 13, int)
        self.operating_units = _parse_fixed(line, 14, 16, int)
        self.active_generation = _parse_fixed(line, 17, 21, float, 2)
        self.reactive_generation = _parse_fixed(line, 22, 26, float, 2)

    def __repr__(self):
        return f"<GeneratorIdentification {self.number} Group={self.group} ActiveGen={self.active_generation:.2f}>"


# -------------------------------------------------------------------------
#  DMOT — Motor Configuration
# -------------------------------------------------------------------------

class PwfMotorConfiguration:
    """
    PwfMotorConfiguration
    """
    __slots__ = (
        "bus",
        "operation",
        "status",
        "group",
        "sign",
        "loading_factor",
        "units",
        "stator_resistance",
        "stator_reactance",
        "magnetizing_reactance",
        "rotor_resistance",
        "rotor_reactance",
        "base_power",
        "engine_type",
        "active_charge_portion",
    )

    def __init__(self):
        self.bus: int = 0
        self.operation: str = "A"
        self.status: str = "L"
        self.group: int = 0
        self.sign: str = "+"
        self.loading_factor: float = 1.0
        self.units: int = 1
        self.stator_resistance: float = 0.0
        self.stator_reactance: float = 0.0
        self.magnetizing_reactance: float = 0.0
        self.rotor_resistance: float = 0.0
        self.rotor_reactance: float = 0.0
        self.base_power: float = 0.0
        self.engine_type: int = 0
        self.active_charge_portion: float = 0.0

    def parse(self, line: str) -> None:
        """
        
        :param line: 
        :return: 
        """
        self.bus = _parse_fixed(line, 1, 5, int)
        self.operation = _parse_fixed(line, 6, 6, str)
        self.status = _parse_fixed(line, 7, 7, str)
        self.group = _parse_fixed(line, 8, 9, int)
        self.sign = _parse_fixed(line, 10, 10, str)
        self.loading_factor = _parse_fixed(line, 11, 13, float, 2)
        self.units = _parse_fixed(line, 14, 16, int)
        self.stator_resistance = _parse_fixed(line, 17, 21, float, 4)
        self.stator_reactance = _parse_fixed(line, 22, 26, float, 4)
        self.magnetizing_reactance = _parse_fixed(line, 27, 31, float, 4)
        self.rotor_resistance = _parse_fixed(line, 32, 36, float, 4)
        self.rotor_reactance = _parse_fixed(line, 37, 41, float, 4)
        self.base_power = _parse_fixed(line, 42, 46, float, 4)
        self.engine_type = _parse_fixed(line, 47, 49, int)
        self.active_charge_portion = _parse_fixed(line, 50, 53, float, 3)

    def __repr__(self):
        return f"<MotorConfiguration {self.bus} Group={self.group} Status={self.status}>"


# -------------------------------------------------------------------------
#  DCMT — Comments
# -------------------------------------------------------------------------

class PwfComment:
    """
    PwfComment
    """
    __slots__ = (
        "comment",
    )

    def __init__(self):
        self.comment: str = ""

    def parse(self, line: str) -> None:
        """
        
        :param line: 
        :return: 
        """
        self.comment = line.strip()

    def __repr__(self):
        return f"<Comment: {self.comment}>"


# -------------------------------------------------------------------------
#  Injection (DINJ)
# -------------------------------------------------------------------------

class PwfInjection:
    """
    PwfInjection
    """
    __slots__ = (
        "number",
        "operation",
        "equivalent_active_injection",
        "equivalent_reactive_injection",
        "equivalent_shunt",
        "equivalent_participation_factor",
    )

    def __init__(self):
        self.number: int = 0
        self.operation: str = "A"
        self.equivalent_active_injection: float = 0.0
        self.equivalent_reactive_injection: float = 0.0
        self.equivalent_shunt: float = 0.0
        self.equivalent_participation_factor: float = 0.0

    def parse(self, line: str) -> None:
        """
        
        :param line: 
        :return: 
        """
        self.number = _parse_fixed(line, 1, 5, int)
        self.operation = _parse_fixed(line, 6, 6, str)
        self.equivalent_active_injection = _parse_fixed(line, 7, 14, float, 2)
        self.equivalent_reactive_injection = _parse_fixed(line, 15, 22, float, 2)
        self.equivalent_shunt = _parse_fixed(line, 23, 29, float, 2)
        self.equivalent_participation_factor = _parse_fixed(line, 30, 36, float, 2)

    def __repr__(self):
        return f"<Injection {self.number} Active={self.equivalent_active_injection} Reactive={self.equivalent_reactive_injection}>"


# -------------------------------------------------------------------------
#  PWFNetwork (container)
# -------------------------------------------------------------------------
class PwfNetwork:
    """
    PwfNetwork
    """
    __slots__ = (
        "buses",
        "lines",
        "generators",
        "transformers",
        "shunts",
        "static_compensators",
        "dc_lines",
        "loads",
        "comments",
        "injections",
        "generator_reactances",
        "voltage_limit_groups",
        "voltage_groups",
        "generator_identifications",
    )

    def __init__(self):
        # Store devices by type (e.g., buses, lines, generators, etc.)
        self.buses = []
        self.lines = []
        self.generators = []
        self.transformers = []
        self.shunts = []
        self.static_compensators = []
        self.dc_lines = []
        self.loads = []
        self.comments = []
        self.injections = []
        self.generator_reactances = []
        self.voltage_limit_groups = []
        self.voltage_groups = []
        self.generator_identifications = []

    def add_device(self, device):
        """
        Add a device to the appropriate list based on its class type.
        :param device:
        :return:
        """

        if isinstance(device, PwfBus):
            self.buses.append(device)
        elif isinstance(device, PwfVoltageGroup):
            self.voltage_groups.append(device)
        elif isinstance(device, PwfLine):
            self.lines.append(device)
        elif isinstance(device, PwfGenerator):
            self.generators.append(device)
        elif isinstance(device, PwfTransformer):
            self.transformers.append(device)
        elif isinstance(device, PwfShunt):
            self.shunts.append(device)
        elif isinstance(device, PwfStaticCompensator):
            self.static_compensators.append(device)
        elif isinstance(device, PwfDCLine):
            self.dc_lines.append(device)
        elif isinstance(device, PwfLoad):
            self.loads.append(device)
        elif isinstance(device, PwfComment):
            self.comments.append(device)
        elif isinstance(device, PwfInjection):
            self.injections.append(device)
        elif isinstance(device, PwfGeneratorReactance):
            self.generator_reactances.append(device)
        elif isinstance(device, PwfVoltageLimitGroup):
            self.voltage_limit_groups.append(device)
        elif isinstance(device, PwfGeneratorIdentification):
            self.generator_identifications.append(device)

        else:
            raise ValueError(f"Unknown device type: {type(device)}")

    def to_veragrid(self) -> MultiCircuit:
        """
        
        :return: 
        """
        mc = MultiCircuit(name="Anarede_Network")

        vg_dict = {vg.char: vg for vg in self.voltage_groups}
        bus_dict = {b.number: b.to_veragrid(vg_dict) for b in self.buses}

        #  BUSES 
        for bus in bus_dict.values():
            mc.add_device(bus)

        #  LINES 
        for line in self.lines:
            mc.add_device(line.to_veragrid(bus_dict))

        #  GENERATORS 
        gen_to_bus = {g.group: g.number for g in self.generator_identifications}
        for g in self.generators:
            elm = g.to_veragrid(bus_dict)
            if elm:
                bus_id = gen_to_bus.get(g.number, g.number)
                bus = bus_dict.get(bus_id)
                mc.add_generator(bus=bus, api_obj=elm)

        #  LOADS 
        for load in self.loads:
            elm = load.to_veragrid(bus_dict)
            if elm:
                bus = bus_dict.get(load.bus)
                mc.add_load(bus=bus, api_obj=elm)

        #  TRANSFORMERS 
        for trafo in self.transformers:
            mc.add_device(trafo.to_veragrid(bus_dict))

        #  SHUNTS 
        for shunt in self.shunts:
            for bus_id, s in shunt.to_veragrid(bus_dict):
                mc.add_shunt(bus=bus_dict.get(bus_id), api_obj=s)

        return mc

    def __repr__(self):
        return (f"<PWFNetwork: {len(self.buses)} buses, "
                f"{len(self.lines)} lines, {len(self.generators)} generators, "
                f"{len(self.transformers)} transformers, "
                f"{len(self.shunts)} shunts, {len(self.static_compensators)} static compensators>")


# -------------------------------------------------------------------------
#  Parser
# -------------------------------------------------------------------------

def _split_sections(file_name: str):
    """
    Splits a PWF file into sections based on the delimiter "99999".
    Each section is stored in a dictionary with the section name as the key
    and the line indices as the value.
    """
    # make a guess of the file encoding
    detection = chardet.detect(open(file_name, "rb").read())

    # read and split by blocks
    with open(file_name, 'r', encoding=detection['encoding']) as io:
        file_lines = io.readlines()
        file_lines = [line.strip() for line in file_lines]  # Remove extra spaces and newlines

        # Initialize a dictionary to store section titles and their line indices
        sections = {}

        # Look for section titles and delimiters
        section_titles_idx = [i for i, line in enumerate(file_lines) if line == "99999"]

        # If section titles are found, add to sections dictionary
        if section_titles_idx:
            sections["title_identifier"] = section_titles_idx

        # Split the file lines based on section delimiters (99999)
        section_delim = [0] + section_titles_idx + [len(file_lines)]

        # Iterate through the sections and capture their lines
        for i in range(len(section_delim) - 1):
            section_name_idx = section_delim[i] + 1
            section_name = file_lines[section_name_idx]  # Extract section name
            section_lines = file_lines[section_delim[i] + 1: section_delim[i + 1]]

            sections[section_name] = section_lines

        return file_lines, sections


class PWFParser:
    """
    PWFParser
    """
    __slots__ = (
        "filepath",
        "network",
        "logger",
        "voltage_group_dict",
    )

    def __init__(self, filepath: str):
        self.filepath: str = filepath
        self.network: PwfNetwork = PwfNetwork()
        self.logger = Logger()

        self.voltage_group_dict: Dict[str, PwfVoltageGroup] = dict()

        file_lines, sections = _split_sections(self.filepath)

        for section_name, txt_lines in sections.items():

            if len(txt_lines) > 2:

                if section_name == "DBAR":  # Bus
                    for txt_line in txt_lines[2:]:
                        bus = PwfBus()
                        bus.parse(txt_line)
                        self.network.add_device(bus)

                if section_name == "DBGT":  # Voltage Group
                    for txt_line in txt_lines[2:]:
                        vg = PwfVoltageGroup()
                        vg.parse(txt_line)
                        self.network.add_device(vg)
                        self.voltage_group_dict[vg.char] = vg

                elif section_name == "DLIN":  # Line
                    for txt_line in txt_lines[2:]:
                        line = PwfLine()
                        line.parse(txt_line)
                        self.network.add_device(line)

                elif section_name == "DGER":  # Generator
                    for txt_line in txt_lines[2:]:
                        generator = PwfGenerator()
                        generator.parse(txt_line)
                        self.network.add_device(generator)

                elif section_name == "DGEI":  # Generator Identification
                    for txt_line in txt_lines[2:]:
                        genid = PwfGeneratorIdentification()
                        genid.parse(txt_line)
                        self.network.add_device(genid)

                elif section_name == "DTRA":  # Transformer
                    for txt_line in txt_lines[2:]:
                        transformer = PwfTransformer()
                        transformer.parse(txt_line)
                        self.network.add_device(transformer)

                elif section_name == "DSHL":  # Shunt
                    for txt_line in txt_lines[2:]:
                        shunt = PwfShunt()
                        shunt.parse(txt_line)
                        self.network.add_device(shunt)

                elif section_name == "DCSC":  # Static Compensator
                    for txt_line in txt_lines[2:]:
                        static_compensator = PwfStaticCompensator()
                        static_compensator.parse(txt_line)
                        self.network.add_device(static_compensator)

                elif section_name == "DCLI":  # DC Line
                    for txt_line in txt_lines[2:]:
                        dc_line = PwfDCLine()
                        dc_line.parse(txt_line)
                        self.network.add_device(dc_line)

                elif section_name == "DELO":  # Load
                    for txt_line in txt_lines[2:]:
                        load = PwfLoad()
                        load.parse(txt_line)
                        self.network.add_device(load)

                elif section_name == "DBRE":  # Comment
                    for txt_line in txt_lines[2:]:
                        comment = PwfComment()
                        comment.parse(txt_line)
                        self.network.add_device(comment)

                elif section_name == "DINJ":  # Injection
                    for txt_line in txt_lines[2:]:
                        injection = PwfInjection()
                        injection.parse(txt_line)
                        self.network.add_device(injection)

                elif section_name == "DGBR":  # Generator Reactance
                    for txt_line in txt_lines[2:]:
                        generator_reactance = PwfGeneratorReactance()
                        generator_reactance.parse(txt_line)
                        self.network.add_device(generator_reactance)

                elif section_name == "DGLT":  # Voltage Limit Group
                    for txt_line in txt_lines[2:]:
                        voltage_limit_group = PwfVoltageLimitGroup()
                        voltage_limit_group.parse(txt_line)
                        self.network.add_device(voltage_limit_group)

            else:
                # not enough data values
                pass

    def to_veragrid(self) -> MultiCircuit:
        """
        Convert Anarede grid to VeraGrid
        :return:
        """
        grid = MultiCircuit(name="Anarede_Network")

        vg_dict = {vg.char: vg for vg in self.network.voltage_groups}

        #  BUSES 
        bus_dict: Dict[int, dev.Bus] = {}
        for b in self.network.buses:
            bus = b.to_veragrid(vg_dict)
            grid.add_bus(bus)
            bus_dict[b.number] = bus

        #  LINES 
        for l in self.network.lines:
            elm = l.to_veragrid(bus_dict)
            grid.add_line(elm)

        #  TRANSFORMERS 
        for t in self.network.transformers:
            elm = t.to_veragrid(bus_dict)
            grid.add_transformer2w(elm)

        #  GENERATORS 
        for g in self.network.generators:
            elm = g.to_veragrid(bus_dict)
            bus = bus_dict.get(g.number, None)
            if bus is not None:
                grid.add_generator(bus=bus, api_obj=elm)

        #  LOADS 
        for ld in self.network.loads:
            elm = ld.to_veragrid(bus_dict)
            bus = bus_dict.get(ld.bus)
            if bus is not None:
                grid.add_load(bus=bus, api_obj=elm)

        #  SHUNTS 
        for sh in self.network.shunts:
            elm = sh.to_veragrid(bus_dict)
            bus = bus_dict.get(sh.bus)
            if bus is not None:
                grid.add_shunt(bus=bus, api_obj=elm)

        #  STATIC COMPENSATORS 
        for sc in self.network.static_compensators:
            elm = sc.to_veragrid(bus_dict)
            bus = bus_dict.get(sc.bus)
            if bus is not None:
                grid.add_controllable_shunt(bus=bus, api_obj=elm)

        #  DC LINES 
        for d in self.network.dc_lines:
            elm = d.to_veragrid(bus_dict)
            grid.add_hvdc(elm)

        return grid
