"""Build the STAMP WSCC/GFOR/GFOL network directly in VeraGrid."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import get_stamp_gfol_rms, get_stamp_gfor_rms, get_stamp_synchronous_generator_rms
from .parameters import STAMP_GFOL, STAMP_GFOR, STAMP_SG, SYSTEM_BASE_MVA


VERAGRID_IEEE9_PATH = str(Path(__file__).resolve().parent / "data/IEEE_9_Christoph.gridcal")


STAMP_LINES = (
    (1, 2, 0.0100, 0.0850, 0.1760),
    (1, 3, 0.0170, 0.0920, 0.1580),
    (2, 4, 0.0320, 0.1610, 0.3060),
    (3, 6, 0.0390, 0.1700, 0.3580),
    (4, 5, 0.0085, 0.0720, 0.1490),
    (5, 6, 0.0119, 0.1008, 0.2090),
)

STAMP_LOADS = (
    (2, 125.0, 50.0),
    (3, 90.0, 30.0),
    (5, 100.0, 35.0),
)


def _get_impedance_load_rms_template(vfactory: Any, name: str) -> Any:
    """Voltage-dependent algebraic load matching an RL load at equilibrium."""
    from VeraGridEngine.Devices.Dynamic.rms_template import RmsModelTemplate
    from VeraGridEngine.Utils.Symbolic.block import Block
    from VeraGridEngine.enumerations import (DeviceType, ParamPowerFlowReferenceType,
                                             VarPowerFlowReferenceType)

    template = RmsModelTemplate()
    template.tpe = DeviceType.LoadDevice
    template.name = name
    vm = vfactory.add_var("Vm", reference=VarPowerFlowReferenceType.Vm)
    va = vfactory.add_var("Va", reference=VarPowerFlowReferenceType.Va)
    pl = vfactory.add_var("Pl", reference=VarPowerFlowReferenceType.P)
    ql = vfactory.add_var("Ql", reference=VarPowerFlowReferenceType.Q)
    vm0 = vfactory.add_var(f"{name}.Vm0")
    pl0 = vfactory.add_var(f"{name}.Pl0")
    ql0 = vfactory.add_var(f"{name}.Ql0")
    voltage_squared_ratio = (vm / vm0) ** vfactory.add_const(2.0)

    block = Block()
    block.name = name
    block.in_vars = [vm, va]
    block.algebraic_vars = [pl, ql]
    block.algebraic_eqs = [pl - pl0 * voltage_squared_ratio,
                           ql - ql0 * voltage_squared_ratio]
    block.out_vars = [pl, ql]
    block.event_dict = {vm0: vm, pl0: pl, ql0: ql}
    template.block.children.append(block)
    template.block.in_vars = [vm, va]
    template.block.out_vars = [pl, ql]
    template.block.external_mapping = {
        VarPowerFlowReferenceType.Vm: vm,
        VarPowerFlowReferenceType.Va: va,
        VarPowerFlowReferenceType.P: pl,
        VarPowerFlowReferenceType.Q: ql,
    }
    template.block.api_obj_mapping = {
        ParamPowerFlowReferenceType.Pl0: pl0,
        ParamPowerFlowReferenceType.Ql0: ql0,
    }
    return template


def _assign_network_rms_models(grid: Any, impedance_loads: bool = False,
                               dynamic_lines: bool = False,
                               full_dynamic_network: bool = False) -> None:
    from VeraGridEngine.Templates.Rms.line_rms_template import get_line_rms_template
    from VeraGridEngine.Templates.Rms.load_rms_template import get_load_rms_template
    from VeraGridEngine.Utils.Symbolic.bus_rms_template import initialize_bus_rms
    from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model

    for bus in grid.buses:
        initialize_bus_rms(bus, vf=grid.var_factory)
    for index, line in enumerate(grid.lines, start=1):
        if dynamic_lines:
            from .dynamic_line import build_stamp_dynamic_line_rms
            bus_f = int(line.bus_from.name.removeprefix("Bus"))
            bus_t = int(line.bus_to.name.removeprefix("Bus"))
            line_name = f"NET.{min(bus_f, bus_t)}{max(bus_f, bus_t)}"
            template = build_stamp_dynamic_line_rms(
                grid.var_factory, resistance=line.R, reactance=line.X,
                shunt_susceptance=line.B, name=line_name,
                dynamic_shunts=full_dynamic_network)
        else:
            template = get_line_rms_template(grid.var_factory, name=f"STAMP_line_{index}")
        set_rms_model(line, template.block, grid.var_factory)
    for index, load in enumerate(grid.loads, start=1):
        name = f"STAMP_load_{index}"
        if load.name.startswith("STAMP bus capacitor"):
            from .dynamic_network import build_dynamic_bus_capacitor_rms
            template = build_dynamic_bus_capacitor_rms(
                grid.var_factory, susceptance=float(load.code), name=load.name)
        elif full_dynamic_network:
            from .dynamic_network import build_dynamic_rl_load_rms
            template = build_dynamic_rl_load_rms(
                grid.var_factory, p_pu=load.P/grid.Sbase, q_pu=load.Q/grid.Sbase,
                name=f"Load{index}")
        else:
            template = (_get_impedance_load_rms_template(grid.var_factory, name)
                        if impedance_loads else get_load_rms_template(grid.var_factory, name=name))
        set_rms_model(load, template.block, grid.var_factory)


def build_stamp_wscc_grid(*, impedance_loads: bool = False,
                          dynamic_lines: bool = False,
                          full_dynamic_network: bool = False,
                          nonlinear_converters: bool = False) -> Any:
    """Import VeraGrid's six-bus IEEE9 case and populate STAMP RMS models."""
    import VeraGridEngine.api as gce
    from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model

    grid = gce.open_file(VERAGRID_IEEE9_PATH)
    grid.name = "STAMP WSCC SG-GFOR-GFOL"
    grid.Sbase = SYSTEM_BASE_MVA
    grid.fBase = 50.0
    buses = {int(bus.name.removeprefix("Bus")): bus for bus in grid.buses}

    # This legacy gridcal stores the STAMP dispatch in pu although VeraGrid's
    # device API expects MW/MVAr.  Apply the authoritative case values.
    for load, (_, p_mw, q_mvar) in zip(grid.loads, STAMP_LOADS):
        load.P, load.Q = p_mw, q_mvar
    generator = grid.generators[0]
    generator.name = "STAMP SG1"
    generator.P = STAMP_SG.p_pu_system * SYSTEM_BASE_MVA
    generator.Vset = STAMP_SG.voltage_pu
    generator.Snom = STAMP_SG.rated_mva

    # The topology file predates the GFOR/GFOL injections, so add those two
    # devices after importing the grid rather than recreating the network.
    converter_devices = []
    for params in (STAMP_GFOR, STAMP_GFOL):
        # STAMP places both VSCs in MATPOWER's generator table: P and |V| are
        # specified and Q is solved.  They must therefore be PV devices here,
        # even though their attached RMS blocks are converter models.
        converter = gce.Generator(
            name=f"STAMP {params.mode}{params.number}",
            P=params.p_pu_system * SYSTEM_BASE_MVA,
            vset=params.voltage_pu,
            Snom=params.rated_mva,
        )
        grid.add_generator(buses[params.bus], converter)
        converter_devices.append((params, converter))

    if full_dynamic_network:
        bus_susceptance = {number: 0.0 for number in buses}
        for bus_f, bus_t, _, _, susceptance in STAMP_LINES:
            bus_susceptance[bus_f] += susceptance/2.0
            bus_susceptance[bus_t] += susceptance/2.0
        for number, susceptance in bus_susceptance.items():
            capacitor = gce.Load(name=f"STAMP bus capacitor {number}", P=0.0, Q=0.0)
            capacitor.code = str(susceptance)
            grid.add_load(buses[number], capacitor)

    _assign_network_rms_models(grid, impedance_loads=impedance_loads,
                               dynamic_lines=dynamic_lines,
                               full_dynamic_network=full_dynamic_network)
    sg_template = get_stamp_synchronous_generator_rms(grid.var_factory)
    set_rms_model(generator, sg_template.block, grid.var_factory)
    reference_omega = next(var for var in sg_template.block.state_vars
                           if var.name == "STAMP_SG1.w_pu")
    for params, converter in converter_devices:
        if nonlinear_converters:
            from .nonlinear_converters import build_stamp_converter_rms
            template = build_stamp_converter_rms(
                grid.var_factory, params, f"STAMP_{params.mode}{params.number}",
                reference_omega=reference_omega)
        else:
            template = (get_stamp_gfor_rms(grid.var_factory, reference_omega=reference_omega)
                        if params.mode == "GFOR"
                        else get_stamp_gfol_rms(grid.var_factory, reference_omega=reference_omega))
        set_rms_model(converter, template.block, grid.var_factory)
    return grid
