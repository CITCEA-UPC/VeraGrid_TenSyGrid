"""Assembly of the STAMP WSCC case with three-phase VeraGrid EMT models."""

from __future__ import annotations

import numpy as np

from .emt_converters import build_stamp_converter_emt
from .emt_generator import build_stamp_generator_emt
from .parameters import STAMP_GFOL, STAMP_GFOR, STAMP_SG
from .wscc_case import build_stamp_wscc_grid


def _adapt_parallel_rl_load_periodic_initialization(grid, model):
    """Give the generic R||L block its sinusoidal steady-state derivatives."""
    from VeraGridEngine.enumerations import VarPowerFlowReferenceType

    block = model.block
    c = grid.var_factory.add_const
    voltage_refs = {
        "A": VarPowerFlowReferenceType.v_A,
        "B": VarPowerFlowReferenceType.v_B,
        "C": VarPowerFlowReferenceType.v_C,
    }
    derivative_refs = {
        "A": VarPowerFlowReferenceType.d_v_A,
        "B": VarPowerFlowReferenceType.d_v_B,
        "C": VarPowerFlowReferenceType.d_v_C,
    }
    omega_base = c(2.0*np.pi*grid.fBase)
    for phase in "ABC":
        inductance = next(var for var in block.event_dict if var.name == f"L_{phase}")
        voltage = block.external_mapping[voltage_refs[phase]]
        voltage_derivative = grid.var_factory.add_var(
            f"d_v_{phase}_{block.name}", reference=derivative_refs[phase])
        block.external_mapping[derivative_refs[phase]] = voltage_derivative
        block.event_dict[voltage_derivative] = c(None)
        inductor_current = next(var for var in block.state_vars
                                if var.name == f"iL_{phase}")
        # With di_L/dt = -v/L and sinusoidal v, d2v/dt2 = -omega^2 v,
        # hence the periodic current is i_L = (dv/dt)/(omega^2 L).
        block.init_eqs[inductor_current] = voltage_derivative/(omega_base**2*inductance)
        block.diff_init_eqs[inductor_current.diff_var] = -voltage/inductance
    return model


def build_stamp_wscc_emt_grid(*, nonlinear_converters: bool = False):
    """Import the validated topology and populate all abc EMT device models."""
    from VeraGridEngine.Templates.Emt.load_RLC_emt_template import get_shunt_rlc_combo_emt_template
    from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
    from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
    from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
    from VeraGridEngine.enumerations import ShuntConnectionType

    grid = build_stamp_wscc_grid()
    for bus in grid.buses:
        get_bus_emt_template(grid=grid, bus=bus)
    for line in grid.lines:
        model = get_pi_line_emt_template(grid.var_factory, phN=False, phA=True,
                                         phB=True, phC=True, name=f"EMT_{line.name}",
                                         numerical_damping_conductance=0.0)
        set_emt_model(line, model.block, grid.var_factory)
    for load in grid.loads:
        # STAMP uses a parallel R || L load.  The template derives R and L from
        # the load's shared PF P/Q at the initialized bus voltage.
        model = get_shunt_rlc_combo_emt_template(
            grid.var_factory, include_r=True, include_l=True, include_c=False,
            phA=True, phB=True, phC=True,
            # The imported buses expose ABC without an explicit neutral.  A
            # balanced floating star is electrically equivalent here and
            # keeps the neutral internal to the load block.
            connection_type=ShuntConnectionType.FloatingStar,
            name=f"EMT_{load.name}")
        model = _adapt_parallel_rl_load_periodic_initialization(grid, model)
        set_emt_model(load, model.block, grid.var_factory)

    generators = {generator.name: generator for generator in grid.generators}
    sg = generators["STAMP SG1"]
    sg_model = build_stamp_generator_emt(grid.var_factory, STAMP_SG, name="STAMP_SG1_EMT")
    set_emt_model(sg, sg_model.block, grid.var_factory)
    omega_ref = next(var for var in sg_model.block.state_vars if var.name.endswith(".w_pu"))

    for params in (STAMP_GFOR, STAMP_GFOL):
        device = generators[f"STAMP {params.mode}{params.number}"]
        if nonlinear_converters:
            from .emt_nonlinear_converters import build_stamp_nonlinear_converter_emt
            model = build_stamp_nonlinear_converter_emt(
                grid.var_factory, params, f"STAMP_{params.mode}{params.number}_EMT",
                reference_omega=omega_ref)
        else:
            model = build_stamp_converter_emt(grid.var_factory, params,
                                              f"STAMP_{params.mode}{params.number}_EMT",
                                              reference_omega=omega_ref)
        set_emt_model(device, model.block, grid.var_factory)
    return grid
