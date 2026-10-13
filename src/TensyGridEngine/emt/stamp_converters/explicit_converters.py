"""Algebraic-free EMT ODEs for the nonlinear STAMP GFOR and GFOL models.

The existing STAMP EMT block is the source of the control and filter state
RHS. Its algebraic voltage conversion, P/Q, and abc output-current equations
are substituted into those RHS expressions. No controller equation is copied
or re-derived here. The returned block has states and prescribed abc-voltage
inputs, with zero algebraic unknowns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from VeraGridEngine.Utils.Symbolic import symbolic as sym
from VeraGridEngine.Utils.Symbolic.block import Block

from .bases import RMS_LL_TO_PEAK_LN
from .emt_nonlinear_converters import build_stamp_nonlinear_converter_emt
from .parameters import STAMP_EMT_Q0_BY_MODE, StampConverterParameters


@dataclass(frozen=True)
class StampExplicitObservables:
    """Symbolic outputs calculated from the ODE states and abc inputs."""

    p: sym.Expr
    q: sym.Expr
    current_abc: tuple[sym.Expr, sym.Expr, sym.Expr]
    voltage_rms: sym.Expr


def _nodes(expression: sym.Expr) -> Iterator[sym.Expr]:
    """Visit an expression so the exact sin(Va)/cos(Va) nodes can be replaced."""
    yield expression
    if isinstance(expression, sym.BinOp):
        yield from _nodes(expression.left)
        yield from _nodes(expression.right)
    elif isinstance(expression, sym.UnOp):
        yield from _nodes(expression.operand)
    elif isinstance(expression, sym.Func):
        yield from _nodes(expression.arg)
    elif isinstance(expression, sym.Func2):
        yield from _nodes(expression.arg1)
        yield from _nodes(expression.arg2)


def _operating_point_references(parameters: StampConverterParameters,
                                q_initial: float) -> dict[str, float]:
    """Evaluate the wrapper's P/Q/voltage references at its PF seed point."""
    grid_vq = RMS_LL_TO_PEAK_LN * parameters.voltage_pu
    grid_iq = parameters.p_pu_system / (1.5 * grid_vq)
    grid_id = q_initial / (1.5 * grid_vq)
    capacitor_bus_q = (
        grid_vq + parameters.transformer_r * grid_iq
        + parameters.transformer_x * grid_id
    )
    capacitor_bus_d = (
        parameters.transformer_r * grid_id
        - parameters.transformer_x * grid_iq
    )
    local_p = 1.5 * (capacitor_bus_q * grid_iq + capacitor_bus_d * grid_id)
    local_q = 1.5 * (capacitor_bus_q * grid_id - capacitor_bus_d * grid_iq)
    voltage_ref = float(np.hypot(capacitor_bus_q, capacitor_bus_d))
    return {"P_ref": local_p, "Q_ref": local_q,
            "V_ref": voltage_ref, "Umag_ref": voltage_ref}


def build_stamp_explicit_converter_emt(
    variable_factory: Any,
    parameters: StampConverterParameters,
    name: str,
) -> tuple[Block, StampExplicitObservables]:
    """Build the STAMP GFOR/GFOL abc-input EMT model as an explicit ODE.

    ``name`` should end in ``_EMT``, matching the established STAMP wrapper.
    The controller references retain the original block's event definitions.
    The same physical states, parameters, and nonlinear control laws are used.
    """
    if parameters.mode not in {"GFOR", "GFOL"}:
        raise ValueError("STAMP explicit model supports GFOR and GFOL")

    block = build_stamp_nonlinear_converter_emt(variable_factory, parameters, name).block
    voltage_a, voltage_b, voltage_c = block.in_vars
    theta_grid = next(var for var in block.state_vars if var.name.endswith(".theta_grid"))
    rms_voltage = next(var for var in block.algebraic_vars if var.name.endswith(".Vm"))
    voltage_angle = next(var for var in block.algebraic_vars if var.name.endswith(".Va"))
    measured_p = next(var for var in block.algebraic_vars if var.name.endswith(".P"))
    measured_q = next(var for var in block.algebraic_vars if var.name.endswith(".Q"))
    current_vars = tuple(var for var in block.out_vars if var.name.startswith("i_"))
    if len(current_vars) != 3:
        raise ValueError("STAMP EMT block must have three abc current outputs")

    # These are exactly the abc-to-q/d expressions in the existing EMT wrapper.
    shift = sym.Const(2.0 * np.pi / 3.0)
    phase_b = theta_grid - shift
    phase_c = theta_grid + shift
    voltage_q = sym.Const(2.0 / 3.0) * (
        sym.sin(theta_grid) * voltage_a
        + sym.sin(phase_b) * voltage_b
        + sym.sin(phase_c) * voltage_c
    )
    voltage_d = -sym.Const(2.0 / 3.0) * (
        sym.cos(theta_grid) * voltage_a
        + sym.cos(phase_b) * voltage_b
        + sym.cos(phase_c) * voltage_c
    )
    voltage_peak = sym.sqrt(voltage_q * voltage_q + voltage_d * voltage_d)
    voltage_rms_expr = voltage_peak / sym.Const(np.sqrt(2.0))

    # The wrapper constrains sqrt(2)*Vm*cos(Va)=vq and
    # -sqrt(2)*Vm*sin(Va)=vd. Use their exact sine/cosine ratios in the ODE
    # instead of inserting atan2 inside trigonometric functions. This keeps
    # the generated solver equations explicit and compilable.
    replacements: dict[sym.Expr, sym.Expr] = {rms_voltage: voltage_rms_expr}
    source_expressions = [*block.state_eqs, *block.algebraic_eqs, *block.init_eqs.values()]
    for expression in source_expressions:
        for node in _nodes(expression):
            if isinstance(node, sym.Func) and isinstance(node.arg, sym.Var):
                if node.arg.uid != voltage_angle.uid:
                    continue
                if node.op == "cos":
                    replacements[node] = voltage_q / voltage_peak
                elif node.op == "sin":
                    replacements[node] = -voltage_d / voltage_peak

    # Bare Va occurs in initialization expressions. atan2 is needed there,
    # but does not appear in any solver RHS after the trig replacements above.
    replacements[voltage_angle] = sym.atan2(-voltage_d, voltage_q)
    explicit_rhs = [equation.subs(replacements) for equation in block.state_eqs]
    forbidden_ids = {variable.uid for variable in block.algebraic_vars}
    unresolved = {
        variable.name
        for equation in explicit_rhs
        for variable in equation.get_vars()
        if variable.uid in forbidden_ids
    }
    if unresolved:
        raise ValueError(f"STAMP state equations still contain algebraic variables: {sorted(unresolved)}")

    # Recover observable expressions from the original algebraic equations
    # 'variable - expression = 0'. They are outputs, not solver unknowns.
    def observable(variable: sym.Var) -> sym.Expr:
        equation = next(
            equation for equation in block.algebraic_eqs
            if any(item.uid == variable.uid for item in equation.get_vars())
        )
        return (-equation.subs({variable: sym.Const(0.0)})).subs(replacements)

    observables = StampExplicitObservables(
        p=observable(measured_p),
        q=observable(measured_q),
        current_abc=(observable(current_vars[0]),
                     observable(current_vars[1]),
                     observable(current_vars[2])),
        voltage_rms=voltage_rms_expr,
    )

    # The existing wrapper's P/Q initialization values are the audited STAMP
    # IEEE-9 operating points. They are used only to initialize the ODE states.
    q_initial = STAMP_EMT_Q0_BY_MODE[parameters.mode]
    init_replacements = {
        **replacements,
        measured_p: sym.Const(parameters.p_pu_system),
        measured_q: sym.Const(q_initial),
    }
    state_ids = {variable.uid for variable in block.state_vars}
    block.state_eqs = explicit_rhs
    block.init_eqs = {
        variable: expression.subs(init_replacements)
        for variable, expression in block.init_eqs.items()
        if variable.uid in state_ids
    }
    # The original wrapper computes controller references from initialized
    # states. EmtModelProblem treats event entries as prescribed runtime
    # values, so set the same operating-point references explicitly here.
    reference_values = _operating_point_references(parameters, q_initial)
    block.event_dict = {
        variable: sym.Const(reference_values[variable.name.rsplit(".", 1)[-1]])
        for variable, expression in block.event_dict.items()
        if variable.name.rsplit(".", 1)[-1] in reference_values
    }
    block.algebraic_vars = []
    block.algebraic_eqs = []
    block.out_vars = []  # Algebraic current outputs are in observables.
    block.name = name + "_explicit"
    return block, observables
