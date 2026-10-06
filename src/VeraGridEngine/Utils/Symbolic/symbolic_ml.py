# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations
import copy
import numpy as np
from dataclasses import dataclass
from typing import Collection, List, Mapping
from VeraGridEngine.Utils.Symbolic.block import (Block)
from VeraGridEngine.Utils.Symbolic.symbolic import (Var, Const, Expr, Func, BinOp)
import VeraGridEngine.Utils.Symbolic.symbolic as sym
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory


@dataclass(frozen=True)
class DynamicLiftRecord:
    variable: Var
    derivative: Var
    original_residual: Expr
    normalized_coefficient: Expr
    time_constant: Expr


@dataclass(frozen=True)
class DynamicLiftResult:
    block: Block
    lifted: tuple[DynamicLiftRecord, ...]
    remaining_algebraic_variables: tuple[Var, ...]


def dynamic_lift_affine_algebraics(
        block: Block,
        vfactory: VarFactory,
        time_constants: float | Mapping[str | Var, float | Expr] = 1.0e-4,
        variables: Collection[str | Var] | None = None,
) -> DynamicLiftResult:
    """Promote selected affine algebraic assignments to relaxation states.

    For an algebraic variable and matched residual ``(z, g)`` with
    ``g = a*z + b`` and a
    coefficient ``a`` independent of every solver variable, create

    ``dot(z) = -g / (a*tau)``.

    This normalizes residual polarity automatically. The returned block is a
    flattened deep copy; the input block is never mutated. Equations nonlinear
    in their paired variable are rejected transactionally.
    """
    lifted_block = copy.deepcopy(block)
    root_solver_uids = {
        variable.uid for variable in (
            lifted_block.state_vars + lifted_block.algebraic_vars + lifted_block.diff_vars
        )
    }
    child_solver_uids = {
        variable.uid
        for child in lifted_block.get_all_blocks()[1:]
        for variable in (child.state_vars + child.algebraic_vars + child.diff_vars)
    }
    # ``Block.unify_blocks`` intentionally retains children. Avoid collecting
    # them a second time when a caller passes an already-unified model.
    if not root_solver_uids.intersection(child_solver_uids):
        lifted_block.unify_blocks()
    lifted_block.children = []
    if len(lifted_block.algebraic_vars) != len(lifted_block.algebraic_eqs):
        raise ValueError(
            "Dynamic lifting requires equally sized algebraic variable/equation lists"
        )

    requested_uids: set[int] | None = None
    requested_names: set[str] | None = None
    if variables is not None:
        requested_uids = {item.uid for item in variables if isinstance(item, Var)}
        requested_names = {str(item) for item in variables if isinstance(item, str)}
        known_names = {variable.name for variable in lifted_block.algebraic_vars}
        missing = requested_names - known_names
        if missing:
            raise KeyError("Unknown algebraic variables: " + ", ".join(sorted(missing)))

    def selected(variable: Var) -> bool:
        return variables is None or variable.uid in requested_uids or variable.name in requested_names

    def tau_for(variable: Var) -> Expr:
        if not isinstance(time_constants, Mapping):
            value = time_constants
        elif variable in time_constants:
            value = time_constants[variable]
        elif variable.name in time_constants:
            value = time_constants[variable.name]
        else:
            raise KeyError(f"Missing relaxation time constant for {variable.name!r}")
        result = value if isinstance(value, Expr) else Const(float(value))
        if isinstance(result, Const) and (result.value is None or float(result.value) <= 0.0):
            raise ValueError(f"Time constant for {variable.name!r} must be positive")
        return result

    solver_uids = {
        variable.uid for variable in (
            lifted_block.state_vars + lifted_block.algebraic_vars + lifted_block.diff_vars
        )
    }
    selected_variables = [
        variable for variable in lifted_block.algebraic_vars if selected(variable)
    ]
    candidate_data = {}
    for variable in selected_variables:
        candidates = []
        for equation_index, residual in enumerate(lifted_block.algebraic_eqs):
            coefficient = residual.diff(variable).simplify()
            if isinstance(coefficient, Const) and coefficient.value in (None, 0, 0.0):
                continue
            second = coefficient.diff(variable).simplify()
            if not isinstance(second, Const) or second.value not in (0, 0.0):
                continue
            if any(item.uid in solver_uids for item in coefficient.get_vars()):
                continue
            residual_width = sum(
                item.uid in solver_uids for item in residual.get_vars()
            )
            candidates.append((residual_width, equation_index, residual, coefficient))
        if not candidates:
            raise ValueError(
                f"No affine residual with solver-independent coefficient defines "
                f"{variable.name!r}; its defining equation may be nonlinear"
            )
        candidate_data[variable.uid] = sorted(candidates, key=lambda item: (item[0], item[1]))

    # Maximum bipartite matching prevents two promoted variables from consuming
    # the same residual in a composed block whose lists are not positionally aligned.
    equation_to_variable = {}
    variable_choice = {}

    def assign(variable: Var, visited: set[int]) -> bool:
        for _width, equation_index, residual, coefficient in candidate_data[variable.uid]:
            if equation_index in visited:
                continue
            visited.add(equation_index)
            incumbent = equation_to_variable.get(equation_index)
            if incumbent is None or assign(incumbent, visited):
                equation_to_variable[equation_index] = variable
                variable_choice[variable.uid] = (
                    equation_index, residual, coefficient
                )
                return True
        return False

    for variable in selected_variables:
        if not assign(variable, set()):
            raise ValueError(
                f"Could not find a one-to-one affine residual matching for {variable.name!r}"
            )

    variable_indices = {
        variable.uid: index
        for index, variable in enumerate(lifted_block.algebraic_vars)
    }
    planned = []
    for variable in selected_variables:
        index, residual, coefficient = variable_choice[variable.uid]
        tau = tau_for(variable)
        derivative = vfactory.add_diff_var(
            f"dt_1_lift_{variable.name}", base_var=variable
        )
        rhs = -residual / (coefficient * tau)
        planned.append((
            variable_indices[variable.uid], index, variable, residual,
            coefficient, tau, derivative, rhs,
        ))

    lifted_variable_indices = {item[0] for item in planned}
    lifted_equation_indices = {item[1] for item in planned}
    lifted_block.algebraic_vars = [
        variable for index, variable in enumerate(lifted_block.algebraic_vars)
        if index not in lifted_variable_indices
    ]
    lifted_block.algebraic_eqs = [
        equation for index, equation in enumerate(lifted_block.algebraic_eqs)
        if index not in lifted_equation_indices
    ]
    records = []
    for (_variable_index, _equation_index, variable, residual, coefficient,
         tau, derivative, rhs) in planned:
        lifted_block.state_vars.append(variable)
        lifted_block.state_eqs.append(rhs)
        lifted_block.diff_vars.append(derivative)
        # On a consistently initialized DAE manifold the relaxation residual,
        # and therefore the new derivative, is zero. Avoid introducing an
        # explicit-initialization dependency cycle through other algebraics.
        lifted_block.diff_init_eqs[derivative] = Const(0.0)
        records.append(DynamicLiftRecord(
            variable=variable,
            derivative=derivative,
            original_residual=residual,
            normalized_coefficient=coefficient,
            time_constant=tau,
        ))

    return DynamicLiftResult(
        block=lifted_block,
        lifted=tuple(records),
        remaining_algebraic_variables=tuple(lifted_block.algebraic_vars),
    )


def dynamic_lift_balanced_dq_to_abc(
        block: Block,
        vfactory: VarFactory,
        abc_variables: Collection[str],
        d_variable: str,
        q_variable: str,
        cosine_variable: str,
        sine_variable: str,
        time_constant: float | Expr = 1.0e-4,
) -> DynamicLiftResult:
    """Replace a coupled zero-sequence-free Park system by three states.

    The source block must contain three algebraic phase variables constrained
    by their d/q Park equations and ``a + b + c = 0``.  Treating those
    equations independently requires division by angle-dependent coefficients;
    this group transformation instead applies the exact inverse Park map and
    relaxes all phases with one time constant.  Equal time constants preserve
    the zero-sum phase invariant.
    """
    names = tuple(abc_variables)
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("abc_variables must contain three distinct names")
    lifted_block = copy.deepcopy(block)
    lifted_block.unify_blocks()
    lifted_block.children = []

    def unique(name: str, candidates: Collection[Var]) -> Var:
        matches = [variable for variable in candidates if variable.name == name]
        if len(matches) != 1:
            raise KeyError(f"Expected one variable named {name!r}, found {len(matches)}")
        return matches[0]

    phases = tuple(unique(name, lifted_block.algebraic_vars) for name in names)
    all_variables = lifted_block.get_all_vars()
    d_value = unique(d_variable, all_variables)
    q_value = unique(q_variable, all_variables)
    cosine = unique(cosine_variable, all_variables)
    sine = unique(sine_variable, all_variables)
    phase_uids = {variable.uid for variable in phases}
    coupled_indices = [
        index for index, equation in enumerate(lifted_block.algebraic_eqs)
        if phase_uids.issubset({variable.uid for variable in equation.get_vars()})
    ]
    if len(coupled_indices) != 3:
        raise ValueError(
            "Expected exactly three coupled d/q/zero-sequence phase equations, "
            f"found {len(coupled_indices)}"
        )

    tau = time_constant if isinstance(time_constant, Expr) else Const(float(time_constant))
    if isinstance(tau, Const) and (tau.value is None or float(tau.value) <= 0.0):
        raise ValueError("time_constant must be positive")
    half = Const(0.5)
    sqrt3half = Const(float(np.sqrt(3.0) / 2.0))
    commands = (
        d_value * cosine + q_value * sine,
        d_value * (-half * cosine + sqrt3half * sine)
        + q_value * (-half * sine - sqrt3half * cosine),
        d_value * (-half * cosine - sqrt3half * sine)
        + q_value * (-half * sine + sqrt3half * cosine),
    )

    phase_indices = {variable.uid for variable in phases}
    lifted_block.algebraic_vars = [
        variable for variable in lifted_block.algebraic_vars
        if variable.uid not in phase_indices
    ]
    coupled_index_set = set(coupled_indices)
    lifted_block.algebraic_eqs = [
        equation for index, equation in enumerate(lifted_block.algebraic_eqs)
        if index not in coupled_index_set
    ]
    records = []
    for variable, command in zip(phases, commands):
        derivative = vfactory.add_diff_var(
            f"dt_1_lift_{variable.name}", base_var=variable
        )
        residual = variable - command
        lifted_block.state_vars.append(variable)
        lifted_block.diff_vars.append(derivative)
        lifted_block.state_eqs.append(-residual / tau)
        lifted_block.diff_init_eqs[derivative] = Const(0.0)
        records.append(DynamicLiftRecord(
            variable=variable,
            derivative=derivative,
            original_residual=residual,
            normalized_coefficient=Const(1.0),
            time_constant=tau,
        ))
    return DynamicLiftResult(
        block=lifted_block,
        lifted=tuple(records),
        remaining_algebraic_variables=tuple(lifted_block.algebraic_vars),
    )


def promote_trig_algebraics_exact(
        block: Block,
        cosine_variable: str,
        sine_variable: str,
        angle_variable: str,
) -> DynamicLiftResult:
    """Promote a differential sine/cosine lift to exact ODE states.

    ``trig_transform`` represents sine and cosine as algebraics constrained by
    equations containing their derivatives.  This transformation changes only
    their structural classification: it reuses those derivative variables and
    installs ``dot(cos)=-dot(angle)*sin`` and
    ``dot(sin)=dot(angle)*cos`` as state equations.  No relaxation or
    approximation is introduced.
    """
    lifted_block = copy.deepcopy(block)
    lifted_block.unify_blocks()
    lifted_block.children = []

    def unique(name: str, candidates: Collection[Var]) -> Var:
        matches = [variable for variable in candidates if variable.name == name]
        if len(matches) != 1:
            raise KeyError(f"Expected one variable named {name!r}, found {len(matches)}")
        return matches[0]

    cosine = unique(cosine_variable, lifted_block.algebraic_vars)
    sine = unique(sine_variable, lifted_block.algebraic_vars)
    angle = unique(angle_variable, lifted_block.get_all_vars())
    if cosine.diff_var is None or sine.diff_var is None or angle.diff_var is None:
        raise ValueError("Cosine, sine, and angle variables must have derivatives")
    d_cosine, d_sine, d_angle = cosine.diff_var, sine.diff_var, angle.diff_var
    angle_state_indices = [
        index for index, variable in enumerate(lifted_block.state_vars)
        if variable.uid == angle.uid
    ]
    if len(angle_state_indices) != 1:
        raise ValueError(
            f"Expected {angle.name!r} to be one explicit state, found "
            f"{len(angle_state_indices)} occurrences"
        )
    angle_state_index = angle_state_indices[0]
    if angle_state_index >= len(lifted_block.state_eqs):
        raise ValueError(f"Missing explicit state equation for {angle.name!r}")
    angle_rhs = lifted_block.state_eqs[angle_state_index]
    differential_uids = {variable.uid for variable in lifted_block.diff_vars}
    if any(variable.uid in differential_uids for variable in angle_rhs.get_vars()):
        raise ValueError(
            f"Angle state equation for {angle.name!r} is not explicit"
        )

    derivative_equation_indices = []
    for derivative in (d_cosine, d_sine):
        matches = [
            index for index, equation in enumerate(lifted_block.algebraic_eqs)
            if any(variable.uid == derivative.uid for variable in equation.get_vars())
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one kinematic equation for {derivative.name!r}, found {len(matches)}"
            )
        derivative_equation_indices.append(matches[0])
    if len(set(derivative_equation_indices)) != 2:
        raise ValueError("Sine and cosine derivatives must have distinct equations")

    promoted_uids = {cosine.uid, sine.uid}
    lifted_block.algebraic_vars = [
        variable for variable in lifted_block.algebraic_vars
        if variable.uid not in promoted_uids
    ]
    removed_equations = set(derivative_equation_indices)
    lifted_block.algebraic_eqs = [
        equation for index, equation in enumerate(lifted_block.algebraic_eqs)
        if index not in removed_equations
    ]
    # trig_transform creates an additional derivative alias for the angle.
    # The angle is already a state, so retain its canonical derivative only.
    lifted_block.diff_vars = [
        derivative for derivative in lifted_block.diff_vars
        if derivative.uid in {d_cosine.uid, d_sine.uid}
        or derivative.base_var is None
        or derivative.base_var.uid != angle.uid
        or derivative.uid == d_angle.uid
    ]
    lifted_block.state_vars.extend([cosine, sine])
    lifted_block.state_eqs.extend([-angle_rhs * sine, angle_rhs * cosine])
    lifted_block.diff_init_eqs[d_cosine] = -angle_rhs * sine
    lifted_block.diff_init_eqs[d_sine] = angle_rhs * cosine
    records = (
        DynamicLiftRecord(cosine, d_cosine, cosine - sym.cos(angle), Const(1.0), Const(0.0)),
        DynamicLiftRecord(sine, d_sine, sine - sym.sin(angle), Const(1.0), Const(0.0)),
    )
    return DynamicLiftResult(
        block=lifted_block,
        lifted=records,
        remaining_algebraic_variables=tuple(lifted_block.algebraic_vars),
    )


def ml_positive_part(
    vf: VarFactory,
    u: Expr,
    name: str = '',
    scale_balance: Expr | None = None,
    scale_complementarity: Expr | None = None,
    scale_equalities: Expr | None = None,
):
    u_plus1 = vf.add_var('u_plus1_'+name)
    u_plus2 = vf.add_var('u_plus2_'+name)
    u_minus1 = vf.add_var('u_minus1_' +name)
    u_minus2 = vf.add_var('u_minus2_' +name)
    s_bal = Const(1.0) if scale_balance is None else scale_balance
    s_comp = Const(1.0) if scale_complementarity is None else scale_complementarity
    s_eq = Const(1.0) if scale_equalities is None else scale_equalities
    positive_part_block = Block(
        algebraic_eqs=[
            (u - (u_plus1*u_plus2 - u_minus1*u_minus2)) / s_bal,
            (u_plus1*u_minus1) / s_comp,
            (u_plus1 - u_plus2) / s_eq,
            (u_minus1 - u_minus2) / s_eq,
        ],
        algebraic_vars=[u_plus1, u_plus2, u_minus1, u_minus2],
        init_eqs={
            u_plus1 : sym.sqrt(sym.max(u, Const(0))),
            u_plus2 : u_plus1,
            u_minus1: sym.sqrt(sym.max(-u, Const(0))),
            u_minus2: u_minus1,
        }
    )
    return positive_part_block, u_plus1*u_plus2, u_minus1*u_minus2

def ml_positive_part_alt(vf: VarFactory, A:Expr, name:str=''):
    u_plus1 = vf.add_var('u_plus1_'+name)
    u_plus2 = vf.add_var('u_plus2_'+name)
    u_plus3 = vf.add_var('abs_'+name)
    positive_part_block = Block(
        algebraic_eqs=[
            u_plus1-u_plus2,
            u_plus1*u_plus2-u_plus3,
            u_plus1*u_plus2*u_plus3 - u_plus3*A,
        ],
        algebraic_vars=[u_plus1, u_plus2, u_plus3]
    )
    return positive_part_block, (u_plus3), (A-u_plus3) 

def ml_max(vf: VarFactory, u:Expr, v:Expr, alternative:bool = True, name:str=''):
    if not alternative:
        hv_var, block_hv = ml_heaviside(vf, u-v)
        max_expr = hv_var*u + (Const(1)-hv_var)*v
    else:
        pos_part_uv, neg_part_uv, block_hv = ml_positive_part_alt(vf, u-v)
        max_expr = u + neg_part_uv
    return max_expr, block_hv

def ml_hard_sat(
    vf: VarFactory,
    u: Expr,
    u_min: Expr,
    u_max: Expr,
    name: str = "",
    alternative_positive_part: bool = False,
    scale_balance: Expr | None = None,
    scale_complementarity: Expr | None = None,
    scale_equalities: Expr | None = None,
):
    if alternative_positive_part:
        block1, expr1_plus, expr1_minus = ml_positive_part_alt(vf, u - u_max, name + "_max")
        block2, expr2_plus, expr2_minus = ml_positive_part_alt(vf, u - u_min, name + "_min")
    else:
        block1, expr1_plus, expr1_minus = ml_positive_part(
            vf,
            u - u_max,
            name + "_max",
            scale_balance=scale_balance,
            scale_complementarity=scale_complementarity,
            scale_equalities=scale_equalities,
        )
        block2, expr2_plus, expr2_minus = ml_positive_part(
            vf,
            u - u_min,
            name + "_min",
            scale_balance=scale_balance,
            scale_complementarity=scale_complementarity,
            scale_equalities=scale_equalities,
        )
    final_expr = u_min + expr2_plus - expr1_plus 
    final_block = Block(
        algebraic_eqs=list(block1.algebraic_eqs) + list(block2.algebraic_eqs),
        algebraic_vars=list(block1.algebraic_vars) + list(block2.algebraic_vars),
        init_eqs={**dict(block1.init_eqs), **dict(block2.init_eqs)},
        children=list(block1.children) + list(block2.children),
    )
    return final_block, final_expr 


def smooth_hard_sat(
    vf: VarFactory,
    u: Expr,
    u_min: Expr,
    u_max: Expr,
    lam: Expr | float = 1e-6,
):
    """Smooth approximation of ``hard_sat(u, u_min, u_max)``.

    The expression converges pointwise to exact clipping on ``[u_min, u_max]``
    as ``lam`` approaches zero:

        0.5 * (u_min + u_max
               + sqrt((u - u_min)^2 + lam)
               - sqrt((u - u_max)^2 + lam))

    ``lam`` has squared units of ``u`` and should be positive.
    """

    lam_expr = vf.add_const(float(lam)) if isinstance(lam, (int, float)) else lam
    half = vf.add_const(0.5)
    return half * (
        u_min
        + u_max
        + sym.sqrt((u - u_min) * (u - u_min) + lam_expr)
        - sym.sqrt((u - u_max) * (u - u_max) + lam_expr)
    )


def ml_smooth_sqrt(
    vf: VarFactory,
    x: Expr,
    name: str = "",
):
    """Multilinear auxiliary representation of ``sqrt(x)``.

    The runtime equations are bilinear/linear:
    ``root_a * root_b = x``, ``root_a = root_b``, ``root_a = pos_a * pos_b``,
    and ``pos_a = pos_b``. The last two equations select the non-negative root
    branch when initialized consistently.
    """

    root_a = vf.add_var("sqrt_a_" + name)
    root_b = vf.add_var("sqrt_b_" + name)
    pos_a = vf.add_var("sqrt_pos_a_" + name)
    pos_b = vf.add_var("sqrt_pos_b_" + name)
    block = Block(
        algebraic_eqs=[
            root_a * root_b - x,
            root_a - root_b,
            root_a - pos_a * pos_b,
            pos_a - pos_b,
        ],
        algebraic_vars=[root_a, root_b, pos_a, pos_b],
        init_eqs={
            root_a: sym.sqrt(x),
            root_b: root_a,
            pos_a: sym.sqrt(root_a),
            pos_b: pos_a,
        },
    )
    return block, root_a


def ml_smooth_hard_sat(
    vf: VarFactory,
    u: Expr,
    u_min: Expr,
    u_max: Expr,
    lam: Expr | float = 1e-6,
    name: str = "",
):
    """Multilinearized smooth approximation of ``hard_sat``.

    This is the auxiliary-variable form of ``smooth_hard_sat``. Runtime equations
    use only linear and bilinear products; direct ``sqrt`` only appears in
    initialization equations.
    """

    lam_expr = vf.add_const(float(lam)) if isinstance(lam, (int, float)) else lam
    half = vf.add_const(0.5)

    du_min = vf.add_var("du_min_" + name)
    du_max = vf.add_var("du_max_" + name)
    du_min_aux = vf.add_var("du_min_aux_" + name)
    du_max_aux = vf.add_var("du_max_aux_" + name)
    rad_min = vf.add_var("rad_min_" + name)
    rad_max = vf.add_var("rad_max_" + name)

    shift_block = Block(
        algebraic_eqs=[
            du_min - (u - u_min),
            du_max - (u - u_max),
            du_min_aux - du_min,
            du_max_aux - du_max,
            rad_min - (du_min * du_min_aux + lam_expr),
            rad_max - (du_max * du_max_aux + lam_expr),
        ],
        algebraic_vars=[du_min, du_max, du_min_aux, du_max_aux, rad_min, rad_max],
        init_eqs={
            du_min: u - u_min,
            du_max: u - u_max,
            du_min_aux: du_min,
            du_max_aux: du_max,
            rad_min: du_min * du_min_aux + lam_expr,
            rad_max: du_max * du_max_aux + lam_expr,
        },
    )
    sqrt_min_block, sqrt_min = ml_smooth_sqrt(vf, rad_min, name + "_min")
    sqrt_max_block, sqrt_max = ml_smooth_sqrt(vf, rad_max, name + "_max")
    block = Block(children=[shift_block, sqrt_min_block, sqrt_max_block])
    return block, half * (u_min + u_max + sqrt_min - sqrt_max)


def ml_soft_sign(
    vf: VarFactory,
    x: Expr,
    lamda: Expr | float | int = 1.0e-6,
    name: str = "",
):
    """Smooth sign approximation with an algebraic auxiliary variable.

    This represents ``softsign_x = x / sqrt(x^2 + lamda)`` without division or
    direct square roots in runtime equations. The generated equations are
    bilinear/linear, with direct ``sqrt`` only used for initialization.
    ``lamda`` has squared units of ``x`` and should be positive.
    """

    lamda_expr = Const(float(lamda)) if isinstance(lamda, (float, int)) else lamda
    softsign_x = vf.add_var("softsign_" + name)
    x_aux = vf.add_var("softsign_x_aux_" + name)
    radicand = vf.add_var("softsign_radicand_" + name)
    denom_a = vf.add_var("softsign_denom_a_" + name)
    denom_b = vf.add_var("softsign_denom_b_" + name)
    denom_pos_a = vf.add_var("softsign_denom_pos_a_" + name)
    denom_pos_b = vf.add_var("softsign_denom_pos_b_" + name)

    block = Block(
        algebraic_vars=[softsign_x, x_aux, radicand, denom_a, denom_b, denom_pos_a, denom_pos_b],
        algebraic_eqs=[
            x_aux - x,
            radicand - (x * x_aux + lamda_expr),
            denom_a * denom_b - radicand,
            denom_a - denom_b,
            denom_a - denom_pos_a * denom_pos_b,
            denom_pos_a - denom_pos_b,
            softsign_x * denom_a - x,
        ],
        init_eqs={
            x_aux: x,
            radicand: x * x_aux + lamda_expr,
            denom_a: sym.sqrt(radicand),
            denom_b: denom_a,
            denom_pos_a: sym.sqrt(denom_a),
            denom_pos_b: denom_pos_a,
            softsign_x: x / sym.sqrt(radicand),
        },
    )

    return block, softsign_x


def ml_three_phase_carrier_pwm_direct(
    vf: VarFactory,
    ref_a: Expr,
    ref_b: Expr,
    ref_c: Expr,
    carrier: Expr,
    lamda: Expr | float | int = 1.0e-6,
    name: str = "",
):
    """Smooth three-phase carrier PWM comparator without boolean guards.

    Gates are computed as ``0.5 * (1 + soft_sign(ref - carrier))`` using
    ``ml_soft_sign``. The outputs approach the boolean comparator gates as
    ``lamda`` approaches zero, while remaining continuous in ``ref - carrier``.
    """

    gate_a = vf.add_var("gate_pwm_a_" + name)
    gate_b = vf.add_var("gate_pwm_b_" + name)
    gate_c = vf.add_var("gate_pwm_c_" + name)

    s_a = ref_a - carrier
    s_b = ref_b - carrier
    s_c = ref_c - carrier

    block_a, softsign_a = ml_soft_sign(vf, s_a, lamda=lamda, name="pwm_a_" + name)
    block_b, softsign_b = ml_soft_sign(vf, s_b, lamda=lamda, name="pwm_b_" + name)
    block_c, softsign_c = ml_soft_sign(vf, s_c, lamda=lamda, name="pwm_c_" + name)

    c_half = Const(0.5)
    c_one = Const(1.0)
    block = Block(
        algebraic_vars=[gate_a, gate_b, gate_c],
        algebraic_eqs=[
            gate_a - c_half * (c_one + softsign_a),
            gate_b - c_half * (c_one + softsign_b),
            gate_c - c_half * (c_one + softsign_c),
        ],
        init_eqs={
            gate_a: c_half * (c_one + softsign_a),
            gate_b: c_half * (c_one + softsign_b),
            gate_c: c_half * (c_one + softsign_c),
        },
        children=[block_a, block_b, block_c],
    )

    return block, gate_a, gate_b, gate_c


def ml_three_phase_carrier_pwm_direct_params(
    vf: VarFactory,
    name: str = "",
    lamda: Expr | float | int = 1.0e-6,
    ref_a0: float = 0.0,
    ref_b0: float = 0.0,
    ref_c0: float = 0.0,
    carrier0: float = 0.0,
):
    """Smooth direct three-phase carrier PWM with runtime-parameter inputs."""

    ref_a = vf.add_var("u_ref_a_pwm_" + name)
    ref_b = vf.add_var("u_ref_b_pwm_" + name)
    ref_c = vf.add_var("u_ref_c_pwm_" + name)
    carrier = vf.add_var("u_carrier_pwm_" + name)

    block, gate_a, gate_b, gate_c = ml_three_phase_carrier_pwm_direct(
        vf=vf,
        ref_a=ref_a,
        ref_b=ref_b,
        ref_c=ref_c,
        carrier=carrier,
        lamda=lamda,
        name=name,
    )
    block.event_dict.update({
        ref_a: Const(float(ref_a0)),
        ref_b: Const(float(ref_b0)),
        ref_c: Const(float(ref_c0)),
        carrier: Const(float(carrier0)),
    })

    return block, gate_a, gate_b, gate_c, ref_a, ref_b, ref_c, carrier


def mti_hard_sat(
    vf: VarFactory,
    u: Expr,
    ul: Expr,
    uu: Expr,
    yl: Expr,
    yu: Expr,
    name: str = "",
):
    """MTI hard saturation following toolbox-style mixed constraints.


    """
    b1 = vf.add_var("b1_" + name)
    b2 = vf.add_var("b2_" + name)
    b3 = vf.add_var("b3_" + name)
    b4 = vf.add_var("b4_" + name)
    y = vf.add_var("y_sat_" + name)


    y_expr = b2 * yl + b1 * b4 * u + b3 * yu

    block = Block(
        algebraic_vars=[y],
        algebraic_eqs=[
            y - y_expr,
            b1 + b2 - 1,
            b3 + b4 - 1,
        ],
        inequalities=[
            -(b1 - b2) * (u - ul),
            -(b3 - b4) * (u - uu),
        ],
        init_eqs={
            y: sym.hard_sat(u, yl, yu),
        },
        boolean_guards= {
            b1: sym.heaviside(u - ul),
            b2: 1 - sym.heaviside(u - ul),
            b3: sym.heaviside(u - uu),
            b4: 1 - sym.heaviside(u - uu),
        }
    )

    return block, y


def mti_three_phase_carrier_pwm_direct(
    vf: VarFactory,
    ref_a: Expr,
    ref_b: Expr,
    ref_c: Expr,
    carrier: Expr,
    eps: Expr | float | int | None = None,
    name: str = "",
):
    """
    Direct three-phase MTI carrier PWM comparator.

    The generated gates follow the memoryless comparator rule
    ``gate = 1`` when ``ref - carrier >= 0`` and ``gate = 0`` when
    ``ref - carrier <= 0``. Inequalities use VeraGrid's MTI convention
    ``G <= 0``.
    """
    b_a = vf.add_var("b_pwm_a_" + name)
    b_b = vf.add_var("b_pwm_b_" + name)
    b_c = vf.add_var("b_pwm_c_" + name)
    gate_a = vf.add_var("gate_pwm_a_" + name)
    gate_b = vf.add_var("gate_pwm_b_" + name)
    gate_c = vf.add_var("gate_pwm_c_" + name)

    s_a = ref_a - carrier
    s_b = ref_b - carrier
    s_c = ref_c - carrier

    if eps is None:
        inequalities = [
            -(2 * b_a - 1) * s_a,
            -(2 * b_b - 1) * s_b,
            -(2 * b_c - 1) * s_c,
        ]
    else:
        eps_expr = Const(float(eps)) if isinstance(eps, (float, int)) else eps
        inequalities = [
            b_a * (-(s_a + eps_expr)) + (1 - b_a) * (s_a - eps_expr),
            b_b * (-(s_b + eps_expr)) + (1 - b_b) * (s_b - eps_expr),
            b_c * (-(s_c + eps_expr)) + (1 - b_c) * (s_c - eps_expr),
        ]

    block = Block(
        algebraic_vars=[gate_a, gate_b, gate_c],
        algebraic_eqs=[
            gate_a - b_a,
            gate_b - b_b,
            gate_c - b_c,
        ],
        inequalities=inequalities,
        init_eqs={
            gate_a: sym.heaviside(s_a),
            gate_b: sym.heaviside(s_b),
            gate_c: sym.heaviside(s_c),
        },
        boolean_guards={
            b_a: sym.heaviside(s_a),
            b_b: sym.heaviside(s_b),
            b_c: sym.heaviside(s_c),
        },
    )

    return block, gate_a, gate_b, gate_c


def mti_three_phase_carrier_pwm_direct_params(
    vf: VarFactory,
    name: str = "",
    eps: Expr | float | int | None = None,
    ref_a0: float = 0.0,
    ref_b0: float = 0.0,
    ref_c0: float = 0.0,
    carrier0: float = 0.0,
):
    """
    Direct three-phase MTI carrier PWM comparator with input signals modeled as runtime parameters.

    This mirrors toolbox-style ``u`` inputs in VeraGrid's current block model by
    declaring ``u_ref_a``, ``u_ref_b``, ``u_ref_c`` and ``u_carrier`` in
    ``event_dict``. The MTI inequalities then switch boolean gate modes from the
    parameter residuals ``u_ref_phase - u_carrier``.
    """
    ref_a = vf.add_var("u_ref_a_pwm_" + name)
    ref_b = vf.add_var("u_ref_b_pwm_" + name)
    ref_c = vf.add_var("u_ref_c_pwm_" + name)
    carrier = vf.add_var("u_carrier_pwm_" + name)

    block, gate_a, gate_b, gate_c = mti_three_phase_carrier_pwm_direct(
        vf=vf,
        ref_a=ref_a,
        ref_b=ref_b,
        ref_c=ref_c,
        carrier=carrier,
        eps=eps,
        name=name,
    )
    block.event_dict.update({
        ref_a: Const(float(ref_a0)),
        ref_b: Const(float(ref_b0)),
        ref_c: Const(float(ref_c0)),
        carrier: Const(float(carrier0)),
    })

    return block, gate_a, gate_b, gate_c, ref_a, ref_b, ref_c, carrier


def mti_three_phase_carrier_pwm_internal_carrier_params(
    vf: VarFactory,
    name: str = "",
    omega_sw0: float = 2.0 * np.pi * 1000.0,
    direction_eps: Expr | float | int = 1.0e-9,
    ref_a0: float = 0.0,
    ref_b0: float = 0.0,
    ref_c0: float = 0.0,
    carrier0: float = -1.0,
    carrier_rising0: float = 1.0,
):
    """
    Three-phase MTI PWM with runtime-parameter references and internal triangular carrier.

    The reference signals are modeled as runtime parameters because the current
    VeraGrid MTI path has no first-class ``u`` input category. The triangular
    carrier direction is not external: it is represented by the internal boolean
    ``b_carrier_rise`` and the carrier is a continuous state with slope selected
    by that boolean.
    """
    ref_a = vf.add_var("u_ref_a_pwm_" + name)
    ref_b = vf.add_var("u_ref_b_pwm_" + name)
    ref_c = vf.add_var("u_ref_c_pwm_" + name)
    omega_sw = vf.add_var("u_omega_sw_pwm_" + name)

    carrier = vf.add_var("carrier_pwm_" + name)
    dcarrier = vf.add_diff_var("dt_carrier_pwm_" + name, base_var=carrier)
    b_carrier_rise = vf.add_var("b_carrier_rise_pwm_" + name)

    b_a = vf.add_var("b_pwm_a_" + name)
    b_b = vf.add_var("b_pwm_b_" + name)
    b_c = vf.add_var("b_pwm_c_" + name)
    gate_a = vf.add_var("gate_pwm_a_" + name)
    gate_b = vf.add_var("gate_pwm_b_" + name)
    gate_c = vf.add_var("gate_pwm_c_" + name)

    c_one = Const(1.0)
    c_two = Const(2.0)
    slope_abs = Const(2.0 / np.pi) * omega_sw
    carrier_slope = (c_two * b_carrier_rise - c_one) * slope_abs
    direction_for_switching_surface = c_one - c_two * b_carrier_rise
    direction_eps_expr = Const(float(direction_eps)) if isinstance(direction_eps, (float, int)) else direction_eps

    s_a = ref_a - carrier + direction_eps_expr * direction_for_switching_surface
    s_b = ref_b - carrier + direction_eps_expr * direction_for_switching_surface
    s_c = ref_c - carrier + direction_eps_expr * direction_for_switching_surface

    block = Block(
        state_vars=[carrier],
        state_eqs=[carrier_slope],
        diff_vars=[dcarrier],
        algebraic_vars=[gate_a, gate_b, gate_c],
        algebraic_eqs=[
            gate_a - b_a,
            gate_b - b_b,
            gate_c - b_c,
        ],
        inequalities=[
            -(c_two * b_a - c_one) * s_a,
            -(c_two * b_b - c_one) * s_b,
            -(c_two * b_c - c_one) * s_c,
            b_carrier_rise * (carrier - c_one) + (c_one - b_carrier_rise) * (-carrier - c_one),
        ],
        event_dict={
            ref_a: Const(float(ref_a0)),
            ref_b: Const(float(ref_b0)),
            ref_c: Const(float(ref_c0)),
            omega_sw: Const(float(omega_sw0)),
        },
        init_eqs={
            carrier: Const(float(carrier0)),
            gate_a: sym.heaviside(s_a),
            gate_b: sym.heaviside(s_b),
            gate_c: sym.heaviside(s_c),
        },
        boolean_guards={
            b_a: sym.heaviside(s_a),
            b_b: sym.heaviside(s_b),
            b_c: sym.heaviside(s_c),
            b_carrier_rise: sym.heaviside(Const(float(carrier_rising0)) - Const(0.5)),
        },
    )

    return block, gate_a, gate_b, gate_c, ref_a, ref_b, ref_c, omega_sw, carrier, b_carrier_rise


def mti_three_phase_carrier_pwm_scheduled_params(
    vf: VarFactory,
    time: Expr,
    name: str = "",
    transition_eps: Expr | float | int = 1.0e-9,
    ref_a0: float = 0.0,
    ref_b0: float = 0.0,
    ref_c0: float = 0.0,
    interval_start0: float = 0.0,
    half_period0: float = 5.0e-4,
    carrier_rising0: float = 1.0,
):
    """
    Scheduled-event MTI approximation of regular-sampled carrier PWM.

    The block models the procedural PWM schedule explicitly: sampled references
    and interval timing are runtime parameters, while the phase transition modes
    are MTI booleans driven by ``time - t_cross`` inequalities.
    """
    ref_a = vf.add_var("u_ref_a_sched_pwm_" + name)
    ref_b = vf.add_var("u_ref_b_sched_pwm_" + name)
    ref_c = vf.add_var("u_ref_c_sched_pwm_" + name)
    interval_start = vf.add_var("u_interval_start_sched_pwm_" + name)
    half_period = vf.add_var("u_half_period_sched_pwm_" + name)

    b_carrier_rise = vf.add_var("b_carrier_rise_sched_pwm_" + name)
    q_a = vf.add_var("q_after_a_sched_pwm_" + name)
    q_b = vf.add_var("q_after_b_sched_pwm_" + name)
    q_c = vf.add_var("q_after_c_sched_pwm_" + name)

    t_cross_a = vf.add_var("t_cross_a_sched_pwm_" + name)
    t_cross_b = vf.add_var("t_cross_b_sched_pwm_" + name)
    t_cross_c = vf.add_var("t_cross_c_sched_pwm_" + name)
    gate_a = vf.add_var("gate_pwm_a_" + name)
    gate_b = vf.add_var("gate_pwm_b_" + name)
    gate_c = vf.add_var("gate_pwm_c_" + name)

    c_one = Const(1.0)
    c_half = Const(0.5)
    eps_expr = Const(float(transition_eps)) if isinstance(transition_eps, (float, int)) else transition_eps

    def cross_time(ref: Expr) -> Expr:
        rising_cross = interval_start + c_half * (ref + c_one) * half_period
        falling_cross = interval_start + c_half * (c_one - ref) * half_period
        return b_carrier_rise * rising_cross + (c_one - b_carrier_rise) * falling_cross

    def gate_expr(q_after: Expr) -> Expr:
        return (c_one - q_after) * b_carrier_rise + q_after * (c_one - b_carrier_rise)

    tau_a = time - t_cross_a + eps_expr
    tau_b = time - t_cross_b + eps_expr
    tau_c = time - t_cross_c + eps_expr

    block = Block(
        algebraic_vars=[t_cross_a, t_cross_b, t_cross_c, gate_a, gate_b, gate_c],
        algebraic_eqs=[
            t_cross_a - cross_time(ref_a),
            t_cross_b - cross_time(ref_b),
            t_cross_c - cross_time(ref_c),
            gate_a - gate_expr(q_a),
            gate_b - gate_expr(q_b),
            gate_c - gate_expr(q_c),
        ],
        inequalities=[
            -(Const(2.0) * q_a - c_one) * tau_a,
            -(Const(2.0) * q_b - c_one) * tau_b,
            -(Const(2.0) * q_c - c_one) * tau_c,
        ],
        event_dict={
            ref_a: Const(float(ref_a0)),
            ref_b: Const(float(ref_b0)),
            ref_c: Const(float(ref_c0)),
            interval_start: Const(float(interval_start0)),
            half_period: Const(float(half_period0)),
        },
        init_eqs={
            t_cross_a: cross_time(ref_a),
            t_cross_b: cross_time(ref_b),
            t_cross_c: cross_time(ref_c),
            gate_a: gate_expr(q_a),
            gate_b: gate_expr(q_b),
            gate_c: gate_expr(q_c),
        },
        boolean_guards={
            q_a: sym.heaviside(tau_a),
            q_b: sym.heaviside(tau_b),
            q_c: sym.heaviside(tau_c),
            b_carrier_rise: sym.heaviside(Const(float(carrier_rising0)) - Const(0.5)),
        },
    )

    return (
        block,
        gate_a,
        gate_b,
        gate_c,
        t_cross_a,
        t_cross_b,
        t_cross_c,
        ref_a,
        ref_b,
        ref_c,
        interval_start,
        half_period,
        q_a,
        q_b,
        q_c,
        b_carrier_rise,
    )

def exponential_ml(vf: VarFactory, x:Expr, name:str=""):
    algebraic_eqs = list()
    algebraic_vars = list()
    y = vf.add_var('exp_' + name)
    dy = vf.add_diff_var('dt_'+y.name, base_var=y)
    
    if not isinstance(x, Var):
        u = vf.add_var(name + '_aux')
        aux_block = Block(
            algebraic_eqs=[u-x],
            algebraic_vars=[u],
            init_eqs={u:x},
        )
    else:
        aux_block = Block()
        u = x

    du = vf.add_diff_var('dt_'+u.name, base_var=u)
    children = [aux_block]
    block = Block(
        algebraic_eqs= [dy - du*y] + algebraic_eqs,
        algebraic_vars=[y] + algebraic_vars,
        diff_vars=[du, dy] ,
        init_eqs={
            y: sym.exp(u)
        }, 
        children=children
    )
    return block, y

def ml_heaviside(vf: VarFactory, u:Expr, name:str=''):
    u_plus1 = vf.add_var('u_plus1_' + name)
    u_plus2 = vf.add_var('u_plus2_' + name)
    u_minus1 = vf.add_var('u_minus1_' + name)
    u_minus2 = vf.add_var('u_minus2_' + name)
    hv = vf.add_var('hv_' + name)
    positive_part_block = Block(
        algebraic_eqs=[
            u - (u_plus1*u_plus2 - u_minus1*u_minus2),
            u - (hv*u_plus1*u_plus2 - (1-hv)*u_minus1*u_minus2),
            u_plus1*u_minus1,
            u_plus1 - u_plus2,
            u_minus1 - u_minus2,
        ],
        algebraic_vars=[hv, u_plus1, u_plus2, u_minus1, u_minus2],
        init_eqs={
            hv      : sym.heaviside(u), 
            u_plus1 : sym.sqrt(sym.max(u, Const(0))),
            u_plus2 : u_plus1,
            u_minus1: sym.sqrt(sym.max(-u, Const(0))),
            u_minus2: u_minus1,
        }
    )
    return positive_part_block, hv


def ml_heaviside_sigmoid(
    vf: VarFactory,
    u: Expr,
    name: str = '',
    k: Expr = Const(20.0),
    exp_clip: Expr = Const(60.0),
):
    hv = vf.add_var('hv_' + name)
    sig_arg = sym.max(sym.min(k * u, exp_clip), -exp_clip)
    hv_rhs = Const(1.0) / (Const(1.0) + sym.exp(-sig_arg))
    block = Block(
        algebraic_eqs=[hv - hv_rhs],
        algebraic_vars=[hv],
        init_eqs={hv: hv_rhs},
    )
    return block, hv


def ml_hard_sat_sigmoid(
    vf: VarFactory,
    u: Expr,
    u_min: Expr,
    u_max: Expr,
    name: str = '',
    k: Expr = Const(20.0),
    exp_clip: Expr = Const(60.0),
):
    block_min, h_min = ml_heaviside_sigmoid(vf, u - u_min, name=f"{name}_min", k=k, exp_clip=exp_clip)
    block_max, h_max = ml_heaviside_sigmoid(vf, u - u_max, name=f"{name}_max", k=k, exp_clip=exp_clip)
    sat_expr = u_min + (u - u_min) * h_min - (u - u_max) * h_max
    block = Block(children=[block_min, block_max])
    return block, sat_expr

def ml_piecewise_aux(vf: VarFactory, x: Expr, name: str = None):
    x, all_blocks, _ = ml_piecewise(vf, x, name=name, counter=0)
    block = Block(children = all_blocks)
    return block, x

def ml_piecewise(
        vf: VarFactory,
        x: Expr,
        name: str = '',
        counter: int = 0
    ) -> tuple[Expr, List[Block], int]:
    """
    Recursively traverse expression x, replacing Heaviside() nodes with
    ml_heaviside() outputs and collecting Blocks.
    Each Heaviside gets a unique suffix via an integer counter.
    """
    # --- base cases ---
    if isinstance(x, Var) or isinstance(x, Const):
        return x, list(), counter

    # --- Heaviside function case ---
    if isinstance(x, Func) and x.op == "heaviside":
        local_name = f"{name or 'hv'}_{counter}"
        block, y = ml_heaviside(vf, x.arg, local_name)
        return y, [block], counter + 1

    # --- General function (like sin, exp, etc.) ---
    if isinstance(x, Func):
        new_arg, blocks, counter = ml_piecewise(vf, x.arg, name, counter)
        x = Func(new_arg, x.op)
        return x, blocks, counter

    # --- Binary operator case (e.g. +, -, *, /, **) ---
    if isinstance(x, BinOp):
        left_new, blocks_left, counter = ml_piecewise(vf, x.left, name, counter)
        right_new, blocks_right, counter = ml_piecewise(vf, x.right, name, counter)
        x = BinOp(left=left_new, op= x.op, right=right_new)
        all_blocks = blocks_left + blocks_right
        return x, all_blocks, counter

    # --- fallback ---
    raise ValueError(f"Unsupported expression type: {type(x)}")

def park_transform(vf: VarFactory, X:Var, ang:Var, delta:Var, multilinear:bool = False):
    Xd = vf.add_var('Xd')
    Xq = vf.add_var('Xq')
    sqrt_2 = Const(np.sqrt(2))
    if not multilinear:
        block = Block(
            algebraic_eqs= [
            Xd - X*sym.cos(delta - ang)*sqrt_2,
            Xq - X*sym.sin(delta - ang)*sqrt_2,
            ],
            algebraic_vars= [Xd, Xq]
        )
    else:
        u_cos = vf.add_var('u_cos')
        u_sin = vf.add_var('u_sin')
        d_u_cos = vf.add_diff_var('d_u_cos', base_var=u_cos)
        d_u_sin = vf.add_diff_var('d_u_sin', base_var=u_sin)
        d_delta = vf.add_diff_var('d_delta', base_var=delta)
        d_ang = vf.add_diff_var('d_ang', base_var=ang)
        block = Block(
            algebraic_eqs= [
            Xd - X*u_cos*sqrt_2,
            Xq - X*u_sin*sqrt_2,
            ],
            algebraic_vars= [u_cos, u_cos, Xd, Xq],
            differential_eqs=[
                d_u_cos + (d_delta - d_ang)*u_sin,
                d_u_sin - (d_delta - d_ang)*u_cos,
            ],
            diff_vars=[d_u_sin, d_u_cos, d_delta, d_ang]
        )
    
    return Xd, Xq, block

def trig_transform(vf: VarFactory, u:Expr, type = 'usual'):
    aux_block = None
    if not isinstance(u, Var):
        x = vf.add_var('x')
        aux_block = Block(
            algebraic_eqs=[x - u],
            algebraic_vars=[x],
            init_eqs={x:u}
        )
    else:
        x = u
    if type == 'usual':
        u_cos = vf.add_var('u_cos')
        u_sin = vf.add_var('u_sin')
        d_u_cos = vf.add_diff_var('d_u_cos', base_var=u_cos)
        d_u_sin = vf.add_diff_var('d_u_sin', base_var=u_sin)
        dx = vf.add_diff_var('d_delta', base_var=x)
        block = Block(
            algebraic_vars= [u_cos, u_sin],
            algebraic_eqs =[
                d_u_cos + (dx)*u_sin,
                d_u_sin - (dx)*u_cos,
            ],
            diff_vars=[d_u_sin, d_u_cos, dx],
            reformulated_vars=[u_sin, u_cos],
            init_eqs={
                #x:u,
                u_cos: sym.cos(x),
                u_sin: sym.sin(x),
            }
        )
    elif type == 'norm':
        u_cos = vf.add_var('u_cos')
        u_sin = vf.add_var('u_sin')
        u_cos_n = vf.add_var('u_cos_n')
        u_sin_n = vf.add_var('u_sin_n')

        norm = vf.add_var('norm')
        u_cos_aux = vf.add_var('u_cos_aux')
        u_sin_aux = vf.add_var('u_sin_aux')
        norm_aux = vf.add_var('norm_aux')

        d_u_cos = vf.add_diff_var('d_u_cos', base_var=u_cos)
        d_u_sin = vf.add_diff_var('d_u_sin', base_var=u_sin)
        dx = vf.add_diff_var('d_delta', base_var=x)
        block = Block(
            algebraic_vars= [u_cos, u_sin],
            differential_eqs=[
                d_u_cos + (dx)*u_sin,
                d_u_sin - (dx)*u_cos,
            ],
            diff_vars=[d_u_sin, d_u_cos, dx],
        )
        norm_block = Block(
            algebraic_eqs=[
                norm - norm_aux,
                u_cos - u_cos_aux,
                u_sin - u_sin_aux,
                norm*norm_aux - (u_cos*u_cos_aux + u_sin*u_sin_aux),
                u_cos_n*norm - u_cos,
                u_sin_n*norm - u_sin,
            ],
            algebraic_vars=[norm, norm_aux, u_cos_n, u_sin_n, u_sin_aux, u_cos_aux],
            init_eqs={
                x:u,
                u_cos: sym.cos(x),
                u_cos_aux: sym.cos(x),
                u_cos_n: sym.cos(x),
                u_sin: sym.sin(x),
                u_sin_aux: sym.sin(x),
                u_sin_n: sym.sin(x),
                norm: Const(1),
                norm_aux: Const(1),
            }
        )
        u_cos = u_cos_n
        u_sin = u_sin_n
        block.add(norm_block)

    elif type =='singular':
        int_xsinx = vf.add_var('int_xsinx')
        int_xcosx = vf.add_var('int_xcosx')
        dt_int_xsinx = vf.add_diff_var('dt_int_xsinx', base_var=int_xsinx)
        dt_int_xcosx = vf.add_diff_var('dt_int_xcosx', base_var=int_xcosx)
        u_cos = vf.add_var('u_cos')
        u_sin = vf.add_var('u_sin')
        d_u_cos = vf.add_diff_var('d_u_cos', base_var=u_cos)
        d_u_sin = vf.add_diff_var('d_u_sin', base_var=u_sin)
        dx = vf.add_diff_var('d_delta', base_var=x)
        block = Block(
            algebraic_vars= [u_cos, u_sin, int_xcosx, int_xsinx],
            algebraic_eqs=[
                #int_xsinx - (u_sin - (x)*u_cos), 
                #int_xcosx - ((x)*u_sin + u_cos),
                (int_xsinx + x*int_xcosx)/(1+x**2) - u_sin,
                (int_xcosx - x*int_xsinx)/(1+x**2) - u_cos,
            ],
            differential_eqs = [
                dt_int_xsinx - dx*(u_sin*(x)),
                dt_int_xcosx - dx*(u_cos*(x)),
            ],
            diff_vars=[ dx, dt_int_xcosx, dt_int_xsinx],
            init_eqs={
                #x:u,
                u_cos: sym.cos(x),
                u_sin: sym.sin(x),
                int_xsinx: sym.sin(x) - x*sym.cos(x),
                int_xcosx: x*sym.sin(x) + sym.cos(x),
                dt_int_xsinx: dx*sym.sin(x)*x,
                dt_int_xcosx: dx*sym.cos(x)*x,
            }
        )
    else:
        u_cos = vf.add_var('u_cos')
        u_sin = vf.add_var('u_sin')
        int_sin2 = vf.add_var('int_sin2')
        int_sin2cos2 = vf.add_var('int_sin2cos2')
        dt_int_sin2 = vf.add_diff_var('dt_int_sin2', base_var=int_sin2)
        dt_int_sin2cos2 = vf.add_diff_var('dt_int_sin2cos2', base_var=int_sin2cos2)
        dx = vf.add_diff_var('d_delta', base_var=x)
        block = Block(
            algebraic_vars= [u_cos, u_sin, int_sin2, int_sin2cos2],
            algebraic_eqs=[
                int_sin2 - 0.5*((x)-u_sin*u_cos), 
                int_sin2cos2 - 1/32*(4*(x) - (4*u_cos**3*u_sin - 4*u_sin**3*u_cos)),
            ],
            differential_eqs = [       
                dt_int_sin2 - (dx)*u_sin**2,
                dt_int_sin2cos2 - (dx)*u_sin**2*u_cos**2
            ],
            init_eqs={
                u_cos: sym.cos(x),
                u_sin: sym.sin(x),
                int_sin2: 0.5*(x - u_sin*u_cos),
                int_sin2cos2: 1/32*(4*x - 4*u_cos**3*u_sin + 4*u_sin**3*u_cos),
                dt_int_sin2: dx*u_sin**2,
                dt_int_sin2cos2: dx*u_sin**2*u_cos**2,
            },
            diff_vars=[dx, dt_int_sin2, dt_int_sin2cos2],
        )

    if aux_block is not None:
        block.add(aux_block)
    return block, u_cos, u_sin

def trig_transform_diff(vf: VarFactory, delta1:Expr, delta2:Expr):
    aux_block = None
    
    if not isinstance(delta1, Var):
        x1 = vf.add_var('x1')
        aux_block1 = Block(
            algebraic_eqs=[x1 - delta1],
            algebraic_vars=[x1],
            init_eqs={x1: delta1}
        )
    else:
        x1 = delta1
        aux_block1 = None
    
    if not isinstance(delta2, Var):
        x2 = vf.add_var('x2')
        aux_block2 = Block(
            algebraic_eqs=[x2 - delta2],
            algebraic_vars=[x2],
            init_eqs={x2: delta2}
        )
    else:
        x2 = delta2
        aux_block2 = None
    
    diff = x1 - x2
    
    u_cos = vf.add_var('u_cos')
    u_sin = vf.add_var('u_sin')
    d_u_cos = vf.add_diff_var('d_u_cos', base_var=u_cos)
    d_u_sin = vf.add_diff_var('d_u_sin', base_var=u_sin)
    d_delta1 = vf.add_diff_var('d_delta1', base_var=x1)
    d_delta2 = vf.add_diff_var('d_delta2', base_var=x2)
    
    block = Block(
        algebraic_vars=[u_cos, u_sin],
        algebraic_eqs=[
            d_u_cos + (d_delta1 - d_delta2) * u_sin,
            d_u_sin - (d_delta1 - d_delta2) * u_cos,
        ],
        diff_vars=[d_u_sin, d_u_cos, d_delta1, d_delta2],
        reformulated_vars=[u_sin, u_cos],
        init_eqs={
            u_cos: sym.cos(diff),
            u_sin: sym.sin(diff),
        }
    )
    
    if aux_block1 is not None:
        block.add(aux_block1)
    if aux_block2 is not None:
        block.add(aux_block2)
    
    return block, u_cos, u_sin

def ml_f_exc(vf: VarFactory, In:Expr):
    ml_block1, hv1 = ml_heaviside(vf, Const(0.433) - In)
    ml_block2, hv2 = ml_heaviside(vf, Const(0.75) - In)
    ml_block3, hv3 = ml_heaviside(vf, Const(1.0) - In)

    In_aux = vf.add_var('In_aux')
    sqrt1 = vf.add_var('sqrt1')
    sqrt2 = vf.add_var('sqrt2')
    exp1 = (Const(1) - Const(0.577) * In)
    exp2 = (Const(0.75)  - In*In_aux)
    exp3 = (Const(1.732) - In * Const(1.732))
    b = (exp1 - sqrt1) * hv1
    c = (sqrt1 - exp3) * hv2
    d =  exp3 * hv3
    res_block = Block(
        algebraic_eqs=[
            In_aux -In,
            sqrt1 -sqrt2,
            hv2*(sqrt1*sqrt2 - exp2),
        ],
        algebraic_vars=[In_aux, sqrt1, sqrt2],
        init_eqs={
            In_aux:In,
            sqrt1: sym.sqrt(sym.max(exp2, Const(1e-6))),
            sqrt2: sqrt1,
        },
        children = [ml_block1, ml_block2, ml_block3]
    )
    return res_block, b + c + d
