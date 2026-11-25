# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Exact, shared auxiliary-variable reformulations of continuous Block equations."""
from __future__ import annotations

import copy
import numpy as np
from typing import Mapping

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic import symbolic as sym
from VeraGridEngine.Utils.Symbolic.symbolic import BinOp, Comparison, Const, Expr, Func, Func2, UnOp, Var


def _expression_key(expr: Expr) -> tuple:
    """Structural identity uses variable UIDs, never names or expression UIDs."""
    if isinstance(expr, Var):
        return ('var', expr.uid)
    if isinstance(expr, Const):
        return ('const', expr.value)
    if isinstance(expr, UnOp):
        return ('unary', expr.op, _expression_key(expr.operand))
    if isinstance(expr, BinOp):
        children = (_expression_key(expr.left), _expression_key(expr.right))
        if expr.op in ('+', '*'):
            children = tuple(sorted(children, key=repr))
        return ('binary', expr.op, *children)
    if isinstance(expr, Func):
        return ('function', expr.op, _expression_key(expr.arg))
    if isinstance(expr, Func2):
        return ('function2', expr.name, _expression_key(expr.arg1), _expression_key(expr.arg2))
    raise ValueError(f'Unsupported expression type: {type(expr).__name__}')


class _BlockMultilinearizer:
    """One conversion context, shared by all equations and children of a block."""

    def __init__(self, block: Block, vf: VarFactory,
                 substitutions: Mapping[Expr, Var] | None,
                 derivatives: Mapping[Var, Expr] | None) -> None:
        self.block = copy.deepcopy(block)
        self.vf = vf
        self.original_blocks = self.block.get_all_blocks()
        self.aux = Block(name='multilinear_auxiliaries')
        self.cache: dict[tuple, Expr] = {}
        self.aliases: dict[int, list[Var]] = {}
        self.counter = 0
        self.parameters = {v.uid for b in self.original_blocks for v in list(b.parameters) + list(b.event_dict) + list(b.mode_dict)}
        self.variables = {v.uid: v for b in self.original_blocks
                          for v in b.state_vars + b.algebraic_vars + b.diff_vars + b.in_vars}
        self.unknowns = set(self.variables) - self.parameters
        self.names = {v.name for v in self.variables.values()} | {
            v.name for b in self.original_blocks for v in list(b.parameters) + list(b.event_dict) + list(b.mode_dict)}
        self.velocities = {v.uid: eq for b in self.original_blocks for v, eq in zip(b.state_vars, b.state_eqs)}
        self.velocities.update({v.uid: expr for v, expr in (derivatives or {}).items()})
        for source, replacement in (substitutions or {}).items():
            if not isinstance(replacement, Var) or replacement.uid not in self.variables:
                raise ValueError('An existing substitution must reference a declared block variable')
            self.cache[_expression_key(source)] = self.variables[replacement.uid]
        self.reuse_existing_trig()

    def reuse_existing_trig(self) -> None:
        """Reuse only initialized state pairs whose exact kinematics also match."""
        from VeraGridEngine.Utils.Symbolic.compare_expressions_structure import equivalent
        candidates: dict[tuple, dict[str, tuple[Var, Expr]]] = {}
        for child in self.original_blocks:
            for var, initial in child.init_eqs.items():
                if var.uid in self.velocities and isinstance(initial, Func) and initial.op in ('sin', 'cos'):
                    candidates.setdefault(_expression_key(initial.arg), {})[initial.op] = (var, initial.arg)
        for pair in candidates.values():
            if 'cos' not in pair or 'sin' not in pair:
                continue
            cosine, arg = pair['cos']
            sine, _ = pair['sin']
            try:
                velocity = self.derivative(arg)
            except ValueError:
                continue
            expected = (-velocity * sine, velocity * cosine)
            actual = (self.velocities[cosine.uid], self.velocities[sine.uid])
            # The existing equivalence utility uses display names. Give it
            # UID-specific names so equal labels cannot create a false match.
            variables = {v.uid: v for expr in actual + expected for v in expr.get_vars()}
            renamed = {var: Var(f'identity_{uid}') for uid, var in variables.items()}
            if all(equivalent(a.subs(renamed), b.subs(renamed)) for a, b in zip(actual, expected)):
                self.cache.setdefault(_expression_key(sym.cos(arg)), cosine)
                self.cache.setdefault(_expression_key(sym.sin(arg)), sine)

    def new_var(self, label: str) -> Var:
        """Allocate a collision-free auxiliary and register its solver role."""
        name = label
        while name in self.names:
            self.counter += 1
            name = f'{label}_{self.counter}'
        self.names.add(name)
        var = self.vf.add_var(name)
        self.variables[var.uid] = var
        self.unknowns.add(var.uid)
        self.aux.reformulated_vars.append(var)
        return var

    def active_uids(self, expr: Expr) -> set[int]:
        return {v.uid for v in expr.get_vars()} & self.unknowns

    def derivative(self, expr: Expr) -> Expr:
        """Total time derivative from explicit state RHSs or caller-supplied inputs."""
        result: Expr = Const(0.)
        for var in expr.get_vars():
            if var.uid in self.parameters:
                continue
            if var.uid not in self.velocities:
                raise ValueError(f'No time derivative is known for {var.name!r}; supply derivatives={{variable: rhs}}')
            result = result + expr.diff(var) * self.velocities[var.uid]
        return result.simplify()

    def product(self, left: Expr, right: Expr) -> Expr:
        """Duplicate shared factors so no monomial contains a squared variable."""
        left_uids, right_uids = self.active_uids(left), self.active_uids(right)
        used = left_uids | right_uids
        replacements = {}
        for uid in sorted(left_uids & right_uids):
            candidates = self.aliases.setdefault(uid, [])
            alias = next((v for v in candidates if v.uid not in used), None)
            if alias is None:
                original = self.variables[uid]
                alias = self.new_var(f'{original.name}_copy')
                candidates.append(alias)
                self.aux.algebraic_vars.append(alias)
                self.aux.algebraic_eqs.append(alias - original)
                self.aux.init_eqs[alias] = original
            replacements[self.variables[uid]] = alias
            used.add(alias.uid)
        if replacements:
            right = right.subs(replacements)
        return left * right

    def reciprocal(self, denominator: Expr) -> Expr:
        key = ('reciprocal', _expression_key(denominator))
        if key not in self.cache:
            value = self.new_var('reciprocal')
            self.cache[key] = value
            self.aux.algebraic_vars.append(value)
            self.aux.init_eqs[value] = Const(1.) / denominator
            self.aux.algebraic_eqs.append(self.product(self.convert(denominator), value) - Const(1.))
        return self.cache[key]

    def power(self, base: Expr, exponent: int) -> Expr:
        if exponent == 0:
            return Const(1.)
        if exponent == 1:
            return base
        half = self.power(base, exponent // 2)
        value = self.product(half, half)
        return self.product(value, base) if exponent % 2 else value

    def trig(self, arg: Expr) -> tuple[Expr, Expr]:
        """One exact sine/cosine state pair per argument, with shared kinematics."""
        cos_key, sin_key = _expression_key(sym.cos(arg)), _expression_key(sym.sin(arg))
        cosine, sine = self.cache.get(cos_key), self.cache.get(sin_key)
        missing = []
        if cosine is None:
            cosine = self.new_var('u_cos')
            self.cache[cos_key] = cosine
            missing.append((cosine, sym.cos(arg)))
        if sine is None:
            sine = self.new_var('u_sin')
            self.cache[sin_key] = sine
            missing.append((sine, sym.sin(arg)))
        # Publish both cache entries before converting the argument derivative:
        # theta'=sin(theta), for example, must reuse this very same sine.
        velocity = self.derivative(arg)
        for var, initial in missing:
            derivative = self.vf.add_diff_var(f'd_{var.name}', base_var=var)
            rhs = -velocity * sine if var.uid == cosine.uid else velocity * cosine
            self.aux.state_vars.append(var)
            self.aux.diff_vars.append(derivative)
            self.aux.state_eqs.append(rhs)
            self.aux.init_eqs[var] = initial
            self.aux.diff_init_eqs[derivative] = rhs
        return cosine, sine

    def validate(self, expr: Expr) -> None:
        """Reject unsupported nodes before allocating any factory auxiliaries."""
        if isinstance(expr, (Var, Const)) or _expression_key(expr) in self.cache or not self.active_uids(expr):
            return
        if isinstance(expr, UnOp) and expr.op == '-':
            self.validate(expr.operand)
        elif isinstance(expr, BinOp) and expr.op in ('+', '-', '*', '/', '**'):
            if expr.op == '**':
                exponent = expr.right
                if not isinstance(exponent, Const) or exponent.value is None:
                    raise ValueError('Only constant integer powers and square roots can be multilinearized')
                value = float(exponent.value)
                if not np.isfinite(value) or not (value.is_integer() or value in (.5, -.5)):
                    raise ValueError(f'Unsupported multilinear power: {value}')
            self.validate(expr.left)
            self.validate(expr.right)
        elif isinstance(expr, Func) and expr.op in ('sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'sign', 'heaviside'):
            if expr.op in ('sin', 'cos', 'tan', 'exp', 'log'):
                self.derivative(expr.arg)
            self.validate(expr.arg)
        else:
            raise ValueError(f'Unsupported multilinear expression: {type(expr).__name__}')

    def trig_values(self, arg: Expr) -> tuple[Expr, Expr]:
        """Reuse a base trig pair for constant phase offsets and negated angles."""
        if isinstance(arg, UnOp) and arg.op == '-':
            cosine, sine = self.trig_values(arg.operand)
            result = cosine, -sine
        elif isinstance(arg, BinOp) and arg.op in ('+', '-'):
            left_active, right_active = self.active_uids(arg.left), self.active_uids(arg.right)
            if left_active and not right_active:
                cosine, sine = self.trig_values(arg.left)
                phase = arg.right if arg.op == '+' else -arg.right
            elif right_active and not left_active:
                cosine, sine = self.trig_values(arg.right)
                phase = arg.left
                if arg.op == '-':
                    sine = -sine
            else:
                return self.trig(arg)
            cos_phase, sin_phase = sym.cos(phase).simplify(), sym.sin(phase).simplify()
            result = (self.convert(cosine*cos_phase-sine*sin_phase),
                      self.convert(sine*cos_phase+cosine*sin_phase))
        else:
            return self.trig(arg)
        self.cache[_expression_key(sym.cos(arg))], self.cache[_expression_key(sym.sin(arg))] = result
        return result

    def function(self, expr: Func) -> Expr:
        # Algebraic helpers use the same exact branch-selected reformulations
        # as manually authored multilinear blocks in symbolic_ml.
        from VeraGridEngine.Utils.Symbolic.symbolic_ml import ml_heaviside, ml_positive_part, ml_smooth_sqrt
        op, arg = expr.op, expr.arg
        key = _expression_key(expr)
        if op in ('sin', 'cos', 'tan'):
            cosine, sine = self.trig_values(arg)
            if op == 'tan':
                return self.convert(BinOp(sine, '/', cosine))
            return sine if op == 'sin' else cosine
        if op in ('exp', 'log'):
            value = self.new_var(f'u_{op}')
            self.cache[key] = value
            velocity = self.derivative(arg)
            rhs = value * velocity if op == 'exp' else velocity / arg
            derivative = self.vf.add_diff_var(f'd_{value.name}', base_var=value)
            self.aux.state_vars.append(value)
            self.aux.diff_vars.append(derivative)
            self.aux.state_eqs.append(rhs)
            self.aux.init_eqs[value] = expr
            self.aux.diff_init_eqs[derivative] = rhs
            return value
        converted = self.convert(arg)
        if op == 'sqrt':
            helper, value = ml_smooth_sqrt(self.vf, converted, name=self.unique_suffix())
        elif op == 'heaviside':
            helper, value = ml_heaviside(self.vf, converted, name=self.unique_suffix())
        elif op in ('abs', 'sign'):
            if op == 'sign':
                return self.convert(Const(2.) * sym.heaviside(arg) - Const(1.))
            helper, positive, negative = ml_positive_part(self.vf, converted, name=self.unique_suffix())
            value = positive + negative
        else:
            raise ValueError(f'Unsupported multilinear reformulation: {op!r}')
        self.add_helper(helper)
        return value

    def unique_suffix(self) -> str:
        self.counter += 1
        suffix = f'auto_ml_{self.counter}'
        while any(name.endswith(suffix) for name in self.names):
            self.counter += 1
            suffix = f'auto_ml_{self.counter}'
        return suffix

    def add_helper(self, helper: Block) -> None:
        """Collect helper equations for the same recursive conversion pass."""
        for child in helper.get_all_blocks():
            for var in child.algebraic_vars:
                self.variables[var.uid] = var
                self.unknowns.add(var.uid)
                self.names.add(var.name)
            self.aux.algebraic_vars.extend(child.algebraic_vars)
            self.aux.algebraic_eqs.extend(child.algebraic_eqs)
            self.aux.init_eqs.update(child.init_eqs)
            self.aux.reformulated_vars.extend(child.algebraic_vars)

    def convert(self, expr: Expr) -> Expr:
        if isinstance(expr, (Var, Const)):
            return expr
        key = _expression_key(expr)
        if key in self.cache:
            return self.cache[key]
        # Expressions in coefficients alone are evaluated through the original
        # parameter machinery, rather than promoted to artificial solver states.
        if not self.active_uids(expr):
            return expr
        if isinstance(expr, UnOp) and expr.op == '-':
            result = -self.convert(expr.operand)
        elif isinstance(expr, BinOp):
            if expr.op == '*':
                result = self.product(self.convert(expr.left), self.convert(expr.right))
            elif expr.op == '/':
                if self.active_uids(expr.right):
                    result = self.product(self.convert(expr.left), self.reciprocal(expr.right))
                else:
                    result = self.convert(expr.left) / expr.right
            elif expr.op == '**':
                if not isinstance(expr.right, Const) or expr.right.value is None:
                    raise ValueError('Only constant integer powers and square roots can be multilinearized')
                exponent = float(expr.right.value)
                if exponent.is_integer():
                    result = (self.reciprocal(BinOp(expr.left, '**', Const(-exponent)))
                              if exponent < 0 else self.power(self.convert(expr.left), int(exponent)))
                elif exponent in (.5, -.5):
                    root = self.convert(sym.sqrt(expr.left))
                    result = self.reciprocal(root) if exponent < 0 else root
                else:
                    raise ValueError(f'Unsupported multilinear power: {exponent}')
            elif expr.op in ('+', '-'):
                result = BinOp(self.convert(expr.left), expr.op, self.convert(expr.right))
            else:
                raise ValueError(f'Unsupported binary operator: {expr.op!r}')
        elif isinstance(expr, Func):
            result = self.function(expr)
        else:
            raise ValueError(f'Unsupported multilinear expression: {type(expr).__name__}')
        self.cache[key] = result
        return result

    def run(self) -> Block:
        """Transform solver expressions; keep initialization and parameter programs."""
        for child in self.original_blocks:
            for expr in child.state_eqs + child.algebraic_eqs + child.differential_eqs + child.inequalities:
                if isinstance(expr, Comparison):
                    self.validate(expr.lhs)
                    self.validate(expr.rhs if isinstance(expr.rhs, Expr) else Const(expr.rhs))
                else:
                    self.validate(expr)
        for child in self.original_blocks:
            for field in ('state_eqs', 'algebraic_eqs', 'differential_eqs'):
                setattr(child, field, [self.convert(expr) for expr in getattr(child, field)])
            child.inequalities = [Comparison(self.convert(expr.lhs), expr.op,
                                    self.convert(expr.rhs if isinstance(expr.rhs, Expr) else Const(expr.rhs)))
                                  if isinstance(expr, Comparison) else self.convert(expr)
                                  for expr in child.inequalities]
        # Either work list may grow while converting the other (nested
        # functions, derivative kinematics, repeated factors). Drain both.
        state_index = algebraic_index = 0
        while state_index < len(self.aux.state_eqs) or algebraic_index < len(self.aux.algebraic_eqs):
            while state_index < len(self.aux.state_eqs):
                self.aux.state_eqs[state_index] = self.convert(self.aux.state_eqs[state_index])
                state_index += 1
            while algebraic_index < len(self.aux.algebraic_eqs):
                self.aux.algebraic_eqs[algebraic_index] = self.convert(self.aux.algebraic_eqs[algebraic_index])
                algebraic_index += 1
        if self.aux.state_vars or self.aux.algebraic_vars:
            self.block.children.append(self.aux)
        return self.block


def multilinearize_block(block: Block, vfactory: VarFactory,
                        substitutions: Mapping[Expr, Var] | None = None,
                        derivatives: Mapping[Var, Expr] | None = None) -> Block:
    """Return a multilinear copy, preserving original variable UIDs and interfaces.

    A structural cache shared across the complete hierarchy reuses every lifted
    expression, including separately constructed ``sin(theta)`` expressions.
    Sine/cosine share one exact differential pair; exp/log use exact differential
    lifts. Their arguments must have known derivatives (explicit state RHSs,
    or ``derivatives`` supplied for input/algebraic signals). Initial conditions
    retain the original functions. No relaxation or smoothing is added.

    Division, integer powers, square roots, abs/sign and Heaviside/min/max use
    algebraic auxiliary constraints. ``hard_sat`` and nested min/max clipping
    are supported through their Heaviside expressions; ``smooth_hard_sat``
    retains its original smoothing parameter through square-root lifts. Repeated factors receive equality-linked
    copies, so equations are multi-affine, not merely polynomial. Domain and
    consistent-initialization requirements of the original functions remain;
    nonsmooth lifts can be singular at their switching points.

    Existing sine/cosine state pairs are reused when their initialization and
    exact kinematic equations match. Other existing substitutions may be
    supplied as ``{sin(theta): u_sin}``; the
    caller is responsible for the existing relation being enforced by the
    source model. A converted block can be passed again without adding lifts.
    Parameter/event/discrete programs and initialization are intentionally not
    converted: these are evaluated outside the continuous solver equations.
    The source block is unchanged; new variables are allocated in ``vfactory``.
    """
    return _BlockMultilinearizer(block, vfactory, substitutions, derivatives).run()
