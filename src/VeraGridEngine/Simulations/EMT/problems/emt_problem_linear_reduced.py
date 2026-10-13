# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Optional exact elimination of a constant-linear EMT algebraic subsystem.

The ordinary full-grid problem is retained. ``build_reduced_model_problem``
creates a separate state-only problem after grid assembly and initialization.
Parameters are frozen at their initialized values, so this path is intended
for fixed-topology, fixed-parameter experiments without runtime events.
"""

from __future__ import annotations

import numpy as np

from VeraGridEngine.Simulations.EMT.problems.emt_model_problem import EmtModelProblem
from VeraGridEngine.Simulations.EMT.problems.emt_problem_multilinear import EmtProblemMultilinear
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import BinOp, Const, Expr, Func, Func2, UnOp, Var


def _state_support(expr: Expr, state_uids: set[int]) -> set[int]:
    """Check multi-affinity without expanding products of large sums.

    Each returned UID may appear once in any product path. Addition may reuse
    a UID across terms. A state-dependent denominator, exponent or function
    cannot be represented by the multilinear S/Phi basis.
    """
    if isinstance(expr, Const):
        return set()
    if isinstance(expr, Var):
        return {expr.uid} if expr.uid in state_uids else set()
    if isinstance(expr, UnOp) and expr.op == "-":
        return _state_support(expr.operand, state_uids)
    if isinstance(expr, BinOp):
        left = _state_support(expr.left, state_uids)
        right = _state_support(expr.right, state_uids)
        if expr.op in ("+", "-"):
            return left | right
        if expr.op == "*":
            repeated = left & right
            if repeated:
                raise ValueError(f"Reduced equation repeats state UID {next(iter(repeated))} in a product")
            return left | right
        if expr.op == "/" and not right:
            return left
        if expr.op == "**" and isinstance(expr.right, Const):
            exponent = float(expr.right.value)
            if not left or exponent == 1.0:
                return left
            if exponent == 0.0:
                return set()
        raise ValueError(f"Reduced equation has state-dependent {expr.op!r}")
    if isinstance(expr, (Func, Func2)):
        support = {variable.uid for variable in expr.get_vars()
                   if variable.uid in state_uids}
        if support:
            raise ValueError("Reduced equation contains a state-dependent function")
        return set()
    support = {variable.uid for variable in expr.get_vars()
               if variable.uid in state_uids}
    if support:
        raise ValueError(f"Cannot verify multi-affinity of {type(expr).__name__}")
    return set()


class EmtProblemLinearReduced(EmtProblemMultilinear):
    """Full-grid EMT problem with an opt-in linear algebraic reduction."""

    def build_reduced_model_problem(self) -> EmtModelProblem:
        """Eliminate every algebraic variable if its frozen Jacobian is invertible.

        Raises ``ValueError`` for a nonlinear or singular algebraic subsystem.
        The returned problem uses the full-grid initialized state as its start.
        """
        if self.grid.emt_events:
            raise ValueError("Linear reduction requires a grid without scheduled EMT events")
        algebraic_vars = self.get_algebraic_vars()
        algebraic_eqs = self._algebraic_eqs
        if len(algebraic_vars) != len(algebraic_eqs):
            raise ValueError("Algebraic variables and equations have different counts")
        if not algebraic_vars:
            raise ValueError("There are no algebraic variables to eliminate")

        # Freeze the parameters used by the assembled equations. A changing
        # event parameter would invalidate the factorization and substitutions.
        parameters = {
            variable: Const(float(value))
            for variable, value in zip(self._variable_parameters, self._event_params_values)
        }
        parameters.update({
            variable: Const(float(value))
            for variable, value in zip(self._constant_parameters, self._constant_params_values)
        })
        equations = [equation.subs(parameters) for equation in algebraic_eqs]
        column = {variable.uid: index for index, variable in enumerate(algebraic_vars)}
        matrix = np.zeros((len(equations), len(algebraic_vars)))
        for row, equation in enumerate(equations):
            involved = {variable.uid: variable for variable in equation.get_vars()
                        if variable.uid in column}
            for variable in involved.values():
                coefficient = equation.diff(variable).simplify()
                if not isinstance(coefficient, Const):
                    raise ValueError(
                        f"Algebraic equation {row} is not constant-linear in {variable.name}"
                    )
                matrix[row, column[variable.uid]] = float(coefficient.value)

        if np.linalg.matrix_rank(matrix) != len(algebraic_vars):
            raise ValueError("The constant-linear algebraic subsystem is singular")

        zero = {variable: Const(0.0) for variable in algebraic_vars}
        right_hand_side = [equation.subs(zero).simplify() for equation in equations]
        inverse = np.linalg.solve(matrix, np.eye(len(algebraic_vars)))
        replacements = {}
        for variable, row in zip(algebraic_vars, inverse):
            expression = Const(0.0)
            for coefficient, rhs in zip(row, right_hand_side):
                if abs(coefficient) > 1.0e-14:
                    expression = expression - Const(float(coefficient)) * rhs
            replacements[variable] = expression.simplify()

        states = self.get_state_vars()
        differentials = self.get_diff_vars()
        reduced = Block(
            name=f"{self.sys_block.name}_linear_reduced",
            state_vars=states,
            diff_vars=differentials,
            state_eqs=[equation.subs(parameters).subs(replacements).simplify()
                       for equation in self._state_eqs],
            algebraic_vars=[],
            algebraic_eqs=[],
            init_eqs={state: Const(float(self.init_guess[state.uid])) for state in states},
            diff_init_eqs={diff: Const(float(self.diff_init_guess[diff.uid]))
                            for diff in differentials},
            reformulated_vars=list(self.sys_block.reformulated_vars),
        )
        unresolved = {variable.uid for equation in reduced.state_eqs
                      for variable in equation.get_vars() if variable.uid in column}
        if unresolved:
            raise ValueError(f"Reduction left {len(unresolved)} algebraic variables in state equations")
        state_uids = {state.uid for state in states}
        for index, equation in enumerate(reduced.state_eqs):
            try:
                _state_support(equation, state_uids)
            except ValueError as exc:
                raise ValueError(f"Reduced state equation {index} is not multi-affine: {exc}") from exc
        return EmtModelProblem(reduced, inputs={}, options=self.options, glob_time=self.glob_time)


__all__ = ["EmtProblemLinearReduced"]
