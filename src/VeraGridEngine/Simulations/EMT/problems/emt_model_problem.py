# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Standalone simulation of one EMT symbolic model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy import sparse

from VeraGridEngine.Devices.Dynamic.emt_template import EmtModelTemplate
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_multilinear import EmtProblemMultilinear
from VeraGridEngine.Simulations.EMT.problems.emt_problem_template import EmtProblemTemplate
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Expr, Var


@dataclass(frozen=True)
class EmtModelSimulationResults:
    """Time-domain result returned by :meth:`EmtModelProblem.simulate`."""

    time: np.ndarray
    values: np.ndarray
    derivatives: np.ndarray
    initialized: bool
    converged: bool


class EmtModelProblem(EmtProblemMultilinear):
    """An EMT problem containing one model and prescribed symbolic inputs.

    Unlike :class:`EmtProblemDae`, this class does not assemble a ``MultiCircuit``.
    Every model input is instead registered as a runtime parameter whose defining
    expression may depend on :attr:`glob_time`.  The class derives from
    ``EmtProblemMultilinear`` so the same ``S``/``Phi`` extraction and Floquet
    Jacobian implementation are available for multilinear model templates.

    ``inputs`` and ``runtime_parameters`` accept either symbolic variables or
    unambiguous variable names as keys. Values may be constants or VeraGrid
    symbolic expressions. Python callables are intentionally not accepted: using
    symbolic expressions keeps all solver and multilinear backends equivalent.
    """

    def __init__(
        self,
        model: EmtModelTemplate | Block,
        inputs: Mapping[Var | str, Any],
        options: EmtOptions | None = None,
        runtime_parameters: Mapping[Var | str, Any] | None = None,
        initial_values: Mapping[Var | str, float] | None = None,
        glob_time: Var | None = None,
    ) -> None:
        block = model.block if isinstance(model, EmtModelTemplate) else model
        block.unify_blocks()
        self.options = EmtOptions() if options is None else options
        time_var = Var(self.TIME_NAME) if glob_time is None else glob_time

        self._install_runtime_expressions(block, inputs, block.in_vars, "input")
        if runtime_parameters is not None:
            self._install_runtime_expressions(
                block, runtime_parameters, list(block.event_dict), "runtime parameter"
            )

        missing = [variable.name for variable in block.in_vars if variable not in block.event_dict]
        if missing:
            raise ValueError("Missing expressions for model inputs: " + ", ".join(missing))

        # Deliberately bypass EmtProblemDae's electrical-network constructor while
        # retaining EmtProblemMultilinear's model-independent utilities.
        EmtProblemTemplate.__init__(
            self,
            sys_block=block,
            static_parameter_values_mapping=dict(block.parameters),
            glob_time=time_var,
        )
        # EmtModelProblem intentionally bypasses EmtProblemMultilinear.__init__
        # because that constructor assembles a complete electrical network.
        # Initialize its model-independent matrix caches explicitly so the
        # inherited multilinear utilities remain usable for standalone blocks.
        self.Phi: sparse.csr_matrix | None = None
        self.S: sparse.csc_matrix | None = None
        self._ml_all_vars_sa = None
        self._ml_all_basis_vars = None
        self._ml_idx_vars = None
        self._ml_uid_to_basis_idx = None
        self._ml_uid_to_idx_full = None
        self._run_model_explicit_initialization()
        # Explicit caller-provided operating-point values are authoritative and
        # may complete or correct template initialization equations that normally
        # depend on a surrounding network power-flow mapping.
        self._apply_initial_values(initial_values or {})

    @staticmethod
    def _resolve_key(block: Block, key: Var | str, candidates: list[Var], kind: str) -> Var:
        if isinstance(key, Var):
            if not any(variable.uid == key.uid for variable in candidates):
                raise KeyError(f"Unknown model {kind} {key.name!r}")
            return key
        matches = [variable for variable in candidates if variable.name == key]
        if len(matches) != 1:
            raise KeyError(f"Expected one model {kind} named {key!r}, found {len(matches)}")
        return matches[0]

    @classmethod
    def _install_runtime_expressions(
        cls,
        block: Block,
        expressions: Mapping[Var | str, Any],
        candidates: list[Var],
        kind: str,
    ) -> None:
        for key, expression in expressions.items():
            variable = cls._resolve_key(block, key, candidates, kind)
            if callable(expression) and not isinstance(expression, (Expr, Var, Const)):
                raise TypeError(f"{kind} {variable.name!r} must be a symbolic expression, not a callable")
            block.event_dict[variable] = expression if isinstance(expression, (Expr, Var, Const)) else Const(float(expression))

    def _apply_initial_values(self, values: Mapping[Var | str, float]) -> None:
        variables = list(self._state_vars) + list(self._algebraic_vars)
        for key, value in values.items():
            variable = self._resolve_key(self.sys_block, key, variables, "variable")
            self.init_guess[variable.uid] = float(value)

    def _run_model_explicit_initialization(self) -> None:
        """Resolve the model's native ``init_eqs`` using inputs sampled at t=0."""
        pending = [
            (variable, expression, self.init_guess)
            for variable, expression in self._init_eqs_flat.items()
        ] + [
            (variable, expression, self.diff_init_guess)
            for variable, expression in self._diff_init_eqs_flat.items()
        ]
        while pending:
            remaining = []
            for variable, expression, destination in pending:
                if isinstance(expression, Expr) and not self._runtime_expression_dependencies_available(expression):
                    remaining.append((variable, expression, destination))
                    continue
                destination[variable.uid] = self._evaluate_runtime_expression(
                    expression, self._event_params_values, 0.0
                )
            if len(remaining) == len(pending):
                names = ", ".join(variable.name for variable, _, _ in remaining)
                raise ValueError(f"Unresolved model initialization equations: {names}")
            pending = remaining

    @property
    def boundary_update(self):
        """A prescribed-input model has no network/event boundary updater."""
        return None

    def reset_boundary_update_state(self, t0: float = 0.0) -> None:
        """Reset runtime inputs to their expressions evaluated at ``t0``."""
        self._event_params_values = self._initialize_runtime_parameter_values(float(t0))
        self._event_params_values = self.def_event_params_fn(self._event_params_values, float(t0))

    def simulate(self, t0: float = 0.0, t_end: float | None = None) -> EmtModelSimulationResults:
        """Build the configured EMT solver and simulate this single model."""
        end = float(self.options.simulation_time if t_end is None else t_end)
        solver = build_emt_solver(
            options=self.options,
            problem=self,
            t0=float(t0),
            t_end=end,
            h=float(self.options.time_step),
            method=self.options.integration_method,
        )
        time, values, derivatives, initialized, converged = solver.simulate(boundary_updater=None)
        return EmtModelSimulationResults(
            time=np.asarray(time),
            values=np.asarray(values),
            derivatives=np.asarray(derivatives),
            initialized=bool(initialized),
            converged=bool(converged),
        )

    def set_runtime_expression(self, parameter: Var | str, expression: Any) -> None:
        """Replace one runtime expression after clean operating-point initialization."""
        variable = self._resolve_key(
            self.sys_block, parameter, list(self._variable_parameters), "runtime parameter"
        )
        value = expression if isinstance(expression, (Expr, Var, Const)) else Const(float(expression))
        index = self.uid2idx_event_params[variable.uid]
        self.sys_block.event_dict[variable] = value
        self._event_parameters_eqs[index] = value
        self._runtime_all_eqs_source[index] = value
        self._rebuild_runtime_parameter_partition()
        self._build_runtime_param_vectors()

    def trace(self, results: EmtModelSimulationResults, variable: Var | str) -> np.ndarray:
        """Return one state/algebraic trajectory by variable object or name."""
        resolved = self._resolve_key(
            self.sys_block,
            variable,
            list(self._state_vars) + list(self._algebraic_vars),
            "variable",
        )
        return results.values[:, self.get_var_idx(resolved)]
