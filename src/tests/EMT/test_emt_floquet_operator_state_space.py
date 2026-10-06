# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Regression tests for the physical-state EMT Floquet operator contract."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
from scipy import sparse

from VeraGridEngine.Simulations.SmallSignalStabilityEmt.emt_floquet_operator import (
    BlockEmtFloquetOperator,
    EmtFloquetOperator,
)
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.PowerFlow.power_flow_results import PowerFlowResults
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_results_3ph import PowerFlowResults3Ph
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import (
    SmallSignalStabilityEmtDriver,
)
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import (
    SmallSignalStabilityEmtOptions,
)
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.enumerations import DynamicIntegrationMethod, EmtProblemTypes


@dataclass(frozen=True)
class _StateVariable:
    uid: int


@dataclass(frozen=True)
class _DifferentialVariable:
    uid: int
    base_var: _StateVariable


class _ScalarProblem:
    """Minimal one-state problem exposing the Floquet operator protocol."""

    def __init__(self) -> None:
        self.state = _StateVariable(uid=1)
        self.diff = _DifferentialVariable(uid=2, base_var=self.state)

    def get_state_vars(self):
        return [self.state]

    def get_diff_vars(self):
        return [self.diff]

    @staticmethod
    def get_states_number() -> int:
        return 1

    @staticmethod
    def get_algebraic_var_number() -> int:
        return 0


def _zero_dynamics_jacobian(*, h=None, **_kwargs):
    """Discrete trapezoidal Jacobian for xdot=0: (2/h) delta-x[k+1]."""
    return sparse.csc_matrix([[2.0 / float(h)]])


def _build_operator(operator_type):
    problem = _ScalarProblem()
    h = 0.1
    trajectory = np.zeros((2, 1), dtype=float)
    derivatives = np.zeros_like(trajectory)
    return operator_type(
        problem=problem,
        trajectory=trajectory,
        h=h,
        n_states=1,
        method=DynamicIntegrationMethod.DaeTrapezoidal,
        jac_evaluator=_zero_dynamics_jacobian,
        static_params=np.empty(0, dtype=float),
        n_event_params=0,
        t_trajectory=np.asarray([0.0, h]),
        d_trajectory=derivatives,
    )


def test_trapezoidal_operator_exposes_only_physical_states() -> None:
    """Derivative history must not enlarge the public Floquet eigenproblem."""
    scalar = _build_operator(EmtFloquetOperator)
    block = _build_operator(BlockEmtFloquetOperator)

    assert scalar.shape == (1, 1)
    assert block.shape == (1, 1)


def test_trapezoidal_zero_dynamics_has_only_the_physical_unit_multiplier() -> None:
    """An internal history coordinate must not add a spurious -1 multiplier."""
    scalar = _build_operator(EmtFloquetOperator)
    block = _build_operator(BlockEmtFloquetOperator)
    initial = np.asarray([1.0])

    np.testing.assert_allclose(scalar.matvec(initial), initial)
    np.testing.assert_allclose(block.matmat(initial[:, None])[:, 0], initial)

    dense = block.matmat(np.eye(1))
    np.testing.assert_allclose(np.linalg.eigvals(dense), [1.0])


def test_emt_ssa_driver_uses_problem_factory_and_forwards_both_pf_results() -> None:
    """The selected EMT problem type must reach the canonical problem factory."""
    grid = MultiCircuit()
    balanced = object.__new__(PowerFlowResults)
    three_phase = object.__new__(PowerFlowResults3Ph)
    options = EmtOptions(problem_type=EmtProblemTypes.Multilinear)
    selected_problem = object()

    with patch(
        "VeraGridEngine.Simulations.SmallSignalStabilityEmt."
        "small_signal_stability_emt_driver.build_emt_problem",
        return_value=selected_problem,
    ) as factory:
        driver = SmallSignalStabilityEmtDriver(
            grid=grid,
            emt_options=options,
            sss_options=SmallSignalStabilityEmtOptions(),
            pf_results=balanced,
            pf_results_3ph=three_phase,
        )

    assert driver.problem is selected_problem
    assert factory.call_args.kwargs["options"].problem_type == EmtProblemTypes.Multilinear
    assert factory.call_args.kwargs["pf_results"] is balanced
    assert factory.call_args.kwargs["pf_results_3ph"] is three_phase
