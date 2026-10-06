# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import numpy as np

import VeraGridEngine.api as vg
from VeraGridEngine.enumerations import EngineType, InvestmentEvaluationMethod, InvestmentsEvaluationObjectives
from VeraGridEngine.Simulations.InvestmentsEvaluation.Methods.NSGA_3 import NSGA_3
from VeraGridEngine.Simulations.InvestmentsEvaluation.Methods.random_eval import random_trial
from VeraGridEngine.Simulations.InvestmentsEvaluation.investments_evaluation_driver import InvestmentsEvaluationDriver
from VeraGridEngine.Simulations.InvestmentsEvaluation.investments_evaluation_options import InvestmentsEvaluationOptions
from VeraGridEngine.Utils.NumericalMethods.MVRSM_mo_pareto import MVRSM_mo_pareto
from VeraGrid.Session.session import GcThread


class CancelProblemStub:
    """
    Minimal investment problem used to verify driver cancellation.
    """

    __slots__ = ("called", "plot_x_idx", "plot_y_idx", "x_min", "x_max")

    def __init__(self) -> None:
        """
        Constructor.

        :return: None.
        """
        self.called: bool = False
        self.plot_x_idx: int = 0
        self.plot_y_idx: int = 1
        self.x_min: np.ndarray = np.zeros(2, dtype=int)
        self.x_max: np.ndarray = np.ones(2, dtype=int)

    def get_objectives_names(self) -> np.ndarray:
        """
        Get fake objective names.

        :return: Objective names.
        """
        return np.array(["capex", "losses"], dtype=str)

    def get_vars_names(self) -> np.ndarray:
        """
        Get fake variable names.

        :return: Variable names.
        """
        return np.array(["group_0", "group_1"], dtype=str)

    def n_vars(self) -> int:
        """
        Get problem dimension.

        :return: Number of variables.
        """
        return 2

    def objective_function(self, x: np.ndarray) -> np.ndarray:
        """
        Record any unexpected objective evaluation.

        :param x: Decision vector.
        :return: Objective vector.
        """
        self.called = True
        return np.array([float(np.sum(x)), float(np.sum(x))], dtype=float)

    def get_investments_for_combination(self, x: np.ndarray) -> list:
        """
        Return no concrete investments for the selected combination.

        :param x: Selected decision vector.
        :return: Empty list.
        """
        return list()


class CountingCancelChecker:
    """
    Cancel after a fixed number of checks.
    """

    __slots__ = ("calls", "cancel_after")

    def __init__(self, cancel_after: int) -> None:
        """
        Constructor.

        :param cancel_after: Number of checks before cancellation.
        :return: None.
        """
        self.calls: int = 0
        self.cancel_after: int = cancel_after

    def check(self) -> bool:
        """
        Return True once enough checks have been observed.

        :return: Cancellation state.
        """
        self.calls += 1
        return self.calls > self.cancel_after


def cancelled() -> bool:
    """
    Cancellation function that is always active.

    :return: True.
    """
    return True


def nsga_objective(x: np.ndarray) -> np.ndarray:
    """
    Objective that should not be called by pre-cancelled NSGA.

    :param x: Decision vector.
    :return: Objective vector.
    """
    raise AssertionError("NSGA objective must not run after pre-cancel")


def random_objective(x: np.ndarray) -> np.ndarray:
    """
    Cheap random-trial objective.

    :param x: Decision vector.
    :return: Objective vector.
    """
    return np.array([float(np.sum(x)), float(np.sum(x))], dtype=float)


def mvrsm_objective(x: np.ndarray) -> np.ndarray:
    """
    Cheap MVRSM objective.

    :param x: Decision vector.
    :return: Objective vector.
    """
    return np.array([float(np.sum(x)), float(np.sum(x))], dtype=float)


def test_nsga3_pre_cancel_skips_pymoo_evaluation() -> None:
    """
    A pre-cancelled pymoo run must return empty arrays without evaluating candidates.
    """
    x_values, f_values = NSGA_3(
        obj_func=nsga_objective,
        n_var=2,
        lb=np.zeros(2, dtype=int),
        ub=np.ones(2, dtype=int),
        n_obj=2,
        n_partitions=1,
        max_evals=10,
        pop_size=2,
        cancel_checker=cancelled,
    )

    assert x_values.shape == (0, 2)
    assert f_values.shape == (0, 2)


def test_random_trial_cancel_returns_evaluated_prefix() -> None:
    """
    Random investments evaluation must stop at cancellation and return only evaluated rows.
    """
    checker = CountingCancelChecker(cancel_after=2)

    x_values, f_values = random_trial(
        obj_func=random_objective,
        n_var=2,
        lb=np.zeros(2, dtype=int),
        ub=np.ones(2, dtype=int),
        n_obj=2,
        max_evals=10,
        cancel_checker=checker.check,
    )

    assert x_values.shape == (2, 2)
    assert f_values.shape == (2, 2)


def test_mvrsm_cancel_and_small_budget_return_evaluated_prefix() -> None:
    """
    MVRSM must not overrun arrays when random evaluations exceed max evaluations.
    """
    y_sorted, x_sorted, y_population, x_population = MVRSM_mo_pareto(
        obj_func=mvrsm_objective,
        x0=np.zeros(3, dtype=float),
        lb=np.zeros(3, dtype=float),
        ub=np.ones(3, dtype=float),
        num_int=3,
        max_evals=2,
        n_objectives=2,
        rand_evals=3,
    )

    assert y_population.shape == (2, 2)
    assert x_population.shape == (2, 3)
    assert y_sorted.shape[1] == 2
    assert x_sorted.shape[1] == 3


def test_investments_driver_cancel_sets_base_flag_and_skips_objective() -> None:
    """
    Investments driver cancellation must use the base DriverTemplate cancel flag.
    """
    grid = vg.MultiCircuit()
    problem = CancelProblemStub()
    options = InvestmentsEvaluationOptions(
        max_eval=10,
        solver=InvestmentEvaluationMethod.Random,
        obj_tpe=InvestmentsEvaluationObjectives.PowerFlow,
    )
    driver = InvestmentsEvaluationDriver(
        grid=grid,
        options=options,
        problem=problem,
        engine=EngineType.VeraGrid,
    )

    driver.cancel()
    driver.run()

    assert driver.is_cancel()
    assert not problem.called
    assert driver.results.current_evaluation == 0


def test_nsga3_does_not_copy_qt_signal_bound_cancel_checker() -> None:
    """
    Ensure pymoo does not deepcopy the Qt-signal-bearing investment driver.

    :return: None.
    """
    grid: vg.MultiCircuit = vg.MultiCircuit()
    problem: CancelProblemStub = CancelProblemStub()
    options: InvestmentsEvaluationOptions = InvestmentsEvaluationOptions(
        max_eval=4,
        solver=InvestmentEvaluationMethod.NSGA3,
        obj_tpe=InvestmentsEvaluationObjectives.PowerFlow,
    )
    driver: InvestmentsEvaluationDriver = InvestmentsEvaluationDriver(
        grid=grid,
        options=options,
        problem=problem,
        engine=EngineType.VeraGrid,
    )
    thread: GcThread = GcThread(driver=driver)
    driver.initialize(max_iter=5)

    x_values, f_values = NSGA_3(
        obj_func=driver.objective_function,
        n_partitions=1,
        n_var=problem.n_vars(),
        lb=problem.x_min,
        ub=problem.x_max,
        n_obj=2,
        max_evals=4,
        pop_size=2,
        cancel_checker=driver.is_cancel,
    )

    assert thread.driver is driver
    assert x_values is not None
    assert f_values is not None
