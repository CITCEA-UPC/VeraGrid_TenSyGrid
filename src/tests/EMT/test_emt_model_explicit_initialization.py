"""Standalone EMT blocks resolve dependent initialization equations."""

import numpy as np

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Simulations.EMT.problems.emt_model_problem import EmtModelProblem
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Const


def test_model_initialization_uses_runtime_input_and_dependency_order() -> None:
    factory = VarFactory()
    x = factory.add_var("x")
    y = factory.add_var("y")
    dx = factory.add_diff_var("dx", base_var=x)
    dy = factory.add_diff_var("dy", base_var=y)
    u = factory.add_var("u")
    block = Block(
        name="dependent_init", state_vars=[x, y], diff_vars=[dx, dy],
        state_eqs=[-x, -y], algebraic_vars=[], algebraic_eqs=[],
        in_vars=[u], out_vars=[], event_dict={u: Const(2)},
        init_eqs={x: u + Const(1), y: x + Const(2)},
        diff_init_eqs={dx: Const(-3), dy: Const(-5)},
    )
    problem = EmtModelProblem(block, inputs={})
    np.testing.assert_allclose(problem.get_x0(), [3, 5])
    np.testing.assert_allclose(problem.get_dx0(), [-3, -5])
