"""The EMT runtime kernel handles the trigonometric expressions used by IEEE-39."""

import numpy as np
import pytest

from VeraGridEngine.Utils.Symbolic.jit_compiler import RuntimeParameterCompiler
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Func, Func2


@pytest.mark.parametrize("python_kernel", [False, True])
def test_runtime_parameter_trigonometric_and_binary_functions(monkeypatch, python_kernel: bool) -> None:
    if python_kernel:
        monkeypatch.setenv("VERAGRID_RUNTIME_PARAMETER_PYTHON", "1")
    else:
        monkeypatch.delenv("VERAGRID_RUNTIME_PARAMETER_PYTHON", raising=False)
    expressions = [
        Func(Const(1.0), "atan"),
        Func2("atan2", Const(1.0), Const(1.0)),
        Func2("min", Const(1.0), Const(2.0)),
        Func2("max", Const(1.0), Const(2.0)),
    ]
    kernel = RuntimeParameterCompiler({}, {}, set()).compile(expressions)
    values = np.zeros(len(expressions))
    kernel(values, np.zeros(0), 0.0)
    np.testing.assert_allclose(values, [np.pi / 4, np.pi / 4, 1.0, 2.0])
