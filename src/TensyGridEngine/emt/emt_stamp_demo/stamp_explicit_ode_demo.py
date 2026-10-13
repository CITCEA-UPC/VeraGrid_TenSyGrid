"""Run the explicit STAMP GFOR and GFOL ODEs with prescribed abc voltage.

Each converter is simulated on its own, driven by a balanced 50 Hz voltage at
its STAMP operating-point magnitude. Results are saved as MATLAB .mat files:
rows are time samples; state/output/input names label the columns. This demo
contains no network dynamics. Edit ``build_inputs`` to impose a disturbance.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.io import savemat

from TensyGridEngine.emt.stamp_converters import (
    STAMP_GFOL,
    STAMP_GFOR,
    build_stamp_explicit_converter_emt,
)
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.problems.emt_model_problem import EmtModelProblem
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Var, sin
from VeraGridEngine.enumerations import (
    DynamicIntegrationMethod, EmtInitializationMethod, EmtSolverTypes,
)

OUTPUT_DIR = Path(__file__).with_name("stamp_explicit_ode_results")
SIMULATION_TIME = 0.01
TIME_STEP = 1.0e-5


def build_inputs(block, time: Var, voltage_rms: float):
    """Supply all three phase voltages; edit here to make input experiments."""
    peak = Const(np.sqrt(2.0) * voltage_rms)
    angle = Const(2.0 * np.pi * 50.0) * time
    shift = Const(2.0 * np.pi / 3.0)
    voltage_a, voltage_b, voltage_c = block.in_vars
    return {
        voltage_a: peak * sin(angle),
        voltage_b: peak * sin(angle - shift),
        voltage_c: peak * sin(angle + shift),
    }


def save_run(folder: Path, problem, result, observables, voltage_rms: float):
    """Write states, prescribed inputs, and observable outputs to one .mat file."""
    folder.mkdir(parents=True, exist_ok=True)
    time = result.time
    angle = 2.0 * np.pi * 50.0 * time
    peak = np.sqrt(2.0) * voltage_rms
    inputs = np.column_stack((
        peak * np.sin(angle),
        peak * np.sin(angle - 2.0 * np.pi / 3.0),
        peak * np.sin(angle + 2.0 * np.pi / 3.0),
    ))
    state_vars = problem.get_state_vars()
    state_names = [variable.name for variable in state_vars]
    states = result.values[:, [problem.get_var_idx(variable) for variable in state_vars]]
    outputs = np.empty((len(time), 5))
    output_exprs = (observables.p, observables.q, *observables.current_abc)
    for sample in range(len(time)):
        bindings = {variable.uid: states[sample, index]
                    for index, variable in enumerate(state_vars)}
        for phase in range(3):
            bindings[problem.sys_block.in_vars[phase].uid] = inputs[sample, phase]
        for column, expression in enumerate(output_exprs):
            outputs[sample, column] = expression.eval_uid(bindings)

    savemat(folder / "time_series.mat", {
        "time": time.reshape(-1, 1),
        "inputs": inputs,
        "input_names": np.asarray([variable.name for variable in problem.sys_block.in_vars],
                                  dtype=object).reshape(1, -1),
        "states": states,
        "state_names": np.asarray(state_names, dtype=object).reshape(1, -1),
        "outputs": outputs,
        "output_names": np.asarray(["P", "Q", "i_A", "i_B", "i_C"],
                                   dtype=object).reshape(1, -1),
    }, do_compression=True)


def main() -> None:
    for parameters in (STAMP_GFOR, STAMP_GFOL):
        name = f"STAMP_{parameters.mode}{parameters.number}_EMT"
        block, observables = build_stamp_explicit_converter_emt(
            VarFactory(), parameters, name
        )
        time = Var("stamp_time")
        options = EmtOptions(
            time_step=TIME_STEP,
            simulation_time=SIMULATION_TIME,
            solver_type=EmtSolverTypes.Automatic,
            integration_method=DynamicIntegrationMethod.DaeBackEuler,
            initialization_method=EmtInitializationMethod.Explicit,
        )
        problem = EmtModelProblem(
            block,
            inputs=build_inputs(block, time, parameters.voltage_pu),
            options=options,
            glob_time=time,
        )
        result = problem.simulate()
        if not (result.initialized and result.converged):
            raise RuntimeError(f"{parameters.mode} explicit ODE did not converge")
        folder = OUTPUT_DIR / parameters.mode.lower()
        save_run(folder, problem, result, observables, parameters.voltage_pu)
        print(f"{parameters.mode}: states={len(problem.get_state_vars())}, "
              f"algebraic={len(problem.get_algebraic_vars())}, "
              f"samples={len(result.time)}, data={folder / 'time_series.mat'}")


if __name__ == "__main__":
    main()
