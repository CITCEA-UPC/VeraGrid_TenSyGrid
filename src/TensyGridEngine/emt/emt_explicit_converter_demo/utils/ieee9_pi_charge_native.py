"""Wrap the charge-reduced IEEE-9 polynomial in EmtModelProblem.

This follows the same symbolic-Block pattern as two_gfl_kcl_ode.py. The
native .simulate() path is checked on a short interval against the direct
103-state vector field. It does not impose the balanced-subspace restriction
used by the 0.2 s event plot.
"""

from __future__ import annotations

import argparse
import numpy as np
from scipy.integrate import solve_ivp

from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_pi_charge_polynomial import extract
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_pi_charge_polynomial import specialize_inputs
from TensyGridEngine.emt.emt_explicit_converter_demo.experiments.ieee9_pi_charge_event_timeseries import (
    compile_terms, polynomial_rhs,
)
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_model_problem import EmtModelProblem
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Const
from VeraGridEngine.enumerations import (
    DynamicIntegrationMethod, EmtInitializationMethod, EmtSolverTypes,
)


def build_native_problem(output, initial_state, *, step=2.5e-6,
                         reference_value=None,
                         input_names=(), input_values=(),
                         solver_type=EmtSolverTypes.Symbolic,
                         explicit=False):
    """Turn extract()'s sparse polynomial rows into a state-only EMT Block.

    ``output[i]`` defines dz_i/dt. The polynomial map is the actual reduced
    equation; this helper rebuilds it as symbolic expressions for .simulate().
    """
    if len(output) != len(initial_state):
        raise ValueError("Initial state and polynomial output sizes differ")
    factory = VarFactory()
    states = [factory.add_var(f"z_{i}") for i in range(len(output))]
    if reference_value is not None:
        if input_names or input_values:
            raise ValueError("Use either reference_value or input_names/input_values")
        input_names, input_values = ("first_gfl_P_ref",), (reference_value,)
    if len(input_names) != len(input_values):
        raise ValueError("Each symbolic input needs a name and initial value")
    inputs = [factory.add_var(name) for name in input_names]
    if any(mode >= len(states) + len(inputs) for row in output
           for support in row for mode in support):
        raise ValueError("Polynomial contains an unknown state/input index")
    differentials = [factory.add_diff_var(f"d_z_{i}", base_var=state)
                     for i, state in enumerate(states)]
    # Trapezoidal integration uses dx0 in its very first step. A zero seed
    # would give the rotating 50 Hz states an artificial half-step lag.
    initial_polynomial = specialize_inputs(output, input_values)
    initial_derivative = polynomial_rhs(
        np.asarray(initial_state, dtype=np.float64), *compile_terms(initial_polynomial))
    # The final mode is the first GFL's actual P_ref (pu), carried through the
    # grid reduction as a symbol. No second set of grid equations is needed.
    equations = []
    for index, row in enumerate(output):
        expression = Const(0.0)
        for support, coefficient in row.items():
            if abs(coefficient) <= 1e-10:
                continue
            term = Const(float(coefficient))
            for mode in support:
                term = term * (inputs[mode - len(states)] if mode >= len(states)
                               else states[mode])
            expression = expression + term
        equations.append(expression)
    # These state_eqs are the final equations integrated by ieee9_full_explicit.
    # P_ref is an input row in explicit S/Phi extraction, so one
    # matrix pair represents both pre- and post-event equations.
    block = Block(
        name="ieee9_pi_charge_explicit",
        state_vars=states, diff_vars=differentials, state_eqs=equations,
        algebraic_vars=[], algebraic_eqs=[],
        in_vars=inputs if explicit else [],
        out_vars=[],
        event_dict={var: Const(float(value)) for var, value in zip(inputs, input_values)},
        init_eqs={state: Const(float(value)) for state, value in zip(states, initial_state)},
        diff_init_eqs={differential: Const(float(value))
                       for differential, value in zip(differentials, initial_derivative)},
    )
    options = EmtOptions(
        time_step=step, simulation_time=1e-4,
        explicit=explicit,
        solver_type=solver_type,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
    )
    problem = EmtModelProblem(block, inputs={}, options=options)
    problem.reference_var = inputs[0] if len(inputs) == 1 else None
    problem.input_vars_by_name = dict(zip(input_names, inputs))
    return problem


def main(*, native_step=2.5e-6):
    full, output, dynamic, _, _, _ = extract(replacement_B=0.1)
    n, m = len(full.get_state_vars()), len(full.get_algebraic_vars())
    step = 2.5e-6
    full.options.time_step = step
    solver = build_emt_solver(options=full.options, problem=full,
                              t0=0.0, t_end=step, h=step,
                              method=full.options.integration_method)
    t, values, _, initialized, converged = solver.simulate(boundary_updater=full)
    if not initialized or not converged:
        raise RuntimeError("Full DAE did not provide an initial state")
    initial = np.r_[values[1, dynamic], values[1, n + m - 27:n + m]]
    native = build_native_problem(output, initial, step=native_step)
    S, Phi = native.build_multilinear_matrices_fast()
    result = native.simulate(t0=float(t[1]), t_end=1e-4)
    if not result.initialized or not result.converged:
        raise RuntimeError("Native reduced EMT simulation failed")
    terms = compile_terms(output)
    reference = solve_ivp(lambda _, z: polynomial_rhs(z, *terms),
                          (t[1], result.time[-1]), initial,
                          t_eval=result.time, method="BDF", rtol=1e-10, atol=1e-12)
    if not reference.success:
        raise RuntimeError(reference.message)
    error = np.max(np.abs(result.values - reference.y.T))
    print(f"native states={len(native.get_state_vars())}, algebraics={len(native.get_algebraic_vars())}")
    print(f"S={S.shape}, Phi={Phi.shape}, samples={len(result.time)}, native step={native_step:g} s")
    print(f"max native/direct state difference={error:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-step", type=float, default=2.5e-6)
    main(native_step=parser.parse_args().native_step)
