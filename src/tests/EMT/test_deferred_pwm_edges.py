"""Verify native PWM pulse area and rejection semantics through real EMT solvers."""
from typing import Tuple

import numpy as np
import pytest

from VeraGridEngine.Simulations.EMT.problems.emt_problem_template import EmtProblemTemplate
from VeraGridEngine.Simulations.EMT.solvers.jit_symbolic_solver import JitSymbolicSolver
from VeraGridEngine.Simulations.EMT.solvers.solver_AD import JitAdSolver
from VeraGridEngine.Simulations.EMT.solvers.StructuralVectorizedSolver import StructuralVectorizedSolver
from VeraGridEngine.Simulations.EMT.solvers.structural_compiled_solver import StructuralCompiledSolver
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Var
from VeraGridEngine.Utils.procedural_logic import BlockProceduralLogicUpdater, ThreePhaseCarrierPwmLogic, build_boundary_updater_from_block
from VeraGridEngine.enumerations import DynamicIntegrationMethod


def build_pwm_area_problem() -> EmtProblemTemplate:
    """Build three voltage-time integrals driven by the actual PWM logic.

    :return: Small DAE with constant modulation and retained native gate modes.
    """
    states: list[Var] = list(Var('area_'+phase) for phase in 'abc')
    differentials: list[Var] = list(Var('d_area_'+phase, base_var=state) for phase, state in zip('abc', states))
    modulation: list[Var] = list(Var('m_'+phase) for phase in 'abc')
    gates: list[Var] = list(Var('gate_'+phase+'_mode') for phase in 'abc')
    omega: Var = Var('omega_sw')
    phase: Var = Var('carrier_phase')
    reference: Tuple[float, float, float] = (.2, -.99, .99)
    block: Block = Block(name='NativePwmArea', state_vars=states, diff_vars=differentials,
                         state_eqs=list(gates), algebraic_vars=modulation,
                         algebraic_eqs=list(var-Const(value) for var, value in zip(modulation, reference)),
                         parameters=dict())
    block.mode_dict = dict((gate, Const(1.)) for gate in gates)
    block.event_dict = dict(((omega, Const(2*np.pi*5000)), (phase, Const(-np.pi/2))))
    block.init_eqs = dict((var, Const(0.)) for var in states)
    block.init_eqs.update(dict((var, Const(value)) for var, value in zip(modulation, reference)))
    block.procedural_logic = list((ThreePhaseCarrierPwmLogic('m_a', 'm_b', 'm_c',
                                  'gate_a_mode', 'gate_b_mode', 'gate_c_mode', 'omega_sw', 'carrier_phase'),))
    block.unify_blocks()
    problem: EmtProblemTemplate = EmtProblemTemplate(sys_block=block, glob_time=Var('t'),
                                                     static_parameter_values_mapping=dict())
    # The generic container does not run a device explicit initializer.
    variable: Var
    value: float
    for variable, value in zip(modulation, reference):
        problem.init_guess[variable.uid] = value
    return problem


@pytest.mark.parametrize('solver_class', (JitSymbolicSolver, JitAdSolver, StructuralVectorizedSolver, StructuralCompiledSolver))
@pytest.mark.parametrize('dense_threshold', (0, 100))
def test_native_pwm_voltage_time_area(
        solver_class: type[JitSymbolicSolver] | type[JitAdSolver] | type[StructuralVectorizedSolver] | type[StructuralCompiledSolver],
        dense_threshold: int) -> None:
    """Integrate exact native pulses across unaligned macro steps in each backend.

    :param solver_class: One of the production EMT integration backends.
    :param dense_threshold: Select sparse or dense Newton factors.
    :return: None.
    """
    problem: EmtProblemTemplate = build_pwm_area_problem()
    updater: BlockProceduralLogicUpdater | None = build_boundary_updater_from_block(problem)
    assert updater is not None
    solver: JitSymbolicSolver | JitAdSolver | StructuralVectorizedSolver | StructuralCompiledSolver = solver_class(
        problem, t0=0., t_end=.001, h=23e-6, method=DynamicIntegrationMethod.DaeBackEuler,
        newton_tolerance=1e-10, dense_threshold=dense_threshold)
    times: np.ndarray
    values: np.ndarray
    derivatives: np.ndarray
    initialized: bool
    converged: bool
    times, values, derivatives, initialized, converged = solver.simulate(boundary_updater=updater)
    assert initialized and converged
    duty: np.ndarray = (1+np.array((.2, -.99, .99)))/2
    period: float = .0002
    cycles: int = int(np.floor((times[-1]+1e-15)/period))
    remainder: float = max(0., float(times[-1]-cycles*period))
    expected: np.ndarray = cycles*period*duty+np.minimum(remainder, period/2*duty)
    expected += np.maximum(0., remainder-(period-period/2*duty))
    np.testing.assert_allclose(values[-1, :3], expected, atol=2e-12, rtol=0.)


def test_rejected_pwm_interval_does_not_consume_edge() -> None:
    """Reject and retry an exact native edge without changing its schedule.

    :return: None.
    """
    problem: EmtProblemTemplate = build_pwm_area_problem()
    updater: BlockProceduralLogicUpdater | None = build_boundary_updater_from_block(problem)
    assert updater is not None
    states: np.ndarray = problem.get_x0().copy()
    parameters: np.ndarray = problem.event_params_values.copy()
    updater.update(0., states, parameters)
    edge: float | None = updater.get_next_forced_event_time(0., .0001)
    assert edge is not None
    before: np.ndarray = parameters.copy()
    updater.prepare_interval(edge, states, parameters)
    updater.resolve_step(False, parameters)
    np.testing.assert_array_equal(parameters, before)
    assert updater.get_next_forced_event_time(0., .0001) == edge
    updater.prepare_interval(edge, states, parameters)
    updater.resolve_step(True, parameters)
    updater.commit_boundary(edge, states, parameters)
    logic: ThreePhaseCarrierPwmLogic = updater.logic_entries[0]
    assert isinstance(logic, ThreePhaseCarrierPwmLogic)
    np.testing.assert_array_equal(parameters[[logic.gate_a_idx, logic.gate_b_idx, logic.gate_c_idx]], (1., 0., 1.))


def test_accepted_carrier_boundary_always_opens_next_half_period() -> None:
    """Replay recursively computed carrier clocks without stale end times.

    :return: None.
    """
    problem: EmtProblemTemplate = build_pwm_area_problem()
    updater: BlockProceduralLogicUpdater | None = build_boundary_updater_from_block(problem)
    assert updater is not None
    states: np.ndarray = problem.get_x0().copy()
    parameters: np.ndarray = problem.event_params_values.copy()
    updater.update(0., states, parameters)
    logic: ThreePhaseCarrierPwmLogic = updater.logic_entries[0]
    assert isinstance(logic, ThreePhaseCarrierPwmLogic)
    half_index: int
    for half_index in range(100):
        boundary: float = float(logic.interval_end_time)
        updater.prepare_interval(boundary, states, parameters)
        updater.commit_boundary(boundary, states, parameters)
        assert logic.interval_end_time > boundary+5e-5
        assert abs(logic.interval_end_time-(half_index+2)*.0001) < 1e-14
