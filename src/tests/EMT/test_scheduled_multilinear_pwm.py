"""Verify analytic event times, held references, gate maps and runtime cloning."""
from itertools import product

import numpy as np
import pytest

from VeraGridEngine.Utils.procedural_logic import (
    ScheduledMultilinearPwmLogic, ThreePhaseCarrierPwmLogic,
    procedural_logic_entry_to_dict, procedural_logic_from_dict,
    scheduled_pwm_gate_update,
)


def runtime(logic_class=ScheduledMultilinearPwmLogic, phase=-np.pi/2):
    logic = logic_class('ma', 'mb', 'mc', 'qa', 'qb', 'qc', 'omega', 'phase')
    logic.mod_a_idx, logic.mod_b_idx, logic.mod_c_idx = 0, 1, 2
    logic.gate_a_idx, logic.gate_b_idx, logic.gate_c_idx = 0, 1, 2
    logic.omega_sw_idx, logic.carrier_phase_idx = 3, 4
    params = np.array([0., 0., 0., 2*np.pi*5000, phase])
    return logic, params


def test_analytic_edges_hold_references_and_consume_each_edge_once():
    logic, params = runtime()
    logic.update(0., np.array([-.5, 0., .5]), params)
    np.testing.assert_array_equal(params[:3], [1, 1, 1])
    np.testing.assert_allclose(logic.pending_transition_time, [25e-6, 50e-6, 75e-6])
    previous = 0.
    for edge, gates in [(25e-6, [0, 1, 1]), (50e-6, [0, 0, 1]), (75e-6, [0, 0, 0])]:
        assert logic.get_next_forced_event_time(previous, 100e-6) == pytest.approx(edge)
        # Changed live references must not alter this half-cycle's schedule.
        logic.update(edge, np.array([.9, -.9, -.9]), params)
        logic.update(edge, np.ones(3), params)
        np.testing.assert_array_equal(params[:3], gates)
        previous = edge
    assert logic.edge_event_count == 3
    logic.update(100e-6, np.array([.25, -.25, 0.]), params)
    np.testing.assert_allclose(logic.pending_transition_time, [137.5e-6, 162.5e-6, 150e-6])
    assert logic.carrier_rising == 0
    assert logic.boundary_event_count == 1


def test_coincident_edges_and_saturated_references():
    logic, params = runtime()
    logic.update(0., np.zeros(3), params)
    logic.update(50e-6, np.zeros(3), params)
    np.testing.assert_array_equal(params[:3], [0, 0, 0])
    assert logic.edge_event_count == 3
    assert logic.get_next_forced_event_time(50e-6, 100e-6) == pytest.approx(100e-6)
    logic.update(100e-6, np.array([-1., 1., 0.]), params)
    np.testing.assert_array_equal(params[:3], [0, 1, 0])
    assert np.isnan(logic.pending_transition_time[:2]).all()
    logic.update(150e-6, np.zeros(3), params)
    np.testing.assert_array_equal(params[:3], [0, 1, 1])


@pytest.mark.parametrize('phase', [-np.pi/2, 0., .4, np.pi/2])
def test_event_trace_matches_native_scheduler_over_many_half_cycles(phase):
    proposed, p = runtime(phase=phase)
    native, q = runtime(ThreePhaseCarrierPwmLogic, phase)
    t = 0.
    for _ in range(150):
        modulation = .8*np.sin(2*np.pi*50*t + np.array([0., -2*np.pi/3, 2*np.pi/3]))
        proposed.update(t, modulation, p)
        native.update(t, modulation, q)
        np.testing.assert_array_equal(p[:3], q[:3])
        next_proposed = proposed.get_next_forced_event_time(t, t+.001)
        next_native = native.get_next_forced_event_time(t, t+.001)
        assert next_proposed == pytest.approx(next_native, abs=1e-15)
        assert next_proposed > t
        t = next_proposed


def test_multilinear_gate_map_truth_table_and_affinity():
    for gate, edge, boundary, direction, initial in product((0., 1.), repeat=5):
        result = scheduled_pwm_gate_update(gate, edge, boundary, direction, initial)
        expected = initial if boundary else (1-direction if edge else gate)
        assert result == expected
    values = [.2, .3, .4, .6, .7]
    for index in range(5):
        left, right = values.copy(), values.copy()
        left[index], right[index] = 0., 1.
        assert scheduled_pwm_gate_update(*values) == pytest.approx(
            (1-values[index])*scheduled_pwm_gate_update(*left)
            + values[index]*scheduled_pwm_gate_update(*right))


def test_codec_and_remap_keep_scheduled_type():
    logic, _ = runtime()
    restored = procedural_logic_from_dict([procedural_logic_entry_to_dict(logic)])[0]
    assert type(restored) is ScheduledMultilinearPwmLogic
    assert type(logic.remap({})) is ScheduledMultilinearPwmLogic


def test_frequency_enable_rebuilds_schedule_at_absolute_carrier_phase():
    logic, p = runtime()
    native, q = runtime(ThreePhaseCarrierPwmLogic)
    p[3] = q[3] = 0.
    modulation = np.array([-.3, .4, .7])
    logic.update(0., modulation, p)
    native.update(0., modulation, q)
    p[3] = q[3] = 2*np.pi*5000
    logic.update(100e-6, modulation, p)
    native.update(100e-6, modulation, q)
    np.testing.assert_array_equal(p[:3], q[:3])
    np.testing.assert_allclose(logic.pending_transition_time, native.pending_transition_time)
    assert logic.carrier_rising == 0.


def test_resynchronization_after_skipped_half_cycles():
    logic, p = runtime()
    native, q = runtime(ThreePhaseCarrierPwmLogic)
    modulation = np.array([-.3, .4, .7])
    for t in [0., 200e-6, 235e-6, 270e-6, 285e-6]:
        logic.update(t, modulation, p)
        native.update(t, modulation, q)
        np.testing.assert_array_equal(p[:3], q[:3])


def test_macro_clock_roundoff_does_not_repeat_half_cycle_updates():
    logic, params = runtime()
    modulation = np.array([-.3, .4, .7])
    logic.update(0., modulation, params)
    previous = target = 0.
    for _ in range(1000):
        target += 1e-5
        while previous < target-1e-15:
            event = logic.get_next_forced_event_time(previous, target)
            accepted = target if event is None else event
            logic.update(accepted, modulation, params)
            logic.update(accepted, modulation, params)
            assert logic.interval_end_time > accepted+1e-12
            previous = accepted
    assert logic.boundary_event_count == 100
    assert logic.edge_event_count == 300


def test_exact_pulse_area_through_real_emt_solver_with_unaligned_steps():
    """Catch early/late gate application, not just correct edge timestamps."""
    from tests.EMT.test_deferred_pwm_edges import build_pwm_area_problem
    from VeraGridEngine.Simulations.EMT.solvers.structural_compiled_solver import (
        StructuralCompiledSolver, StructuralCompiledWarmupPolicy)
    from VeraGridEngine.Utils.procedural_logic import build_boundary_updater_from_block
    from VeraGridEngine.enumerations import DynamicIntegrationMethod
    problem = build_pwm_area_problem()
    data = procedural_logic_entry_to_dict(problem.sys_block.procedural_logic[0])
    data['multilinear_gate_updates'] = True
    problem.sys_block.procedural_logic = procedural_logic_from_dict([data])
    updater = build_boundary_updater_from_block(problem)
    assert type(updater.logic_entries[0]) is ScheduledMultilinearPwmLogic
    solver = StructuralCompiledSolver(
        problem, t0=0., t_end=.001, h=23e-6, method=DynamicIntegrationMethod.DaeBackEuler,
        newton_tolerance=1e-10, warmup_policy=StructuralCompiledWarmupPolicy.Off)
    time, values, _, initialized, converged = solver.simulate(boundary_updater=updater)
    assert initialized and converged
    assert .001 <= time[-1] <= .001+23e-6
    duty = (1+np.array([.2, -.99, .99]))/2
    period = .0002
    cycles = int(np.floor((time[-1]+1e-15)/period))
    remainder = max(0., time[-1]-cycles*period)
    expected = cycles*period*duty + np.minimum(remainder, period/2*duty)
    expected += np.maximum(0., remainder-(period-period/2*duty))
    np.testing.assert_allclose(values[-1, :3], expected, atol=2e-12, rtol=0.)
