"""Check one-sided native PWM edge publication and pulse voltage-time area."""
import numpy as np

from TensyGridEngine.hybrid_boolean_mti.pwm_dq_comparison.diagnose_native_controls import RecordingUpdater
from VeraGridEngine.Utils.procedural_logic import ThreePhaseCarrierPwmLogic


class NativePwmDelegate:
    def __init__(self):
        self.pwm = ThreePhaseCarrierPwmLogic('a', 'b', 'c', 'ga', 'gb', 'gc', 'omega', 'phase')
        p = self.pwm
        p.mod_a_idx, p.mod_b_idx, p.mod_c_idx = 0, 1, 2
        p.gate_a_idx, p.gate_b_idx, p.gate_c_idx = 0, 1, 2
        p.omega_sw_idx, p.carrier_phase_idx = 3, 4
        self.logic_entries = [p]

    def update(self, t, x, params):
        self.pwm.update(t, x, params)

    def get_next_forced_event_time(self, start, end):
        return self.pwm.get_next_forced_event_time(start, end)

    def commit_boundary(self, t, x, params):
        pass


def test_pwm_edge_is_published_only_after_acceptance():
    updater = RecordingUpdater(NativePwmDelegate(), defer_pwm_edges=True)
    invalidations = []
    updater.invalidate_factorization = lambda: invalidations.append(True)
    x = np.array([.2, -.3, .7])
    params = np.array([0., 0., 0., 2*np.pi*5000, -np.pi/2])
    updater.update(0., x, params)
    old = params[:3].copy()
    edge = updater.get_next_forced_event_time(0., .0001)
    updater.update(edge, x, params)
    np.testing.assert_array_equal(params[:3], old)
    before_commit = len(invalidations)
    updater.commit_boundary(edge, x, params)
    np.testing.assert_array_equal(params[:3], [1., 0., 1.])
    np.testing.assert_array_equal(x, [.2, -.3, .7])
    assert updater.edge_commits == 1
    assert len(invalidations) == before_commit+1


def test_native_pulse_area_is_exact_despite_unaligned_macro_steps():
    updater = RecordingUpdater(NativePwmDelegate(), defer_pwm_edges=True)
    x = np.array([.2, -.3, .7])
    params = np.array([0., 0., 0., 2*np.pi*5000, -np.pi/2])
    updater.update(0., x, params)
    area = np.zeros(3)
    previous = 0.
    for target in np.r_[np.arange(23e-6, .0002, 23e-6), .0002]:
        while previous < target-1e-15:
            edge = updater.get_next_forced_event_time(previous, target)
            current = target if edge is None else edge
            updater.update(current, x, params)
            area += (current-previous)*params[:3]
            updater.commit_boundary(current, x, params)
            previous = current
    np.testing.assert_allclose(area, .0002*(1+x)/2, rtol=0., atol=1e-15)
