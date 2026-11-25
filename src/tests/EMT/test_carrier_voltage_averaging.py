"""Check time weighting and partial-interval clipping for PWM diagnostics."""
import numpy as np

from TensyGridEngine.hybrid_boolean_mti.pwm_dq_comparison.analyze_voltage_delivery import interval_averages


def test_carrier_averages_weight_irregular_event_times():
    time = np.array([0., .1, .3, 1.])
    values = np.array([[0., 1.], [2., 1.], [4., 1.]])
    means = interval_averages(time, values, np.array([0., .2, .5, 1.]))
    np.testing.assert_allclose(means, [[1., 1.], [10/3, 1.], [4., 1.]])


def test_carrier_average_resolves_a_short_gate_pulse():
    # A 20% duty pulse must average to 0.2 even with only one 'high' endpoint.
    time = np.array([0., .4, .6, 1.])
    np.testing.assert_allclose(interval_averages(time, np.array([0., 1., 0.]), [0., 1.]), [.2])
