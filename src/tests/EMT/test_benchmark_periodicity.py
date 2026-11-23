"""One-period diagnostics must distinguish angle winding from physical drift."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('multilinear_benchmark',
    Path(__file__).resolve().parents[3] / 'trunk/benchmarks/multilinear_ssa.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_periodicity_allows_only_declared_absolute_angle_winding():
    times = np.linspace(0., .02, 101)
    phase = 2 * np.pi * times / .02
    trajectory = np.column_stack((phase + .3, np.sin(phase), np.full(101, .4)))
    report = benchmark.periodicity_report(trajectory, times,
        ['theta_generator_0', 'v_A', 'relative_angle'], [0], .02, 1e-8)
    assert report['periodicity_validated']
    assert report['last_period_scaled_closure_inf'] < 1e-12
    trajectory[-1, 2] += .01
    report = benchmark.periodicity_report(trajectory, times,
        ['theta_generator_0', 'v_A', 'relative_angle'], [0], .02, 1e-8)
    assert not report['periodicity_validated']
    assert report['largest_period_closure_errors'][0]['name'] == 'relative_angle'


def test_equal_endpoints_do_not_validate_the_wrong_period():
    report = benchmark.periodicity_report(np.zeros((3, 1)), np.array([0., .01, .02]),
                                         ['voltage'], [], .04, 1e-6)
    assert not report['periodicity_validated']
