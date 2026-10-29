"""End-to-end checks for fixed-load-step RMS response metrics."""

import pytest

from VeraGridEngine.api import open_file

from hackaton_files import BasicRmsTuner
from hackaton_files.control_response_experiment import GRID_PATH, SetGeneratorGains


@pytest.fixture(scope="module")
def baseline():
    return BasicRmsTuner().test_tuning(open_file(str(GRID_PATH)))


def test_unchanged_values_match_no_op(baseline) -> None:
    same_values = SetGeneratorGains({"Kp": 0.0, "Ki": 0.0}).test_tuning(
        open_file(str(GRID_PATH))
    )
    assert same_values.min_oscillatory_damping == baseline.min_oscillatory_damping
    assert same_values.max_eigenvalue_real == baseline.max_eigenvalue_real
    assert same_values.peak_frequency_deviation_hz == baseline.peak_frequency_deviation_hz


def test_higher_kp_worsens_this_case(baseline) -> None:
    changed = SetGeneratorGains({"Kp": 20.0}).test_tuning(open_file(str(GRID_PATH)))
    assert changed.transient_converged
    assert changed.min_oscillatory_damping < baseline.min_oscillatory_damping
    assert changed.peak_frequency_deviation_hz > baseline.peak_frequency_deviation_hz


def test_short_response_can_miss_unstable_mode(baseline) -> None:
    changed = SetGeneratorGains({"Ki": 5.0}).test_tuning(open_file(str(GRID_PATH)))
    assert changed.transient_converged
    assert changed.max_eigenvalue_real > 0.0
    assert changed.peak_frequency_deviation_hz < baseline.peak_frequency_deviation_hz
