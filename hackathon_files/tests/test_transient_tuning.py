"""End-to-end checks for small-signal response metrics."""

import pytest

from VeraGridEngine.api import open_file

from hackathon_files import BasicRmsTuner
from hackathon_files.intro.examples.control_response_experiment import GRID_PATH, SetGeneratorGains


@pytest.fixture(scope="module")
def baseline():
    return BasicRmsTuner().test_tuning(open_file(str(GRID_PATH)))


def test_unchanged_values_match_no_op(baseline) -> None:
    same_values = SetGeneratorGains({"Kp": 0.0, "Ki": 0.0}).test_tuning(
        open_file(str(GRID_PATH))
    )
    assert same_values.min_oscillatory_damping == baseline.min_oscillatory_damping
    assert same_values.max_eigenvalue_real == baseline.max_eigenvalue_real


def test_higher_kp_worsens_this_case(baseline) -> None:
    changed = SetGeneratorGains({"Kp": 20.0}).test_tuning(open_file(str(GRID_PATH)))
    assert changed.min_oscillatory_damping < baseline.min_oscillatory_damping


def test_ki_can_introduce_unstable_mode(baseline) -> None:
    changed = SetGeneratorGains({"Ki": 5.0}).test_tuning(open_file(str(GRID_PATH)))
    assert changed.max_eigenvalue_real > 0.0
