"""Regression checks for scenario isolation and the introductory modal workflow."""
import numpy as np
import pytest
from scipy.linalg import eigvals
from hackathon_files import RmsTuning
from hackathon_files.intro.evaluator import SCENARIOS, load_benchmark, make_scenario
from hackathon_files.intro.evaluator import FinalTuner, initialize_grid, make_evaluator


def test_scenarios_do_not_modify_benchmark():
    benchmark = load_benchmark()
    original = [(load.P, load.Q) for load in benchmark.loads]
    heavy = make_scenario(benchmark, SCENARIOS[2])
    assert [(load.P, load.Q) for load in benchmark.loads] == original
    assert np.allclose([load.P for load in heavy.loads],
                       [p * 1.05 for p, q in original])
    heavy.loads[0].P = 0
    assert benchmark.loads[0].P == original[0][0]


def test_descriptor_matches_diagnostic_physical_modes():
    tunable = initialize_grid(load_benchmark())
    problem = tunable.problem
    x = problem.get_x0().copy()
    dx = np.zeros_like(x)
    values = eigvals(problem.get_static_state_matrix(x, dx).toarray(),
                     b=-problem.get_E_matrix(x, dx))
    finite = values[np.isfinite(values)]
    reduced = FinalTuner().get_modal_analysis(tunable)["eigenvalues"]
    reduced = reduced[np.isfinite(reduced)]
    assert len(finite) == len(reduced)
    for value in reduced:
        assert np.min(np.abs(finite - value)) < 1e-6


def test_evaluator_skips_invalid_parameter():
    class InvalidTuner(FinalTuner):
        def tune(self, grid):
            parameter = grid.tunable_parameters[0]
            return RmsTuning(parameters={parameter: parameter.maximum + 1})

    baseline = FinalTuner().test_tuning(load_benchmark(), make_evaluator())
    with pytest.warns(UserWarning, match="Skipped parameter"):
        sanitized = InvalidTuner().test_tuning(load_benchmark(), make_evaluator())
    assert sanitized.max_eigenvalue_real == baseline.max_eigenvalue_real
    assert sanitized.active_losses_mw == baseline.active_losses_mw
    assert sanitized.stable == baseline.stable


def test_benchmark_has_dynamic_gfm_controls():
    tunable = initialize_grid(load_benchmark())
    controls = {p.name for p in tunable.tunable_parameters if p.device_name == "gen 1"}
    assert {"Kdp", "Kdq", "Kp_icl", "Ki_icl", "tau_P", "tau_Q"} <= controls
    assert tunable.problem.get_diff_var_number() > 0


def test_full_evaluator_accepts_descriptor_constraints():
    metrics = FinalTuner().test_tuning(load_benchmark(), make_evaluator())
    assert metrics.eigenvalue_count > 4  # Includes the converter's physical modes.
    assert np.isfinite(metrics.max_eigenvalue_real)
    assert np.isfinite(metrics.active_losses_mw)
    assert metrics.min_oscillatory_damping is not None


def test_combined_runner_covers_both_collections(monkeypatch, capsys):
    from types import SimpleNamespace
    from hackathon_files.intro import evaluator

    calls = []
    metrics = SimpleNamespace(
        stable=True, voltage_violations=0, thermal_violations=0,
        min_oscillatory_damping=0.1, active_losses_mw=1.0,
        max_eigenvalue_real=-0.1, peak_frequency_deviation_hz=0.01,
        structural_changes=2, max_structural_changes=8, changes_score=0.75,
    )
    monkeypatch.setattr(evaluator, "load_benchmark", lambda: object())
    monkeypatch.setattr(evaluator, "make_scenario", lambda grid, scenario: scenario.name)
    monkeypatch.setattr(evaluator, "make_evaluator", lambda: object())

    def intro(self, grid, rules):
        calls.append(grid)
        return metrics

    def basic(path):
        calls.append(path.stem)
        if len(calls) == 12:
            raise RuntimeError("case failed")
        return metrics

    monkeypatch.setattr(evaluator.FinalTuner, "test_tuning", intro)
    monkeypatch.setattr(evaluator, "evaluate_basic", basic)
    assert evaluator.main() == 1
    assert len(calls) == 61
    output = capsys.readouterr().out
    assert "changes=2/8" in output
    assert "S_changes=0.750" in output
    assert "Mean change score over 60 successfully evaluated cases: 0.750" in output
    assert "peak dV" not in output
    assert "intro: 11/11 passed" in output
    assert "basic: 49/50 passed" in output
    assert "Evaluation: 60/61 passed" in output
