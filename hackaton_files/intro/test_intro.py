"""Regression checks for scenario isolation and the introductory modal workflow."""
import numpy as np
import pytest
from scipy.linalg import eigvals
from hackaton_files import RmsTuning
from .evaluate_quick import evaluate_quick
from .scenarios import QUICK_SCENARIOS, LOCAL_SCENARIOS, load_benchmark, make_scenario
from .starter_script import StarterTuner, initialize_grid


def test_scenarios_do_not_modify_benchmark():
    benchmark = load_benchmark()
    original = [(load.P, load.Q) for load in benchmark.loads]
    heavy = make_scenario(benchmark, QUICK_SCENARIOS[2])
    assert [(load.P, load.Q) for load in benchmark.loads] == original
    assert np.allclose([load.P for load in heavy.loads],
                       [p * 1.05 for p, q in original])
    heavy.loads[0].P = 0
    assert benchmark.loads[0].P == original[0][0]
    assert set(QUICK_SCENARIOS).issubset(LOCAL_SCENARIOS)


def test_descriptor_matches_diagnostic_physical_modes():
    tunable = initialize_grid(load_benchmark())
    problem = tunable.problem
    x = problem.get_x0().copy()
    dx = np.zeros_like(x)
    values = eigvals(problem.get_static_state_matrix(x, dx).toarray(),
                     b=-problem.get_E_matrix(x, dx))
    finite = values[np.isfinite(values)]
    reduced = StarterTuner().get_modal_analysis(tunable)["eigenvalues"]
    reduced = reduced[np.isfinite(reduced)]
    assert len(finite) == len(reduced)
    for value in reduced:
        assert np.min(np.abs(finite - value)) < 1e-6


def test_quick_rejects_invalid_proposal():
    class InvalidTuner(StarterTuner):
        def tune(self, grid):
            parameter = grid.tunable_parameters[0]
            return RmsTuning(parameters={parameter: parameter.maximum + 1})

    with pytest.raises(ValueError, match="out-of-range"):
        evaluate_quick(InvalidTuner(), load_benchmark())


def test_benchmark_has_dynamic_gfm_controls():
    tunable = initialize_grid(load_benchmark())
    controls = {p.name for p in tunable.tunable_parameters if p.device_name == "gen 1"}
    assert {"Kdp", "Kdq", "Kp_icl", "Ki_icl", "tau_P", "tau_Q"} <= controls
    assert tunable.problem.get_diff_var_number() > 0


def test_full_evaluator_accepts_descriptor_constraints():
    from .settings import make_evaluator

    metrics = StarterTuner().test_tuning(load_benchmark(), make_evaluator())
    assert metrics.eigenvalue_count > 4  # Includes the converter's physical modes.
    assert np.isfinite(metrics.max_eigenvalue_real)
    assert np.isfinite(metrics.active_losses_mw)
    assert metrics.min_oscillatory_damping is not None
