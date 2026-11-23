"""Check trial enforcement independently of expensive electrical scoring."""
from dataclasses import FrozenInstanceError
import pytest
from hackathon_files import RmsTuning
from hackathon_files.intro.evaluator import initialize_grid
from hackathon_files.intro.evaluator import load_benchmark
from hackathon_files.intro.evaluator import make_evaluator


def test_budget_reset_and_authoritative_baseline(monkeypatch):
    baseline = initialize_grid(load_benchmark(), enable_trials=False)
    evaluator = make_evaluator()
    scored = []

    def score(grid, proposal):
        assert grid is baseline
        scored.append(proposal)
        if proposal is None:
            raise ValueError("invalid proposal")
        return "metrics"

    monkeypatch.setattr(evaluator, "evaluate_proposal", score)
    first = evaluator.start_session(baseline)
    assert first.max_test_tunings == 5
    assert not hasattr(first, "evaluator")
    for name, value in [("tests_used", 0), ("max_test_tunings", 100),
                        ("max_open_lines", 100)]:
        with pytest.raises(FrozenInstanceError):
            setattr(first, name, value)
    original_load = baseline.problem.grid.loads[0].P
    first.problem.grid.loads[0].P = 0
    assert baseline.problem.grid.loads[0].P == original_load
    with pytest.raises(ValueError):
        first.test_tuning(None)
    for _ in range(4):
        assert first.test_tuning(RmsTuning()) == "metrics"
    with pytest.raises(RuntimeError, match="budget"):
        first.test_tuning(RmsTuning())
    assert len(scored) == 5
    assert first.tests_used == 5

    # Reusing the same organizer evaluator creates an independent new budget.
    second = evaluator.start_session(baseline)
    assert second.tests_used == 0
    assert second.test_tuning(RmsTuning()) == "metrics"
    assert first.tests_used == 5
    with pytest.raises(RuntimeError, match="budget"):
        first.test_tuning(RmsTuning())
    # Final scoring is separate and still works after exhausting all trials.
    assert evaluator.evaluate_proposal(baseline, RmsTuning()) == "metrics"
    assert second.tests_used == 1
