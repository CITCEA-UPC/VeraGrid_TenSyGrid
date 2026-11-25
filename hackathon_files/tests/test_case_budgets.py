"""Finite and unlimited trial accounting, and the balanced basic-case policy."""
from types import SimpleNamespace
import math
import pytest
from hackathon_files.basic.cases import case_files, case_trial_budget
from hackathon_files import RmsTunerEvaluator, RmsTuning, TunableGrid
from hackathon_files.src import rms_tuning


def test_balanced_case_budgets():
    paths = case_files()
    budgets = [case_trial_budget(p.stem) for p in paths]
    assert budgets.count(math.inf) == budgets.count(20) == 25
    for suffix in range(1, 6):
        group = [case_trial_budget(p.stem) for p in paths if p.stem.endswith(f'_{suffix:02d}')]
        assert group.count(math.inf) == group.count(20) == 5


@pytest.mark.parametrize('limit', [0, 20, math.inf])
def test_session_budget_and_rebinding(monkeypatch, limit):
    problem = SimpleNamespace(grid=SimpleNamespace(copy=lambda: object()), power_flow_results=None)
    monkeypatch.setattr(rms_tuning, 'SmallSignalStabilityRmsDriver',
                        lambda **kwargs: SimpleNamespace(problem=problem))
    baseline = TunableGrid(problem, (), (), (), (), ())
    evaluator = RmsTunerEvaluator()
    monkeypatch.setattr(evaluator, 'evaluate_proposal', lambda *args: 'result')
    session = evaluator.start_session(baseline, limit=limit)
    rebound = evaluator.start_session(session)
    assert rebound.max_test_tunings == limit
    if limit:
        for _ in range(25 if math.isinf(limit) else limit):
            assert rebound.test_tuning(RmsTuning()) == 'result'
    if not math.isinf(limit):
        with pytest.raises(RuntimeError, match='exhausted'):
            rebound.test_tuning(RmsTuning())
    else:
        assert rebound.tests_used == 25
    assert session.tests_used == 0


@pytest.mark.parametrize('limit', [-1, True, 2.5, -math.inf, math.nan])
def test_invalid_budgets(limit):
    with pytest.raises(ValueError, match='budget'):
        RmsTunerEvaluator().start_session(None, limit=limit)
