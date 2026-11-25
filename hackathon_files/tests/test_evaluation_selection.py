"""Small-set membership and case grading, including failed-case accounting."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from hackathon_files.basic import case_files
from hackathon_files.intro import evaluator
from hackathon_files.src.rms_scoring import score_components,case_mark


def metrics(**changes):
    values=dict(stable=True,voltage_violations=0,thermal_violations=0,
                min_oscillatory_damping=.05,active_losses_mw=8.,changes_score=.75)
    return SimpleNamespace(**(values|changes))


def test_limited_contains_exactly_ten_smallest_public_grids():
    small=case_files('limited');all_cases=case_files()
    assert len(small)==len(set(small))==10
    assert set(small)<set(all_cases)
    assert len(all_cases)==60 and all(p.exists() for p in all_cases)
    directory=Path(__file__).resolve().parents[1]/'basic'
    records=json.loads((directory/'validation.json').read_text())+json.loads((directory/'switching_validation.json').read_text())
    sizes={r['file']:r['buses'] for r in records}
    assert sorted(sizes[p.name] for p in small)==[5]*6+[6]*4
    assert max(sizes[p.name] for p in small)<=min(sizes[p.name] for p in all_cases if p not in small)
    assert any('switching' in p.name for p in small)
    with pytest.raises(ValueError,match='selection'):
        case_files('unknown')


def test_marks_match_weights_and_loss_baseline():
    components=score_components(metrics(),10.)
    assert components==pytest.approx(dict(damping=.5,loss=.2,robustness=1.,changes=.75))
    assert case_mark(components)==pytest.approx(58.75)
    assert score_components(metrics(stable=False),10.)['damping']==0
    assert score_components(metrics(thermal_violations=1),10.)['robustness']==0
    assert score_components(metrics(min_oscillatory_damping=None),10.)['damping']==1
    assert score_components(metrics(active_losses_mw=11.),10.)['loss']==0
    with pytest.raises(ValueError,match='baseline losses'):
        score_components(metrics(),0.)


def test_limited_runner_skips_intro_and_reports_ten_marks(monkeypatch,capsys):
    calls=[]
    def forbidden():
        raise AssertionError('Limited must not load the intro benchmark')
    monkeypatch.setattr(evaluator,'load_benchmark',forbidden)
    result=metrics(active_losses_mw=1.,min_oscillatory_damping=.1,
                   max_eigenvalue_real=-.1,peak_frequency_deviation_hz=.01,
                   structural_changes=1,max_structural_changes=4)
    def basic(path,*,with_baseline=False):
        calls.append(path)
        assert with_baseline
        return result,2.
    monkeypatch.setattr(evaluator,'evaluate_basic',basic)
    assert evaluator.main(['--cases','limited'])==0
    assert calls==list(case_files('limited'))
    output=capsys.readouterr().out
    assert output.count('mark=83.75/100')==10
    assert 'basic: 10/10 passed' in output
    assert 'Overall mark (limited, 10 cases): 83.75/100' in output
    assert 'intro/' not in output
