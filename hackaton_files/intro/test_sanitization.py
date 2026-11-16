"""Sanitization keeps valid independent actions without changing submitted values."""
from dataclasses import replace
import pytest
from hackaton_files import RmsTuning
from .scenarios import load_benchmark
from .starter_script import initialize_grid
from .settings import make_evaluator


@pytest.fixture(scope="module")
def baseline():
    return initialize_grid(load_benchmark(), enable_trials=False)


def test_mixed_actions_and_order(baseline):
    first, second = baseline.tunable_parameters[:2]
    unknown = replace(first, device_idtag="unknown")
    lines = baseline.openable_lines
    proposal = RmsTuning(
        parameters={first: first.initial_value, second: float("nan"), unknown: 1.0},
        open_lines=(lines[0], lines[0], lines[1], lines[2]),
        add_lines=("not a handle",),
    )
    with pytest.warns(UserWarning) as captured:
        clean = make_evaluator().sanitize_tuning(baseline, proposal)
    assert clean.parameters == {first: first.initial_value}
    assert clean.open_lines == (lines[0], lines[1])
    assert clean.add_lines == ()
    assert len(captured) == 5
    assert clean.is_accepted(baseline)
    assert len(proposal.open_lines) == 4  # No in-place rewriting.


@pytest.mark.parametrize("value", ["bad", float("inf"), -1e9])
def test_invalid_redispatch_is_dropped_as_group(baseline, value):
    first, second = baseline.dispatch_options[:2]
    parameter = baseline.tunable_parameters[0]
    proposal = RmsTuning(parameters={parameter: parameter.initial_value},
                        dispatch_mw={first: value, second: second.initial_mw})
    with pytest.warns(UserWarning, match="entire redispatch"):
        clean = make_evaluator().sanitize_tuning(baseline, proposal)
    assert clean.dispatch_mw == {}
    assert clean.parameters == proposal.parameters


def test_unbalanced_and_balanced_dispatch(baseline):
    first, second = baseline.dispatch_options[:2]
    evaluator = make_evaluator()
    with pytest.warns(UserWarning, match="entire redispatch"):
        clean = evaluator.sanitize_tuning(baseline, RmsTuning(
            dispatch_mw={first: first.initial_mw + 0.1}))
    assert clean.dispatch_mw == {}
    proposal = RmsTuning(dispatch_mw={first: first.initial_mw + 0.1,
                                    second: second.initial_mw - 0.1})
    assert evaluator.sanitize_tuning(baseline, proposal) == proposal


def test_wrong_return_type_remains_error(baseline):
    with pytest.raises(ValueError, match="RmsTuning"):
        make_evaluator().sanitize_tuning(baseline, {})
