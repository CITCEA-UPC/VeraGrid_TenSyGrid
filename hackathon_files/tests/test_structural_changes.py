"""PDF structural-action counting and per-scenario normalization."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from hackathon_files import BasicRmsTuner, LineCandidate, LineOption, RmsTuning, ShuntOption
from hackathon_files.intro.evaluator import initialize_grid, load_benchmark


@pytest.fixture(scope='module')
def grid():
    snapshot = initialize_grid(load_benchmark(), enable_trials=False)
    # The intro catalog has no additions. Supply explicit test-only permissions
    # for counting; end-to-end addition tests use the real Kundur catalogs.
    first, second = snapshot.problem.grid.buses[:2]
    return replace(
        snapshot,
        line_candidates=(LineCandidate(first.idtag, second.idtag, 'New circuit',
                                       .01, .1, .1, 400.),),
        shunt_options=(ShuntOption(first.idtag, first.name, 1., 10.),),
    )


def test_pdf_example_two_of_eight_changes(grid):
    scenario = replace(
        grid,
        openable_lines=tuple(LineOption(str(i), f'Open {i}') for i in range(4)),
        closable_lines=tuple(LineOption(str(i), f'Close {i}') for i in range(4, 6)),
        max_open_lines=4, max_close_lines=2, max_new_lines=1, max_new_shunts=1,
        line_candidates=grid.line_candidates[:1], shunt_options=grid.shunt_options[:1],
    )
    assert scenario.max_structural_changes == 8
    report = BasicRmsTuner().validate_proposal(
        scenario, RmsTuning(open_lines=scenario.openable_lines[:2])
    )
    assert report.accepted
    assert report.structural_changes == 2
    assert report.max_structural_changes == 8
    assert report.changes_score == .75
    assert .15 * report.changes_score == pytest.approx(.1125)


def test_all_supported_structural_actions_count_once(grid):
    scenario = replace(grid, max_open_lines=1, max_close_lines=1,
                       max_new_lines=1, max_new_shunts=1)
    # This fixture has no initially open lines, so supply a separate catalog handle.
    scenario = replace(scenario, closable_lines=(LineOption('standby', 'Standby'),))
    proposal = RmsTuning(
        open_lines=scenario.openable_lines[:1], close_lines=scenario.closable_lines,
        add_lines=scenario.line_candidates[:1], add_shunts=scenario.shunt_options[:1],
    )
    report = BasicRmsTuner().validate_proposal(scenario, proposal)
    assert report.accepted
    assert report.structural_changes == report.max_structural_changes == 4
    assert report.changes_score == 0
    assert scenario.tests_used == 0


def test_controls_and_balanced_redispatch_are_free(grid):
    first, second = grid.dispatch_options[:2]
    amount = min(.01, first.maximum_mw - first.initial_mw,
                 second.initial_mw - second.minimum_mw)
    proposal = RmsTuning(
        parameters={p: p.initial_value for p in grid.tunable_parameters},
        dispatch_mw={first: first.initial_mw + amount, second: second.initial_mw - amount},
    )
    report = BasicRmsTuner().validate_proposal(grid, proposal)
    assert report.accepted
    assert report.structural_changes == 0
    assert report.changes_score == 1


def test_count_only_sanitized_actions(grid):
    scenario = replace(grid, max_open_lines=1)
    first, second = scenario.openable_lines[:2]
    report = BasicRmsTuner().validate_proposal(scenario, RmsTuning(
        open_lines=(first, first, second, LineOption('unknown', 'Unknown')),
    ))
    assert not report.accepted
    assert len(report.issues) == 3
    assert report.effective_proposal.open_lines == (first,)
    assert report.structural_changes == 1
    assert 0 <= report.changes_score <= 1


def test_no_available_actions_gets_full_score_despite_nonzero_default_limits(grid):
    scenario = replace(grid, openable_lines=(), closable_lines=(),
                       line_candidates=(), shunt_options=())
    assert scenario.max_open_lines > 0
    assert scenario.max_structural_changes == 0
    report = BasicRmsTuner().validate_proposal(scenario, RmsTuning())
    assert report.accepted
    assert report.structural_changes == 0
    assert report.changes_score == 1


def test_maximum_uses_unique_available_actions_and_prevents_double_switch_count(grid):
    line = grid.openable_lines[0]
    scenario = replace(grid, openable_lines=(line, line), closable_lines=(line,),
                       line_candidates=(), shunt_options=(),
                       max_open_lines=5, max_close_lines=5)
    assert scenario.max_structural_changes == 1
    report = BasicRmsTuner().validate_proposal(scenario, RmsTuning(
        open_lines=(line,), close_lines=(line,),
    ))
    assert not report.accepted
    assert report.structural_changes == 1
    assert report.changes_score == 0


def test_saved_validation_records_use_change_metric():
    path = Path(__file__).resolve().parents[1] / 'basic/validation.json'
    records = json.loads(path.read_text())
    assert len(records) == 50
    for record in records:
        assert 'witness_peak_voltage_pu' not in record
        count = record['witness_structural_changes']
        maximum = record['max_structural_changes']
        assert count == len(record['witness_close_lines'])
        assert 0 <= count <= maximum
        assert record['witness_changes_score'] == pytest.approx(
            1 if maximum == 0 else 1 - count / maximum
        )
        if record['file'].endswith('_02.veragrid'):
            assert maximum == 0
