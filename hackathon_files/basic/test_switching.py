"""Permissions and saved evidence for the ten fixed-controller network cases."""
from hashlib import sha256
import json
from pathlib import Path
import pytest
from VeraGridEngine.api import open_file
from hackathon_files import BasicRmsTuner, RmsTuning
from hackathon_files.basic.cases import case_files,initialize_grid,reference_tuning,case_trial_budget

PATHS=[p for p in case_files() if p.stem.startswith('basic_switching_')]


def test_collection_has_ten_additional_network_cases():
    assert len(case_files())==60
    assert len(PATHS)==10
    records=json.loads(Path(__file__).with_name('switching_validation.json').read_text())
    assert len(records)==10
    for r in records:
        assert sha256((PATHS[0].parent/r['file']).read_bytes()).hexdigest()==r['sha256']
        assert r['challenge_type']=='network_actions'
        assert r['baseline_metrics']['thermal_violations']>0
        assert r['tunable_parameter_count']==0 and not r['witness_parameters']
        assert r['witness_metrics']['stable'] and r['witness_metrics']['transient_converged']
        assert r['witness_metrics']['thermal_violations']==r['witness_metrics']['voltage_violations']==0


@pytest.mark.parametrize('path',PATHS,ids=lambda p:p.stem)
def test_network_actions_are_available_without_control_or_dispatch(path):
    case=initialize_grid(open_file(str(path)))
    assert not case.tunable_parameters and not case.dispatch_options
    assert case.closable_lines and case.line_candidates and case.shunt_options
    assert (case.max_open_lines,case.max_close_lines,case.max_new_lines,case.max_new_shunts)==(1,1,1,1)
    assert 3<=case.max_structural_changes<=4
    assert case.max_test_tunings==case_trial_budget(path.stem)
    proposal=reference_tuning(case)
    assert not proposal.parameters and len(proposal.close_lines)==1
    assert proposal.is_accepted(case)
    assert BasicRmsTuner().validate_proposal(case,proposal).accepted
    assert case.tests_used==0
    for addition in (RmsTuning(add_lines=case.line_candidates[:1]),RmsTuning(add_shunts=case.shunt_options[:1])):
        assert addition.is_accepted(case)
