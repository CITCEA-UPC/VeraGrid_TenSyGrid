"""Proposal power-flow checks share scoring actions without consuming trials."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from VeraGridEngine.api import open_file
from hackathon_files import BasicRmsTuner, RmsTuning
from hackathon_files.basic import initialize_grid, basic_evaluator
from hackathon_files.intro.examples.tuning_examples import action_examples


@pytest.fixture(scope='module')
def grid():
    path = Path(__file__).resolve().parents[1] / 'basic/testgrids/basic_ieee9_01.veragrid'
    return initialize_grid(open_file(str(path)))


def test_powerflow_proposals_match_full_scoring_and_isolate_grid(grid):
    tuner = BasicRmsTuner()
    before_p = [g.P for g in grid.problem.grid.generators]
    before_active = [line.active for line in grid.problem.grid.lines]
    before_counts = len(grid.problem.grid.lines), len(grid.problem.grid.shunts)
    for proposal in [RmsTuning(), *action_examples(grid).values()]:
        pf = tuner.run_powerflow(proposal, grid=grid)
        assert pf.converged
        losses = tuner.get_losses(proposal, grid=grid)
        full = basic_evaluator().evaluate_proposal(grid, proposal)
        assert losses == pytest.approx(full.active_losses_mw, abs=1e-7)
    assert [g.P for g in grid.problem.grid.generators] == before_p
    assert [line.active for line in grid.problem.grid.lines] == before_active
    assert (len(grid.problem.grid.lines), len(grid.problem.grid.shunts)) == before_counts
    assert grid.tests_used == tuner.tests_used == 0


def test_losses_include_active_shunt_consumption(grid):
    tuner = BasicRmsTuner()
    shunt = replace(grid.shunt_options[0], conductance_mw=0.5)
    snapshot = replace(grid, shunt_options=(shunt,))
    proposal = RmsTuning(add_shunts=(shunt,))
    pf = tuner.run_powerflow(proposal, grid=snapshot)
    index = next(i for i, b in enumerate(grid.problem.grid.buses) if b.idtag == shunt.bus_idtag)
    expected = np.sum(pf.losses.real) + 0.5 * abs(pf.voltage[index]) ** 2
    assert tuner.get_losses(proposal, grid=snapshot) == pytest.approx(expected)


def test_powerflow_nonconvergence_and_invalid_proposals(grid, monkeypatch):
    tuner = BasicRmsTuner()
    with pytest.raises(ValueError, match="RmsTuning"):
        tuner.run_powerflow(None, grid=grid)
    monkeypatch.setattr(tuner, '_proposal_powerflow',
                        lambda *args: (None, SimpleNamespace(converged=False)))
    assert not tuner.run_powerflow(RmsTuning(), grid=grid).converged
    with pytest.raises(RuntimeError, match='did not converge'):
        tuner.get_losses(RmsTuning(), grid=grid)
