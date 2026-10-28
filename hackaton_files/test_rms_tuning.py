"""Small end-to-end check of the participant tuning interface."""

from pathlib import Path

import pytest

from VeraGridEngine.api import open_file

from hackaton_files import BasicRmsTuner, RmsTuner, RmsTuning, RmsTunerEvaluator, TunableGrid, test_tuning as evaluate_participant


GRID_PATH = Path(__file__).with_name("stable_rms_grid.veragrid")


class ChangeGen1Kp(RmsTuner):
    def tune(self, tunable_grid: TunableGrid) -> RmsTuning:
        parameter_1 = tunable_grid.tunable_parameters[0]
        return RmsTuning({parameter_1: 20.0})


class RecordingEvaluator(RmsTunerEvaluator):
    def apply_tuning(self, tunable_grid: TunableGrid, tuner):
        tuned_problem = super().apply_tuning(tunable_grid, tuner)
        assert tuned_problem is not tunable_grid.problem
        assert tuned_problem.grid is tunable_grid.problem.grid
        self.tuned_problem = tuned_problem
        return tuned_problem


def test_one_control_changes_small_signal_damping() -> None:
    grid = open_file(str(GRID_PATH))
    evaluator = RecordingEvaluator()

    baseline = evaluate_participant(BasicRmsTuner(), grid)
    changed = evaluate_participant(ChangeGen1Kp(), grid, evaluator)

    assert baseline.eigenvalue_count > 0
    assert changed.eigenvalue_count == baseline.eigenvalue_count
    assert changed.min_oscillatory_damping is not None
    assert baseline.min_oscillatory_damping is not None
    assert changed.min_oscillatory_damping != baseline.min_oscillatory_damping
    assert changed.active_losses_mw == baseline.active_losses_mw
    gen1_id = grid.generators[0].idtag
    tuned_kp = [
        evaluator.tuned_problem._variable_parameters_values[index]
        for index, variable in enumerate(evaluator.tuned_problem._variable_parameters)
        if variable.name == "Kp"
        and evaluator.tuned_problem._event_parameter_device_idtags[variable.uid] == gen1_id
    ]
    assert tuned_kp == [20.0]
    assert grid.generators[0].rms_model.children[0].event_dict[next(
        variable for variable in grid.generators[0].rms_model.children[0].event_dict
        if variable.name == "Kp"
    )].value == 0.0


class TryGridActions(RmsTuner):
    def tune(self, tunable_grid: TunableGrid) -> RmsTuning:
        self.tunable_grid = tunable_grid
        shunt = RmsTuning(add_shunts=(tunable_grid.shunt_options[0],))
        line = RmsTuning(add_lines=(tunable_grid.line_candidates[0],))
        self.shunt_metrics = tunable_grid.test_tuning(shunt)
        self.line_metrics = tunable_grid.test_tuning(line)
        with pytest.raises(RuntimeError, match="budget"):
            tunable_grid.test_tuning(RmsTuning())
        assert not RmsTuning(add_shunts=tunable_grid.shunt_options).is_accepted(tunable_grid)
        assert not RmsTuning(open_lines=tunable_grid.openable_lines[:3]).is_accepted(tunable_grid)
        return line if self.line_metrics.active_losses_mw < self.shunt_metrics.active_losses_mw else shunt


def test_grid_actions_and_candidate_budget() -> None:
    grid = open_file(str(GRID_PATH))
    tuner = TryGridActions()
    evaluator = RmsTunerEvaluator(max_test_tunings=2)
    selected = evaluate_participant(tuner, grid, evaluator)

    assert tuner.tunable_grid.tests_used == 2
    assert selected.active_losses_mw == min(
        tuner.line_metrics.active_losses_mw,
        tuner.shunt_metrics.active_losses_mw,
    )
    assert len(grid.lines) == 14
    assert len(grid.shunts) == 0


class Redispatch(RmsTuner):
    def tune(self, tunable_grid: TunableGrid) -> RmsTuning:
        assert {parameter.name for parameter in tunable_grid.tunable_parameters} == {"Kp", "Ki"}
        first, second = tunable_grid.dispatch_options[:2]
        self.proposal = RmsTuning(dispatch_mw={
            first: first.initial_mw + 10.0,
            second: second.initial_mw - 10.0,
        })
        assert self.proposal.is_accepted(tunable_grid)
        assert not RmsTuning(dispatch_mw={first: first.initial_mw + 10.0}).is_accepted(tunable_grid)
        return self.proposal


def test_balanced_dispatch_changes_power_flow_and_losses() -> None:
    grid = open_file(str(GRID_PATH))
    baseline = evaluate_participant(BasicRmsTuner(), grid)
    changed = evaluate_participant(Redispatch(), grid)
    assert changed.active_losses_mw != baseline.active_losses_mw
