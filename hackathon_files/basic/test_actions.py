"""General basic actions must change both the PF and RMS physical circuit."""
from dataclasses import replace
from enum import Enum

import pytest

from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Utils.Symbolic.symbolic import Const
from hackathon_files.basic.cases import basic_evaluator, build_network, initialize_grid, make_variant, network_specs, reference_tuning
from hackathon_files.basic.validation import equilibrium_error
from hackathon_files.src.rms_tuning import LineOption, RmsTunerEvaluator, RmsTuning, RmsTuningMetrics, TunableGrid


class ActionKind(Enum):
    """Distinct physical action paths exercised by this workflow."""

    CLOSE = 'close'
    SWAP = 'swap'
    NEW_LINE = 'new_line'
    SHUNT = 'shunt'
    DISPATCH = 'dispatch'


@pytest.fixture(scope='module')
def general_case() -> TunableGrid:
    """Prepare a real mixed-converter network with a normally-open circuit.

    :return: Independent IEEE9 scenario with all physical action catalogs.
    """
    grid: MultiCircuit = make_variant(build_network(network_specs()[2]), 'ieee9', 4)
    return initialize_grid(grid)


def proposal_for(case: TunableGrid, kind: ActionKind) -> RmsTuning:
    """Use actual current-snapshot handles to build each physical operation.

    :param case: Initialized mixed-converter case.
    :param kind: Action path to exercise alongside its stabilizing controller gains.
    :return: Complete, permission-checked proposal.
    """
    controls: RmsTuning = reference_tuning(case)
    if kind == ActionKind.CLOSE:
        result: RmsTuning = RmsTuning(parameters=controls.parameters, close_lines=(case.closable_lines[0],))
    elif kind == ActionKind.SWAP:
        standby = next(line for line in case.problem.grid.lines if not line.active)
        original = next(line for line in case.problem.grid.lines if line.active
                        and line.bus_from is standby.bus_from and line.bus_to is standby.bus_to
                        and line.R == standby.R and line.X == standby.X)
        opening: LineOption = next(option for option in case.openable_lines if option.idtag == original.idtag)
        result = RmsTuning(parameters=controls.parameters, open_lines=(opening,), close_lines=(case.closable_lines[0],))
    elif kind == ActionKind.NEW_LINE:
        result = RmsTuning(parameters=controls.parameters, add_lines=(case.line_candidates[0],))
    elif kind == ActionKind.SHUNT:
        result = RmsTuning(parameters=controls.parameters, add_shunts=(case.shunt_options[1],))
    else:
        first, second = case.dispatch_options[:2]
        delta: float = min(0.01, first.maximum_mw - first.initial_mw, second.initial_mw - second.minimum_mw)
        assert delta > 0.0
        result = RmsTuning(parameters=controls.parameters,
                           dispatch_mw={first: first.initial_mw + delta, second: second.initial_mw - delta})
    return result


@pytest.mark.parametrize('kind', tuple(ActionKind))
def test_general_action_changes_pf_and_rms(general_case: TunableGrid, kind: ActionKind) -> None:
    """Run PF, equilibrium, full SSA and load-step checks after a real action.

    :param general_case: Saved-topology-equivalent participant scenario.
    :param kind: Selected operation, including a genuine open/close transfer.
    """
    proposal: RmsTuning = proposal_for(general_case, kind)
    assert proposal.is_accepted(general_case)
    evaluator: RmsTunerEvaluator = basic_evaluator()
    assert evaluator.sanitize_tuning(general_case, proposal) == proposal
    snapshot: TunableGrid = initialize_grid(general_case.problem.grid.copy())
    problem = evaluator.apply_tuning(snapshot, proposal)
    assert equilibrium_error(problem) < 1e-6
    for option in proposal.open_lines:
        assert not next(line for line in problem.grid.lines if line.idtag == option.idtag).active
    for option in proposal.close_lines:
        assert next(line for line in problem.grid.lines if line.idtag == option.idtag).active
    assert len(problem.grid.lines) == len(general_case.problem.grid.lines) + len(proposal.add_lines)
    assert len(problem.grid.shunts) == len(general_case.problem.grid.shunts) + len(proposal.add_shunts)
    metrics: RmsTuningMetrics = evaluator.evaluate_proposal(general_case, proposal)
    assert metrics.stable
    assert metrics.transient_converged
    assert metrics.voltage_violations == metrics.thermal_violations == 0
    assert not next(line for line in general_case.problem.grid.lines if line.idtag == general_case.closable_lines[0].idtag).active
    baseline_line = next(line for line in general_case.problem.grid.lines
                         if line.idtag == general_case.closable_lines[0].idtag)
    baseline_status: tuple[float, ...] = tuple(float(expression.value)
        for block in baseline_line.rms_model.get_all_blocks()
        for variable, expression in block.event_dict.items()
        if variable.name == 'u' and isinstance(expression, Const))
    assert baseline_status
    assert all(value == 0.0 for value in baseline_status)


def test_closing_permissions_and_conflicts(general_case: TunableGrid) -> None:
    """Closing cannot bypass line state, membership, uniqueness or budgets.

    :param general_case: Actual catalogs used by participant trials.
    """
    closing: LineOption = general_case.closable_lines[0]
    assert not RmsTuning(close_lines=(general_case.openable_lines[0],)).is_accepted(general_case)
    assert not RmsTuning(close_lines=(closing, closing)).is_accepted(general_case)
    assert not RmsTuning(close_lines=(LineOption('unknown', 'unknown'),)).is_accepted(general_case)
    conflicting: TunableGrid = replace(general_case, openable_lines=(closing,))
    proposal: RmsTuning = RmsTuning(open_lines=(closing,), close_lines=(closing,))
    assert not proposal.is_accepted(conflicting)
    with pytest.warns(UserWarning, match='conflicting close'):
        cleaned: RmsTuning = basic_evaluator().sanitize_tuning(conflicting, proposal)
    assert cleaned.open_lines == (closing,)
    assert cleaned.close_lines == tuple()
    assert cleaned.is_accepted(conflicting)
