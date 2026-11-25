"""Verify the saved participant workflow and permitted grid-action and controller references."""
from pathlib import Path

import pytest

from VeraGridEngine.api import open_file

from hackathon_files.basic.cases import basic_evaluator, build_network, case_files, initialize_grid, make_variant, network_specs, reference_tuning
from hackathon_files.src.rms_tuning import RmsTunerEvaluator, RmsTuning, RmsTuningMetrics, TunableGrid, build_rms_tunables
from hackathon_files.basic.controllers import ConverterKind, attach_converter
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from hackathon_files.basic.validation import validate_case
from hackathon_files.basic.cases import case_trial_budget


@pytest.mark.parametrize('path', case_files(), ids=list(path.stem for path in case_files()))
def test_saved_case_can_be_stabilized(path: Path) -> None:
    """Reopen each grid and verify PF, equilibrium, all modes and load step.

    :param path: Unstable serialized participant starting grid.
    """
    record: dict = validate_case(path)
    assert record['baseline_max_real'] > 1e-4
    assert record['witness_max_real'] < -1e-5
    assert record['witness_transient_converged']
    assert record['witness_structural_changes'] == len(record['witness_close_lines'])
    assert 0 <= record['witness_structural_changes'] <= record['max_structural_changes']
    assert 0 <= record['witness_changes_score'] <= 1
    if record['converters']:
        assert record['baseline_growing_converter_participation'] >= 0.1
        assert all(parameter['name'] != 'Kp' for parameter in record['tunable_parameters'])
    else:
        pass


@pytest.mark.parametrize('index', (0, 39))
def test_public_trial_session_preserves_baseline(index: int) -> None:
    """Exercise participant trials on a small and a medium saved grid.

    :param index: Deterministic case index in the fifty-file collection.
    """
    tunable: TunableGrid = initialize_grid(open_file(str(case_files()[index])))
    initial_values: tuple[float, ...] = tuple(parameter.initial_value for parameter in tunable.tunable_parameters)
    assert tunable.tests_used == 0
    assert tunable.max_test_tunings == case_trial_budget(tunable.problem.grid.name)
    assert len(tunable.dispatch_options) >= 2
    assert tunable.closable_lines
    assert tunable.line_candidates
    assert tunable.shunt_options
    metrics: RmsTuningMetrics = tunable.test_tuning(reference_tuning(tunable))
    assert metrics.stable
    assert metrics.voltage_violations == 0
    assert metrics.thermal_violations == 0
    assert tunable.tests_used == 1
    allowed_keys: set[tuple[str, str]] = set((parameter.device_idtag, parameter.name) for parameter in tunable.tunable_parameters)
    actual_values: tuple[float, ...] = tuple(
        parameter.initial_value
        for parameter in build_rms_tunables(tunable.problem.grid, basic_evaluator().control_ranges)
        if (parameter.device_idtag, parameter.name) in allowed_keys
    )
    assert actual_values == initial_values


def test_minimum_damping_is_configurable_for_real_gfm_case() -> None:
    """Accept a 2.5%-damped GFM tuning while the default 5% policy rejects it.

    :return: None.
    """
    grid: MultiCircuit = make_variant(build_network(network_specs()[1]), 'six_bus', 2)
    # Use the smaller converter configuration that motivated the requirement
    # change; the bundled six-bus cases retain their larger damping margin.
    attach_converter(grid, grid.generators[-1], ConverterKind.GFM, 4.0)
    tunable: TunableGrid = initialize_grid(grid)
    reference: RmsTuning = reference_tuning(tunable)
    tuning: RmsTuning = RmsTuning(parameters={
        parameter: 0.5 if parameter.name == 'Kp_icl' else value
        for parameter, value in reference.parameters.items()
    })
    relaxed: RmsTunerEvaluator = basic_evaluator()
    strict: RmsTunerEvaluator = RmsTunerEvaluator(
        control_ranges=relaxed.control_ranges, shunt_bus_names=(), line_corridors=(),
    )
    accepted: RmsTuningMetrics = relaxed.evaluate_proposal(tunable, tuning)
    rejected: RmsTuningMetrics = strict.evaluate_proposal(tunable, tuning)
    assert accepted.min_oscillatory_damping is not None
    assert 0.025 <= accepted.min_oscillatory_damping < 0.05
    assert accepted.stable
    assert not rejected.stable
    assert accepted.transient_converged
    assert rejected.transient_converged


@pytest.mark.parametrize('path', [p for p in case_files() if p.stem.endswith('_02')],
                         ids=lambda p: p.stem)
def test_controller_only_cases_hide_all_physical_actions(path):
    case = initialize_grid(open_file(str(path)))
    assert case.tunable_parameters
    assert not any((case.dispatch_options, case.openable_lines, case.closable_lines,
                    case.line_candidates, case.shunt_options))
    assert (case.max_open_lines, case.max_close_lines,
            case.max_new_lines, case.max_new_shunts) == (0, 0, 0, 0)
    assert case.max_test_tunings == case_trial_budget(case.problem.grid.name)
    proposal = reference_tuning(case)
    assert not proposal.close_lines
    metrics = case.test_tuning(proposal)
    assert metrics.stable
    assert metrics.voltage_violations == metrics.thermal_violations == 0
