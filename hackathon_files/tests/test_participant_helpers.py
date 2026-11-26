"""Participant discovery, capability and validation contracts on real grids."""
from dataclasses import replace
from pathlib import Path
import warnings

import numpy as np
import pytest

from VeraGridEngine.api import open_file
from hackathon_files import (
    AnalysisCapability, BasicRmsTuner, ControllerInfo, ControllerRole,
    ProposalValidationReport, RmsTuning,
)
from hackathon_files.basic import initialize_grid
from hackathon_files.intro.evaluator import (
    initialize_grid as initialize_intro, load_benchmark, make_evaluator,
)


@pytest.fixture(scope='module')
def mixed_grid():
    path = Path(__file__).resolve().parents[1] / 'basic/testgrids/basic_ieee9_05.veragrid'
    return initialize_grid(open_file(str(path)))


@pytest.fixture(scope='module')
def intro_grid():
    return initialize_intro(load_benchmark(), enable_trials=False)


def test_discovery_returns_existing_views_and_permitted_handles(mixed_grid):
    tuner = BasicRmsTuner()
    used = mixed_grid.tests_used
    before = tuple(str(g.rms_model) for g in mixed_grid.problem.grid.generators)
    controllers = tuner.get_controllers(mixed_grid)
    assert controllers
    assert all(isinstance(info, ControllerInfo) for info in controllers)
    expected = {
        (device.idtag, controller.path)
        for device in mixed_grid.problem.grid.get_all_elements_iter()
        if hasattr(device, 'rms_model')
        for controller in tuner.get_controller_blocks(device)
    }
    assert {(info.controller.device_idtag, info.controller.path) for info in controllers} == expected
    for info in controllers:
        assert info.parameters == tuner.get_controller_parameters(mixed_grid, info.controller)
        assert all(any(handle is p for p in mixed_grid.tunable_parameters) for handle in info.parameters)
    plls = tuner.get_controllers(mixed_grid, role=ControllerRole.PLL)
    assert plls
    assert all(info.controller.role == ControllerRole.PLL for info in plls)
    assert all(info.parameters for info in plls)
    assert len(plls) < len(controllers)
    assert before == tuple(str(g.rms_model) for g in mixed_grid.problem.grid.generators)
    assert mixed_grid.tests_used == used
    with pytest.raises(TypeError, match='ControllerRole'):
        tuner.get_controllers(mixed_grid, role='pll')


def test_capabilities_are_cheap_and_match_descriptor_support(intro_grid, monkeypatch):
    tuner = BasicRmsTuner()
    # Capabilities must inspect structure without assembling symbolic matrices or SSA.
    def forbidden(*args, **kwargs):
        raise AssertionError('Capability discovery must not run an analysis')
    monkeypatch.setattr(tuner, 'get_modal_analysis', forbidden)
    monkeypatch.setattr(tuner, 'get_parameterized_state_matrix', forbidden)
    capabilities = tuner.get_analysis_capabilities(intro_grid)
    assert all(isinstance(value, AnalysisCapability) for value in capabilities.values())
    assert capabilities['modal_analysis'].supported
    assert capabilities['state_matrix'].supported
    for name in ('symbolic_state_matrices', 'parameter_overrides', 'analytic_sensitivities'):
        assert not capabilities[name].supported
        assert 'descriptor' in capabilities[name].reason
    assert not capabilities['full_trials'].supported
    with pytest.raises(NotImplementedError, match='descriptor'):
        tuner.get_symbolic_state_matrices(intro_grid)
    assert intro_grid.tests_used == 0


def test_capabilities_and_numeric_matrix_on_both_model_families(intro_grid):
    tuner = BasicRmsTuner()
    analysis = tuner.get_modal_analysis(intro_grid)
    np.testing.assert_allclose(tuner.get_state_matrix(intro_grid), analysis['state_matrix'])
    path = Path(__file__).resolve().parents[1] / 'basic/testgrids/basic_ieee9_01.veragrid'
    grid = initialize_grid(open_file(str(path)))
    used = grid.tests_used
    capabilities = tuner.get_analysis_capabilities(grid)
    assert capabilities['symbolic_state_matrices'].supported
    assert capabilities['parameter_overrides'].supported
    assert capabilities['analytic_sensitivities'].supported
    assert capabilities['full_trials'].supported
    tuner.get_parameterized_state_matrix(grid)
    assert grid.tests_used == used


def test_validation_explains_each_omission_and_matches_scoring(intro_grid):
    tuner = BasicRmsTuner()
    first, second, third = intro_grid.tunable_parameters[:3]
    unknown = replace(first, device_idtag='unknown')
    lines = intro_grid.openable_lines
    proposal = RmsTuning(
        parameters={first: first.initial_value, second: second.maximum + 1,
                    third: float('nan'), unknown: 0.0},
        open_lines=(lines[0], lines[0], lines[1], lines[2]),
        add_lines=('unknown',),
    )
    before = intro_grid.problem._variable_parameters_values.copy()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        report = tuner.validate_proposal(intro_grid, proposal)
    assert not captured  # Validation returns diagnostics instead of warning.
    assert isinstance(report, ProposalValidationReport)
    assert not report.accepted
    assert len(report.issues) == 6
    joined = '\n'.join(report.issues)
    for phrase in ('outside', 'finite real', 'unavailable handle', 'duplicate', 'budget exceeded'):
        assert phrase in joined
    with pytest.warns(UserWarning) as captured:
        sanitized = make_evaluator().sanitize_tuning(intro_grid, proposal)
    assert report.issues == tuple(str(item.message) for item in captured)
    assert report.effective_proposal == sanitized
    assert sanitized.parameters == {first: first.initial_value}
    assert tuner.validate_proposal(intro_grid, sanitized).accepted
    assert len(proposal.open_lines) == 4
    np.testing.assert_array_equal(before, intro_grid.problem._variable_parameters_values)
    assert intro_grid.tests_used == 0


def test_validation_handles_balanced_dispatch_and_reports_imbalance(intro_grid):
    tuner = BasicRmsTuner()
    first, second = intro_grid.dispatch_options[:2]
    delta = min(.01, first.maximum_mw - first.initial_mw,
                second.initial_mw - second.minimum_mw)
    assert delta > 0
    proposal = RmsTuning(dispatch_mw={first: first.initial_mw + delta,
                                      second: second.initial_mw - delta})
    assert tuner.validate_proposal(intro_grid, proposal).accepted
    proposal = RmsTuning(dispatch_mw={first: first.initial_mw + delta})
    report = tuner.validate_proposal(intro_grid, proposal)
    assert not report.accepted
    assert 'unbalanced total MW' in report.issues[0]
    assert not report.effective_proposal.dispatch_mw
    report = tuner.validate_proposal(intro_grid, RmsTuning(
        dispatch_mw={first: first.maximum_mw + 1, second: float('inf')}
    ))
    assert 'outside' in report.issues[0]
    assert 'finite real' in report.issues[0]
    assert not report.effective_proposal.dispatch_mw


def test_validation_malformed_containers_conflicts_and_types(intro_grid):
    tuner = BasicRmsTuner()
    report = tuner.validate_proposal(intro_grid, RmsTuning(parameters=None, dispatch_mw=[], open_lines=5))
    assert not report.accepted
    assert len(report.issues) == 3
    assert report.effective_proposal == RmsTuning()
    assert not tuner.validate_proposal(intro_grid, None).accepted
    assert tuner.validate_proposal(intro_grid, RmsTuning()).accepted
    line = intro_grid.openable_lines[0]
    grid = replace(intro_grid, closable_lines=(line,))
    report = tuner.validate_proposal(grid, RmsTuning(open_lines=(line,), close_lines=(line,)))
    assert not report.accepted
    assert 'conflicting close' in report.issues[0]
    assert report.effective_proposal.open_lines == (line,)
    assert not report.effective_proposal.close_lines
    for call in (tuner.get_controllers, tuner.get_analysis_capabilities):
        with pytest.raises(TypeError, match='TunableGrid'):
            call(None)
    with pytest.raises(TypeError, match='TunableGrid'):
        tuner.validate_proposal(None, RmsTuning())


@pytest.mark.parametrize('active_mode, reactive_mode, expected_gains', [
    ('Pac', 'Qac', {'Kp_pol', 'Ki_pol'}),
    ('Pdc', 'Vm_ac', {'Kp_pol', 'Ki_pol', 'Kp_vac', 'Ki_vac'}),
    ('Vm_dc', 'Qac', {'Kp_vdc', 'Ki_vdc', 'Kp_pol', 'Ki_pol'}),
    ('Vm_dc', 'Vm_ac', {'Kp_vdc', 'Ki_vdc', 'Kp_vac', 'Ki_vac'}),
])
def test_gfl_discovery_tracks_configured_outer_loops(active_mode, reactive_mode, expected_gains):
    from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
    from VeraGridEngine.Devices.Injections.generator import Generator
    from VeraGridEngine.Templates.Rms.vsc_gfl import build_gfl_converter_model
    from VeraGridEngine.enumerations import ConverterControlType

    factory = VarFactory()
    core, *_ = build_gfl_converter_model(
        factory, inputs=[factory.add_var(name) for name in ('Vm', 'Va', 'Vdc', 'P', 'Q')],
        control1=ConverterControlType[active_mode], control2=ConverterControlType[reactive_mode],
    )
    generator = Generator(name='GFL test')
    generator.rms_model = core
    views = {view.role: view for view in BasicRmsTuner().get_controller_blocks(generator)}
    assert set(views) == {ControllerRole.PLL, ControllerRole.OUTER_POWER, ControllerRole.INNER_CURRENT}
    assert set(views[ControllerRole.OUTER_POWER].parameter_names) == expected_gains
    for view in views.values():
        assert view.variables
        assert view.algebraic_equations or view.state_equations


def test_discovery_finds_gfm_loops_on_saved_grids(mixed_grid, intro_grid):
    for grid in (mixed_grid, intro_grid):
        devices = {}
        for info in BasicRmsTuner().get_controllers(grid):
            devices.setdefault(info.controller.device_idtag, {})[info.controller.role] = info
        gfm, = (views for views in devices.values() if ControllerRole.DROOP in views)
        for role, gains in (
            (ControllerRole.DROOP, {'Kdp', 'Kdq', 'tau_P', 'tau_Q'}),
            (ControllerRole.OUTER_VOLTAGE, {'Kp_icl', 'Ki_icl'}),
            (ControllerRole.INNER_CURRENT, {'Kp_icl', 'Ki_icl'}),
        ):
            info = gfm[role]
            assert info.controller.variables and info.controller.algebraic_equations
            assert {handle.name for handle in info.parameters} == gains
