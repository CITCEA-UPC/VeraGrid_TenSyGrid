"""Controller discovery must support the real participant tuning workflow."""
from pathlib import Path

import numpy as np
import pytest

from VeraGridEngine.api import open_file
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from hackathon_files import BasicRmsTuner, ControllerRole, ParticipationScope, RmsTuning, TunableGrid
from hackathon_files.basic import initialize_grid
from hackathon_files.src.rms_controllers import ControllerBlock, ControllerMode
from hackathon_files.src.rms_tuning import RmsTuningMetrics, TunableParameter


@pytest.fixture(scope='module')
def mixed_case() -> tuple[BasicRmsTuner, TunableGrid, dict]:
    """Initialize a saved mixed GFM/GFL case and reuse one full eigensolve.

    :return: Participant interface, initialized snapshot and modal analysis.
    """
    path: Path = (Path(__file__).resolve().parents[1] / 'basic') / 'testgrids/basic_ieee9_05.veragrid'
    tunable: TunableGrid = initialize_grid(open_file(str(path)))
    tuner: BasicRmsTuner = BasicRmsTuner()
    return tuner, tunable, tuner.get_modal_analysis(tunable)


def test_controller_discovery_parameters_and_modes(mixed_case: tuple[BasicRmsTuner, TunableGrid, dict]) -> None:
    """Inspect both converter families with device-local permitted handles.

    :param mixed_case: Real saved converter scenario and cached modal snapshot.
    """
    tuner: BasicRmsTuner
    tunable: TunableGrid
    analysis: dict
    tuner, tunable, analysis = mixed_case
    trials_before: int = tunable.tests_used
    roots_before: tuple[str, ...] = tuple(str(generator.rms_model) for generator in tunable.problem.grid.generators)
    families: set[ControllerRole] = set()
    controller: ControllerBlock
    for generator in tunable.problem.grid.generators:
        for controller in tuner.get_controller_blocks(generator):
            if controller.role != ControllerRole.UNCLASSIFIED:
                families.add(controller.role)
                assert controller.variables
                assert controller.state_equations or controller.algebraic_equations
                handles: tuple[TunableParameter, ...] = tuner.get_controller_parameters(tunable, controller)
                assert all(handle in tunable.tunable_parameters and handle.device_idtag == generator.idtag for handle in handles)
                if 'Basic dynamic' in generator.rms_model.name:
                    assert handles
                    assert all(handle.name != 'Kp' for handle in handles)
                    modes: tuple[ControllerMode, ...] = tuner.get_controller_modes(analysis, controller)
                    assert modes
                    mode: ControllerMode
                    for mode in modes:
                        assert np.isfinite(mode.eigenvalue)
                        assert abs(mode.eigenvalue) > 1e-8
                        assert 0.01 <= mode.participation <= 1.0 + 1e-12
                        selected: np.ndarray = np.array(list(label['device_idtag'] == generator.idtag
                                                            and label['uid'] in mode.variable_uids
                                                            for label in analysis['variables']))
                        factors: np.ndarray = analysis['participation_factors'][:, mode.index]
                        assert mode.participation == pytest.approx(float(factors[selected].sum() / factors.sum()))
                    assert all(modes[index].participation >= modes[index + 1].participation for index in range(len(modes) - 1))
                else:
                    # Stable synchronous gains are deliberately not editable
                    # in this converter scenario, even though equations use Kp.
                    assert handles == tuple()
                    governor_modes: tuple[ControllerMode, ...] = tuner.get_controller_modes(analysis, controller)
                    assert governor_modes
                    assert all(mode.scope == ParticipationScope.AFFECTED for mode in governor_modes)
            else:
                pass
    assert {ControllerRole.PLL, ControllerRole.OUTER_POWER, ControllerRole.OUTER_VOLTAGE,
            ControllerRole.INNER_CURRENT, ControllerRole.DROOP, ControllerRole.GOVERNOR}.issubset(families)
    assert roots_before == tuple(str(generator.rms_model) for generator in tunable.problem.grid.generators)
    assert tuner.tests_used == 0
    assert tunable.tests_used == trials_before


def test_controller_handles_can_stabilize_a_saved_mixed_case(mixed_case: tuple[BasicRmsTuner, TunableGrid, dict]) -> None:
    """Build an accepted proposal solely from the discovered inner-loop handles.

    :param mixed_case: Real saved converter scenario.
    """
    tuner: BasicRmsTuner
    tunable: TunableGrid
    analysis: dict
    tuner, tunable, analysis = mixed_case
    values: dict[TunableParameter, float] = dict()
    growing_inner_modes: int = 0
    for generator in tunable.problem.grid.generators:
        for controller in tuner.get_controller_blocks(generator):
            if controller.role == ControllerRole.INNER_CURRENT:
                for parameter in tuner.get_controller_parameters(tunable, controller):
                    values[parameter] = 0.5 if parameter.name == 'Kp_icl' else 5.0
                growing_inner_modes += sum(mode.eigenvalue.real > 1e-4
                                           for mode in tuner.get_controller_modes(analysis, controller))
            else:
                pass
    assert growing_inner_modes > 0
    assert len(values) == 4
    proposal: RmsTuning = RmsTuning(parameters=values)
    assert proposal.is_accepted(tunable)
    metrics: RmsTuningMetrics = tunable.test_tuning(proposal)
    assert metrics.stable
    assert metrics.voltage_violations == metrics.thermal_violations == 0
    assert tunable.tests_used == 1
    assert all(parameter.initial_value != value for parameter, value in values.items())


def test_static_governor_modes_report_affected_states() -> None:
    """An algebraic governor must not claim ownership of rotor states.

    :return: None.
    """
    path: Path = (Path(__file__).resolve().parents[1] / 'basic') / 'testgrids/basic_ieee9_01.veragrid'
    tunable: TunableGrid = initialize_grid(open_file(str(path)))
    tuner: BasicRmsTuner = BasicRmsTuner()
    controller: ControllerBlock = tuner.get_controller_blocks(tunable.problem.grid.generators[0])[0]
    assert controller.role == ControllerRole.GOVERNOR
    parameters: tuple[TunableParameter, ...] = tuner.get_controller_parameters(tunable, controller)
    assert tuple(parameter.name for parameter in parameters) == ('Kp',)
    assert parameters[0].minimum < 0.0 < parameters[0].initial_value
    modes: tuple[ControllerMode, ...] = tuner.get_controller_modes(tunable, controller)
    assert modes
    assert all(mode.scope == ParticipationScope.AFFECTED for mode in modes)
    assert any(mode.eigenvalue.real > 1e-4 for mode in modes)


def test_unknown_blocks_are_not_classified_by_display_name(mixed_case: tuple[BasicRmsTuner, TunableGrid, dict]) -> None:
    """Names alone must not turn a fixed source into a PLL or current loop.

    :param mixed_case: Source of an actual fixed-voltage reference model.
    """
    tuner: BasicRmsTuner
    tunable: TunableGrid
    analysis: dict
    tuner, tunable, analysis = mixed_case
    grid: MultiCircuit = tunable.problem.grid.copy()
    grid.generators[0].rms_model.name = 'PLL governor inner current controller'
    controllers: tuple[ControllerBlock, ...] = tuner.get_controller_blocks(grid.generators[0])
    assert controllers
    assert all(controller.role == ControllerRole.UNCLASSIFIED for controller in controllers)
