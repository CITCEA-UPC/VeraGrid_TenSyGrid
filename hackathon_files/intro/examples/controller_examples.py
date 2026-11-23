"""Inspect the saved mixed IEEE9 controllers without consuming tuning trials.

Run from the repository root with this checkout's engine importable:
    python -m hackathon_files.intro.examples.controller_examples
"""
from pathlib import Path

from VeraGridEngine.api import open_file
from VeraGridEngine.Devices.Injections.generator import Generator
from hackathon_files import BasicRmsTuner, ControllerRole, TunableGrid
from hackathon_files.basic import initialize_grid
from hackathon_files.src.rms_controllers import ControllerBlock, ControllerMode
from hackathon_files.src.rms_tuning import TunableParameter


def main() -> int:
    """Show discovery, permitted parameters, equations and cached modal results.

    :return: Zero after inspecting the real saved mixed-converter scenario.
    """
    path: Path = (Path(__file__).resolve().parents[2] / 'basic') / 'testgrids/basic_ieee9_05.veragrid'
    grid: TunableGrid = initialize_grid(open_file(str(path)))
    tuner: BasicRmsTuner = BasicRmsTuner()
    # Reuse this eigensolve for every device and controller. Diagnostics neither
    # evaluate a proposal nor consume the snapshot's five-trial allowance.
    analysis: dict = tuner.get_modal_analysis(grid)
    generator: Generator
    controller: ControllerBlock
    for generator in grid.problem.grid.generators:
        for controller in tuner.get_controller_blocks(generator):
            if controller.role != ControllerRole.UNCLASSIFIED:
                print(f'\n{generator.name}: {controller.role.name} ({controller.path})')
                parameter: TunableParameter
                for parameter in tuner.get_controller_parameters(grid, controller):
                    print(f'  {parameter.name}: {parameter.initial_value:g} '
                          f'[{parameter.minimum:g}, {parameter.maximum:g}]')
                print(f'  Equations: {len(controller.state_equations)} state, '
                      f'{len(controller.algebraic_equations)} connected algebraic')
                mode: ControllerMode
                for mode in tuner.get_controller_modes(analysis, controller)[:3]:
                    print(f'  lambda={mode.eigenvalue:+.5g}, f={mode.frequency_hz:.3f} Hz, '
                          f'damping={mode.damping_ratio:.2%}, participation={mode.participation:.2%}, '
                          f'scope={mode.scope.name}')
            else:
                # Unclassified blocks remain available for inspection; their
                # display names are never treated as proof of a controller role.
                pass
    print(f'\nTrials consumed: {grid.tests_used}/{grid.max_test_tunings}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
