"""Inspect the saved mixed IEEE9 controllers without consuming tuning trials.

Run from the repository root with this checkout's engine importable:
    python -m hackathon_files.intro.examples.controller_examples
"""
from pathlib import Path

from VeraGridEngine.api import open_file
from hackathon_files import BasicRmsTuner, ControllerRole, TunableGrid
from hackathon_files.basic import initialize_grid


def main() -> int:
    """Show discovery, permitted parameters, equations and cached modal results.

    :return: Zero after inspecting the real saved mixed-converter scenario.
    """
    # 1. Load a mixed-converter case and prepare its permitted controls.
    path = (
        Path(__file__).resolve().parents[2]
        / 'basic/testgrids/basic_ieee9_05.veragrid'
    )
    grid: TunableGrid = initialize_grid(open_file(str(path)))
    tuner: BasicRmsTuner = BasicRmsTuner()
    # Reuse this eigensolve for every device and controller. Diagnostics neither
    # evaluate a proposal nor consume the snapshot's trial allowance.
    analysis: dict = tuner.get_modal_analysis(grid)
    # 2. Discover installed views and their permitted handles in one call.
    # Use role=ControllerRole.PLL to select only PLLs. The lower-level
    # get_controller_blocks(device) remains available for device inspection.
    for info in tuner.get_controllers(grid):
        controller = info.controller
        # Unclassified blocks can be inspected separately. Their names do
        # not establish a controller role, so skip them in this walkthrough.
        if controller.role == ControllerRole.UNCLASSIFIED:
            continue

        print(f'\n{controller.device_name} / {controller.role.name}')
        print(f'  Block: {controller.path}')

        # 3. Discover handles and bounds instead of guessing parameter names.
        parameters = info.parameters
        print('  Permitted parameters (initial value; allowed range):')
        for parameter in parameters:
            print(
                f'    {parameter.name}: {parameter.initial_value:g}; '
                f'[{parameter.minimum:g}, {parameter.maximum:g}]'
            )
        if not parameters:
            print('    No editable parameters for this block.')

        print(
            f'  Equations: {len(controller.state_equations)} state, '
            f'{len(controller.algebraic_equations)} connected algebraic'
        )

        # 4. Reuse the analysis to inspect up to three associated modes.
        modes = tuner.get_controller_modes(analysis, controller)[:3]
        print('  Associated modes (up to three):')
        for mode in modes:
            print(
                f'    Eigenvalue: {mode.eigenvalue:+.5g} | '
                f'frequency: {mode.frequency_hz:.3f} Hz'
            )
            print(
                f'      Damping: {mode.damping_ratio:.2%} | '
                f'participation: {mode.participation:.2%} | '
                f'scope: {mode.scope.name}'
            )
        if not modes:
            print('    No associated modes reported.')
    print(f'\nTrials consumed: {grid.tests_used}/{grid.max_test_tunings}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
