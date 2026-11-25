"""Discover controllers, check analysis support and validate without full trials."""
from pathlib import Path

from VeraGridEngine.api import open_file
from hackathon_files import BasicRmsTuner, ControllerRole, RmsTuning
from hackathon_files.basic import initialize_grid


def main() -> int:
    path = (
        Path(__file__).resolve().parents[2]
        / 'basic/testgrids/basic_ieee9_05.veragrid'
    )
    grid = initialize_grid(open_file(str(path)))
    tuner = BasicRmsTuner()

    # 1. Discover support before attempting an expensive or unsupported analysis.
    # A supported API can still fail numerically; this only checks structure.
    print('1. Analysis capabilities')
    capabilities = tuner.get_analysis_capabilities(grid)
    for name, capability in capabilities.items():
        status = 'available' if capability.supported else 'unavailable'
        print(f'   {name}: {status}\n     {capability.reason}')

    # 2. Obtain installed PLLs, not newly created standalone PLL templates.
    print('\n2. Installed PLLs and their permitted controls')
    plls = tuner.get_controllers(grid, role=ControllerRole.PLL)
    for info in plls:
        print(f'   {info.controller.device_name} / {info.controller.path}')
        for parameter in info.parameters:
            print(
                f'     {parameter.name}: initial={parameter.initial_value:g}, '
                f'range=[{parameter.minimum:g}, {parameter.maximum:g}]'
            )
    if not plls:
        print('   No recognized PLLs in this case.')

    # 3. Deliberately submit an out-of-range value to see the validation report.
    # No warnings, grid changes, solves, or full trials occur during validation.
    print('\n3. Preview a rejected proposal')
    handle = grid.tunable_parameters[0]
    proposal = RmsTuning(parameters={handle: handle.maximum + 1})
    report = tuner.validate_proposal(grid, proposal)
    print(f'   Accepted unchanged: {report.accepted}')
    for issue in report.issues:
        print(f'   {issue}')
    print(f'   Effective proposal: {report.effective_proposal}')

    # 4. Correct the proposal using an allowed value (baseline for this example).
    # Passing validation says nothing about electrical feasibility or improvement.
    print('\n4. Validate an allowed proposal')
    proposal = RmsTuning(parameters={handle: handle.initial_value})
    report = tuner.validate_proposal(grid, proposal)
    print(f'   Accepted unchanged: {report.accepted}')
    print(f'   Structural changes: {report.structural_changes}/{report.max_structural_changes}')
    print(f'   Change score: {report.changes_score:.3f}')
    print(f'   Full trials consumed: {grid.tests_used}/{grid.max_test_tunings}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
