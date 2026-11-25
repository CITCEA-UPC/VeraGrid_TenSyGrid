"""Check static feasibility before spending full RMS trials.

Read in order: load a case, build independent proposals, solve each proposal,
then report losses, voltage and loading. Dynamic stability needs a full trial.
"""
from pathlib import Path
import numpy as np

from VeraGridEngine.api import open_file
from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions

from hackathon_files import BasicRmsTuner, RmsTuning
from hackathon_files.basic import initialize_grid
from hackathon_files.intro.examples.tuning_examples import action_examples


def main() -> int:
    # 1. Prepare the saved case and power-flow solver settings.
    path = Path(__file__).resolve().parents[2] / 'basic/testgrids/basic_ieee9_01.veragrid'
    grid = initialize_grid(open_file(str(path)))
    tuner = BasicRmsTuner()
    options = PowerFlowOptions(tolerance=1e-9, max_iter=50)

    # 2. Compare the unchanged baseline with independent physical actions.
    proposals = {'baseline': RmsTuning(), **action_examples(grid)}
    print('Each candidate is solved on its own copy of the grid.')
    for name, proposal in proposals.items():
        print(f'\nCandidate: {name}')
        # 3. Solve and check convergence before reading the result arrays.
        results = tuner.run_powerflow(proposal, grid=grid, options=options)
        if not bool(results.converged):
            print('  Power flow did not converge; skipping this candidate.')
            continue
        # get_losses solves the supplied proposal independently; it does not
        # return a cached result from another proposal. Losses include shunt G.
        losses_mw = tuner.get_losses(proposal, grid=grid, options=options)
        voltage_magnitudes = np.abs(results.voltage)
        branch_loading = np.abs(results.loading)
        print(f'  Active losses: {losses_mw:.6f} MW')
        print(
            f'  Bus voltage range: {voltage_magnitudes.min():.4f}'
            f'--{voltage_magnitudes.max():.4f} pu (target: 0.95--1.05)'
        )
        if branch_loading.size:
            print(f'  Maximum branch loading: {branch_loading.max():.2%} (limit: 100%)')
        else:
            print('  No branches to check for thermal loading.')
    print(f'Full RMS trials consumed: {grid.tests_used}/{grid.max_test_tunings}')
    # Controller gains alone usually do not change static power-flow losses.
    # A converged power flow does not establish dynamic stability.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
