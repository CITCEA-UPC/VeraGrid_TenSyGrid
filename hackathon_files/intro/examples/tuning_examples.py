"""Implement tune(), inspect catalogs, and test proposals within a trial budget."""
from pathlib import Path
from VeraGridEngine.api import open_file
from hackathon_files import RmsTuner, RmsTuning, TunableGrid
from hackathon_files.basic import basic_evaluator


def action_examples(grid: TunableGrid) -> dict[str, RmsTuning]:
    """Build independent examples from permitted handles, without changing grid.

    Catalog acceptance checks permissions and bounds, not electrical feasibility.
    These actions are alternatives, not a combination promised to stabilize a case.
    """
    proposals = {}
    for label, catalog, field in (
        ('open', grid.openable_lines, 'open_lines'),
        ('close', grid.closable_lines, 'close_lines'),
        ('reinforce', grid.line_candidates, 'add_lines'),
        ('shunt', grid.shunt_options, 'add_shunts'),
    ):
        if catalog:
            proposals[label] = RmsTuning(**{field: (catalog[0],)})
    if len(grid.dispatch_options) >= 2:
        first, second = grid.dispatch_options[:2]
        amount = min(0.1, first.maximum_mw - first.initial_mw,
                     second.initial_mw - second.minimum_mw)
        if amount > 0:
            proposals['redispatch'] = RmsTuning(dispatch_mw={
                first: first.initial_mw + amount,
                second: second.initial_mw - amount,
            })
    return proposals


class TrialTuner(RmsTuner):
    """Small demonstration search; contestants should implement their own strategy."""

    def tune(self, grid: TunableGrid) -> RmsTuning:
        print('Controls [initial, minimum, maximum]:', grid.control_tensor)
        for name, proposal in action_examples(grid).items():
            print(name, 'permitted:', proposal.is_accepted(grid))
        # Control tensor rows align exactly with tunable_parameters.
        proposals = [RmsTuning(), RmsTuning(parameters={
            parameter: max(parameter.minimum, min(-400.0, parameter.maximum))
            for parameter in grid.tunable_parameters if parameter.name == 'Kp'
        })]
        best, best_rank = RmsTuning(), None
        for proposal in proposals:
            if grid.tests_used >= grid.max_test_tunings:
                break
            if not proposal.is_accepted(grid):
                continue
            # Inside tune(), use grid.test_tuning(). Calling self.test_tuning()
            # here would recursively invoke tune(). Failed trials also cost budget.
            try:
                metrics = grid.test_tuning(proposal)
            except Exception as error:
                print('Trial failed:', error)
                continue
            rank = (not metrics.stable, metrics.voltage_violations + metrics.thermal_violations,
                    metrics.max_eigenvalue_real, metrics.active_losses_mw)
            if best_rank is None or rank < best_rank:
                best, best_rank = proposal, rank
        print(f'Trials consumed: {grid.tests_used}/{grid.max_test_tunings}')
        return best


def main() -> int:
    # This saved synchronous case is unstable initially. Its case-local catalogs
    # are prepared with the basic initializer, just as in the combined evaluator.
    from hackathon_files.basic import initialize_grid
    path = Path(__file__).resolve().parents[2] / 'basic/testgrids/basic_ieee9_01.veragrid'
    baseline = initialize_grid(open_file(str(path)))
    evaluator = basic_evaluator()
    participant = evaluator.start_session(baseline)
    proposal = TrialTuner().tune(participant)
    metrics = evaluator.evaluate_proposal(baseline, proposal)
    print('Final stable:', metrics.stable, 'losses MW:', metrics.active_losses_mw)
    # For raw grids with standard catalogs, the outer evaluation API is
    # tuner.test_tuning(raw_grid, evaluator). The load-response example shows it.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
