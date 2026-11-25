"""Build a proposal, test two control candidates, and return the better one.

Start with TrialTuner.tune() for the search loop. action_examples() separately
shows the permitted physical actions; these are not combined with the search.
The ranking is illustrative, not the competition's composite scoring formula.
"""
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
    # Each proposal uses one handle from the scenario's permitted catalog.
    # Keep these explicit so each RmsTuning field is easy to copy and adapt.
    if grid.openable_lines:
        proposals['open'] = RmsTuning(open_lines=(grid.openable_lines[0],))
    if grid.closable_lines:
        proposals['close'] = RmsTuning(close_lines=(grid.closable_lines[0],))
    if grid.line_candidates:
        proposals['reinforce'] = RmsTuning(add_lines=(grid.line_candidates[0],))
    if grid.shunt_options:
        proposals['shunt'] = RmsTuning(add_shunts=(grid.shunt_options[0],))

    # Transfer the same MW amount between two generators to keep dispatch balanced.
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
        # 1. Inspect controls in the same order as control_tensor rows.
        print('1. Available controls (initial value; allowed range)')
        for parameter in grid.tunable_parameters:
            print(
                f'   {parameter.name}: {parameter.initial_value:g}; '
                f'[{parameter.minimum:g}, {parameter.maximum:g}]'
            )
        print('\n2. Independent physical actions (catalog checks only)')
        for name, proposal in action_examples(grid).items():
            report = self.validate_proposal(grid, proposal)
            print(name, 'permitted:', report.accepted)
            for issue in report.issues:
                print('   ', issue)
        # 3. Compare the baseline with one case-specific gain setting.
        # -400 is a demonstration for this saved case, not a universal gain.
        gain_values = {}
        for parameter in grid.tunable_parameters:
            if parameter.name == 'Kp':
                gain_values[parameter] = max(
                    parameter.minimum, min(-400.0, parameter.maximum)
                )
        candidates = [
            ('baseline', RmsTuning()),
            ('adjust Kp', RmsTuning(parameters=gain_values)),
        ]
        best = RmsTuning()
        best_rank = None
        print('\n3. Full control trials')
        for name, proposal in candidates:
            if grid.tests_used >= grid.max_test_tunings:
                print('   Trial budget exhausted; returning the best proposal so far.')
                break
            report = self.validate_proposal(grid, proposal)
            if not report.accepted:
                print(f'   {name}: skipped because {report.issues}')
                continue
            # Inside tune(), use grid.test_tuning(). Calling self.test_tuning()
            # here would recursively invoke tune(). Failed trials also cost budget.
            try:
                metrics = grid.test_tuning(proposal)
            except Exception as error:
                print('Trial failed:', error)
                continue
            # Lower tuples win, comparing each criterion from left to right.
            # Prefer stability, then fewer violations, then a smaller real part
            # of the leading eigenvalue, then lower losses.
            rank = (
                not metrics.stable,
                metrics.voltage_violations + metrics.thermal_violations,
                metrics.max_eigenvalue_real,
                metrics.active_losses_mw,
            )
            print(
                f'   {name}: stable={metrics.stable}, '
                f'violations={rank[1]}, max real part={metrics.max_eigenvalue_real:.6g}, '
                f'losses={metrics.active_losses_mw:.6f} MW'
            )
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
