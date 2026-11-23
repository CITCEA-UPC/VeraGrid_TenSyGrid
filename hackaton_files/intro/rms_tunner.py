"""Implement your algorithm in FinalTuner.tune()."""
from hackaton_files import RmsTuner, RmsTuning, TunableGrid


class FinalTuner(RmsTuner):
    """Participant tuner; the default proposal leaves the grid unchanged."""

    def tune(self, tunable_grid: TunableGrid) -> RmsTuning:
        """Choose a proposal using the current scenario's permitted handles.

        :param tunable_grid: Initialized grid and catalog of allowed actions.
        :return: Controller, dispatch or network changes to evaluate.
        """
        # Use self.get_modal_analysis(tunable_grid) to inspect the modes.
        # Build RmsTuning(parameters={handle: value}) with catalog handles.
        # tunable_grid.test_tuning(proposal) runs a full trial (budget: five).
        # Return the chosen proposal; do not modify the input grid directly.
        return RmsTuning()
