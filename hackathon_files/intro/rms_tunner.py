"""Implement your algorithm in FinalTuner.tune()."""
from hackathon_files import RmsTuner, RmsTuning, TunableGrid


class FinalTuner(RmsTuner):
    """Participant tuner; the default proposal leaves the grid unchanged."""

    def tune(self, tunable_grid: TunableGrid) -> RmsTuning:
        """Choose a proposal using the current scenario's permitted handles.

        :param tunable_grid: Initialized grid and catalog of allowed actions.
        :return: Controller, dispatch or network changes to evaluate.
        """
        # Example: tune one controller gain and close one existing circuit.
        # Use handles from this case. Some catalogs are empty.
        # parameters = {}
        # if tunable_grid.tunable_parameters:
        #     gain = tunable_grid.tunable_parameters[0]
        #     parameters[gain] = (gain.minimum + gain.maximum) / 2
        #
        # proposal = RmsTuning(
        #     parameters=parameters,                       # TunableParameter -> value
        #     dispatch_mw={},                              # DispatchOption -> MW
        #     open_lines=(),                               # from openable_lines
        #     close_lines=tunable_grid.closable_lines[:1],   # existing open circuits
        #     add_lines=(),                                # from line_candidates
        #     add_shunts=(),                               # from shunt_options
        # )
        # Nonempty dispatch_mw must keep total scheduled generation balanced.
        # self.validate_proposal(tunable_grid, proposal) checks permissions.
        # tunable_grid.test_tuning(proposal) evaluates it and uses one trial.
        # return proposal
        return RmsTuning()
