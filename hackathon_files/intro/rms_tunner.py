"""Implement your algorithm in FinalTuner.tune()."""
from hackathon_files import RmsTuner, RmsTuning, TunableGrid


class FinalTuner(RmsTuner):
    """Participant tuner; the default proposal leaves the grid unchanged."""

    def tune(self, tunable_grid: TunableGrid) -> RmsTuning:
        """Choose a proposal using the current scenario's permitted handles.

        :param tunable_grid: Initialized grid and catalog of allowed actions.
        :return: Controller, dispatch or network changes to evaluate.
        """
        return RmsTuning()


def example_proposal(grid: TunableGrid) -> RmsTuning:
    """Build a proposal for the public Lynn 5-bus synchronous case."""
    gains = {
        handle: -400.0
        for handle in grid.tunable_parameters
        if handle.name == "Kp" and handle.minimum <= -400.0 <= handle.maximum
    }
    restore_line = next(
        line for line in grid.closable_lines if line.name.startswith("Restore outage:")
    )
    return RmsTuning(parameters=gains, close_lines=(restore_line,))


def main() -> None:
    """Run with PYTHONPATH=src:. python -m hackathon_files.intro.rms_tunner."""
    from VeraGridEngine.api import open_file
    from hackathon_files.basic import case_files, initialize_grid

    path = next(path for path in case_files("limited") if path.stem == "basic_lynn5_01")
    grid = initialize_grid(open_file(str(path)))
    proposal = example_proposal(grid)
    report = FinalTuner().validate_proposal(grid, proposal)
    if not report.accepted:
        raise ValueError(report.issues)

    print("Controller settings:", {f"{p.device_name}: {p.name}": value
                                   for p, value in proposal.parameters.items()})
    print("Close line:", proposal.close_lines[0].name)
    print("Trial result:", grid.test_tuning(proposal))


if __name__ == "__main__":
    main()
