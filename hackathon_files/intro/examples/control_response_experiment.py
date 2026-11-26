"""Compare Kundur control gains using small-signal analysis.

Damping and eigenvalues describe the local linearized operating point.
All setups only adjust existing gains, so their structural-change count is zero.
"""

from pathlib import Path

from VeraGridEngine.api import open_file

from hackathon_files import BasicRmsTuner, RmsTuner, RmsTuning, TunableGrid


GRID_PATH = (Path(__file__).resolve().parents[2] / "stable_rms_grid.veragrid")


class SetGeneratorGains(RmsTuner):
    """Apply the same selected controller gains to all four generators."""

    def __init__(self, gains: dict[str, float]) -> None:
        super().__init__()
        self.gains = gains

    def tune(self, tunable_grid: TunableGrid) -> RmsTuning:
        return RmsTuning({
            parameter: self.gains[parameter.name]
            for parameter in tunable_grid.tunable_parameters
            if parameter.name in self.gains
        })


def main() -> None:
    # 1. Load once; the outer test API evaluates each tuner independently.
    grid = open_file(str(GRID_PATH))
    # 2. Vary one gain at a time, keeping the same grid.
    setups = (
        ("baseline", BasicRmsTuner()),
        ("Kp=5", SetGeneratorGains({"Kp": 5.0})),
        ("Kp=20", SetGeneratorGains({"Kp": 20.0})),
        ("Ki=1", SetGeneratorGains({"Ki": 1.0})),
        ("Ki=5", SetGeneratorGains({"Ki": 5.0})),
    )
    # 3. Report units in the header and align values for side-by-side comparison.
    print(
        f"{'Setup':<12} {'Damping':>10} {'Max Re (1/s)':>14} "
        f"{'Changes n/max':>14} {'S_changes':>12} {'SSA stable':>12}"
    )
    for name, tuner in setups:
        metrics = tuner.test_tuning(grid)
        print(
            f"{name:<12} {metrics.min_oscillatory_damping:>10.2%} "
            f"{metrics.max_eigenvalue_real:>14.6g} "
            f"{str(metrics.structural_changes) + '/' + str(metrics.max_structural_changes):>14} "
            f"{metrics.changes_score:>12.3f} "
            f"{str(metrics.stable):>12}"
        )



if __name__ == "__main__":
    main()
