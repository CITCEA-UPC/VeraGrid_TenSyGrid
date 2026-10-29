"""Compare Kundur control gains using the same load-step disturbance."""

from pathlib import Path

from VeraGridEngine.api import open_file

from hackaton_files import BasicRmsTuner, RmsTuner, RmsTuning, TunableGrid


GRID_PATH = Path(__file__).with_name("stable_rms_grid.veragrid")


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
    grid = open_file(str(GRID_PATH))
    setups = (
        ("baseline", BasicRmsTuner()),
        ("Kp=5", SetGeneratorGains({"Kp": 5.0})),
        ("Kp=20", SetGeneratorGains({"Kp": 20.0})),
        ("Ki=1", SetGeneratorGains({"Ki": 1.0})),
        ("Ki=5", SetGeneratorGains({"Ki": 5.0})),
    )
    print("setup,damping_ratio,max_real,peak_frequency_hz,peak_voltage_pu,ssa_stable")
    for name, tuner in setups:
        metrics = tuner.test_tuning(grid)
        print(
            f"{name},{metrics.min_oscillatory_damping:.8f},"
            f"{metrics.max_eigenvalue_real:.8f},"
            f"{metrics.peak_frequency_deviation_hz:.8f},"
            f"{metrics.peak_voltage_deviation_pu:.8f},"
            f"{metrics.stable}"
        )


if __name__ == "__main__":
    main()
