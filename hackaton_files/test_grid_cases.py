"""Checks for reproducible, viable scrambled RMS cases."""

import pytest

from VeraGridEngine.api import open_file

from hackaton_files import (
    BasicRmsTuner,
    RmsTuner,
    RmsTuning,
    RmsTunerEvaluator,
    STABLE_GRID_FILES,
    UNSTABLE_CONTROL_RANGES,
    UNSTABLE_GRID_FILE,
    TunableGrid,
    make_unstable_grid,
    scramble_dispatch,
    scramble_grid,
)


def test_scramble_grid_is_reproducible_and_viable() -> None:
    first = scramble_grid(17)
    second = scramble_grid(17)

    assert first.baseline_metrics.stable
    assert first.baseline_metrics.voltage_violations == 0
    assert first.baseline_metrics.thermal_violations == 0
    assert first.baseline_metrics.neutral_mode_count == 1
    assert first.source_file == second.source_file
    assert first.baseline_metrics == second.baseline_metrics
    assert [(line.R, line.X, line.active) for line in first.grid.lines] == [
        (line.R, line.X, line.active) for line in second.grid.lines
    ]
    assert first.evaluator.line_corridors == second.evaluator.line_corridors
    assert first.evaluator.shunt_bus_names == second.evaluator.shunt_bus_names

    source = open_file(str(first.source_file))
    assert [bus.idtag for bus in first.grid.buses] != [bus.idtag for bus in source.buses]
    assert [bus.idtag for bus in first.grid.buses] == [bus.idtag for bus in second.grid.buses]
    assert {bus.idtag for bus in first.grid.buses} == {bus.idtag for bus in source.buses}
    assert {
        (line.idtag, line.bus_from.idtag, line.bus_to.idtag, line.active)
        for line in first.grid.lines
    } == {
        (line.idtag, line.bus_from.idtag, line.bus_to.idtag, line.active)
        for line in source.lines
    }


def test_scramble_dispatch_preserves_source_and_rebalances_generation() -> None:
    source = open_file(str(STABLE_GRID_FILES[1]))
    original_loads = [(load.P, load.Q) for load in source.loads]
    original_generation = [generator.P for generator in source.generators]

    first = scramble_dispatch(source, seed=23)
    second = scramble_dispatch(source, seed=23)

    assert [(load.P, load.Q) for load in source.loads] == original_loads
    assert [generator.P for generator in source.generators] == original_generation
    assert [(load.P, load.Q) for load in first.loads] == [
        (load.P, load.Q) for load in second.loads
    ]
    assert [generator.P for generator in first.generators] == [
        generator.P for generator in second.generators
    ]
    assert [load.P for load in first.loads] != [load.P for load in source.loads]
    assert sum(generator.P for generator in first.generators) / sum(load.P for load in first.loads) == pytest.approx(
        sum(original_generation) / sum(P for P, _ in original_loads)
    )


class RepairControls(RmsTuner):
    def tune(self, tunable_grid: TunableGrid) -> RmsTuning:
        targets = {"Kp": 0.0, "Ki": 0.0}
        return RmsTuning({parameter: targets[parameter.name]
                          for parameter in tunable_grid.tunable_parameters})


def test_unstable_grid_can_be_repaired_by_control_tuning() -> None:
    generated = make_unstable_grid(seed=2026)
    assert generated.baseline_metrics.max_eigenvalue_real > 1e-5
    assert not generated.baseline_metrics.stable
    assert generated.baseline_metrics.voltage_violations == 0

    saved = open_file(str(UNSTABLE_GRID_FILE))
    evaluator = RmsTunerEvaluator(control_ranges=UNSTABLE_CONTROL_RANGES)
    saved_baseline = BasicRmsTuner().test_tuning(saved, evaluator)
    assert saved_baseline.max_eigenvalue_real > 1e-5
    assert not saved_baseline.stable
    repaired = RepairControls().test_tuning(saved, evaluator)
    assert repaired.stable
    assert repaired.voltage_violations == 0
    assert repaired.max_eigenvalue_real < -1e-5
