"""Verified RMS starting cases and reproducible challenge variations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from VeraGridEngine.api import open_file
from VeraGridEngine.Devices.multi_circuit import MultiCircuit

from .rms_tuning import BasicRmsTuner, KUNDUR_CONTROL_RANGES, RmsTunerEvaluator, RmsTuningMetrics


GRID_DIRECTORY = Path(__file__).with_name("grids")
STABLE_GRID_FILES = (
    GRID_DIRECTORY / "kundur_rms_60.veragrid",
    GRID_DIRECTORY / "kundur_rms_65.veragrid",
    GRID_DIRECTORY / "kundur_rms_70.veragrid",
)
UNSTABLE_GRID_FILE = GRID_DIRECTORY / "kundur_rms_unstable.veragrid"
UNSTABLE_CONTROL_RANGES = {
    "Kp": (0.0, 150.0),
    "Ki": (0.0, 10.0),
}

_CORRIDOR_POOL = (("Bus6", "Bus8"), ("Bus7", "Bus9"), ("Bus8", "Bus10"))
_SHUNT_BUS_POOL = ("Bus6", "Bus7", "Bus8", "Bus9")


def _permuted(items: list, rng: np.random.Generator) -> list:
    """Return the same objects in a non-identity order when possible."""
    order = rng.permutation(len(items))
    if len(items) > 1 and np.array_equal(order, np.arange(len(items))):
        order = np.roll(order, 1)
    return [items[int(index)] for index in order]


def _scramble_dispatch_in_place(
    grid: MultiCircuit,
    rng: np.random.Generator,
    variation: float,
) -> None:
    """Vary demand and redistribute generation at a similar total output."""
    original_load_mw = sum(load.P for load in grid.loads)
    original_generation_mw = sum(generator.P for generator in grid.generators)
    for load in grid.loads:
        factor = float(rng.uniform(1.0 - variation, 1.0 + variation))
        load.P *= factor
        load.Q *= factor
    for generator in grid.generators:
        generator.P *= float(rng.uniform(1.0 - variation, 1.0 + variation))
    if original_load_mw > 0 and original_generation_mw > 0:
        target_generation_mw = original_generation_mw * (
            sum(load.P for load in grid.loads) / original_load_mw
        )
        provisional_generation_mw = sum(generator.P for generator in grid.generators)
        for generator in grid.generators:
            generator.P *= target_generation_mw / provisional_generation_mw


def scramble_dispatch(grid: MultiCircuit, seed: int, variation: float = 0.03) -> MultiCircuit:
    """Return a reproducible dispatch variation without modifying ``grid``."""
    if seed < 0 or not 0 <= variation < 1:
        raise ValueError("seed must be non-negative and variation must be in [0, 1)")
    scrambled = grid.copy()
    _scramble_dispatch_in_place(scrambled, np.random.default_rng(seed), variation)
    return scrambled


@dataclass(frozen=True)
class ScrambledRmsCase:
    """A reproducible, validated grid and its challenge settings."""

    grid: MultiCircuit
    evaluator: RmsTunerEvaluator
    source_file: Path
    seed: int
    baseline_metrics: RmsTuningMetrics


@dataclass(frozen=True)
class UnstableRmsCase:
    """An RMS case with acceptable power flow and unstable physical modes."""

    grid: MultiCircuit
    evaluator: RmsTunerEvaluator
    seed: int
    baseline_metrics: RmsTuningMetrics


def make_unstable_grid(seed: int = 2026, max_attempts: int = 8) -> UnstableRmsCase:
    """Randomize controller gains until SSA confirms instability."""
    if seed < 0 or max_attempts < 1:
        raise ValueError("seed must be non-negative and max_attempts positive")
    rng = np.random.default_rng(seed)
    evaluator = RmsTunerEvaluator(control_ranges=UNSTABLE_CONTROL_RANGES)
    for _ in range(max_attempts):
        grid = open_file(str(STABLE_GRID_FILES[1]))
        for generator in grid.generators:
            generator.rms_model.set_parameter_in_model("Kp", float(rng.uniform(100.0, 130.0)))
            generator.rms_model.set_parameter_in_model("Ki", float(rng.uniform(0.0, 1.0)))
        try:
            metrics = BasicRmsTuner().test_tuning(grid, evaluator)
        except (RuntimeError, ValueError):
            continue
        if (metrics.max_eigenvalue_real > 1e-5
                and metrics.voltage_violations == 0
                and metrics.thermal_violations == 0):
            return UnstableRmsCase(grid, evaluator, seed, metrics)
    raise RuntimeError(f"Could not produce an unstable RMS grid for seed {seed}")


def scramble_grid(seed: int, max_attempts: int = 8) -> ScrambledRmsCase:
    """Vary one verified Kundur case and retain only a viable RMS snapshot.

    The seed controls dispatch, branch impedances, ordering of buses
    and devices, and which grid actions the tuner may consider. The source
    files remain unchanged. A candidate is returned only after PF and SSA
    converge with acceptable voltage, thermal, and physical-mode metrics.
    """
    if seed < 0 or max_attempts < 1:
        raise ValueError("seed must be non-negative and max_attempts positive")

    source_file = STABLE_GRID_FILES[seed % len(STABLE_GRID_FILES)]
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        grid = open_file(str(source_file))
        _scramble_dispatch_in_place(grid, rng, variation=0.03)
        for line in grid.lines:
            line.R *= float(rng.uniform(0.95, 1.05))
            line.X *= float(rng.uniform(0.95, 1.05))

        # Reordering changes numeric indices and list positions, never the
        # physical bus references or active state of any branch.
        grid.buses = _permuted(grid.buses, rng)
        grid.lines = _permuted(grid.lines, rng)
        grid.generators = _permuted(grid.generators, rng)
        grid.loads = _permuted(grid.loads, rng)

        corridor_indices = rng.permutation(len(_CORRIDOR_POOL))[:int(rng.integers(1, 3))]
        shunt_indices = rng.permutation(len(_SHUNT_BUS_POOL))[:int(rng.integers(1, 4))]
        corridors = tuple(_CORRIDOR_POOL[int(i)] for i in corridor_indices)
        shunt_buses = tuple(_SHUNT_BUS_POOL[int(i)] for i in shunt_indices)
        control_count = int(rng.integers(1, len(KUNDUR_CONTROL_RANGES) + 1))
        control_names = tuple(KUNDUR_CONTROL_RANGES)[:control_count]
        evaluator = RmsTunerEvaluator(
            control_ranges={name: KUNDUR_CONTROL_RANGES[name] for name in control_names},
            line_corridors=corridors,
            shunt_bus_names=shunt_buses,
            max_open_lines=int(rng.integers(1, 4)),
            max_new_lines=int(rng.integers(1, len(corridors) + 1)),
            max_new_shunts=int(rng.integers(1, min(2, len(shunt_buses)) + 1)),
            max_test_tunings=int(rng.integers(3, 11)),
        )
        try:
            metrics = BasicRmsTuner().test_tuning(grid, evaluator)
        except (RuntimeError, ValueError):
            continue
        if metrics.stable and metrics.voltage_violations == 0 and metrics.thermal_violations == 0:
            return ScrambledRmsCase(grid, evaluator, source_file, seed, metrics)

    raise RuntimeError(f"Could not produce a viable RMS grid for seed {seed}")
