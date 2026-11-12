"""Shared, deterministic sample operating points (no hidden random cases)."""
from dataclasses import dataclass
from VeraGridEngine.api import open_file
from .starter_script import BENCHMARK_PATH


@dataclass(frozen=True)
class Scenario:
    name: str
    load_scale: float
    reactive_scale: float


# Quick screening uses five points spanning the local sample envelope.
QUICK_SCENARIOS = (
    Scenario("nominal", 1.00, 1.00),
    Scenario("light", 0.95, 0.95),
    Scenario("heavy", 1.05, 1.05),
    Scenario("low-reactive", 1.00, 0.90),
    Scenario("high-reactive", 1.00, 1.10),
)
LOCAL_SCENARIOS = QUICK_SCENARIOS + (
    Scenario("light-high-q", 0.95, 1.10),
    Scenario("heavy-low-q", 1.05, 0.90),
    Scenario("light-low-q", 0.95, 0.90),
    Scenario("heavy-high-q", 1.05, 1.10),
    Scenario("mid-light", 0.975, 1.00),
    Scenario("mid-heavy", 1.025, 1.00),
)


def load_benchmark():
    return open_file(str(BENCHMARK_PATH))


def make_scenario(benchmark, scenario):
    """Copy before editing so cases and repeated runs cannot contaminate others.

    Scale active generation with active demand; AC power flow lets the slack
    supply the remaining mismatch and losses. Reactive demand varies separately.
    RMS initialization then derives model injections from this operating point.
    """
    grid = benchmark.copy()
    for load in grid.loads:
        load.P *= scenario.load_scale
        load.Q *= scenario.reactive_scale
    for generator in grid.generators:
        generator.P *= scenario.load_scale
    return grid
