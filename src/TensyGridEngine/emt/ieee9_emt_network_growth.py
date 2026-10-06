"""Locate the first IEEE9 network section that breaks the EMT baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
for import_path in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import VeraGridEngine.api as gce
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph

from ieee9_emt_from_scratch import (
    attach_baseline_emt_models,
    build_emt_options,
)
from ieee9_emt_simulation import build_power_flow_options


@dataclass(frozen=True)
class Section:
    kind: str
    bus_from: int
    bus_to: int
    r: float
    x: float
    b: float
    rate: float


SECTIONS = (
    Section("transformer", 0, 3, 0.005, 0.0576, 0.0, 250.0),
    Section("line", 3, 4, 0.0170, 0.0920, 0.158, 250.0),
    Section("line", 4, 5, 0.0390, 0.1700, 0.358, 150.0),
    Section("transformer", 2, 5, 0.005, 0.0586, 0.0, 300.0),
    Section("line", 5, 6, 0.0119, 0.1008, 0.209, 150.0),
    Section("line", 6, 7, 0.0085, 0.0720, 0.149, 250.0),
    Section("transformer", 1, 7, 0.005, 0.0625, 0.0, 250.0),
    Section("line", 7, 8, 0.0320, 0.1610, 0.306, 250.0),
    Section("line", 8, 3, 0.0100, 0.0850, 0.176, 250.0),
)

REAL_LOADS = {4: 90.0, 6: 100.0, 8: 125.0}


def build_stage(stage: int) -> gce.MultiCircuit:
    """Build sections 1..stage, terminating every non-source leaf bus."""
    selected = SECTIONS[:stage]
    active_bus_indices = sorted({idx for section in selected for idx in (section.bus_from, section.bus_to)})
    grid = gce.MultiCircuit(name=f"IEEE9 EMT growth stage {stage}", Sbase=100.0, fbase=50.0)
    buses = {
        idx: gce.Bus(name=f"bus {idx}", Vnom=345.0, is_slack=(idx == 0))
        for idx in active_bus_indices
    }
    for bus in buses.values():
        grid.add_bus(bus)

    grid.add_generator(
        bus=buses[0],
        api_obj=gce.Generator(
            name="thevenin source",
            P=72.3,
            vset=1.04,
            Snom=grid.Sbase,
            freq=grid.fBase,
            r1=0.002,
            x1=0.25,
            Qmin=-999.0,
            Qmax=999.0,
        ),
    )

    degree = {idx: 0 for idx in active_bus_indices}
    for idx, section in enumerate(selected, start=1):
        degree[section.bus_from] += 1
        degree[section.bus_to] += 1
        if section.kind == "line":
            grid.add_line(
                gce.Line(
                    name=f"stage line {idx}",
                    bus_from=buses[section.bus_from],
                    bus_to=buses[section.bus_to],
                    r=section.r,
                    x=section.x,
                    b=section.b,
                    rate=section.rate,
                )
            )
        else:
            grid.add_transformer2w(
                gce.Transformer2W(
                    name=f"stage transformer {idx}",
                    bus_from=buses[section.bus_from],
                    bus_to=buses[section.bus_to],
                    r=section.r,
                    x=section.x,
                    g=0.0005,
                    b=0.02,
                    rate=section.rate,
                    HV=345.0,
                    LV=345.0,
                )
            )

    for bus_idx, p_mw in REAL_LOADS.items():
        if bus_idx in buses:
            grid.add_load(buses[bus_idx], gce.Load(name=f"load bus {bus_idx}", P=p_mw, Q=0.0))

    loaded = set(REAL_LOADS).intersection(buses)
    for bus_idx, connections in degree.items():
        if bus_idx != 0 and connections == 1 and bus_idx not in loaded:
            grid.add_load(
                buses[bus_idx],
                gce.Load(name=f"leaf termination bus {bus_idx}", P=1.0, Q=0.0),
            )
    return grid


def run_stage(stage: int) -> tuple[bool, bool, float]:
    grid = build_stage(stage)
    attach_baseline_emt_models(grid)
    power_flow = PowerFlowDriver(grid=grid, options=build_power_flow_options())
    power_flow.run()
    if not bool(power_flow.results.converged):
        raise RuntimeError(f"Power flow failed at growth stage {stage}")
    power_flow_3ph = PowerFlowDriver3Ph(grid=grid, options=build_power_flow_options())
    power_flow_3ph.run()
    if not bool(power_flow_3ph.results.converged):
        raise RuntimeError(f"Three-phase power flow failed at growth stage {stage}")

    options = build_emt_options()
    options.simulation_time = 1.0e-3
    problem = EmtProblemDae(
        grid=grid,
        options=options,
        pf_results_3ph=power_flow_3ph.results,
        pf_results=power_flow.results,
    )
    solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=options.simulation_time,
        h=options.time_step,
        method=options.integration_method,
    )
    _t, y, _dy, initialized, converged = solver.simulate(boundary_updater=cast(Any, problem))
    return bool(initialized), bool(converged), float(abs(y).max())


def main() -> None:
    for stage, section in enumerate(SECTIONS, start=1):
        initialized, converged, max_abs = run_stage(stage)
        print(
            f"stage={stage} added={section.kind}:{section.bus_from}-{section.bus_to} "
            f"initialized={initialized} converged={converged} max_abs_y={max_abs:.6e}",
            flush=True,
        )
        if not (initialized and converged):
            break


if __name__ == "__main__":
    main()
