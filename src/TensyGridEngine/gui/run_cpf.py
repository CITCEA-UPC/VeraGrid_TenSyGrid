#!/usr/bin/env python3
"""
Run Continuation Power Flow (VeraGridEngine) and write P-V curve data to a result file.

Usage:
    python run_cpf.py <grid_filename> <output_json_path>

Output: JSON written to <output_json_path>
"""

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = _PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

GRIDS_DIR = _PROJECT_ROOT / "Grids_and_profiles" / "grids"

import VeraGridEngine.api as gce


def run_cpf(grid_filename: str) -> dict:
    t0 = time.perf_counter()
    _log("Loading grid %s ...", grid_filename)
    grid = gce.FileOpen(str(GRIDS_DIR / grid_filename)).open()

    _log("Running power flow ...")
    pf_options = gce.PowerFlowOptions()
    pf_drv = gce.PowerFlowDriver(grid, pf_options)
    pf_drv.run()
    pf_results = pf_drv.results

    if pf_results is None:
        return {"status": "failed", "error": "Power flow did not converge"}

    bus_names = [b.name for b in grid.buses]
    bus_index = {b: i for i, b in enumerate(grid.buses)}

    Sload_per_bus = np.zeros(len(grid.buses), dtype=complex)
    for ld in grid.loads:
        i = bus_index.get(ld.bus)
        if i is not None:
            Sload_per_bus[i] += complex(ld.P, ld.Q)

    alpha = 3.0
    Sbase = pf_results.Sbus / grid.Sbase
    Sgen_solved = pf_results.Sbus + Sload_per_bus
    Starget = (Sgen_solved - Sload_per_bus * alpha) / grid.Sbase

    vc_options = gce.ContinuationPowerFlowOptions(
        step=0.1,
        approximation_order=gce.CpfParametrization.Natural,
        adapt_step=True,
        step_min=0.02,
        step_max=0.5,
        error_tol=1e-3,
        tol=1e-6,
        max_it=30,
        stop_at=gce.CpfStopAt.Nose,
        verbose=0,
    )

    vc_inputs = gce.ContinuationPowerFlowInput(
        Sbase=Sbase, Vbase=pf_results.voltage, Starget=Starget,
    )

    _log("Running Continuation Power Flow ...")
    drv = gce.ContinuationPowerFlowDriver(grid, vc_options, vc_inputs, pf_options)
    drv.run()
    results = drv.results

    if results is None or results.voltages is None:
        return {"status": "failed", "error": "CPF did not converge"}

    lambdas = results.lambdas
    if lambdas is None or len(lambdas) == 0:
        return {"status": "failed", "error": "CPF produced no continuation steps"}

    elapsed = time.perf_counter() - t0

    Vm = np.abs(results.voltages)
    P = results.Sbus.real * grid.Sbase
    nose_idx = int(np.argmax(lambdas))

    bus_data = []
    for i, name in enumerate(bus_names):
        bus_data.append({
            "name": name,
            "V_path": [round(float(v), 6) for v in Vm[:, i]],
            "P_path": [round(float(p), 4) for p in P[:, i]],
        })

    _log("CPF completed: %d steps, nose at step %d, elapsed %.1fs", len(lambdas), nose_idx, elapsed)

    return {
        "status": "completed",
        "grid_name": grid_filename,
        "lambda_path": [float(v) for v in lambdas],
        "bus_data": bus_data,
        "nose_idx": nose_idx,
        "n_steps": len(lambdas),
        "n_buses": len(bus_names),
        "elapsed_seconds": round(elapsed, 2),
    }


def _log(fmt: str, *args):
    """Log to stderr so it does not interfere with the stdout result file."""
    sys.stderr.write((fmt % args) + "\n")
    sys.stderr.flush()


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: run_cpf.py <grid_filename> <output_json_path>\n")
        sys.exit(1)

    grid_filename = sys.argv[1]
    output_path = Path(sys.argv[2])

    try:
        result = run_cpf(grid_filename)
    except Exception as e:
        tb = traceback.format_exc()
        result = {"status": "failed", "error": str(e), "traceback": tb}

    output_path.write_text(json.dumps(result, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
