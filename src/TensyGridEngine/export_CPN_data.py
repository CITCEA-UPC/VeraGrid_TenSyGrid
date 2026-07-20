# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import os
import sys
import pickle
import hashlib
from pathlib import Path

import scipy
from matplotlib import pyplot as plt
import numpy as np

from VeraGridEngine import PowerFlowResults, RmsOptions, RmsProblemMultilinear

project_base = Path(__file__).resolve().parents[2]
src_path = project_base / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import VeraGridEngine.api as vge
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model


def ensure_unique_device_names(grid) -> None:
    def _rename(devices, prefix: str) -> None:
        seen: dict[str, int] = {}
        for i, dev in enumerate(devices):
            base = str(dev.name).strip() if getattr(dev, "name", None) else f"{prefix}_{i}"
            if base not in seen:
                seen[base] = 0
                dev.name = base
            else:
                seen[base] += 1
                dev.name = f"{base}_{seen[base]}"

    _rename(list(grid.generators), "gen")
    _rename(list(grid.lines), "line")
    _rename(list(grid.transformers2w), "trafo")
    _rename(list(grid.loads), "load")
    _rename(list(grid.shunts), "shunt")


def build_problems(grid_filename: str, grid=None) -> tuple[RmsProblemMultilinear, PowerFlowResults, RmsOptions]:
    """Build RMS problems for small-signal analysis.

    Parameters
    ----------
    grid_filename:
        Name of the grid file (used to locate the file when *grid* is None).
    grid:
        Optional pre-loaded (and pre-modified) grid object.  When provided the
        file is **not** re-read from disk, so any element already deactivated on
        the object will be excluded from the RMS model construction.
    """
    if grid is None:
        grid_path = project_base / "Grids_and_profiles" / "grids" / grid_filename
        grid = vge.open_file(str(grid_path))
    ensure_unique_device_names(grid)

    for bus in grid.buses:
        if bus.rms_model.empty():
            vge.initialize_bus_phasor_rms(bus, vf=grid.var_factory)

    for igen, gen in enumerate(grid.generators):
        if not gen.active:
            continue
        if not gen.rms_model.empty():
            continue
        gen_mdl = vge.get_complete_generator_template_phasor(grid.var_factory, name=f"Gen{igen}").block
        grid.var_factory.add_connections([gen_mdl.in_vars[0]], [gen.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([gen_mdl.in_vars[1]], [gen.bus.rms_model.out_vars[1]])
        gen_mdl = vge.to_implicit(gen_mdl, grid.var_factory)
        set_rms_model(device=gen, model=gen_mdl, var_factory=grid.var_factory)

    for line in grid.lines:
        if not line.active:
            continue
        if not line.rms_model.empty():
            continue
        line_mdl = vge.get_line_phasor_rms_template(grid.var_factory, name=line.name).block
        grid.var_factory.add_connections([line_mdl.in_vars[0]], [line.bus_from.rms_model.out_vars[0]])
        grid.var_factory.add_connections([line_mdl.in_vars[1]], [line.bus_from.rms_model.out_vars[1]])
        grid.var_factory.add_connections([line_mdl.in_vars[2]], [line.bus_to.rms_model.out_vars[0]])
        grid.var_factory.add_connections([line_mdl.in_vars[3]], [line.bus_to.rms_model.out_vars[1]])
        line_mdl = vge.to_implicit(line_mdl, grid.var_factory)
        set_rms_model(device=line, model=line_mdl, var_factory=grid.var_factory)

    for load in grid.loads:
        if not load.active:
            continue
        if not load.rms_model.empty():
            continue
        load_mdl = vge.get_load_phasor_current_rms_template(grid.var_factory, name=load.name).block
        grid.var_factory.add_connections([load_mdl.in_vars[0]], [load.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([load_mdl.in_vars[1]], [load.bus.rms_model.out_vars[1]])
        set_rms_model(device=load, model=load_mdl, var_factory=grid.var_factory)

    for trafo in grid.transformers2w:
        if not trafo.active:
            continue
        if not trafo.rms_model.empty():
            continue
        trafo_mdl = vge.initialize_trafo_rms(trafo, grid.var_factory, use_phasor_template=True).block
        grid.var_factory.add_connections([trafo_mdl.in_vars[0]], [trafo.bus_from.rms_model.out_vars[0]])
        grid.var_factory.add_connections([trafo_mdl.in_vars[1]], [trafo.bus_from.rms_model.out_vars[1]])
        grid.var_factory.add_connections([trafo_mdl.in_vars[2]], [trafo.bus_to.rms_model.out_vars[0]])
        grid.var_factory.add_connections([trafo_mdl.in_vars[3]], [trafo.bus_to.rms_model.out_vars[1]])
        trafo_mdl = vge.to_implicit(trafo_mdl, grid.var_factory)
        set_rms_model(device=trafo, model=trafo_mdl, var_factory=grid.var_factory)

    for shunt in grid.shunts:
        if not shunt.active:
            continue
        if not shunt.rms_model.empty():
            continue
        shunt_mdl = vge.get_shunt_template(grid.var_factory, name=shunt.name, phasor=True).block
        grid.var_factory.add_connections([shunt_mdl.in_vars[0]], [shunt.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([shunt_mdl.in_vars[1]], [shunt.bus.rms_model.out_vars[1]])
        shunt_mdl = vge.to_implicit(shunt_mdl, grid.var_factory)
        set_rms_model(device=shunt, model=shunt_mdl, var_factory=grid.var_factory)

    pf_results = vge.power_flow(grid, vge.PowerFlowOptions(tolerance=1e-5))
    if not pf_results.converged:
        raise RuntimeError("Power flow did not converge")

    rms_options_ml = vge.RmsOptions(
        time_step=0.01,
        simulation_time=1.0,
        tolerance=1e-6,
        max_iter=20,
        problem_type=vge.RmsProblemTypes.Multilinear,
    )

    problem_ml = vge.RmsProblemMultilinear(grid=grid, options=rms_options_ml, pf_results=pf_results)
    return problem_ml, pf_results, rms_options_ml


def run_small_signal_from_driver(problem, pf_results: vge.PowerFlowResults, rms_options: vge.RmsOptions):
    ss_options = vge.RmsSmallSignalStabilityOptions(ss_assessment_time=0, verbose=0)
    driver = vge.SmallSignalStabilityRmsDriver(
        grid=vge.MultiCircuit(Sbase=problem.grid.Sbase),
        rms_options=rms_options,
        sss_options=ss_options,
        pf_results=pf_results,
    )
    driver.problem = problem
    driver.k = problem.get_states_number()
    driver.run()
    state_var_names = [str(v.name) if hasattr(v, 'name') else f"state_{i}"
                       for i, v in enumerate(problem.state_and_algebraic_vars)]
    return driver.results.eigenvalues, driver.results.participation_factors, state_var_names


def _hash_operating_point(x: np.ndarray) -> str:
    return hashlib.sha256(x.tobytes()).hexdigest()


def main() -> None:
    grid_filename = sys.argv[1] if len(sys.argv) > 1 else "IEEE 9 Bus.gridcal"
    problem_ml, pf_results, rms_options_ml = build_problems(grid_filename=grid_filename)
    x = problem_ml.get_x0()

    problem_ml.build_multilinear_matrices()

    eqs_array = [str(eq) for eq in problem_ml.get_algebraic_eqs]
    vars_array = [str(var) for var in problem_ml.algebraic_vars]

    eqs_array = np.array(eqs_array, dtype=object)
    vars_array = np.array(vars_array, dtype=object)
    scipy.io.savemat(f'CPN1_computations/CPN1_{problem_ml.grid}.mat', {
        'S': problem_ml.S,
        'Phi': problem_ml.Phi,
        'eqs': eqs_array,
        'var': vars_array
    })

    print("CPN1 file saved!")


if __name__ == "__main__":
    #sys.stdout = open(os.devnull, 'w')
    main()
