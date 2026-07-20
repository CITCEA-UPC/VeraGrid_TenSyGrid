# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Small Signal Analysis for Phasor-based RMS Grid.

This script creates the phasor problem manually and injects it into the SmallSignalStabilityRmsDriver.
"""

import sys
from pathlib import Path
import pickle

project_base = Path(__file__).resolve().parents[2]
script_dir = Path(__file__).resolve().parent
src_path = project_base / "src"
if src_path.exists():
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
else:
    print(f"[ERROR] Engine not found: {src_path}")
    sys.exit(1)

import time, pickle, os
import multiprocessing as mp
from typing import Tuple, Any, List
import numpy as np
try:
    from memory_profiler import profile, LineProfiler, show_results
except ModuleNotFoundError:
    def profile(func):
        return func

    class LineProfiler:
        def add_function(self, func):
            return None

        def __call__(self, func):
            return func

    def show_results(lp):
        return None
# from cryptography.x509 import name
project_base = Path(__file__).resolve().parents[2]
src_path = project_base / "src"
if src_path.exists():
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
else:
    print(f"[ERROR] Engine not found: {src_path}")
    sys.exit(1)

from PolynomialMatrixBuilder import PolynomialMatrixBuilder
from matplotlib import pyplot as plt
import VeraGridEngine.api as vge
from VeraGridEngine.basic_structures import Logger
from VeraGridEngine.IO.veragrid.pack_unpack import parse_veragrid_data
from VeraGridEngine.IO.veragrid.zip_interface import get_frames_from_zip
from VeraGridEngine.Simulations.Rms.numerical.back_euler_fx import BackEulerImplicitIntegration
from phasor_diagnostics import (
    check_pf_line_equations,
    check_equation_consistency,
    debug_exciter_equations,
    check_generator_current_residuals as compare_generator_bus_currents,
    check_shunt_pf_currents,
    print_buses_with_many_injections,
    print_bus_current_balance,
)
from scipy.optimize import linear_sum_assignment
from types import MethodType
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model


grid_folder = project_base / "Grids_and_profiles" / "grids"
GRID = sys.argv[1] if len(sys.argv) > 1 else "IEEE 9 Bus.gridcal"
USE_CACHED_BUILDER = False
BUILDER_CACHE_FILE = f'trunk/tensygrid/precomputed_builds/{GRID}.pkl'

REPORT_TIME = False

FAST_MODE = False
LOGS = False
FULL_GRID_PATH = str(grid_folder / GRID)
FAST_GRID_PATH = FULL_GRID_PATH
PRINT_ALL_BUSES = LOGS
ENABLE_POST_COMPARE_PLOT = LOGS
RUN_DYNAMIC_STEP = True
RUN_CONSISTENCY_CHECK = False
RUN_PF_LINE_EQ_CHECK = False
RUN_EXCITER_DIAGNOSTICS = False
RUN_SMALL_SIGNAL_DRIVER = False
PF_LINE_FOCUS_INDEX = 0
DYNAMIC_SIM_TIME_FAST = 0.05
DYNAMIC_SIM_TIME_FULL = 0.20

EXEC_METRICS = {
    "n_vars": 0,
    "t_builder": 0.0,
    "t_solve": 0.0,
    "is_sparse": False,
    "stops": 0
}


def open_grid_rebuilding_symbolic_data(grid_path):
    """Open a grid, ignoring stale serialized symbolic blocks if they cannot be parsed."""
    try:
        return vge.open_file(grid_path)
    except KeyError as ex:
        logger = Logger()
        data_dictionary, _, has_multiverse_data = get_frames_from_zip(file_name_zip=grid_path, logger=logger)
        if data_dictionary is None or has_multiverse_data:
            raise

        if data_dictionary.get("symbolic_data") is None:
            raise

        print(f"  ! Ignoring stale serialized symbolic data ({ex}); rebuilding RMS models from grid devices")
        data_dictionary["symbolic_data"] = {}
        return parse_veragrid_data(data=data_dictionary, logger=logger)


def define_problem_with_path(grid_path):
    """Creates the grid, runs the power flow, and returns the phasor problem."""
    print("\n[1] Creating phasor grid_path...")
    grid = open_grid_rebuilding_symbolic_data(grid_path)
    ensure_unique_device_names(grid)

    # Build models

    ## BUSES:
    for bus in grid.buses:
        if bus.rms_model.empty():
            busec = vge.initialize_bus_phasor_rms(bus, vf=grid.var_factory)

    ## -GENERATORS
    for igen, gen in enumerate(grid.generators):
        if not gen.rms_model.empty():
            continue
        genqec = vge.get_complete_generator_template_phasor(grid.var_factory, name=f"Gen{igen}").block

        grid.var_factory.add_connections([genqec.in_vars[0]], [gen.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([genqec.in_vars[1]], [gen.bus.rms_model.out_vars[1]])
        genqec = vge.to_implicit(genqec, grid.var_factory)
        set_rms_model(device=gen, model=genqec, var_factory=grid.var_factory)

    ## -LINES:
    for line in grid.lines:
        if not line.rms_model.empty():
            continue
        lineec = vge.get_line_phasor_rms_template(grid.var_factory, name=line.name).block
        grid.var_factory.add_connections([lineec.in_vars[0]], [line.bus_from.rms_model.out_vars[0]])
        grid.var_factory.add_connections([lineec.in_vars[1]], [line.bus_from.rms_model.out_vars[1]])
        grid.var_factory.add_connections([lineec.in_vars[2]], [line.bus_to.rms_model.out_vars[0]])
        grid.var_factory.add_connections([lineec.in_vars[3]], [line.bus_to.rms_model.out_vars[1]])
        lineec = vge.to_implicit(lineec, grid.var_factory)
        set_rms_model(device=line, model=lineec, var_factory=grid.var_factory)
    ## -LOADS:
    for load in grid.loads:
        if not load.rms_model.empty():
            continue
        loadec = vge.get_load_phasor_current_rms_template(grid.var_factory, name=load.name).block
        grid.var_factory.add_connections([loadec.in_vars[0]], [load.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([loadec.in_vars[1]], [load.bus.rms_model.out_vars[1]])
        # loadec = vge.to_implicit(loadec, grid.var_factory)
        set_rms_model(device=load, model=loadec, var_factory=grid.var_factory)
    ## -TRANSFORMERS 2w:
    for trafo in grid.transformers2w:
        if not trafo.rms_model.empty():
            continue
        trafoec = vge.initialize_trafo_rms(trafo, grid.var_factory, use_phasor_template=True).block
        grid.var_factory.add_connections([trafoec.in_vars[0]], [trafo.bus_from.rms_model.out_vars[0]])
        grid.var_factory.add_connections([trafoec.in_vars[1]], [trafo.bus_from.rms_model.out_vars[1]])
        grid.var_factory.add_connections([trafoec.in_vars[2]], [trafo.bus_to.rms_model.out_vars[0]])
        grid.var_factory.add_connections([trafoec.in_vars[3]], [trafo.bus_to.rms_model.out_vars[1]])
        trafoec = vge.to_implicit(trafoec, grid.var_factory)
        set_rms_model(device=trafo, model=trafoec, var_factory=grid.var_factory)
    ## -SHUNTS (load like):
    for shunt in grid.shunts:
        if not shunt.rms_model.empty():
            continue
        shuntec = vge.get_shunt_template(grid.var_factory, name=shunt.name, phasor=True).block
        grid.var_factory.add_connections([shuntec.in_vars[0]], [shunt.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([shuntec.in_vars[1]], [shunt.bus.rms_model.out_vars[1]])
        shuntec = vge.to_implicit(shuntec, grid.var_factory)
        set_rms_model(device=shunt, model=shuntec, var_factory=grid.var_factory)

    # TRANSFORMERS 3W: Not present in existing grids and not supported
    sanitize_runtime_parameter_names(grid)
    print("  ✓ Grid with path loaded")

    # ===================================================================
    # POWER FLOW
    # ===================================================================

    print("\n[2] Running power flow...")
    # pf_options = vge.PowerFlowOptions(tolerance=1e-13)
    pf_options = vge.PowerFlowOptions(tolerance=1e-5)
    pf_results = vge.power_flow(grid, pf_options)


    if pf_results.converged:
        print(f"  ✓ Power flow converged")
        if PRINT_ALL_BUSES:
            for i, bus in enumerate(grid.buses):
                v = pf_results.voltage[i]
                print(f"    {bus.name}: Vm={abs(v):.4f} pu")
        else:
            vm = np.abs(pf_results.voltage)
            print(f"    Buses: {len(grid.buses)}, Vm[min/avg/max]=({vm.min():.4f}/{vm.mean():.4f}/{vm.max():.4f}) pu")
    else:
        print("  ✗ Power flow failed")
        return False, None, None


    if CHRISTOPH_COMPARISON:
        if CHRISTOPH_PF_FILE.exists():
            with open(CHRISTOPH_PF_FILE, "rb") as f:
                pf_results = pickle.load(f)
        else:
            print(f"  ! Christoph PF reference not found at {CHRISTOPH_PF_FILE}; using computed power flow")

    if RUN_PF_LINE_EQ_CHECK:
        check_pf_line_equations(grid=grid, pf_results=pf_results)

    check_shunt_pf_currents(grid=grid, pf_results=pf_results)
    print("\n[2.7] Buses with multiple injection devices...")
    print_buses_with_many_injections(grid=grid)
    print_bus_current_balance(grid=grid, pf_results=pf_results)
    # ===================================================================
    # CREATE PHASOR PROBLEM
    # ===================================================================
    print("\n[3] Creating phasor problem...")

    rms_options = vge.RmsOptions(
        time_step=0.01,
        simulation_time=1.0,
        tolerance=1e-6,
        max_iter=20,
        verbose=True,
    )

    start_time = time.perf_counter()
    phasor_problem = vge.RmsProblemPhasor(grid=grid, options=rms_options, pf_results=pf_results)
    EXEC_METRICS["n_vars"] = phasor_problem.get_all_vars_number()
    end_time = time.perf_counter()
    print(f"\n[3.1.0] ⏰ Func phasor_problem took {end_time - start_time:.2f} seconds")
    print(f"  ✓ Phasor problem created with {phasor_problem.get_states_number()} states")
    if RUN_CONSISTENCY_CHECK:
        check_equation_consistency(phasor_problem)
    if RUN_EXCITER_DIAGNOSTICS:
        debug_exciter_equations(phasor_problem)
        compare_generator_bus_currents(problem=phasor_problem, pf_results=pf_results)
    print("\n[3.0] Exciter diagnostics...")
    # debug_exciter_equations(problem=phasor_problem)
    # ===================================================================
    # DYNAMIC VALIDATION
    # ===================================================================
    if RUN_DYNAMIC_STEP:
        print("\n[3.1.1] Running dynamic simulation...")
        ensure_problem_event_api(phasor_problem)
        dyn_t_end = DYNAMIC_SIM_TIME_FAST if FAST_MODE else DYNAMIC_SIM_TIME_FULL
        solver = BackEulerImplicitIntegration(
            problem=phasor_problem,
            t0=0,
            t_end=dyn_t_end,
            h=rms_options.time_step,
            max_iter=rms_options.max_iter,
            tolerance=rms_options.tolerance,
        )
        t_dyn, y_dyn, well_initialized, converged = solver.simulate()
        print(
            f"  ✓ Dynamic simulation finished: steps={len(t_dyn)}, "
            f"well_initialized={well_initialized}, converged={converged}, "
            f"final|x|={np.linalg.norm(y_dyn[-1, :]):.4e}"
        )
    return phasor_problem, pf_results, grid


def ensure_problem_event_api(problem):
    """Provide no-op event API methods expected by BackEulerImplicitIntegration."""
    if not hasattr(problem, "get_next_forced_event_time"):
        def _get_next_forced_event_time(self, t_local_prev, t_macro_target):
            return None
        problem.get_next_forced_event_time = MethodType(_get_next_forced_event_time, problem)

    if not hasattr(problem, "update"):
        def _update(self, t, x_snapshot, variable_parameters):
            return None
        problem.update = MethodType(_update, problem)


def sanitize_runtime_parameter_names(grid):
    """Avoid name collisions between vars and runtime parameters per device."""
    def _sanitize_block(block, tag):
        block.unify_blocks()
        all_syms = list(block.algebraic_vars) + list(block.state_vars) + list(block.diff_vars)
        all_syms += list(block.event_dict.keys()) + list(block.mode_dict.keys())
        seen = {}
        for k, sym in enumerate(all_syms):
            nm = str(sym.name)
            if nm not in seen:
                seen[nm] = 0
                continue
            seen[nm] += 1
            sym.name = f"{nm}__{tag}_{seen[nm]}_{k}"

    for i, dev in enumerate(grid.get_branches_iter(add_vsc=True, add_hvdc=True, add_switch=True)):
        if not dev.rms_model.empty():
            _sanitize_block(dev.rms_model, f"br{i}")

    for i, dev in enumerate(grid.get_injection_devices_iter()):
        if not dev.rms_model.empty():
            _sanitize_block(dev.rms_model, f"inj{i}")


def ensure_unique_device_names(grid):
    def _rename(devices, prefix):
        seen = {}
        for i, dev in enumerate(devices):
            base = str(dev.name).strip() if getattr(dev, 'name', None) else f"{prefix}_{i}"
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

def define_problem():
    """Creates the grid, runs the power flow, and returns the phasor problem."""
    print("\n[1] Creating phasor grid...")
    Sbase = 100.0
    grid = vge.MultiCircuit(Sbase=Sbase, fbase=50.0)

    bus0 = vge.Bus(name="Bus0", Vnom=10, is_slack=True)
    bus1 = vge.Bus(name="Bus1", Vnom=10)
    grid.add_bus(bus0)
    grid.add_bus(bus1)

    for bus in grid.buses:
        vge.initialize_bus_phasor_rms(bus, vf=grid.var_factory)

    line = vge.Line(name="Line", bus_from=bus0, bus_to=bus1,
                    r=0.029585798816568046, x=0.07100591715976332, b=0.03, rate=900.0)
    grid.add_line(line)

    load = vge.Load(P=9.999999, Q=0.999999)
    grid.add_load(bus=bus1, api_obj=load)

    gen = vge.Generator(name="Gen0", P=10, vset=1.0, Snom=900)
    grid.add_generator(bus=bus0, api_obj=gen)

    # Build RMS models
    genqec = vge.get_complete_generator_template_phasor(grid.var_factory, name="Gen0").block
    genqec = vge.to_implicit(genqec, grid.var_factory)

    line_mdl = vge.get_line_phasor_rms_template(grid.var_factory).block
    grid.var_factory.add_connections([line_mdl.in_vars[0]], [bus0.rms_model.out_vars[0]])
    grid.var_factory.add_connections([line_mdl.in_vars[1]], [bus0.rms_model.out_vars[1]])
    grid.var_factory.add_connections([line_mdl.in_vars[2]], [bus1.rms_model.out_vars[0]])
    grid.var_factory.add_connections([line_mdl.in_vars[3]], [bus1.rms_model.out_vars[1]])

    # Attach line model to the line device
    set_rms_model(device=line, model=line_mdl, var_factory=grid.var_factory)

    grid.var_factory.add_connections([genqec.in_vars[0]], [bus0.rms_model.out_vars[0]])
    grid.var_factory.add_connections([genqec.in_vars[1]], [bus0.rms_model.out_vars[1]])

    set_rms_model(device=gen, model=genqec, var_factory=grid.var_factory)

    load_mdl = vge.get_load_phasor_current_rms_template(grid.var_factory).block
    grid.var_factory.add_connections([load_mdl.in_vars[0]], [bus1.rms_model.out_vars[0]])
    grid.var_factory.add_connections([load_mdl.in_vars[1]], [bus1.rms_model.out_vars[1]])
    # Convert power to current: I = S* / V* (approximation for initialization)
    # For V ≈ 1.0, Ir ≈ P, Ii ≈ Q
    load_mdl.set_parameter_in_model(var_name="Ir0", new_value=-0.1)
    load_mdl.set_parameter_in_model(var_name="Ii0", new_value=-0.01)
    set_rms_model(device=load, model=load_mdl, var_factory=grid.var_factory)

    print("  ✓ Grid created with multilinear generator")

    # ===================================================================
    # POWER FLOW
    # ===================================================================
    print("\n[2] Running power flow...")
    pf_options = vge.PowerFlowOptions()
    pf_results = vge.power_flow(grid, pf_options)

    if pf_results.converged:
        print(f"  ✓ Power flow converged")
        for i, bus in enumerate(grid.buses):
            v = pf_results.voltage[i]
            print(f"    {bus.name}: Vm={abs(v):.4f} pu")
    else:
        print("  ✗ Power flow failed")
        return False

    # ===================================================================
    # CREATE PHASOR PROBLEM
    # ===================================================================
    print("\n[3] Creating phasor problem...")

    rms_options = vge.RmsOptions(
        time_step=0.01,
        simulation_time=1.0,
        tolerance=1e-6,
        max_iter=20,
    )

    phasor_problem = vge.RmsProblemPhasor(grid=grid, options=rms_options, pf_results=pf_results)
    print(f"  ✓ Phasor problem created with {phasor_problem.get_states_number()} states")

    return phasor_problem


def solve_problem(problem: Any = None) -> Any:
    builder = None

    if USE_CACHED_BUILDER and os.path.exists(BUILDER_CACHE_FILE):
        print(f"\n[📦] Loading builder object from '{BUILDER_CACHE_FILE}'...")
        with open(BUILDER_CACHE_FILE, 'rb') as f:
            builder = pickle.load(f)

        # Reassign the problem we just created/loaded in the main script
        print("  ✓ Builder restored successfully (skipping linearization)")

    else:

        print("\n[4] Rebuilding equations and extracting exact physics...")
        builder = PolynomialMatrixBuilder(problem=problem, verbose=True)
        v_dict: dict = PolynomialMatrixBuilder.op_extraction(problem=problem)

        print("\n[*] Linearizing model with real parameters...")
        builder.linearize(v_dict=v_dict)
        print(f"Vi_2 at equilibrium: {v_dict.get('Vi_2', 'Missing!')}")
        print(f"In_aux at equilibrium: {v_dict.get('In_aux', 'Missing!')}")
        # Also check if there's a base 'In'
        print(f"In at equilibrium: {v_dict.get('In', 'Missing!')}")
        # 2. If it does not exist, build it from scratch and save it

        if USE_CACHED_BUILDER:
            print(f"\n[📦] Saving full builder object to '{BUILDER_CACHE_FILE}'...")
            with open(BUILDER_CACHE_FILE, 'wb') as f:
                pickle.dump(builder, f)
            print("  ✓ Builder saved successfully")

    # Continue with the normal stability analysis
    # Use automatic routing: sparse for large systems, dense fallback otherwise.
    res = builder.compute_stability()

    eigenvalues = res[0]
    # print(eigenvalues)
    stable = res[4]
    margin = res[5] if res[5] is not None else 0.0
    participation_matrix = res[3] if res[3] is not None else None

    print("\n" + "=" * 80)
    print("SMALL SIGNAL ANALYSIS COMPLETE - POLYNOMIAL BUILDER")
    print("=" * 80)
    print(f"  Found {len(eigenvalues)} finite eigenvalues")
    print(f"  System is stable: {stable}")
    if -1e-6 < margin < 1e-6:
        print("Expected bifurcation. Further analysis is needed.")
    print(f"  Stability margin: {margin:.4f}")
    print("=" * 80)

    # Extract the mapping using the class's corrected function
    pf_map: dict = builder.get_participation_mapping(eigenvalues, participation_matrix)

    print("\n" + "!" * 80)
    print("!!! UNSTABLE MODE ANALYSIS (Re > 0)")
    print("!" * 80)

    unstable_found = False
    for ev, participations in pf_map.items():
        # Strictly filter poles in the right half-plane
        if ev.real > 1e-6:
            unstable_found = True
            print(f"\n🚨 Critical mode: {ev.real:.4f} + {ev.imag:.4f}j")
            if not participations:
                print("  -> [!] No dominant physical variables (check structural patch)")
            for var_name, pf in participations:
                print(f"  -> {var_name}: {pf * 100:.1f}%")

    if not unstable_found:
        print("\n✅ The system is stable! There is no real part greater than 0.")
    return res, pf_map


def run_small_signal_driver(problem: vge.RmsProblemPhasor) -> None:
    """Run SmallSignalStabilityRmsDriver by injecting the phasor problem."""
    print("\n" + "=" * 80)
    print("SMALL SIGNAL ANALYSIS - DRIVER (INJECTED PHASOR PROBLEM)")
    print("=" * 80)

    pf_results = getattr(problem, "power_flow_results", None)
    if pf_results is None:
        print("[WARN] power_flow_results not available in problem, skipping driver run")
        return

    ss_options = vge.RmsSmallSignalStabilityOptions(ss_assessment_time=0)
    driver = vge.SmallSignalStabilityRmsDriver(
        grid=vge.MultiCircuit(Sbase=problem.grid.Sbase),
        rms_options=problem.options,
        sss_options=ss_options,
        pf_results=pf_results,
    )

    driver.problem = problem
    driver.k = problem.get_states_number()
    driver.run()

    eigenvalues = driver.results.eigenvalues
    participation_factors = driver.results.participation_factors

    finite_mask = np.isfinite(eigenvalues) & (np.abs(eigenvalues) < 1e6)
    filtered_eigs = eigenvalues[finite_mask]
    print(f"Finite eigenvalues: {len(filtered_eigs)}")

    print("All finite eigenvalues (driver):")
    for i, ev in enumerate(filtered_eigs, start=1):
        print(f"  mode={i:<4} ev={np.real(ev):.6f}+{np.imag(ev):.6f}j")

    unstable = [(i, ev) for i, ev in enumerate(filtered_eigs) if np.real(ev) > 1e-6]
    if not unstable:
        print("No unstable eigenvalues found by driver")
        return

    state_vars = problem.state_and_algebraic_vars
    print("Unstable modes (driver):")
    for i, ev in unstable:
        dominant_state = "N/A"
        pf_val = 0.0
        if participation_factors is not None:
            original_idx = np.flatnonzero(finite_mask)[i]
            pf = np.abs(participation_factors[:, original_idx])
            max_pf_idx = int(np.argmax(pf))
            pf_val = float(pf[max_pf_idx])
            if max_pf_idx < len(state_vars):
                dominant_state = str(state_vars[max_pf_idx])
            else:
                dominant_state = f"State {max_pf_idx}"

        print(f"  mode={i + 1} ev={np.real(ev):.6f}+{np.imag(ev):.6f}j dominant={dominant_state} PF={pf_val:.3f}")

def _hash_operating_point(x: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(x.tobytes()).hexdigest()


def _damping(re: float, im: float) -> float:
    mag = np.sqrt(re * re + im * im)
    return -re / mag if mag > 1e-12 else 0.0


def _freq_hz(im: float) -> float:
    return abs(im) / (2 * np.pi)


def main():
    # --- START CHRONO ---
    start_time = time.perf_counter()

    grid_filename = GRID
    grid_path = FAST_GRID_PATH if FAST_MODE else FULL_GRID_PATH
    print(f"[FAST_MODE={FAST_MODE}] Using grid: {grid_path}")

    problem = None
    pf_results = None
    grid = None
    calculated_evs = []

    # 1. Check whether the builder cache exists BEFORE creating the problem
    if USE_CACHED_BUILDER and os.path.exists(BUILDER_CACHE_FILE):
        print(f"\n[⏩] File '{BUILDER_CACHE_FILE}' found.")
        print("[⏩] Skipping Phasor Problem creation...")
        # Keep problem as None. solve_problem will load the .pkl internally.
        res, pf_map = solve_problem(problem=problem)
        calculated_evs = res[0]
    else:
        # 2. If there is no cache, run the normal heavy simulation
        problem, pf_results, grid = define_problem_with_path(grid_path)
        if problem:
            if RUN_SMALL_SIGNAL_DRIVER:
                run_small_signal_driver(problem)
            # Pass the newly created problem so it can be linearized and cached
            res, pf_map = solve_problem(problem=problem)
            calculated_evs = res[0]

    # --- END CHRONO ---
    end_time = time.perf_counter()
    total_duration = end_time - start_time

    print(f"\n========================================")
    print(f"EXECUTION TIME: {total_duration:.4f} seconds")
    print(f"========================================")

    # --- GENERATE .tensygrid FILE FOR GUI ---
    if problem is not None and pf_results is not None and grid is not None:
        print("\n[5] Generating .tensygrid file for GUI...")
        
        # Extract eigenvalues
        finite_evs = calculated_evs[np.isfinite(calculated_evs) & (np.abs(calculated_evs) < 1e6)] if len(calculated_evs) > 0 else np.array([])
        stable = bool(res[4]) if len(res) > 4 else False
        margin = float(res[5]) if len(res) > 5 and res[5] is not None else 0.0
        
        eigenvalues_list = [
            {'re': float(np.real(ev)), 'im': float(np.imag(ev))}
            for ev in finite_evs
        ]
        
        # Extract critical modes with participation factors
        critical_modes = []
        for i, ev in enumerate(finite_evs):
            re, im = float(np.real(ev)), float(np.imag(ev))
            if abs(im) > 0.1:
                damping = float(_damping(re, im))
                
                participation = {}
                if pf_map and ev in pf_map:
                    for var_name, pf_val in pf_map[ev]:
                        participation[str(var_name)] = float(pf_val)
                
                top_participation = dict(sorted(participation.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
                
                critical_modes.append({
                    'mode': i,
                    're': re,
                    'im': im,
                    'damping': damping,
                    'freq_hz': float(_freq_hz(im)),
                    'critical': bool(abs(damping) < 0.05),
                    'participation_factors': top_participation,
                })
        critical_modes.sort(key=lambda m: m['damping'])
        
        # Extract bus voltages
        bus_voltages = []
        voltages = pf_results.voltage
        bus_names = pf_results.bus_names
        bus_types = pf_results.bus_types
        
        for i, bus in enumerate(grid.buses):
            v_complex = voltages[i] if i < len(voltages) else 1.0 + 0j
            vm = float(abs(v_complex))
            theta = float(np.angle(v_complex))
            bus_type = str(bus_types[i]) if i < len(bus_types) else 'Unknown'
            bus_name = str(bus_names[i]) if i < len(bus_names) else str(bus.name)
            
            bus_voltages.append({
                'id': bus.idtag if hasattr(bus, 'idtag') else str(i),
                'name': bus_name,
                'type': bus_type,
                'vm_pu': round(vm, 4),
                'theta_deg': round(float(np.degrees(theta)), 2),
            })
        
        # Extract grid info
        grid_info = {
            'n_buses': len(list(grid.buses)),
            'n_generators': len(list(grid.generators)),
            'n_lines': len(list(grid.lines)),
            'n_loads': len(list(grid.loads)),
            'n_transformers': len(list(grid.transformers2w)),
        }
        
        # Get operating point for hash
        try:
            x = problem.get_x0()
            op_hash = _hash_operating_point(x)
            n_state_vars = len(x)
        except:
            op_hash = "unknown"
            n_state_vars = problem.get_states_number()
        
        # Build results dictionary
        results = {
            'grid_name': grid_filename,
            'stable': stable,
            'margin': margin,
            'matches': 0,  # Not applicable for tensygrid method
            'op_hash': op_hash,
            'eigenvalues': eigenvalues_list,
            'critical_modes': critical_modes[:10],
            'bus_voltages': bus_voltages,
            'grid_info': grid_info,
            'n_eigenvalues': len(finite_evs),
            'n_state_vars': n_state_vars,
            'elapsed_seconds': round(total_duration, 2),
        }
        
        # Save to .tensygrid file
        output_dir = Path('precomputed_builds')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{Path(grid_filename).stem}_{results['op_hash'][:6]}.tensygrid"
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"  ✓ Results saved to {output_path}")

    if not ENABLE_POST_COMPARE_PLOT:
        print("[FAST_MODE] Skipping eigenvalue reference matching and plotting")
        return

    reference_evs = [complex(x) for x in
                     ["(-20.000000001+0j)", "(-10.000000001+0j)", "(-50.00000013636582+0j)", "(-49.9999998638391+0j)",
                      "(-12.168789431034414+0j)", "(-10.142655446318065+0j)", "(-6.854035294550814+0j)",
                      "(-5.029702766536969+0j)", "(-3.6156680497265823+0j)", "(0.7258649757244431+0j)",
                      "(-1.8851882786542178+0j)", "(-1.719135993985083+0.32085026247445086j)",
                      "(-1.7191359939850832-0.3208502624744508j)", "(-0.49201422052160765+0.8707873679911879j)",
                      "(-0.49201422052160765-0.870787367991188j)", "(-0.48381848641530006+0.465928959924564j)",
                      "(-0.48381848641530006-0.46592895992456396j)", "(-0.10000000099999705+0j)",
                      "(-0.11161483998140341+0j)", "(-0.9999999991690246+0j)", "(-1.0000000028309748+0j)",
                      "(-0.39635072300328184+0j)", "(-10.00000000099956+0j)", "(-1.0000000000000023e-09+0j)",
                      "(-9.999999999999996e-10+0j)", "(-9.999999999999992e-10+0j)"]]

    # --- MATCHING LOGIC (Hungarian Algorithm) ---
    finite_evs = calculated_evs[np.isfinite(calculated_evs) & (np.abs(calculated_evs) < 1e6)] if len(
        calculated_evs) > 0 else []

    tol = 0.06  # Tolerance to consider eigenvalues as matching
    matched_calc = []
    unmatched_calc = []
    unmatched_ref = list(reference_evs)

    if len(finite_evs) > 0 and len(reference_evs) > 0:
        # 1. Build distance (cost) matrix
        cost_matrix = np.zeros((len(finite_evs), len(reference_evs)))
        for i, c_ev in enumerate(finite_evs):
            for j, r_ev in enumerate(reference_evs):
                cost_matrix[i, j] = np.abs(c_ev - r_ev)

        # 2. Optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 3. Filter by tolerance
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < tol:
                matched_calc.append(finite_evs[i])
                unmatched_ref[j] = None  # Mark as matched
            else:
                unmatched_calc.append(finite_evs[i])

        # 4. Clean unmatched lists
        unmatched_ref = [x for x in unmatched_ref if x is not None]

        # Add computed eigenvalues that were not assigned at all
        for i in range(len(finite_evs)):
            if i not in row_ind:
                unmatched_calc.append(finite_evs[i])
    else:
        unmatched_calc = list(finite_evs)

    n_matches = len(matched_calc)
    n_unmatched_calc = len(unmatched_calc)
    n_unmatched_ref = len(unmatched_ref)

    # --- PLOT: Create 1x2 figure ---
    # Use wider spacing and reserve more room for the scatter plot (2:1 ratio)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2.5, 1]})

    # --- SUBPLOT 1: SCATTER (Original plot adapted to ax1) ---
    ax1.scatter(np.real(reference_evs), np.imag(reference_evs),
                color='gray', marker='x', alpha=0.5, label="Pablo's computation")

    if len(finite_evs) > 0:
        ax1.scatter(np.real(finite_evs), np.imag(finite_evs),
                    c=(np.real(finite_evs) > 1e-6), cmap='coolwarm',
                    marker='+', s=100, linewidths=2, label="TenSyGrid's computation")

    ax1.axvline(0, color='k', linestyle='--')
    ax1.axhline(0, color='k', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left')

    # Note: Added 'r' prefix to avoid SyntaxWarning for '\R' and '\I'
    ax1.set_xlabel(r'$\Re({\lambda})$', fontsize=12)
    ax1.set_ylabel(r'$\Im({\lambda})$', fontsize=12)
    ax1.set_title("Eigenvalue comparison")

    # --- SUBPLOT 2: BARS AND RESULTS ---
    labels = ['Matching', 'Exceeding\n(TenSyGrid)', 'Missing\n(Pablo)']
    counts = [n_matches, n_unmatched_calc, n_unmatched_ref]
    colors = ["#439C46", '#F44336', '#FFC107']  # Green, Red, Yellow

    bars = ax2.bar(labels, counts, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_title(f"Coincidence\n(Tolerance: {tol})")
    ax2.set_ylabel("Amount")

    # Add the number above each bar
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval), ha='center', va='bottom', fontweight='bold')

    # Hide top and right spines for a cleaner bar chart
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("scatter_and_bars.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- CONSOLE OUTPUT (So you can see the exact values) ---
    print("=== EIGENVALUE REPORT ===")
    print(f"Total computed (TenSyGrid): {len(finite_evs)}")
    print(f"Total reference (Pablo): {len(reference_evs)}")
    print("-" * 30)
    print(f"✅ Matches: {n_matches}")

    print(f"\n❌ EXTRA POLES (Computed by TenSyGrid, not in Pablo's reference): {n_unmatched_calc}")
    for ev in unmatched_calc:
        print(f"   {ev.real:.4f} + {ev.imag:.4f}j")

    print(f"\n⚠️ MISSING POLES (In Pablo's reference, not found by TenSyGrid): {n_unmatched_ref}")
    for ev in unmatched_ref:
        print(f"   {ev.real:.4f} + {ev.imag:.4f}j")


if __name__ == "__main__":
    try:
        mp.freeze_support()
        if REPORT_TIME:
            lp = LineProfiler()
            lp.add_function(main)
            lp.add_function(solve_problem)
            lp.add_function(define_problem_with_path)
            lp(main)()
        else:
            main()
    except SystemExit:
        print("SystemExit")
        # print("Error during execution")
    finally:
        if REPORT_TIME:
            print("\n" + "=" * 80)
            print("      DETAILED MEMORY PROFILE REPORT (VeraGrid Analysis)")
            print("=" * 80)
            show_results(lp)
            print(
                f"N_variables: {EXEC_METRICS['n_vars']}, "
                f"exec_time_builder: {EXEC_METRICS['t_builder']:.4f} sec, "
                f"exec_time_solve: {EXEC_METRICS['t_solve']:.4f} sec, "
                f"sparse: {EXEC_METRICS['is_sparse']}"
            )
