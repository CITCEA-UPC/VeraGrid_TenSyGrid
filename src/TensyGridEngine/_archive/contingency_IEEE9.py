# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Small Signal Analysis for Phasor-based RMS Grid.

This script creates the phasor problem manually and injects it into the SmallSignalStabilityRmsDriver.
"""

import sys
import math
from pathlib import Path
import scienceplots

project_base = Path(__file__).resolve().parents[2]
src_path = project_base / "src"
if src_path.exists():
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
else:
    print(f"[ERROR] Engine not found: {src_path}")
    sys.exit(1)

import time, pickle, os
from typing import Tuple, Any, List
import numpy as np
from memory_profiler import profile, LineProfiler, show_results
# from cryptography.x509 import name
from PolynomialMatrixBuilder import PolynomialMatrixBuilder
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import VeraGridEngine.api as vge
from VeraGridEngine.Templates.Rms.genqec_phasor_rms_template import get_genqec_phasor
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

plt.style.use(['science', 'ieee', 'no-latex'])
# GRID = "Texas 2000 + VSC"
# GRID = "IEEE 14.xlsx"
# GRID = "case_ACTIVSg2000.matpower"
"""GRIDS: 
    "IEEE 14.xlsx"
    "case_ACTIVSg2000.matpower"
    "Texas 2000 + VSC"
    "IEEE 118.xlsx"
"""
grid_folder = project_base / "Grids_and_profiles" / "grids"
GRID = "IEEE 9 Bus.gridcal"
USE_CACHED_BUILDER = False
BUILDER_CACHE_FILE = f'../trunk/tensygrid/precomputed_builds/{GRID}.pkl'

REPORT_TIME = False

FAST_MODE = False
LOGS = False
FULL_GRID_PATH = str(grid_folder / GRID)
FAST_GRID_PATH = FULL_GRID_PATH
PRINT_ALL_BUSES = LOGS
ENABLE_POST_COMPARE_PLOT = LOGS
RUN_DYNAMIC_STEP = False
RUN_CONSISTENCY_CHECK = False
RUN_PF_LINE_EQ_CHECK = False
RUN_EXCITER_DIAGNOSTICS = False
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


def debug_input(input_bool):
    """
    Function to debug
    """
    EXEC_METRICS["stops"] = EXEC_METRICS["stops"] + 1
    if input_bool:
        input(f"Hello, we stopped {EXEC_METRICS['stops']} times")
    else:
        print(f"Hello, we stopped {EXEC_METRICS['stops']} times")


def define_problems_with_path(grid_path):
    """Creates the grid, runs the power flow, and returns the phasor problem."""
    print("\n[1] Creating phasor grid_path...")
    grid = vge.open_file(grid_path)

    # Build RMS models
    ## BUSES:
    for bus in grid.buses:
        busec = vge.initialize_bus_phasor_rms(bus, vf=grid.var_factory)

    ## -GENERATORS
    for igen, gen in enumerate(grid.generators):
        genqec = vge.get_complete_generator_template_phasor(grid.var_factory, name=f"Gen{igen}").block
        grid.var_factory.add_connections([genqec.in_vars[0]], [gen.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([genqec.in_vars[1]], [gen.bus.rms_model.out_vars[1]])
        vge.set_rms_model(device=gen, model=genqec, var_factory=grid.var_factory)

    ## -LINES:
    for line in grid.lines:
        lineec = vge.get_line_phasor_rms_template(grid.var_factory, name=line.name).block
        grid.var_factory.add_connections([lineec.in_vars[0]], [line.bus_from.rms_model.out_vars[0]])
        grid.var_factory.add_connections([lineec.in_vars[1]], [line.bus_from.rms_model.out_vars[1]])
        grid.var_factory.add_connections([lineec.in_vars[2]], [line.bus_to.rms_model.out_vars[0]])
        grid.var_factory.add_connections([lineec.in_vars[3]], [line.bus_to.rms_model.out_vars[1]])
        lineec = vge.to_implicit(lineec, grid.var_factory)
        vge.set_rms_model(device=line, model=lineec, var_factory=grid.var_factory)

    ## -LOADS:
    for load in grid.loads:
        loadec = vge.get_load_phasor_current_rms_template(grid.var_factory, name=load.name).block
        grid.var_factory.add_connections([loadec.in_vars[0]], [load.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([loadec.in_vars[1]], [load.bus.rms_model.out_vars[1]])
        vge.set_rms_model(device=load, model=loadec, var_factory=grid.var_factory)

    ## -TRANSFORMERS 2w:
    for trafo in grid.transformers2w:
        trafoec = vge.initialize_trafo_rms(trafo, grid.var_factory, use_phasor_template=True).block
        grid.var_factory.add_connections([trafoec.in_vars[0]], [trafo.bus_from.rms_model.out_vars[0]])
        grid.var_factory.add_connections([trafoec.in_vars[1]], [trafo.bus_from.rms_model.out_vars[1]])
        grid.var_factory.add_connections([trafoec.in_vars[2]], [trafo.bus_to.rms_model.out_vars[0]])
        grid.var_factory.add_connections([trafoec.in_vars[3]], [trafo.bus_to.rms_model.out_vars[1]])
        trafoec = vge.to_implicit(trafoec, grid.var_factory)
        vge.set_rms_model(device=trafo, model=trafoec, var_factory=grid.var_factory)

    ## -SHUNTS (load like):
    for shunt in grid.shunts:
        shuntec = vge.get_shunt_template(grid.var_factory, name=shunt.name, phasor=True).block
        grid.var_factory.add_connections([shuntec.in_vars[0]], [shunt.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([shuntec.in_vars[1]], [shunt.bus.rms_model.out_vars[1]])
        shuntec = vge.to_implicit(shuntec, grid.var_factory)
        vge.set_rms_model(device=shunt, model=shuntec, var_factory=grid.var_factory)

    print("  ✓ Grid with path loaded")

    contingencies = [
                        vge.Contingency(name=f"Fault_{generator.name}", device=generator) for generator in
                        grid.generators] + [
                        vge.Contingency(name=f"Fault_{load.name}", device=load) for load in grid.loads] + [
                        vge.Contingency(name=f"Fault_{shunt.name}", device=shunt) for shunt in grid.shunts] + [
                        vge.Contingency(name=f"Fault_{line.name}", device=line) for line in grid.lines
                    ]

    phasor_problems = []

    for ct in contingencies:
        print(f"\n" + "=" * 50)
        print(f"--- Evaluating contingency: {ct.name} ---")
        print("=" * 50)

        target_device = ct.device
        original_state = target_device.active

        # Apply the contingency (disconnect)
        target_device.active = False

        print("\n[2] Running power flow...")
        pf_options = vge.PowerFlowOptions(tolerance=1e-5)
        pf_results = vge.power_flow(grid, pf_options)

        if pf_results.converged:
            print(f"  ✓ Power flow converged")

            print("\n[3] Creating phasor problem...")
            rms_options = vge.RmsOptions(
                time_step=0.01,
                simulation_time=1.0,
                tolerance=1e-6,
                max_iter=20,
                verbose=False,
            )

            phasor_problem = vge.RmsProblemPhasor(grid=grid, options=rms_options, pf_results=pf_results)
            print(f"  ✓ Phasor problem created with {phasor_problem.get_states_number()} states")

            # CORRECT APPEND: Appends the tuple containing the problem AND the name
            phasor_problems.append((phasor_problem, ct.name))
        else:
            print(f"  ✗ Power flow failed for {ct.name}. The system collapses in steady state.")

        # --- FUNDAMENTAL: RESTORE THE SYSTEM ---
        target_device.active = original_state

    # Return the correctly formatted list of tuples
    return phasor_problems


def op_extraction(problem: vge.RmsProblemPhasor) -> dict:
    """
    Returns the operating point dictionary from a defined VeraGrid problem.
    """
    v_dict: dict = dict()

    def safe_get_value(val):
        try:
            return float(val)
        except:
            if hasattr(val, 'value'):
                try:
                    return float(val.value)
                except:
                    pass
        return None

    # 3. EXTRACT x0 (Equilibrium point)
    try:
        x0 = problem.get_x0()
        for i, sym in enumerate(problem.state_and_algebraic_vars):
            v_dict[str(sym.name)] = float(x0[i])
    except Exception:
        pass

    # 4. EXTRACTION FROM THE PHOTO PATHS
    if hasattr(problem, '_variable_parameters') and hasattr(problem, '_variable_parameters_values'):
        names = problem._variable_parameters
        values = problem._variable_parameters_values
        if names is not None and values is not None:
            # If values is a list/array and names contains the symbols
            for name_obj, val_obj in zip(names, values):
                s_name = str(name_obj.name) if hasattr(name_obj, 'name') else str(name_obj)
                val = safe_get_value(val_obj)
                if val is not None:
                    v_dict[s_name] = val

    # Backup just in case there are also fixed constants
    if hasattr(problem, '_constant_parameters') and hasattr(problem, '_constant_params'):
        for n, v in zip(problem._constant_parameters, problem._constant_params):
            s_name = str(n.name) if hasattr(n, 'name') else str(n)
            val = safe_get_value(v)
            if val is not None:
                v_dict[s_name] = val

    return v_dict


def solve_problem(problem: Any = None) -> Tuple[np.ndarray, dict]:
    builder = None

    print("\n[4] Rebuilding equations and extracting exact physics...")
    builder = PolynomialMatrixBuilder(problem=problem, verbose=True)
    v_dict: dict = op_extraction(problem=problem)

    print("\n[*] Linearizing model with real parameters...")
    builder.linearize(v_dict=v_dict)

    # Continue with the normal stability analysis
    # Use automatic routing: sparse for large systems, dense fallback otherwise.
    res = builder.compute_stability()

    eigenvalues = res[0]
    #print(eigenvalues)
    stable = res[4]
    margin = res[5] if res[5] is not None else 0.0
    participation_matrix = res[3] if res[3] is not None else None

    print("\n" + "=" * 80)
    print("SMALL SIGNAL ANALYSIS COMPLETE - POLYNOMIAL BUILDER")
    print("=" * 80)
    print(f"  Found {len(eigenvalues)} finite eigenvalues")
    print(f"  System is stable: {stable}")
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

    return eigenvalues, pf_map


def plot_contingency_severity_and_pf(results_dict: dict):
    """
    Creates a severity ranking based on the dominant pole,
    followed by a grid of participation factor bar charts for all unstable modes.
    """
    names = []
    max_reals = []
    unstable_modes_info = []

    # Extract the dominant pole and unstable mode participations for each contingency
    for name, (evs, pf_map) in results_dict.items():
        if evs is not None and len(evs) > 0:
            dominant_pole = np.max(np.real(evs))
            names.append(name)
            max_reals.append(dominant_pole)

            if pf_map:
                # Track seen conjugate pairs so we don't plot identical participation charts
                seen_modes = set()
                for ev, participations in pf_map.items():
                    if ev.real > 1e-6:
                        # Identificador para pares conjugados basado en magnitud
                        mode_id = (round(ev.real, 4), round(abs(ev.imag), 4))
                        if mode_id not in seen_modes:
                            seen_modes.add(mode_id)
                            unstable_modes_info.append((name, ev, participations))

    if not names:
        print("[!] No data available to plot the severity ranking.")
        return

    # Sort severity ranking (from highest to lowest real part)
    sorted_data = sorted(zip(names, max_reals), key=lambda x: x[1], reverse=True)
    sorted_names, sorted_values = zip(*sorted_data)

    # --- Setup the Figure and Grid Layout ---
    num_unstable = len(unstable_modes_info)
    cols = min(3, max(1, num_unstable))  # Up to 3 columns
    pf_rows = math.ceil(num_unstable / cols) if num_unstable > 0 else 0
    total_rows = 1 + pf_rows

    # Dynamic figure height based on rows
    fig = plt.figure(figsize=(14, 6 + 4 * pf_rows))

    # GridSpec gives more height to the main top plot
    height_ratios = [1.5] + [1] * pf_rows
    gs = GridSpec(total_rows, cols, figure=fig, height_ratios=height_ratios)

    # --- 1. Plot Contingency Severity Ranking ---
    ax_main = fig.add_subplot(gs[0, :])
    # Define colors: Red for unstable (Re > 0), Green for stable
    colors = ['#e74c3c' if val > 0 else '#2ecc71' for val in sorted_values]
    bars = ax_main.bar(sorted_names, sorted_values, color=colors, edgecolor='black', alpha=0.8)

    # Reference line at 0 (Stability limit)
    ax_main.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

    ax_main.set_title("Contingency severity ranking", fontsize=14, fontweight='bold')
    ax_main.set_ylabel(r"Real part of dominant pole ($\sigma$)", fontsize=12)
    ax_main.set_xlabel("Contingency", fontsize=12)
    ax_main.set_xticks(range(len(sorted_names)))
    ax_main.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax_main.grid(axis='y', linestyle='--', alpha=0.6)

    # Add value labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax_main.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}',
                     va='bottom' if yval > 0 else 'top', ha='center', fontsize=9)

    # --- 2. Plot Participation Factors Grid for Unstable Modes ---
    if num_unstable > 0:
        for idx, (ct_name, ev, participations) in enumerate(unstable_modes_info):
            r = 1 + (idx // cols)
            c = idx % cols
            ax_sub = fig.add_subplot(gs[r, c])

            if not participations:
                ax_sub.text(0.5, 0.5, "No dominant variables found", ha='center', va='center')
                ax_sub.set_title(f"[{ct_name}]\nMode: {ev.real:.3f} $\pm$ {abs(ev.imag):.3f}j", fontsize=11)
                ax_sub.axis('off')
                continue

            # Sort ascending so the highest factor is at the top of horizontal bar chart
            sorted_pf = sorted(participations, key=lambda x: x[1], reverse=False)

            # Keep top 5 to avoid cluttering the plot
            sorted_pf = sorted_pf[-5:]
            var_names = [x[0] for x in sorted_pf]
            pf_values = [x[1] * 100 for x in sorted_pf]  # convert to %

            bars_sub = ax_sub.barh(var_names, pf_values, color='#3498db', edgecolor='black', alpha=0.8)
            ax_sub.set_title(f"[{ct_name}]\nMode: {ev.real:.3f} $\pm$ {abs(ev.imag):.3f}j", fontsize=11,
                             fontweight='bold')
            ax_sub.set_xlabel("Participation Factor (%)", fontsize=10)
            ax_sub.set_xlim(0, max(pf_values) * 1.2 if pf_values else 100)

            # Add percentage text next to each bar
            for i, v in enumerate(pf_values):
                ax_sub.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9)

            # Truncate y-labels if they are too long to maintain grid neatness
            ax_sub.set_yticklabels([str(lbl)[:20] + '...' if len(str(lbl)) > 20 else str(lbl) for lbl in var_names],
                                   fontsize=9)

    plt.tight_layout()
    plt.show()


def main():
    # --- START CHRONO ---
    start_time = time.perf_counter()

    grid_path = FAST_GRID_PATH if FAST_MODE else FULL_GRID_PATH
    print(f"[FAST_MODE={FAST_MODE}] Using grid: {grid_path}")

    # Dictionary to store eigenvalues and pf_maps linked to their contingency names
    all_results = {}

    problems_with_names = define_problems_with_path(grid_path)

    if problems_with_names:
        # Unpack the problem and its name
        for problem, ct_name in problems_with_names:
            print(f"\n>>> Solving small signal for: {ct_name}")

            # solve_problem returns both eigenvalues and participation factors dictionary
            eigenvalues, pf_map = solve_problem(problem)

            # Save them into the results dictionary
            all_results[ct_name] = (eigenvalues, pf_map)

    # --- END CHRONO ---
    end_time = time.perf_counter()
    total_duration = end_time - start_time

    print(f"\n========================================")
    print(f"EXECUTION TIME: {total_duration:.4f} seconds")
    print(f"========================================")

    # Finally, plot the severity ranking and participation factors grid
    if all_results:
        plot_contingency_severity_and_pf(all_results)


if __name__ == "__main__":
    try:
        if REPORT_TIME:
            lp = LineProfiler()
            lp.add_function(main)
            lp.add_function(solve_problem)
            lp.add_function(define_problems_with_path)
            lp(main)()
        else:
            main()
    except SystemExit:
        print("Error during execution")
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