#!/usr/bin/env python3
"""
Script per calcular contingències de xarxa elèctrica.
Desactiva un element (generator/line/load) i recalcula l'estabilitat.
No desa resultats a disc (és temporal).
"""

import sys
import json
import copy
import argparse
import numpy as np
from pathlib import Path
import os

# Redirigir stdout a stderr per evitar contaminació de DEBUG messages
_real_stdout = sys.stdout
sys.stdout = sys.stderr

# Afegir el path del projecte
project_base = Path(__file__).resolve().parents[2]
src_path = project_base / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import VeraGridEngine.api as vge
from small_signal_multilinear import build_problems, run_small_signal_from_driver


def create_contingency_problem(base_problem, element_to_trip):
    contingency_problem = copy.deepcopy(base_problem)
    contingency_problem.grid = copy.deepcopy(base_problem.grid)
    contingency_problem._algebraic_eqs = list(base_problem._algebraic_eqs)
    contingency_problem._state_eqs = list(base_problem._state_eqs)

    if isinstance(element_to_trip, vge.Generator):
        original_collection = list(base_problem.grid.generators)
        copied_collection = list(contingency_problem.grid.generators)
    elif isinstance(element_to_trip, vge.Line):
        original_collection = list(base_problem.grid.lines)
        copied_collection = list(contingency_problem.grid.lines)
    elif isinstance(element_to_trip, vge.Transformer2W):
        original_collection = list(base_problem.grid.transformers2w)
        copied_collection = list(contingency_problem.grid.transformers2w)
    else:
        raise ValueError(f"Unsupported device type: {type(element_to_trip)}")

    try:
        idx = original_collection.index(element_to_trip)
    except ValueError:
        raise RuntimeError(f"Element {element_to_trip.idtag} not found in original grid")

    if idx >= len(copied_collection):
        raise RuntimeError(f"Index {idx} out of range in copied grid")

    copied_collection[idx].active = False

    model = element_to_trip.rms_model
    vars_to_zero = []
    
    if isinstance(element_to_trip, vge.Generator):
        patterns = ('Irg_', 'Iig_')
    elif isinstance(element_to_trip, (vge.Line, vge.Transformer2W)):
        patterns = ('Irf_', 'Iif_', 'Irt_', 'Iit_')
    else:
        raise NotImplementedError(f"Contingency logic for device type {type(element_to_trip)} is not implemented.")

    for var in model.algebraic_vars:
        if any(var.name.startswith(p) for p in patterns):
            vars_to_zero.append(var)

    if not vars_to_zero:
        raise RuntimeError(f"No current variables found for {type(element_to_trip).__name__} '{element_to_trip.name}'")

    model_eqs = list(model.algebraic_eqs)
    for var_sym in vars_to_zero:
        defining_eq = None
        for eq in model_eqs:
            if eq.contains_var(var_sym):
                defining_eq = eq
                break

        if defining_eq is None:
            print(f"Warning: No equation found for '{var_sym.name}'. Skipping.", file=sys.stderr)
            continue

        try:
            eq_idx = contingency_problem._algebraic_eqs.index(defining_eq)
        except ValueError:
            try:
                eq_idx = next(i for i, e in enumerate(contingency_problem._algebraic_eqs) if e is defining_eq)
            except StopIteration:
                print(f"Warning: Equation for '{var_sym.name}' not in problem. Skipping.", file=sys.stderr)
                continue

        contingency_problem._algebraic_eqs[eq_idx] = var_sym

    return contingency_problem


def main():
    parser = argparse.ArgumentParser(description='Compute network contingencies')
    parser.add_argument('--grid', required=True, help='Nom del fitxer de xarxa')
    parser.add_argument('--type', required=True, choices=['generator', 'line', 'transformer'],
                        help='Element type to disconnect')
    parser.add_argument('--element', required=True, help='Element index (e.g., gen_0, line_2)')

    args = parser.parse_args()

    try:
        # Carregar la xarxa des de disc
        # --- New Workflow: Build the base problem ONCE ---
        print(f"Building base case problem for {args.grid}...", file=sys.stderr)
        base_problem_ml, _, base_pf_results, base_rms_options_ml, _ = build_problems(
            grid_filename=args.grid,
        )
        print("Base case built. Starting contingency analysis.", file=sys.stderr)

        # --- Identify the element to trip from the base grid ---
        element_idx = int(args.element.split("_")[1])
        element_map = {
            "generator": list(base_problem_ml.grid.generators),
            "line": list(base_problem_ml.grid.lines),
            "load": list(base_problem_ml.grid.loads),
            "transformer": list(base_problem_ml.grid.transformers2w),
        }

        if args.type not in element_map:
            raise ValueError(f"Invalid element type: {args.type}")

        elements = element_map[args.type]
        if not (0 <= element_idx < len(elements)):
            raise ValueError(f"{args.type.capitalize()} index {element_idx} out of range ({len(elements)} elements)")

        element_to_trip = elements[element_idx]
        print(f"Identified contingency: Trip {args.type} '{element_to_trip.name}'", file=sys.stderr)

        # --- Create the contingency problem by modifying the base matrices ---
        print("Creating contingency problem via matrix surgery...", file=sys.stderr)
        problem_ml = create_contingency_problem(base_problem_ml, element_to_trip)

        # The power flow of the base case is no longer valid for the post-contingency state.
        # We must re-solve the power flow for the new topology.
        print("Re-solving power flow for N-1 topology...", file=sys.stderr)
        pf_results = vge.power_flow(problem_ml.grid, vge.PowerFlowOptions(tolerance=1e-5))

        # Use the options from the base build
        rms_options_ml = base_rms_options_ml

        # --- Run the analysis on the modified problem ---
        print("Running small-signal analysis on contingency problem...", file=sys.stderr)
        eig_driver_ml, _, state_var_names_ml = run_small_signal_from_driver(
            problem=problem_ml,
            pf_results=pf_results,
            rms_options=rms_options_ml
        )

        # Filtrar eigenvalues vàlids
        fin_drv_ml = eig_driver_ml[np.isfinite(eig_driver_ml) & (np.abs(eig_driver_ml) < 1e6)]

        # Calcular estabilitat
        stable_ml = bool(np.all(np.real(fin_drv_ml) <= 0.0)) if len(fin_drv_ml) > 0 else False
        margin_ml = float(np.max(np.real(fin_drv_ml))) if len(fin_drv_ml) > 0 else float("nan")

        # Convertir eigenvalues a llista
        eigenvalues_list = [
            {"re": float(np.real(ev)), "im": float(np.imag(ev))}
            for ev in fin_drv_ml
        ]

        # Funcions auxiliars
        def _damping(re, im):
            mag = np.sqrt(re * re + im * im)
            return -re / mag if mag > 1e-12 else 0.0

        def _freq_hz(im):
            return abs(im) / (2 * np.pi)

        # Calcular modes crítics (sense participation factors)
        critical_modes = []
        for i, ev in enumerate(fin_drv_ml):
            re, im = float(np.real(ev)), float(np.imag(ev))
            if abs(im) > 0.1:
                damping = float(_damping(re, im))
                critical_modes.append({
                    "mode": i,
                    "re": re,
                    "im": im,
                    "damping": damping,
                    "freq_hz": float(_freq_hz(im)),
                    "critical": bool(abs(damping) < 0.05),
                })
        critical_modes.sort(key=lambda m: m["damping"])

        # Preparar resultat
        result = {
            "grid_name": args.grid,
            "stable": stable_ml,
            "margin": margin_ml,
            "n_eigenvalues": len(fin_drv_ml),
            "n_state_vars": len(problem_ml.get_x0()),
            "eigenvalues": eigenvalues_list,
            "critical_modes": critical_modes[:10],
            "grid_info": {
                "n_buses": len(list(problem_ml.grid.buses)),
                "n_generators": len([g for g in problem_ml.grid.generators if g.active]),
                "n_lines": len([l for l in problem_ml.grid.lines if l.active]),
                "n_loads": len([l for l in problem_ml.grid.loads if l.active]),
                "n_transformers": len(list(problem_ml.grid.transformers2w)),
            },
        }

        # Determinar status
        if result["stable"]:
            result["status"] = "STABLE"
        elif result["margin"] < 0.001:
            result["status"] = "MARGINAL"
        else:
            result["status"] = "UNSTABLE"

        # Restaurar stdout i imprimir JSON
        sys.stdout = _real_stdout
        print(json.dumps(result))

    except Exception as e:
        sys.stdout = _real_stdout
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
