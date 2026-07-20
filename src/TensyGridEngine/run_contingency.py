#!/usr/bin/env python3
"""
Script per calcular contingències de xarxa elèctrica.
Desactiva un element (generator/line/transformer) i recalcula l'estabilitat.
No desa resultats a disc (és temporal).
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

_real_stdout = sys.stdout
sys.stdout = sys.stderr

project_base = Path(__file__).resolve().parents[2]
src_path = project_base / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import VeraGridEngine.api as vge
from small_signal_analysis_main import load_grid, set_models, run_small_signal_analysis


def _damping(re_: float, im: float) -> float:
    mag = np.sqrt(re_ * re_ + im * im)
    return -re_ / mag if mag > 1e-12 else 0.0


def _freq_hz(im: float) -> float:
    return abs(im) / (2 * np.pi)


def main():
    parser = argparse.ArgumentParser(description='Compute network contingencies')
    parser.add_argument('--grid', required=True, help='Nom del fitxer de xarxa')
    parser.add_argument('--type', required=True, choices=['generator', 'line', 'transformer'],
                       help='Element type to disconnect')
    parser.add_argument('--element', required=True, help='Element index (e.g., gen_0, line_2, transformer_1)')

    args = parser.parse_args()

    try:
        grid = load_grid(args.grid)

        element_idx = int(args.element.split("_")[1])

        if args.type == "generator":
            gens = list(grid.generators)
            if element_idx >= len(gens):
                raise ValueError(f"Generator index {element_idx} out of range ({len(gens)} generators)")
            gens[element_idx].active = False
        elif args.type == "line":
            lines = list(grid.lines)
            if element_idx >= len(lines):
                raise ValueError(f"Line index {element_idx} out of range ({len(lines)} lines)")
            lines[element_idx].active = False
        elif args.type == "transformer":
            trafos = list(grid.transformers2w)
            if element_idx >= len(trafos):
                raise ValueError(f"Transformer index {element_idx} out of range ({len(trafos)} transformers)")
            trafos[element_idx].active = False

        set_models(grid)
        results = run_small_signal_analysis(grid)

        eigenvalues = results["eigenvalues"]
        pf_matrix = results["participation_factors"]

        eigenvalues_list = [
            {"re": float(np.real(ev)), "im": float(np.imag(ev))}
            for ev in eigenvalues
        ]

        state_var_names = [
            str(v.name) if hasattr(v, "name") else f"state_{i}"
            for i, v in enumerate(results["problem"].state_and_algebraic_vars)
        ]

        critical_modes = []
        for i, ev in enumerate(eigenvalues):
            re, im = float(np.real(ev)), float(np.imag(ev))
            if abs(im) <= 0.1:
                continue
            damping = float(_damping(re, im))

            participation = {}
            if pf_matrix is not None and pf_matrix.ndim == 2:
                pf_col = pf_matrix[:, i]
                for j, var_name in enumerate(state_var_names):
                    if j < len(pf_col):
                        participation[str(var_name)] = float(pf_col[j])

            top_participation = dict(
                sorted(participation.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            )

            critical_modes.append({
                "mode": i,
                "re": re,
                "im": im,
                "damping": damping,
                "freq_hz": float(_freq_hz(im)),
                "critical": bool(abs(damping) < 0.05),
                "participation_factors": top_participation,
            })
        critical_modes.sort(key=lambda m: m["damping"])

        result = {
            "grid_name": args.grid,
            "stable": results["stable"],
            "margin": results["margin"],
            "n_eigenvalues": len(eigenvalues),
            "n_state_vars": len(results["problem"].get_x0()),
            "eigenvalues": eigenvalues_list,
            "critical_modes": critical_modes[:10],
            "grid_info": {
                "n_buses": len(list(grid.buses)),
                "n_generators": len([g for g in grid.generators if g.active]),
                "n_lines": len([l for l in grid.lines if l.active]),
                "n_loads": len(list(grid.loads)),
                "n_transformers": len(list(grid.transformers2w)),
            },
        }

        if result["stable"]:
            result["status"] = "STABLE"
        elif result["margin"] < 0.001:
            result["status"] = "MARGINAL"
        else:
            result["status"] = "UNSTABLE"

        sys.stdout = _real_stdout
        print(json.dumps(result))

    except Exception as e:
        sys.stdout = _real_stdout
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
