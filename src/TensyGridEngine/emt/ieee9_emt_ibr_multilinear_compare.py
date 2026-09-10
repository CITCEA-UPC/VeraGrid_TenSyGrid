"""Compare nonlinear and multilinear inverter formulations on IEEE9 EMT."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ieee9_emt_ibr_simulation as case


def _traces(result):
    problem, time, values, devices, _gfl_buses, gfm_buses = result
    traces = {}
    for _generator, bus, model, _target in devices:
        for name in ("omega", "P", "Q", "Vdc_cap"):
            values_i = case._get_signal(problem, values, model, name)
            if values_i is not None:
                traces[f"{bus.name}:{name}"] = np.asarray(values_i, dtype=float)
    for generator in problem.grid.generators:
        if generator.bus.name not in gfm_buses:
            continue
        for name in ("omega", "P", "Q"):
            values_i = case._get_signal(problem, values, generator.emt_model, name)
            if values_i is not None:
                traces[f"{generator.bus.name}:{name}"] = np.asarray(values_i, dtype=float)
    return np.asarray(time, dtype=float), traces


def main() -> None:
    reference = case.run_case(case.N_GFL, case.N_GFM, multilinear_inverters=False)
    multilinear = case.run_case(case.N_GFL, case.N_GFM, multilinear_inverters=True)
    t_ref, ref = _traces(reference)
    t_ml, ml = _traces(multilinear)
    if not np.array_equal(t_ref, t_ml):
        raise RuntimeError("Reference and multilinear simulations returned different time grids")

    common = sorted(set(ref).intersection(ml))
    errors = {name: float(np.max(np.abs(ml[name] - ref[name]))) for name in common}
    print("IEEE9 nonlinear versus multilinear inverter response:")
    for name, error in errors.items():
        print(f"  max |ML-reference| {name}: {error:.9e}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for name, axis in zip(("omega", "P", "Q"), axes):
        for key in common:
            if key.endswith(f":{name}"):
                label = key.split(":", 1)[0]
                axis.plot(1e3 * t_ref, ref[key], label=f"nonlinear {label}")
                axis.plot(1e3 * t_ml, ml[key], "--", label=f"multilinear {label}")
        axis.set_ylabel(f"{name} (p.u.)")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Time (ms)")
    axes[0].set_title("IEEE9 EMT: nonlinear vs multilinear inverter response")
    output = Path(__file__).with_name(
        f"ieee9_emt_ibr_multilinear_compare_gfl{case.N_GFL}_gfm{case.N_GFM}.png"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"plot={output}")


if __name__ == "__main__":
    main()
