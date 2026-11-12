"""Five-case PF + modal screen, omitting the local evaluator's load-step test.

Run: PYTHONPATH=src:. python -m hackaton_files.intro.evaluate_quick
This is a screening result, not a substitute for full local validation.
"""
from time import perf_counter
import numpy as np
from hackaton_files import TunableGrid
from .starter_script import StarterTuner, initialize_grid
from .scenarios import QUICK_SCENARIOS, load_benchmark, make_scenario


def evaluate_quick(tuner, grid):
    """Validate/apply a proposal and recompute its operating point and modes."""
    tunable = initialize_grid(grid)
    proposal = tuner.tune(tunable)
    # apply_tuning validates the handles/bounds and reruns PF for grid actions.
    problem = tunable.evaluator.apply_tuning(tunable, proposal)
    tuned = TunableGrid.from_problem(
        problem, tunable.evaluator.control_ranges,
        shunt_bus_names=(), line_corridors=(),
    )
    analysis = tuner.get_modal_analysis(tuned)
    values = analysis["eigenvalues"]
    values = values[np.isfinite(values)]
    neutral_count = int(np.count_nonzero(np.abs(values) <= 1e-8))
    physical = values[np.abs(values) > 1e-8]
    if not physical.size:
        raise RuntimeError("No finite physical modes")
    oscillatory = physical[np.abs(physical.imag) / (2 * np.pi) > 0.05]
    damping = float(np.min(-oscillatory.real / np.abs(oscillatory))) if oscillatory.size else None
    max_real = float(np.max(physical.real))
    pf = problem.power_flow_results
    voltage = np.abs(pf.voltage)
    violations = int(np.count_nonzero((voltage < 0.95) | (voltage > 1.05)))
    violations += int(np.count_nonzero(np.abs(pf.loading) > 1.0))
    passed = (neutral_count <= 1 and max_real < -1e-5
              and (damping is None or damping >= 0.05) and violations == 0)
    return max_real, damping, passed


def main():
    start = perf_counter()
    benchmark = load_benchmark()
    failures = 0
    for scenario in QUICK_SCENARIOS:
        try:
            max_real, damping, passed = evaluate_quick(
                StarterTuner(), make_scenario(benchmark, scenario))
            damping_text = "n/a" if damping is None else f"{100 * damping:.2f}%"
            print(f"{scenario.name:18s} max Re(lambda)={max_real:+.5f}  "
                  f"damping={damping_text:>8s}  {'PASS' if passed else 'FAIL'}")
            failures += not passed
        except Exception as error:
            # One invalid proposal/case must not hide results from the others.
            failures += 1
            print(f"{scenario.name:18s} ERROR: {type(error).__name__}: {error}")
    print(f"Quick screen: {len(QUICK_SCENARIOS) - failures}/{len(QUICK_SCENARIOS)} passed; "
          f"{perf_counter() - start:.3f} s (includes loading/initialization).")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
