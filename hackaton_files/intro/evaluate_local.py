"""Full sample validation: losses, damping, limits and RMS load-step response.

Run: PYTHONPATH=src:. python -m hackaton_files.intro.evaluate_local
The eleven samples include every quick case. They are practice cases, not an
assertion about the organizer's hidden scenarios or final scoring formula.
"""
from time import perf_counter
from .settings import make_evaluator
from .starter_script import StarterTuner
from .scenarios import LOCAL_SCENARIOS, load_benchmark, make_scenario


def main():
    start = perf_counter()
    benchmark = load_benchmark()
    results = []
    failures = 0
    for scenario in LOCAL_SCENARIOS:
        try:
            # Fresh tuner per case: budgets and algorithm state do not leak.
            # This public helper validates the proposal, recomputes PF and SSA,
            # and simulates the standard 2% load increase over three seconds.
            metrics = StarterTuner().test_tuning(
                make_scenario(benchmark, scenario), make_evaluator())
            results.append(metrics)
            passed = (metrics.stable and metrics.voltage_violations == 0
                      and metrics.thermal_violations == 0)
            failures += not passed
            damping = metrics.min_oscillatory_damping
            damping_text = "n/a" if damping is None else f"{100 * damping:.2f}%"
            print(f"{scenario.name:18s} losses={metrics.active_losses_mw:8.3f} MW  "
                  f"damping={damping_text:>8s}  max Re={metrics.max_eigenvalue_real:+.5f}  "
                  f"voltage/thermal violations={metrics.voltage_violations}/{metrics.thermal_violations}  "
                  f"peak df={metrics.peak_frequency_deviation_hz:.4f} Hz  "
                  f"peak dV={metrics.peak_voltage_deviation_pu:.4f} pu  "
                  f"{'PASS' if passed else 'FAIL'}")
        except Exception as error:
            failures += 1
            print(f"{scenario.name:18s} ERROR: {type(error).__name__}: {error}")
    print(f"\nLocal validation: {len(LOCAL_SCENARIOS) - failures}/{len(LOCAL_SCENARIOS)} passed; "
          f"{perf_counter() - start:.3f} s.")
    if results:
        print(f"Mean losses over {len(results)} successfully evaluated cases: "
              f"{sum(m.active_losses_mw for m in results) / len(results):.3f} MW")
        damping_values = [m.min_oscillatory_damping for m in results
                          if m.min_oscillatory_damping is not None]
        if damping_values:
            print(f"Worst oscillatory damping: {100 * min(damping_values):.2f}%")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
