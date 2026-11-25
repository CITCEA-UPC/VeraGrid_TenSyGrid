# PWM benchmark after upstream initialization port

Upstream source: SanPen/VeraGrid `3fc0e970c`. Local ports retain the existing module layout. Raw trajectories, status checks and per-case logs accompany this report.

## Standalone PWM

4 ms, 1 kHz carrier, 50 Hz sinusoidal references, modulation 0.8. One first call and three repeated calls construct fresh models. Imports are excluded. Setup and integration are measured separately. Times depend on machine load and caches.

| Variant | Step (µs) | First total (ms) | Median setup (ms) | Median integration (ms) | Median total (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| pwm_procedural | 10 | 48.530 | 2.880 | 41.459 | 44.848 |
| pwm_live_comparator | 10 | 56.016 | 2.407 | 50.294 | 53.080 |
| pwm_internal_mti | 10 | 240.466 | 0.351 | 233.732 | 234.102 |
| pwm_scheduled_mti | 10 | 186.321 | 0.555 | 180.465 | 181.065 |

The live comparator uses continuous references. Procedural PWM and both standalone MTI adapters request the same held sinusoidal reference samples. Runtime event handling and output recording can still differ. The 0.1 µs plotting grid is a previous-value reconstruction, not the integration step. The event-equation aligned test is a separate check and does not time the MTI integrator.

| Variant | Gate mismatch A / B / C (%) |
| --- | --- |
| pwm_procedural | 0.000 / 0.000 / 0.000 |
| pwm_internal_mti | 0.995 / 1.202 / 1.125 |
| pwm_live_comparator | 1.997 / 1.305 / 2.902 |
| pwm_scheduled_mti | 21.684 / 5.997 / 35.964 |

## Reproduce

From the repository root, in the VeraGrid Python environment:

```sh
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/pwm-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/benchmark_solvers.py --timeout 180
```

Use `--output PATH` for separate results, `--pwm-step-us VALUE` to refine the standalone step, and `--cases CASE...` for selected cases. The original results remain in the parent directory. Initialization consistency is not a claim of PWM-periodic steady state, nor does convergence alone establish equivalent solutions.
