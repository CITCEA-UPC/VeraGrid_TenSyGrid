# Matched PWM converter benchmarks

Upstream source: SanPen/VeraGrid `3fc0e970c`. Local ports retain the existing module layout. Raw trajectories, status checks and per-case logs accompany this report.

## Standalone PWM

4 ms, 1 kHz carrier, 50 Hz sinusoidal references, modulation 0.8. One first call and three repeated calls construct fresh models. Imports are excluded. Setup and integration are measured separately. Times depend on machine load and caches.

| Variant | Step (µs) | First total (ms) | Median setup (ms) | Median integration (ms) | Median total (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| pwm_procedural | 50 | 22.050 | 4.038 | 14.425 | 18.362 |
| pwm_live_comparator | 50 | 33.176 | 3.908 | 24.333 | 29.136 |
| pwm_internal_mti | 50 | 270.308 | 0.611 | 254.231 | 254.909 |
| pwm_scheduled_mti | 50 | 90.979 | 0.735 | 91.717 | 92.764 |

The live comparator uses continuous references. Procedural PWM and both standalone MTI adapters request the same held sinusoidal reference samples. Runtime event handling and output recording can still differ. The 0.1 µs plotting grid is a previous-value reconstruction, not the integration step. The event-equation aligned test is a separate check and does not time the MTI integrator.

| Variant | Gate mismatch A / B / C (%) |
| --- | --- |
| pwm_procedural | 0.000 / 0.000 / 0.000 |
| pwm_internal_mti | 4.737 / 4.697 / 3.365 |
| pwm_live_comparator | 6.090 / 2.130 / 4.102 |
| pwm_scheduled_mti | 18.687 / 2.747 / 29.469 |

## Full converter in the AC/DC grid

Three buses, 100 MVA base, 230 kV AC and 320 kV DC, 30 MW DC load. The upstream converter defaults use a 5 kHz carrier starting at an extremum, longer measurement/PLL filtering and startup gain scheduling. Backward Euler, 10 µs nominal step, 2 ms horizon, handover at 0.1 ms. All variants use the same corrected power base, loss convention and nominal limits. The MTI bridge retains a continuous comparator and integrated carrier; it does not use the standalone scheduled-MTI adapter. The multilinear row is a trig-state/auxiliary-equation converter model using the same procedural PWM. Its trig states now follow the PLL angular rate and its PWM phase advance is an algebraic rotation. MTI uses the same carrier-synchronized held modulation inputs. The soft-sign variant changes only the comparator and keeps the shared controls. Soft smoothing and finite-step integration remain approximation differences.

Soft-sign uses `gate = (1 + margin / sqrt(margin**2 + lambda)) / 2`, where `margin = held_modulation - carrier`. The default `lambda=1e-6` has normalized width 0.001 (about 50 ns at 5 kHz). The nominal 10 µs step does not resolve that transition; convergence must be checked before claiming equivalence to hard switching.

StructuralCompiled uses warmup off; MTI uses its dedicated integrator. Grid times are single runs. MTI uses 12 event bisections. Initialization residual must be below 1e-7, initialization/convergence flags must pass, values must be finite and the requested final time must be reached to enter the comparison.

| Case | Status | Initial algebraic residual | Setup (s) | Integration (s) |
| --- | --- | ---: | ---: | ---: |
| grid_mti_StructuralCompiled | ok | 6.15e-09 | 66.199 | 132.814 |
| grid_multilinear_StructuralCompiled | ok | 6.15e-09 | 36.587 | 9.798 |
| grid_softsign_StructuralCompiled | ok | 6.15e-09 | 59.745 | 8.070 |
| grid_standard_StructuralCompiled | ok | 6.15e-09 | 62.912 | 32.494 |

Baseline: `grid_standard_StructuralCompiled`. Continuous outputs are interpolated onto its timestamps.

| Case | Max phase-A current error (pu) | Max DC voltage error (pu) | Max filtered P error (pu) |
| --- | ---: | ---: | ---: |
| grid_standard_StructuralCompiled | 0 | 0 | 0 |
| grid_multilinear_StructuralCompiled | 0.000134289 | 1.50482e-06 | 4.52731e-05 |
| grid_mti_StructuralCompiled | 0.0329569 | 0.000207343 | 0.000853432 |
| grid_softsign_StructuralCompiled | 0.0105769 | 2.52444e-05 | 7.60709e-05 |

## Reproduce

From the repository root, in the VeraGrid Python environment:

```sh
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/pwm-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/benchmark_solvers.py --timeout 400 --grid-step-us 10 --grid-duration-ms 2 --output src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/benchmark_results/matched_pwm_models
```

Use `--output PATH` for separate results, `--pwm-step-us VALUE` to refine the standalone step, and `--cases CASE...` for selected cases. The original results remain in the parent directory. Initialization consistency is not a claim of PWM-periodic steady state, nor does convergence alone establish equivalent solutions.

See [validation and failed-attempt notes](VALIDATION.md) and the [5 us refinement](step_5us/README.md). The procedural refinement fails at 1.215 ms, so full-horizon step convergence is not established.
