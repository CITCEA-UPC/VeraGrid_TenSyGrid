# Matched PWM converter benchmarks

Upstream source: SanPen/VeraGrid `3fc0e970c`. Local ports retain the existing module layout. Raw trajectories, status checks and per-case logs accompany this report.


## Full converter in the AC/DC grid

Three buses, 100 MVA base, 230 kV AC and 320 kV DC, 30 MW DC load. The upstream converter defaults use a 5 kHz carrier starting at an extremum, longer measurement/PLL filtering and startup gain scheduling. Backward Euler, 10 µs nominal step, 10 ms horizon, handover at 0.1 ms. All variants use the same corrected power base, loss convention and nominal limits. The inequality-based MTI bridge retains a continuous comparator and integrated carrier; it does not use the standalone scheduled-MTI adapter. The multilinear row is a trig-state/auxiliary-equation converter model using the same procedural PWM. Its trig states now follow the PLL angular rate and its PWM phase advance is an algebraic rotation. MTI uses the same carrier-synchronized held modulation inputs. The scheduled variant uses analytic edge times and a multilinear gate map with the MTI continuous solver, without switching Boolean unknowns. The soft-sign variant changes only the comparator and keeps the shared controls. Soft smoothing and finite-step integration remain approximation differences.

Soft-sign uses `gate = (1 + margin / sqrt(margin**2 + lambda)) / 2`, where `margin = held_modulation - carrier`. The default `lambda=1e-6` has normalized width 0.001 (about 50 ns at 5 kHz). The nominal 10 µs step does not resolve that transition; convergence must be checked before claiming equivalence to hard switching.

StructuralCompiled uses warmup off; MTI uses its dedicated integrator. Grid times are single runs. Inequality-based MTI uses 12 event bisections; scheduled MTI has no switching inequalities. Initialization residual must be below 1e-7, initialization/convergence flags must pass, values must be finite and the requested final time must be reached to enter the comparison.

| Case | Status | Initial algebraic residual | Setup (s) | Integration (s) |
| --- | --- | ---: | ---: | ---: |
| grid_standard_StructuralCompiled | ok | 6.15e-09 | 40.515 | 173.215 |
| grid_scheduled_StructuralCompiled | ok | 6.15e-09 | 51.511 | 82.198 |

Baseline: `grid_standard_StructuralCompiled`. Continuous outputs are interpolated onto its timestamps.

| Case | Max phase-A current error (pu) | Max DC voltage error (pu) | Max filtered P error (pu) |
| --- | ---: | ---: | ---: |
| grid_standard_StructuralCompiled | 0 | 0 | 0 |
| grid_scheduled_StructuralCompiled | 0.00220981 | 2.13621e-05 | 0.000120672 |

## Reproduce

From the repository root, in the VeraGrid Python environment:

```sh
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/pwm-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/benchmark_solvers.py --timeout 1200 --grid-only --cases grid_standard_StructuralCompiled grid_scheduled_StructuralCompiled --grid-step-us 10 --grid-duration-ms 10 --output src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/benchmark_results/scheduled_multilinear_pwm/run_10ms
```

Use `--output PATH` for separate results, `--pwm-step-us VALUE` to refine the standalone step, and `--cases CASE...` for selected cases. The original results remain in the parent directory. Initialization consistency is not a claim of PWM-periodic steady state, nor does convergence alone establish equivalent solutions.

[Prototype design and validation](../DESIGN.md).
