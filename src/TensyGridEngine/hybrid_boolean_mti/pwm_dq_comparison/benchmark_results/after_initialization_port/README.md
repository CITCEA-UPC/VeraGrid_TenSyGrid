# PWM benchmark after upstream initialization port

Upstream source: SanPen/VeraGrid `3fc0e970c`. Local ports retain the existing module layout. Raw trajectories, status checks and per-case logs accompany this report.

## Standalone PWM

4 ms, 1 kHz carrier, 50 Hz sinusoidal references, modulation 0.8. One first call and three repeated calls construct fresh models. Imports are excluded. Setup and integration are measured separately. Times depend on machine load and caches.

| Variant | Step (µs) | First total (ms) | Median setup (ms) | Median integration (ms) | Median total (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| pwm_procedural | 50 | 17.819 | 3.466 | 11.608 | 15.064 |
| pwm_live_comparator | 50 | 24.353 | 2.403 | 18.655 | 20.901 |
| pwm_internal_mti | 50 | 149.015 | 0.365 | 143.581 | 143.980 |
| pwm_scheduled_mti | 50 | 70.752 | 0.557 | 66.918 | 67.511 |

The live comparator uses continuous references. Procedural PWM and both standalone MTI adapters request the same held sinusoidal reference samples. Runtime event handling and output recording can still differ. The 0.1 µs plotting grid is a previous-value reconstruction, not the integration step. The event-equation aligned test is a separate check and does not time the MTI integrator.

| Variant | Gate mismatch A / B / C (%) |
| --- | --- |
| pwm_procedural | 0.000 / 0.000 / 0.000 |
| pwm_internal_mti | 4.737 / 4.697 / 3.365 |
| pwm_live_comparator | 6.090 / 2.130 / 4.102 |
| pwm_scheduled_mti | 18.687 / 2.747 / 29.469 |

## Full converter in the AC/DC grid

Three buses, 100 MVA base, 230 kV AC and 320 kV DC, 30 MW DC load. The upstream converter defaults use a 5 kHz carrier starting at an extremum, longer measurement/PLL filtering and startup gain scheduling. Backward Euler, 10 µs nominal step, 2 ms horizon, handover at 0.1 ms. All variants use the same corrected power base, loss convention and nominal limits. The MTI bridge retains a continuous comparator and integrated carrier; it does not use the standalone scheduled-MTI adapter. The multilinear row is a trig-state/auxiliary-equation converter model using the same procedural PWM, not another PWM algorithm. Its trig states still advance at nominal frequency rather than following the PLL angle directly, so these model timings are not an equal-accuracy PWM solver ranking.

StructuralCompiled uses warmup off; MTI uses its dedicated integrator. Grid times are single runs. Initialization residual must be below 1e-7, initialization/convergence flags must pass, values must be finite and the requested final time must be reached to enter the comparison.

| Case | Status | Initial algebraic residual | Setup (s) | Integration (s) |
| --- | --- | ---: | ---: | ---: |
| grid_standard_StructuralCompiled | ok | 6.15e-09 | 40.944 | 30.892 |
| grid_multilinear_StructuralCompiled | ok | 6.15e-09 | 24.747 | 9.375 |
| grid_mti_StructuralCompiled | ok | 6.15e-09 | 48.708 | 48.740 |

Baseline: `grid_standard_StructuralCompiled`. Continuous outputs are interpolated onto its timestamps.

| Case | Max phase-A current error (pu) | Max DC voltage error (pu) | Max filtered P error (pu) |
| --- | ---: | ---: | ---: |
| grid_standard_StructuralCompiled | 0 | 0 | 0 |
| grid_multilinear_StructuralCompiled | 0.0309426 | 0.000139585 | 0.000869162 |
| grid_mti_StructuralCompiled | 0.0780109 | 0.000401115 | 0.00195634 |

## Reproduce

From the repository root, in the VeraGrid Python environment:

```sh
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/pwm-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/benchmark_solvers.py --timeout 180
```

Use `--output PATH` for separate results, `--pwm-step-us VALUE` to refine the standalone step, and `--cases CASE...` for selected cases. The original results remain in the parent directory. Initialization consistency is not a claim of PWM-periodic steady state, nor does convergence alone establish equivalent solutions.

## Validation and step refinement

Standard, trig-state and MTI full-grid initialization checks now pass. The first combined regression run passed eight checks and exposed an MTI common-mode seed error. After correcting that seed, the targeted MTI initialization check passed. The original bridge/controller checks passed in the combined run; both logs are retained.

The step refinement in `step_10us` reduces internal-carrier MTI gate disagreement to approximately 1.0–1.2%, but scheduled MTI still disagrees for up to 36% of samples. Its adapter remains unresolved; the initialization port does not repair that runtime behavior. Smaller steps increase runtime and do not establish equivalent solutions.

`initialization_before_after.png` compares the old benchmark with the ported model. This includes the upstream default frequency/filtering/startup changes as well as initialization corrections; it is not an isolated initialization-only experiment.
