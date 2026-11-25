# PWM and full converter benchmark

Run on 25 November 2026 with one BLAS/OpenMP thread. Timings are machine-specific.

## PWM solver timing

Sinusoidal references, 1 kHz switching, 50 Hz frame, modulation 0.8; 4 ms duration with a nominal 50 µs step. Each call constructs a fresh problem. Three repeats follow the first call; imports are excluded.

| Variant | First call (s) | Median repeat (s) |
| --- | ---: | ---: |
| pwm_internal_mti | 0.149095 | 0.145658 |
| pwm_live_comparator | 0.024285 | 0.021385 |
| pwm_procedural | 0.074959 | 0.013047 |
| pwm_scheduled_mti | 0.068417 | 0.067442 |

These variants have different sampling/event semantics. See `pwm_errors.json` and `pwm_comparison.png`; timings alone do not establish equivalent solutions.

The separate aligned-event test uses identical held references and ramp boundaries for both sinusoidal and variable-dq inputs. See `aligned_events.log`. It checks the event equations, rather than the solver adapters.

## Full switched converter in an AC/DC grid

Three buses: 230 kV AC source behind a Thevenin equivalent and transformer, a 100 MVA VSC, and a 320 kV DC bus with a 30 MW load. Power-flow-seeded initialization; full bridge, RL filter, PLL, controls, DC dynamics and startup handover at 0.1 ms. Backward Euler, 10 µs nominal step, 2 ms duration (two switching periods; this is a startup comparison, not a steady-state or stability study).

The MTI variant uses `EmtProblemMTI` and `EmtMtiBackEulerSolver`. Other variants use the named EMT backend. For the short run, StructuralCompiled uses its supported warmup-off policy (generated Python kernels). The default adaptive warmup attempt exceeded 300 s during compilation. Setup includes PF, model assembly, initialization and solver construction. Simulation includes any compilation deferred until first use. Persistent caches and concurrent machine load can affect setup times. Grid timings are single runs. Earlier default-policy attempts are archived in `initial_attempts.json` and `*_adaptive_timeout.log`.

| Case | Status | Setup (s) | Simulation (s) |
| --- | --- | ---: | ---: |
| grid_mti_StructuralCompiled | invalid | 48.316 | 18.844 |
| grid_mti_Symbolic | skipped | — | — |
| grid_multilinear_StructuralCompiled | ok | 23.859 | 6.950 |
| grid_multilinear_Symbolic | ok | 23.653 | 5.267 |
| grid_standard_Automatic | error | — | — |
| grid_standard_StructuralAD | error | — | — |
| grid_standard_StructuralCompiled | ok | 40.546 | 13.212 |
| grid_standard_Symbolic | ok | 39.531 | 21.780 |

Only runs that initialize, converge, remain finite and reach the requested end time enter the waveform comparison. Errors and timeouts are retained in `timings.json` and per-case logs. These status checks do not establish accuracy.

Waveforms are interpolated onto the baseline timestamps for max-absolute and RMS errors; gate discontinuities use previous-value sampling in the PWM comparison. Grid currents/powers use the model’s per-unit convention.

Baseline: `grid_standard_StructuralCompiled`.

| Case | Phase-A current max error (pu) | DC current max error (pu) | P max error (pu) |
| --- | ---: | ---: | ---: |
| grid_multilinear_StructuralCompiled | 0.180443 | 0.00388511 | 0.891118 |
| grid_multilinear_Symbolic | 0.193882 | 0.00395111 | 0.907463 |
| grid_standard_StructuralCompiled | 0 | 0 | 0 |
| grid_standard_Symbolic | 1.2574e-09 | 2.56249e-10 | 8.59843e-07 |

The full-grid MTI run failed initialization and is excluded from agreement comparisons. Automatic and StructuralAD crashed during native compilation. The multilinear converter differs substantially from the standard converter; its lower runtime does not establish an equivalent solution.

## Reproduce

From the repository root:

```sh
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/pwm-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/benchmark_solvers.py --timeout 300
```

Use `--cases CASE...` to retry selected cases while retaining other saved results. Local changes were present in the working tree during these runs; this is not a clean-checkout benchmark.

Validation: six regression tests passed; see `validation_tests.log`. Both aligned-event cases passed.
