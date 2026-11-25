# Converter comparison over 10 ms

The procedural PWM trajectory is the reference. Errors below are maximum absolute differences up to each stated time, using interpolation onto the reference timestamps. Integration times exclude model/solver setup and include first-use compilation. All four cases pass initialization and convergence, reach 10 ms, and contain 1,001 finite output points. Their first 2 ms exactly reproduces the earlier runs; see [verification](verification.json).

| Variant | Setup (s) | Integration (s) | Current error at 2 ms (pu) | At 5 ms (pu) | At 10 ms (pu) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Procedural PWM | 43.21 | 161.78 | 0 | 0 | 0 |
| Trig-state model | 25.52 | 47.18 | 0.000134289 | 0.000136645 | 0.0183919 |
| MTI switching | 48.59 | 514.62 | 0.0329569 | 0.127359 | 0.210126 |
| Soft-sign | 41.51 | 27.19 | 0.0105769 | 0.0107255 | 0.0252809 |

Current error means phase-A current error, not a percentage.

| Variant | Final DC voltage (pu) | Final current envelope (pu) | Final filtered P (pu) | Final filtered Q (pu) |
| --- | ---: | ---: | ---: | ---: |
| Procedural PWM | 0.990261 | 0.712545 | 0.298113 | -0.00348635 |
| Trig-state model | 0.990547 | 0.820755 | 0.3007 | -0.00417742 |
| MTI switching | 0.999413 | 0.476649 | 0.326148 | -0.000236686 |
| Soft-sign | 0.98999 | 0.747535 | 0.298458 | -0.00374203 |

The current envelope is `sqrt(2/3 * (ia² + ib² + ic²))`. These are startup trajectories; numerical completion does not imply a settled response.

[Trajectories](grid_comparison.png) · [Error growth](grid_error_growth.png) · [Timing plot](timing_comparison.png) · [Method and run status](README.md)

Recreate this analysis from the repository root:

```sh
PYTHONPATH=src MPLBACKEND=Agg python src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/analyze_grid_comparison.py src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/benchmark_results/matched_pwm_models_10ms
```
