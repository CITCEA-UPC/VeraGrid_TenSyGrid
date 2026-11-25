# Converter comparison over 10 ms

The procedural PWM trajectory is the reference. Errors below are maximum absolute differences up to each stated time, using interpolation onto the reference timestamps. Integration times exclude model/solver setup and include first-use compilation.

| Variant | Setup (s) | Integration (s) | Current error at 2 ms (pu) | At 5 ms (pu) | At 10 ms (pu) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Procedural PWM | 40.52 | 173.21 | 0 | 0 | 0 |
| Scheduled MTI | 51.51 | 82.20 | 0.00220981 | 0.00220981 | 0.00220981 |

Current error means phase-A current error, not a percentage.

| Variant | Final DC voltage (pu) | Final current envelope (pu) | Final filtered P (pu) | Final filtered Q (pu) |
| --- | ---: | ---: | ---: | ---: |
| Procedural PWM | 0.999176 | 0.466249 | 0.324908 | -0.000818042 |
| Scheduled MTI | 0.999181 | 0.46615 | 0.324951 | -0.000865998 |

The current envelope is `sqrt(2/3 * (ia² + ib² + ic²))`. These are startup trajectories; numerical completion does not imply a settled response.

[Trajectories](grid_comparison.png) · [Error growth](grid_error_growth.png) · [Timing plot](timing_comparison.png) · [Method and run status](README.md)

Recreate this analysis from the repository root:

```sh
PYTHONPATH=src MPLBACKEND=Agg python src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/analyze_grid_comparison.py src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/benchmark_results/scheduled_multilinear_pwm/run_10ms
```
