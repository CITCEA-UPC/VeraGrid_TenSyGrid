# Converter comparison over 2 ms

The procedural PWM trajectory is the reference. Errors below are maximum absolute differences up to each stated time, using interpolation onto the reference timestamps. Integration times exclude model/solver setup and include first-use compilation.

| Variant | Setup (s) | Integration (s) | Current error at 2 ms (pu) | At 5 ms (pu) | At 10 ms (pu) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Procedural PWM | 42.88 | 35.20 | 0 | — | — |
| Scheduled MTI | 49.00 | 16.11 | 0.00220981 | — | — |

Current error means phase-A current error, not a percentage.

| Variant | Final DC voltage (pu) | Final current envelope (pu) | Final filtered P (pu) | Final filtered Q (pu) |
| --- | ---: | ---: | ---: | ---: |
| Procedural PWM | 0.999808 | 0.464246 | 0.325874 | 0.000256371 |
| Scheduled MTI | 0.999791 | 0.463708 | 0.325764 | 0.000166891 |

The current envelope is `sqrt(2/3 * (ia² + ib² + ic²))`. These are startup trajectories; numerical completion does not imply a settled response.

[Trajectories](grid_comparison.png) · [Error growth](grid_error_growth.png) · [Timing plot](timing_comparison.png) · [Method and run status](README.md)

Recreate this analysis from the repository root:

```sh
PYTHONPATH=src MPLBACKEND=Agg python src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/analyze_grid_comparison.py src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/benchmark_results/scheduled_multilinear_pwm
```
