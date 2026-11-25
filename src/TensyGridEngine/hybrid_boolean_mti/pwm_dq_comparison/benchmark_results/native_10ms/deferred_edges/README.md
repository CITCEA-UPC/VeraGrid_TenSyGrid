# One-sided native PWM edge trial

Historical opt-in prototype from commit 416f8d741. The current diagnostic wrapper no longer accepts --defer-pwm-edges or changes gates/factors; production prepare/commit logic supersedes it. The measurements below describe that historical prototype, not the newer production result.

The trial completed 10 ms with initialization and convergence flags passing. Controller gains, 1 ms current filter, voltage and PLL filters, grid, 5 kHz carrier, startup handover, and 10 us Backward Euler step are unchanged. Production defaults were not modified: enable the experiment with diagnose_native_controls.py --defer-pwm-edges --output <directory>.

## Integration change

The native scheduler still prepares each target boundary and its next gate sequence. Its new gates are withheld from the continuous solve ending at that boundary, which retains the previous gates. After acceptance the prepared gates are published for the next interval. Continuous states are unchanged by publication. Algebraic outputs recorded exactly at edges are left limits; the next positive-time global DAE solve recomputes algebraics with the new modes. There is no artificial tiny-time state reset or zero-time global projection. The prototype is scoped to the native procedural PWM/Backward Euler diagnostic, not other solvers or FMU runtimes.

Newton factorization is invalidated before every local solve and after a gate change. This avoids reusing a factor assembled at a different local h or mode. The first attempt without invalidation stopped at 1.13 ms; invalidating only after gate changes extended it to 5.32 ms but still failed. Those failed trials are retained separately and are not stability results. The successful conservative factorization policy took 249.627 s integration versus 170.805 s for the recorded baseline; this is not a performance optimization.

The successful run records 298 deferred gate-change commits and 1298 accepted initial/substep states. The original carrier half-index rounding behavior remains present, so that separate defect is not repaired or isolated here.

## Results against the completed original 10 ms baseline

| Metric | Original ordering | Deferred ordering |
| --- | ---: | ---: |
| Minimum DC voltage (pu) | 0.9882205 | 0.9978528 |
| Final DC voltage (pu) | 0.9902613 | 0.9979292 |
| d-current tracking RMSE, 0.5–10 ms (pu) | 0.1277359 | 0.0206425 |
| q-current tracking RMSE, 0.5–10 ms (pu) | 0.0372494 | 0.0221118 |
| Maximum absolute filtered Q (pu) | 0.0070221 | 0.0032856 |
| Final filtered Q (pu) | -0.0034863 | -0.0017580 |
| Final id / reference (pu) | 0.710506 / 0.500795 | 0.473258 / 0.468410 |
| Carrier-average d-voltage tracking RMSE, 2–10 ms (pu) | 0.0877303 | 0.0075522 |
| Carrier-average q-voltage tracking RMSE, 2–10 ms (pu) | 0.0524810 | 0.0055385 |

The DC drop is much smaller and the large late current overshoot is removed, supporting the switching-edge voltage-time-area diagnosis. Residual DC/Q offsets remain. Ten milliseconds is a finite-horizon improvement, not an asymptotic stability proof. The corrected interval voltages independently match the phase RL current-increment balance within 9.75e-8 pu.

See ../deferred_edge_comparison/filter_comparison.png and filter_metrics.json. Here voltage_delivery.png / voltage_delivery.json show the carrier-voltage comparison; controls.npz contains macro and accepted-substep states plus scheduling rows. Five tests pass, covering exact pulse area with unaligned macro steps, deferred publication/factor invalidation, irregular averaging, and current-loop sign/decoupling/base checks.
