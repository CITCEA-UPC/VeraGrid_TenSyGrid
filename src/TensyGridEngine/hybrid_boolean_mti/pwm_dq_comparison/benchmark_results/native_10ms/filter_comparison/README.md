# Current-feedback filter experiment

Only the current-feedback filter time constant was changed; gains, voltage/PLL filters, PWM scheduler, initialization and 10 us Backward Euler step were held fixed. The normal unsaturated controller signs and decoupling were independently checked against the phase-domain plant in test_native_current_control_signs.py (100 random operating points, one passing test). See controller_equation_review.md.

| Current filter | Completed time | Converged | Minimum Vdc (pu) | Final available Vdc (pu) | Maximum absolute filtered Q (pu) |
| --- | ---: | --- | ---: | ---: | ---: |
| 1000 us baseline | 10 ms | yes | 0.988221 | 0.990261 | 0.007022 |
| 500 us | 10 ms | yes | 0.988380 | 0.989655 | 0.007083 |
| 200 us | 8.98 ms | no | 0.989905 | 0.990071 | 0.007551 |

On the common 0.5 to 10 ms interval, changing 1000 to 500 us reduces raw d-current tracking RMSE from 0.127736 to 0.117886 pu (7.71%). q-current tracking RMSE is essentially unchanged (0.0372494 vs 0.0372404 pu). The final d current/reference is 0.604496/0.495156 pu versus baseline 0.710506/0.500795 pu. The early deficit and DC drop persist; peak filtered Q is not improved. The 500 us final filtered Q is closer to zero, but neither trace establishes steady state.

The 200 us simulation loses numerical convergence at the next step after the final published 8.98 ms sample; its final boundary attempt is 8.99 ms. Its states remain finite before failure. Do not compare its endpoint with a completed 10 ms run or interpret the solver failure as proof of physical instability. Its tracking metrics in filter_metrics.json use the same 0.5 to 8.98 ms horizon for both cases. The red vertical line marks termination in filter_comparison.png.

Conclusion: shortening current feedback changes damping/late overshoot but does not remove the principal DC-voltage deficit or Q excursion. The experiment does not support assigning the main disturbance solely to the 1 ms current filter. No production default or gain was changed. Next isolation should compare the carrier-averaged applied bridge dq voltage with the commanded voltage and assess event/integration timing before retuning gains.

## Artifacts and reproduction

- This folder: baseline vs 200 us plot and metrics; controller equation review.
- ../filter_comparison_500us: completed baseline vs 500 us comparison.
- ../current_filter_200us and ../current_filter_500us: full state and PWM boundary records, metadata and per-case diagnostic plots.
- ../controls: baseline full state and PWM boundary records.

Run diagnose_native_controls.py with --current-filter-periods 1 (200 us) or 2.5 (500 us) and separate --output directories, then compare_current_filters.py BASELINE EXPERIMENT OUTPUT. Use PYTHONPATH=src, MPLBACKEND=Agg, OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 in the VeraGrid Python environment. Time constants scale with the carrier frequency; these values refer to 5 kHz. The factory checks that only the three current-filter derivatives are modified.
