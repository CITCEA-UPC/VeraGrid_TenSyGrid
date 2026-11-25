# Production native PWM boundary handling

This is the final production run, not the opt-in prototype. The recorder only forwards prepare/resolve/commit and records outputs; it contains no gate override or factorization callback. The native scheduler consumes edges and samples modulation from the accepted endpoint state in the shared BlockProceduralLogicUpdater commit path. Interval preparation retains incoming gates. Rejected intervals do not consume transitions. Startup and direct standalone updates remain immediate.

All four EMT backends route Backward Euler through the shared prepare/commit API. StructuralCompiled compares both runtime parameters and local h against the retained factorization snapshot on every interval. Local h comparisons allow only tiny grid-rounding changes (relative 1e-6); parameter comparison uses the existing absolute 1e-14 threshold. This replaces the prototype's unconditional private invalidation callback. The carrier half-index is snapped only within four machine-epsilon-scaled ulps of an integer; both native PWM and held-modulation clocks use this helper.

The same changes were ported to /home/pablo/Desktop/eroots/VeraGrid, preserving its existing numerical helpers and unrelated staged work. That checkout's 10 ms case used the default problem-owned boundary updater, without an external wrapper. It initialized and converged; minimum DC voltage 0.99910564 pu, peak absolute filtered Q 0.00170048 pu. Those results are in its benchmarks/pwm_boundary_integration/native_10ms folder.

## TenSyGrid 10 ms validation

Unchanged grid, gains, filters, 5 kHz carrier and 10 us Backward Euler step. Initialization and convergence passed; initial residual 6.15e-9. Minimum DC voltage 0.99910565 pu (original 0.98822052), final 0.99917625 pu. Peak absolute filtered Q 0.00170113 pu (original 0.00702209), final -0.00081804 pu. d-current tracking RMSE over 0.5–10 ms is 0.01673869 pu (original 0.12773590). Final id/reference are 0.46624857/0.46536911 pu. Carrier-average d/q voltage tracking RMS errors over 2–10 ms are 0.00066701/0.00208456 pu. The independent phase RL voltage balance agrees within 9.75e-8 pu.

The run took 44.45 s setup and 173.12 s integration on this machine. These are individual-run timings, not a portable performance comparison with the upstream checkout, which has other compiler/runtime changes. Production_edges records the accepted-endpoint implementation before the clock snap; its measurements are intermediate results.

## Tests and scope

109 tests pass in this checkout; 136 pass upstream. The new native PWM workflow is tested through Symbolic, AD, StructuralVectorized and StructuralCompiled, dense and sparse, integrating narrow pulses over five carriers with unaligned 23 us macro steps. A rejection/retry test preserves the pending edge; 100 recursively computed half-period clocks always open the next interval. Existing alignment/rejection and procedural-logic regressions pass. Logs accompany this report.

The one-sided change is scoped to Backward Euler. Other methods retain their derivative-history behavior: trapezoidal requires a separate right-limit derivative restart before this convention can be extended safely. External/FMU providers without prepare_interval retain their existing transactional update/resolve API. Algebraic outputs exactly at edges use left limits; the following positive-time DAE solve resolves the new gates. No artificial tiny-time state reset is introduced. Ten milliseconds does not establish asymptotic stability.

Reproduce from the repository root with PYTHONPATH=src and the VeraGrid Python environment: diagnose_native_controls.py --output <directory>, then analyze_voltage_delivery.py <directory>. No edge-policy flag is required or available. See ../production_final_comparison/filter_comparison.png and filter_metrics.json for the original/final overlay. Baseline and prototype data are historical results from commits 9f2b52ca5 and 416f8d741 respectively; running the current source produces the corrected production path.
