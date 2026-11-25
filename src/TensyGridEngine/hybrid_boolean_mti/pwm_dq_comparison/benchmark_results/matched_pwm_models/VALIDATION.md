# Validation notes

The final 10 us, 2 ms grid traces all pass initialization, convergence, finite-value and final-time checks. Initial algebraic residual: 6.150067e-9 for all four models.

Focused checks:
- Angle tracking, PWM phase advance, shared soft bridge equations and comparator law: 7 passed (`angle_and_soft_sign_regressions.log`).
- Variant wiring/name checks plus the earlier six model checks: 10 passed (`model_regressions.log`).
- Full-grid initialization and explicit initializer regressions: 18 passed with one test lookup failure in the first combined run (`initialization_test_first_attempt.log`). The failed lookup used an algebraic-variable finder for an event parameter; the corrected soft-sign test passes (`soft_sign_clock_regression.log`). It verifies the carrier clock in both interpreted and compiled runtime evaluation.

The MTI process crashed once during setup and passed on a separate retry; the cause of that crash is undetermined. See `attempt_history.json` and `mti_first_attempt_crash.log`. The soft-sign development run with a frozen carrier was rejected and replaced by the corrected trace.

The 5 us run is a separate refinement attempt. Its procedural reference stops at 1.215 ms, so it cannot establish full-horizon convergence or support an error comparison against that reference. Consult its per-case status files rather than treating solver completion flags from other variants as an equivalence claim.

Timings are single grid runs under variable machine load. Setup is separate from integration. MTI uses 12 event bisections, whereas the earlier initialization-only comparison used the default two; the timing difference includes that accuracy change. Soft-sign uses lambda=1e-6; smooth and hard comparators differ at finite lambda. The standalone scheduled-MTI adapter is unchanged and remains distinct from the full-grid MTI model.
