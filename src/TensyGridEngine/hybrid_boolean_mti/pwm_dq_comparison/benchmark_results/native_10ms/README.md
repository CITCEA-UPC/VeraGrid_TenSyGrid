# Native procedural PWM diagnostics

The native ThreePhaseCarrierPwmLogic converter was run in the 100 MVA, three-bus AC/DC grid at 5 kHz PWM, 10 us Backward Euler steps and a 10 ms requested horizon. The Q reference is zero, Vdc reference 1 pu, and DC load 30 MW.

- [Baseline stability check](STABILITY.md): original completed 10 ms trajectory, DC voltage and Q response.
- [Internal control diagnostic](controls/README.md): current references, feedback, voltage limits and live PWM scheduling records; verified against the original trace.
- [Controller/filter experiment](filter_comparison/README.md): unchanged gains, only current feedback shortened to 500 or 200 us. The 500 us case completes 10 ms but does not remove the voltage deficit; the 200 us case fails numerical convergence after 8.98 ms. Defaults remain unchanged.
- [Final production result](production_edges_snapped_clock/README.md): the native PWM/Backward Euler prepare/commit correction is now in the shared logic and all four EMT backends, also ported to the sibling VeraGrid checkout. The diagnostic recorder no longer modifies gates or factors. Results are in production_edges_snapped_clock and production_final_comparison. Other integration methods keep their prior derivative-history behavior.
- [One-sided edge trial](deferred_edges/README.md): completed 10 ms with unchanged gains/filters; substantially smaller DC drop and current-tracking error. Conservative fresh factorization is needed; production defaults remain unchanged.
- [Controller equation review](filter_comparison/controller_equation_review.md): normal unsaturated feedback signs, dq decoupling and power-base scaling checked against the phase-domain equations.

Full numerical state trajectories and event records are retained in each case folder. Rejected first-attempt recording artifacts are excluded from version control and do not support these findings. Timings are individual machine-dependent runs; no stability claim follows solely from numerical convergence.
