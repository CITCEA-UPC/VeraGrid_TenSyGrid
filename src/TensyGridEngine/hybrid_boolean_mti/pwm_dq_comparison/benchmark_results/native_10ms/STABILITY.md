# Native procedural PWM: 10 ms check

Native `ThreePhaseCarrierPwmLogic` in the corrected switched converter, connected to the shared three-bus AC/DC grid. No MTI or soft-sign switching is used. Q reference is zero, DC-voltage reference is 1 pu, DC load is 30 MW on a 100 MVA system base. PWM is 5 kHz, handover is 0.1 ms, Backward Euler nominal step is 10 microseconds.

The simulation reached 10 ms with 1001 saved points. Initialization and convergence flags passed; initial algebraic residual was 6.150067e-9. Setup took 53.532 s, integration 211.084 s on this run.

Filtered Q reached -0.0070221 pu (-0.70221 MVAr) and ended at -0.00348635 pu (-0.348635 MVAr). Its final 1 ms mean was -0.00458927 pu. Instantaneous Q ranged from -0.317915 to +0.377045 pu and contains substantial switching ripple. These are different quantities; filtered Q has not settled to its zero reference.

DC voltage ranged from 0.988221 to 1.000099 pu and ended at 0.990261 pu. The three-phase current envelope started at 0.463509 pu, reached 0.728897 pu, and ended at 0.712545 pu, with a pronounced rise near 9 ms.

This confirms numerical completion, not asymptotic stability. The response remains transient over 10 ms; neither the Q response nor the late current rise justifies calling it settled. No Q-reference step was applied, so this is not a commanded-Q step-response test.

See `native_reactive_power.png`, `native_stability.png`, `stability_metrics.json`, and the saved NPZ trace. Reproduce using `benchmark_solvers.py --grid-duration-ms 10 --grid-step-us 10 --cases grid_standard_StructuralCompiled --timeout 600 --output <this-directory>`, then `analyze_native_stability.py <this-directory>`, with PYTHONPATH=src and the VeraGrid Python environment. The source includes the pending converter initialization port; this is not an unmodified upstream checkout.
