# Native control diagnostic, 10 ms

Corrected recording wrapper forwards forced-event queries and accepted-boundary commits explicitly. The production solver detects these APIs by inspecting the wrapper type/dir; generic __getattr__ forwarding alone was insufficient. The rejected first attempt is retained separately with a rejection notice.

This run initialized and converged through 10 ms. It reproduces the original saved instantaneous P and filtered Q exactly; capacitor voltage differs from the saved bus voltage only by about 3.04e-11 pu (the terminal resistance drop), and phase A current differs by at most 2.83e-7 pu. It recorded 1298 updates, including 297 updates away from the nominal 10 us grid. The recorder does not change the electrical or controller equations.

## Findings

- Current tracking is poor: at 6 ms, id reference/actual/filtered measurement are 0.465712/0.284744/0.321317 pu. At 10 ms they are 0.500795/0.710506/0.574919 pu. The outer DC loop requests more current as voltage falls, but delivered active current initially remains below reference and later overshoots.
- At 10 ms, iq reference/actual are 0.003007/0.053872 pu. Significant q-current tracking error exists despite the small reactive-current reference.
- Neither d nor q voltage command was clipped at any saved macro point: before/after-limit values agree to numerical precision. d voltage command peaked at 1.445285 pu, while the voltage ceiling stayed above 1.591080 pu. This rules out voltage-command saturation at those saved points, not at every unsaved Newton/internal substep.
- The PLL stays near nominal: omega is 313.868 to 315.278 rad/s. The current-loop measurement filter has a 1 ms time constant at 5 kHz; filtered feedback visibly lags actual currents.
- 26 live PWM updates leave interval_end <= current_time within 1e-12 s. Each is followed by the next update 10 us later. These occur at carrier boundaries including 2.0, 2.1, 2.2, 8.8, 8.9 and 9.1 ms. The interval descriptor uses floor((omega*t + phase + pi/2)/pi); rounding just below an integer selects the previous interval. This is a confirmed scheduling defect; its contribution to the physical trajectory is not isolated by this diagnostic.

No controller gains, production scheduling code, or plant equations were changed for this diagnostic. A controlled run correcting only the carrier-boundary selection is needed before assigning the entire voltage drop to it. Feedback lag and finite-step treatment of discontinuities remain additional candidates.

## Files

`current_and_voltage_commands.png`, `pwm_timing_zoom.png`, `diagnosis.json`, full continuous state/derivative trajectories and PWM update log in `controls.npz`, variable index mapping in `metadata.json`.

Reproduce with PYTHONPATH=src in the VeraGrid environment: `python src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/diagnose_native_controls.py --output <directory>`, followed by `python src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/plot_native_controls.py <directory>`.
