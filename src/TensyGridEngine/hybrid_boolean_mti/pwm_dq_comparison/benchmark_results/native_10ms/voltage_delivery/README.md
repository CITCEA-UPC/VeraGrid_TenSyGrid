# Native PWM voltage delivery

The unchanged baseline was rerun for 10 ms, recording all 1298 accepted initial/substep states. Its saved macro trajectory is bit-for-bit identical to the prior baseline full-state trajectory. No plant, controller, filter, PWM scheduling or integration rule was changed. The recorder now captures accepted states as well as scheduling updates.

## Carrier averages

Averages use actual substep durations over complete 200 us carrier periods. The applied dq voltage is the voltage entering the native plant after the startup blend: blend*v_conv_dq + (1-blend)*v_cmd_dq. Statistics exclude startup and cover carrier centers from 2 ms onward.

An independent check reconstructs phase bridge voltage from bus voltage, phase-current increments and the RL equation, using the benchmark R=0.02 pu, X=0.08 pu, f=50 Hz. The maximum difference from the right-endpoint voltage used by Backward Euler is 9.75e-8 pu. This confirms that the reported applied voltage reflects the simulated current dynamics, rather than just a plotting artifact. Name aliases for the two series branches agree within the solver tolerance.

| Axis | Applied-minus-command RMS (pu) | Left-held reconstruction-minus-command RMS (pu) |
| --- | ---: | ---: |
| d | 0.0877303 | 0.00579333 |
| q | 0.0524810 | 0.00396544 |

From 2 to 8.5 ms the mean applied d voltage exceeds command by 0.0624003 pu. A higher converter-side d voltage reduces incoming active current under this branch convention, consistent with the current deficit and capacitor discharge. From 8.5 to 10 ms that mean error becomes -0.0237771 pu, consistent with the late current rise. Carrier-period d error reaches 0.196988 pu in absolute value.

## Edge/integration interpretation

The native procedural scheduler updates retained gate modes at the target edge before the solver advances the continuous state to that edge. The Backward Euler residual consequently uses the new gate value over the entire preceding local interval. Even an exactly located edge can therefore add/remove up to one local step's worth of voltage-time area.

For a diagnostic reconstruction, the previous accepted gate is held over each recorded interval, with midpoint angle/DC scaling. This reconstructs the same recorded gate sequence on the same state trajectory; it is not an alternate simulation or proof of corrected closed-loop stability. Its much smaller delivery error strongly implicates one-sided switching-edge treatment. The remaining small error includes sampling/hold effects, frame rotation, DC scaling and midpoint quadrature.

The PWM reference already includes a nominal half-hold phase advance. To test a simple remaining command-lag explanation, commands were shifted by 0, 25, 50 and 100 us before averaging. d-axis applied-voltage RMS errors remain 0.08773, 0.08850, 0.08938 and 0.09083 pu respectively; those small delays do not explain the discrepancy. This is a sensitivity check, not an exact resimulation of held commands.

The separate carrier-half-index rounding defect remains present in this run; these results do not isolate its contribution. The edge-integration area error is a distinct mechanism. The next controlled change should advance to an edge with the old gates, then apply the new gates for the following interval and restore algebraic consistency before continuing. Gains should stay unchanged for that comparison.

## Reproduce

Use diagnose_native_controls.py with the default 5-period current filter and --output <directory>, then analyze_voltage_delivery.py <directory>, in the VeraGrid Python environment with PYTHONPATH=src and MPLBACKEND=Agg. The saved controls.npz contains both macro and accepted-substep trajectories. carrier_averages.npz contains the three plotted averages; voltage_delivery.json contains metrics. Two averaging tests verify irregular time weighting and clipping/short-pulse duty.
