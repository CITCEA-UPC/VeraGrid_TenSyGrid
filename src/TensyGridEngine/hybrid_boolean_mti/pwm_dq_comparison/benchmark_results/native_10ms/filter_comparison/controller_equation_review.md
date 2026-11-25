# Native current-loop equation review

The unsaturated controller is consistent with the phase-domain plant. With the sine / minus-cosine Park convention:

    di_d/dt = omega_base/L * (v_d - v_conv_d - R*i_d) - omega_pll*i_q
    di_q/dt = omega_base/L * (v_q - v_conv_q - R*i_q) + omega_pll*i_d

The controller commands are:

    v_cmd_d = v_d - R*i_d - (omega_pll/omega_base)*L*i_q - PI_d
    v_cmd_q = v_q - R*i_q + (omega_pll/omega_base)*L*i_d - PI_q

With ideal instantaneous measurement and actuation these give di_d/dt = omega_base/L*PI_d and di_q/dt = omega_base/L*PI_q. Positive reference-minus-current error therefore drives the current upward. The code's L is per-unit reactance; omega_base/L supplies the time-domain conversion. Independent numerical differentiation of the rotating Park transform, compared with the actual symbolic command equations for 100 random operating points, passed in test_native_current_control_signs.py.

For balanced phase-peak variables on the system base, total power is sum(v_phase*i_phase)/3 = 0.5*(v_d*i_d+v_q*i_q). This agrees with the full converter's factor 0.5 and id feedforward factor 2. With v_d positive, increasing iq increases Q under the code's convention. The DC controller increases id reference when Vdc falls.

These checks establish the unsaturated algebraic signs and base scaling, not closed-loop stability with filtered measurements, sampled actuation or the connected grid. Anti-windup behavior under saturation was not exercised by these runs. Existing native diagnostics found no d/q voltage-command clipping at saved macro points.

## Controlled filter experiments

Only the state derivatives of i_d_meas_f, i_q_meas_f and i_0_meas_f are scaled. The 5-period baseline has a 1 ms time constant at 5 kHz. One period gives 200 us; 2.5 periods gives 500 us. Voltage and PLL filters, current and outer-loop gains, carrier phase, handover, grid, initialization and integration step remain unchanged. The experiment modifies the constructed model only; no production default was changed.

The known PWM half-period rounding defect is intentionally present in all runs. Thus these experiments isolate sensitivity to the current-feedback filter but do not establish behavior after a scheduler correction. A failed simulation is not evidence by itself of physical instability. See the saved run metadata for convergence and final simulated time before comparing trajectories.
