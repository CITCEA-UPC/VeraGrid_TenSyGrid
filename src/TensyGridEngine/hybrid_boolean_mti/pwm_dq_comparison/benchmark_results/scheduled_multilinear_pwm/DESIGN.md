# Scheduled multilinear PWM prototype

This variant shares the native switched converter's electrical equations, controls, ports, initialization and carrier convention. It uses the EMT MTI Backward Euler solver for continuous integration, with zero switching Boolean unknowns and zero PWM inequalities. There is no integrated triangular carrier or tick counter.

## Event and update equations

At an accepted half-cycle boundary, sample the three modulation references. For a rising-direction bit `b` and half-period `Th`, calculate each interior phase edge:

```
s = 2*b - 1
edge_i = half_start + Th/2 * (1 + s*u_i)
next_event = min(half_end, pending phase edges)
```

The event selector operates on known timestamps. The electrical solver still respects its nominal maximum integration step; PWM edges shorten that step exactly, without introducing a separate PWM tick grid. It is piecewise, and is not claimed to be a globally multilinear map. It does not ask the electrical Newton solve to choose Boolean modes.

At an interior edge, the known phase indicator `A_i` selects the multilinear gate update:

```
q_i_next = (1-A_i)*q_i + A_i*(1-b)
```

At a normal half-cycle boundary, `R=1` reverses direction:

```
b_next = b + R - 2*b*R
```

The combined gate map implemented in `scheduled_pwm_gate_update` is:

```
q_i_next = (1-R)*((1-A_i)*q_i + A_i*(1-b)) + R*q_boundary_i
```

For interior references at a half-cycle start, `q_boundary_i=b_next`. Saturated references select their rail, without zero-width pulses. Initialization or a frequency change may begin mid-ramp, so the analytic clock determines the correct initial gate and direction there.

Incoming electrical intervals use the old gate values. An accepted endpoint consumes every coincident phase edge, or samples references for the next half-cycle. Rejected solver preparations do not consume edges. References remain held until the next sampling boundary.

## Implementation

- `ScheduledMultilinearPwmLogic` reuses the native event calendar and endpoint contract, with explicit multilinear gate updates and analytic crossing times.
- `get_switched_emt_converter_scheduled` injects this bridge into the shared converter stack; no global builder patching is used.
- The runtime codec preserves the variant using `multilinear_gate_updates=true` on the existing three-phase PWM port contract. The native implementation remains the default.
- The benchmark case is `grid_scheduled_StructuralCompiled`. Its actual integrator is explicitly recorded as `EmtMtiBackEulerSolver`; the final case-name component is retained for compatibility with the existing benchmark naming convention.

## Verification

Scheduler tests cover ascending/descending ramps, held inputs, mid-ramp starts, coincident edges, saturation, repeated updates, frequency enable, codec/remapping, and the multilinear map's truth table and affinity. A real EMT solver verifies exact pulse area with unaligned macro steps and narrow pulses. Full-grid initialization verifies a consistent PF seed and zero Boolean unknowns.

The first 2 ms comparison integrates the scheduled model in 16.111 s versus 35.200 s for the fresh procedural baseline. Maximum phase-A current difference is 0.00220981 pu and maximum DC-voltage difference is 1.69859e-5 pu. These are different continuous solver implementations; the scheduled event trace itself matches the native scheduler in the isolated event tests.

Diagnostic event counters are telemetry; they do not drive the clock or gates.

The baseline includes commit `8821788cc`'s accepted-endpoint PWM integration correction. Earlier benchmark results from before that correction are historical and do not provide a matched baseline for this prototype. The same numerical settings are used here: Backward Euler, nominal 10 us step, 5 kHz PWM, warmup off for StructuralCompiled and one BLAS/OMP thread. Grid times are single runs with setup separated from integration; integration includes first-use compilation.

See `run_10ms/` for the longer check and `tests.log` / `initialization_test.log` for regression results.

## Final 10 ms result

Both models reach 10 ms with 1,001 finite output points and pass initialization and convergence. Scheduled MTI uses zero switching Boolean unknowns and records exactly 297 phase edges and 99 half-cycle updates after enable. Integration takes 82.198 s versus 173.215 s for the fresh procedural baseline; setup takes 51.511 s versus 40.515 s. Maximum phase-A difference is 0.00220981 pu, DC-voltage difference 2.13621e-5 pu, and filtered-P difference 0.000120672 pu. The maximum phase-A difference is already present in the 2 ms comparison and does not increase over 10 ms. Both longer-run prefixes exactly reproduce the short runs.

The first long development attempt converged numerically but repeated clock-boundary updates (213 rather than 99). It was rejected. The fixed implementation snaps accepted near-boundary times to the scheduler's canonical timestamp before advancing the half-cycle. A regression test checks repeated updates and accumulated macro-clock roundoff over 100 half-cycles. The rejected trace is retained in `run_10ms/attempt_before_boundary_snap/`.

The final regression suite reports 23 passing checks, plus the full-grid initialization check. `verification.json` independently checks the saved waveform metrics, finite output, binary gate values, final times, event counts and prefix reproducibility.
