# EMT periodicity corrections and validation

> **Performance update (23 November 2026):** Sparse Jacobian assembly and a smaller mode search reduce the corrected IEEE118 total from 634.26 s to **104.72 s**, and IEEE39 from 81.09 s to **26.97 s**. Both retain 600 steps per period and the same trajectories. See [performance results](../performance/README.md).


The earlier closure diagnostic missed the multilinear generator states named
`theta_generator_*`. It therefore counted their normal 2π advance as drift.
The corrected diagnostic declares the absolute angle indices explicitly,
checks that the captured time span is one requested period, subtracts one
turn only from those indices, and measures
`max(abs(end - start - winding) / max(1, max(abs(trajectory))))`.
The normalization is a numerical scaling convention, not a percentage of
rated current or voltage.

The original saved one-period captures have corrected errors 0.00143502
(IEEE39) and 0.04185804 (IEEE118). The previously reported 1.17 and 1.45
were diagnostic errors and are superseded.

## Transformer correction

IEEE118's remaining error was traced to off-nominal transformer leakage
scaling in `xfmr_emt_multilinear_template.py`. The standard PF equivalent is
`If = (Vf/t - Vt)/(t*Z)`. This template instead uses winding voltage
`Vf - t*Vt` and current `If`; its leakage impedance must consequently be
`t**2 * Z`. The missing factor affected both resistance and inductance.
The correction reduces the largest initial fundamental derivative mismatch
from about 174 to 0.025. The new tests check the actual template dynamics
against an independent PF phasor formula for taps 0.935, 1.0 and 1.069.

## One-period capture checks

All captures start from the PF-seeded explicit initialization. IEEE39 uses
50 Hz as stored in the input case; IEEE118 uses 60 Hz. These are one-period
captures, not demonstrations that a long transient settles to steady state.

| Model | Steps per period | Scaled closure error |
|---|---:|---:|
| IEEE39 | 120 | 0.00143502 |
| IEEE39 | 600 | 0.0000574313 |
| IEEE39 | 1200 | 0.0000145637 |
| IEEE118 before tap correction | 600 | 0.0438461 |
| IEEE118 before tap correction | 1200 | 0.0438605 |
| IEEE118 after tap correction | 600 | 0.0000881518 |
| IEEE118 after tap correction | 1200 | 0.0000822701 |

IEEE39 shows approximately second-order closure convergence with step
refinement. IEEE118's remaining error largely plateaus; it meets the chosen
1e-4 benchmark tolerance, but this does not establish a more accurate periodic
orbit. Initialization/regularization error remains a limitation, especially
for modes near the stability boundary.

The benchmark now stops before SSA if the closure exceeds
`--periodicity-tolerance` (default 1e-4). `--allow-nonperiodic` explicitly
permits diagnostic mode calculations and preserves the false validation flag.
The capture code also rounds an almost-integer number of steps per period
instead of truncating it; genuinely noninteger periods are rejected.

## Mode calculations

The corrected IEEE39 600-step run returned six modes in 81.09 s total
(73.79 s SSA). Its dominant real growth rate is +243.043 /s, compared with
+243.402 /s at 120 steps, a 0.15% change. The multiplier magnitude is 129.135.
This supports instability of this particular benchmark model and operating
point. The benchmark uses generic machine data and held PF-seeded inputs;
it does not establish instability of every dynamic realization of IEEE39.
A five-period free transient also grows and eventually fails integration;
such a transient is unsuitable for finding an unstable periodic orbit.

The corrected IEEE118 calculation was interrupted during SSA before returning
modes; its completed closure check and partial timings are under `ieee118_600/`.
There is no completed corrected IEEE118 SSA timing or stability conclusion.
Earlier IEEE118 modes used the faulty transformer scaling and must not be used
as the final stability conclusion.

Run commands:

```sh
/home/pablo/myenv/bin/python trunk/benchmarks/multilinear_ssa.py \
  --grid ieee39 --formulation emt --steps-per-period 600 \
  --assessment-periods 1 --modes 6 \
  --output trunk/benchmarks/periodicity_validation/ieee39_600
/home/pablo/myenv/bin/python trunk/benchmarks/multilinear_ssa.py \
  --grid ieee118 --formulation emt --steps-per-period 600 \
  --assessment-periods 1 --modes 6 \
  --output trunk/benchmarks/periodicity_validation/ieee118_600
```

Validation: 47 regression tests passed in 17.23 s, including generator
operating points, transformer terminal attachment and tap physics, period
closure, full-period capture under floating-point roundoff, sparse consistency
reduction and backend snapshots. See `regression_tests.log`.
