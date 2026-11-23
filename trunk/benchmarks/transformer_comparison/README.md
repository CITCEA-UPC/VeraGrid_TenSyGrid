# Transformer equivalence and coupling comparison

The tests do not identify transformer coupling as a source of self-excited
instability. They do not establish full-network IEEE118 stability.

## Input cases

The loaded IEEE39 `.gridcal` case contains **zero transformer devices**; its
branches are represented as lines. The coupled-core transformer template is
therefore absent from that particular benchmark and cannot explain its modes.
This statement concerns the loaded device representation, not the physical
New England system.

IEEE118 contains nine transformers, all grounded-star/grounded-star, with
zero specified shunt G/B, copper losses, core losses, open-circuit current and
phase shift. The attached coupled model defaults to **linear core enabled**;
saturation is disabled. Off-nominal taps are retained.

## Method and isolated results

`transformer_equivalence.py` differentiates the actual symbolic DAE equations,
including their multilinear auxiliary constraints, at a consistent zero-input
origin. For the default linear core, the physical terminal response is linear;
this is not a saturation study around rated flux.

It extracts the six-port frequency response from the descriptor system and
compares it with an independent RL/ideal-tap transformer satisfying
`If = (Vf/t - Vt)/(t*Z)`, `It = -t*If`. Both use the same static R, X and tap.
The deliberately restricted simple model rejects unsupported winding groups,
shunts, loss-test data or phase shifts rather than silently dropping them.

Across the nine IEEE118 transformers:

- Simple model versus static admittance: largest absolute entry error
  **7.11e-15**.
- Coupled model versus static admittance: largest absolute entry difference
  **5.80e-4**. The coupled model includes a weak magnetizing branch even with
  zero open-circuit current data because its constitutive regularization uses
  `sqrt(... + 1e-9)`. Leakage regularization also changes the impedance slightly.
  These are admittance entries, not percentages of network voltage.
- Changing return-path geometry by a factor of 1000 changes the balanced
  response by at most **4.32e-15**, while changing its zero-sequence response.
- The three core shunt susceptance eigenvalues are approximately
  **3.1624e-5, 3.1624e-5, 7.9059e-5 pu**: the coupling is inductive.
- The minimum eigenvalue of the Hermitian part of terminal admittance over
  0.1, 1 and 10 times the fundamental frequency is **-2.85e-14**, consistent
  with nonnegative absorbed power within numerical roundoff.
- Grounding both ports through positive 1 pu resistors gives **no positive
  finite descriptor pole** in either model. The least-damped coupled pole has
  real part **-0.0055609 /s**. This is a specific passive termination test,
  not a proof of stability with arbitrary network loads or controllers.

## Full-network trajectory comparison

Both IEEE118 runs use the same balanced and three-phase PF, dispatch,
generator/load/line models, StructuralAD solver, trapezoidal integration,
600 steps per 60 Hz period and one captured period. Only the nine attached
transformer models differ. No SSA modes are calculated in these capture-only
runs.

| Metric | Coupled core | Simple linear |
|---|---:|---:|
| Scaled one-period closure | 8.8152e-5 | 5.7421e-5 |
| Meets declared 1e-4 tolerance | Yes | Yes |
| Total wall time | 64.10 s | 95.79 s |
| Capture/Jacobian setup wall time | 32.15 s | 62.97 s |

These runs overlapped on the same machine; the times are recorded for
traceability and **are not a controlled performance comparison**.

The maximum absolute bus-voltage difference over all 118 buses, all three
phases and all 601 samples is **3.7958e-5 pu**. The maximum scaled difference
is **2.5562e-5**, at Breed phase A. This small one-period difference does not
exclude a long-term network instability or a sensitive mode.

## Additional mapping fix

The synthetic nonzero-resistance audit exposed an independent issue:
`xfmr_sc_resistance_pct` was derived solely from copper-loss test data, so
branch R could disappear when that data was absent. The multilinear template
now requests the existing static resistance-percentage mapping, whose fallback
preserves branch R. Tests cover taps 0.935, 1.0 and 1.069 with R=0.01 and
X=0.1. The nine IEEE118 transformers used above have R=0, so their results
are unaffected by this additional fix.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/pablo/myenv/bin/python \
  trunk/benchmarks/transformer_equivalence.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/pablo/myenv/bin/python \
  trunk/benchmarks/multilinear_ssa.py --grid ieee118 --formulation emt \
  --steps-per-period 600 --assessment-periods 1 --capture-only \
  --transformer-model simple_linear \
  --output trunk/benchmarks/transformer_comparison/simple_linear
```

Repeat the capture with `--transformer-model coupled_core` and its own output
directory. `terminal_audit.json` contains every device result;
`network_comparison.json` compares the saved bus-voltage trajectories.
Validation: **51 regression tests passed in 15.91 s**; one existing autograd/
NumPy deprecation warning. See `regression_tests.log`.
