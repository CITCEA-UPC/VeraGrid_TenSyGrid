# Kundur RMS tuning case

`stable_rms_grid.veragrid` is a copy of
`Grids_and_profiles/grids/kundur_rms_sss.veragrid`.

`KUNDUR_CONTROL_RANGES` in `rms_tuning.py` selects the allowed controls and
their bounds. `build_rms_tunables()` discovers the matching values and device
IDs from the loaded grid. The four generators (Gen1–Gen4) each expose:

| Parameter | Model meaning | Initial value | Allowed range |
| --- | --- | ---: | ---: |
| `Kp` | Proportional speed control gain | 0 | 0–150 |
| `Ki` | Integral speed control gain | 0 | 0–10 |

The ranges are initial search limits for the hackathon. They do not certify
stability; candidate settings still need RMS or small-signal evaluation.
The tuner selects a `TunableParameter` from `tunable_grid.tunable_parameters`
and returns `RmsTuning({parameter: 20.0})`. For numeric tuning,
`tunable_grid.control_tensor` is a read-only NumPy tensor with one row per
parameter and columns `[initial_value, minimum, maximum]`. A tuner can compute
a vector of candidate values and pair each value with the parameter at the
same row index. The evaluator still owns the full grid for simulation.
It can also select existing lines
from `openable_lines`, a new corridor from `line_candidates`, or shunt
placements from `shunt_options`. Generator setpoints are listed in
`dispatch_options`; use `dispatch_mw={option: new_mw}` to move generation
between units. Each value must stay within its generator limits and within
10% of its starting value, and total scheduled generation must remain fixed.
`BasicRmsTuner` proposes no changes. Machine parameters `M`, `D`, `R1`, and
`X1` are fixed.

`TunableGrid` groups an initialized `RmsProblem` with its permitted actions.
Its default limits are two opened lines, one new line, one new shunt, and ten
candidate tests per tuner run. `RmsTuning.is_accepted()` checks these limits.
Inside `tune()`, participants may call `tunable_grid.test_tuning(proposal)` to
compare proposals. Each call consumes one test; exceeding the budget raises
`RuntimeError`. The final proposal is evaluated separately. `apply_tuning()`
returns an RMS problem rebuilt from the modified grid. All proposal evaluations
use copies, so they start from the same baseline.

The Kundur case offers all 14 existing lines for opening, one candidate line
from Bus7 to Bus9, and shunt placements at Bus7 or Bus9. Each candidate shunt
has `G=1.0 MW` and `B=10 MVAr`; its real conductance consumption is included
in `active_losses_mw`. These are initial challenge choices and can be changed
through the evaluator's configuration.

`active_losses_mw` currently measures branch losses plus shunt conductance
consumption. Generator internal losses would also count in a total-loss
objective, but the supplied Kundur RMS generator template sets stator
resistance `R1=0`, so it models no stator copper loss. Reactances alone do
not dissipate real power. A grid with nonzero generator resistance needs an
explicit generator-loss calculation before this metric can be called total
system losses.

For another RMS grid, define its control names and bounds and pass them to
`RmsTunerEvaluator(control_ranges=...)`. The evaluator discovers matching
parameters from that grid at runtime.

`RmsTunerEvaluator.evaluate_tuning(grid, tuner)` applies dispatch, topology, and shunt
choices before power flow, then applies controller values before building the
final RMS problem and running SSA. It returns active losses,
the largest eigenvalue real part, minimum damping for modes above 0.05 Hz,
voltage violations outside 0.95–1.05 p.u., thermal violations, and a stability
flag using the brief's `-1e-5` real-part and 5% damping thresholds. It also
runs a three-second RMS simulation with a 2% increase in the first load at
0.2 seconds, reporting peak generator frequency and bus voltage deviations
from their initial values. A failed transient marks the tuning unstable.
The peak deviations measure excursion size, not settled-value overshoot.
Scenario robustness and code-quality scores require separate evaluation.

Run `python -m hackaton_files.control_response_experiment` with `PYTHONPATH=src:.`
to compare baseline, increased `Kp`, and increased `Ki` on the Kundur grid.
In the current case, raising `Kp` from 0 to 20 reduces minimum damping from
5.43% to 4.23% and increases peak frequency deviation from 0.0246 to
0.0292 Hz. Raising `Ki` to 5 reduces that three-second peak to 0.0235 Hz,
but introduces a positive-real-part SSA mode. Thus these settings do not
support the hypothesis that stronger gains improve stability at the cost of
larger excursions.

Participants can subclass `RmsTuner`, implement `tune(tunable_grid)`, and call
`test_tuning(my_tuner, grid)` to run that single-grid evaluation themselves.
`my_tuner.test_tuning(grid)` is an equivalent convenience method.

## Grid pool and scrambling

`grids/kundur_rms_60.veragrid`, `kundur_rms_65.veragrid`, and
`kundur_rms_70.veragrid` are three verified operating points derived from the
same Kundur RMS network. Each passed power flow and SSA with no voltage or
thermal violations. SSA includes one neutral reference-angle mode; stability
is assessed on the remaining physical modes.

`scramble_grid(seed)` selects one of these files and varies loads, generation,
line impedances. It also permutes the bus, line,
generator, and load ordering while preserving every bus connection and line
status.
`scramble_dispatch(grid, seed)` is available separately when only demand and
generation setpoints should change; it returns a copy and preserves the
original grid.
It also varies the available line corridors, shunt locations, and action
limits, plus the available control types and number of candidate tests. The
function retries and returns only a case whose baseline passes
power flow, voltage, thermal, and RMS small-signal checks. The same seed gives
the same case. For example:

```python
case = scramble_grid(17)
metrics = test_tuning(MyTuner(), case.grid, case.evaluator)
```

These cases share one underlying network. Hidden seeds and varied operating
conditions reduce the usefulness of memorizing one answer, but do not prevent
all strategies from transferring between similar cases.

## Additional verified RMS grids

`ADDITIONAL_STABLE_GRID_FILES` contains IEEE9, IEEE30, and IEEE39 snapshots,
including two IEEE9 inverter variants.
`build_additional_grid()` regenerates them from the repository's source grids;
`verify_additional_grid()` requires converged power flow, stable physical SSA
modes, no voltage violations, and no thermal violations. The saved `.veragrid`
files were reloaded and verified after serialization.

The IEEE9 cases have three generator step-up branches represented by actual
transformer devices, plus line and constant-power load RMS models. The
`ieee9_rms_gfl.veragrid` variant replaces one synchronous generator model
with the dynamic PVD1 grid-following inverter model. The
`ieee9_rms_mixed_ibr.veragrid` variant also replaces a second generator
model with an ideal grid-forming voltage source. This source holds terminal
voltage and angle but has no converter inner-loop states; it is an idealized
grid-forming limit, not a detailed GFM converter.
IEEE30 has six generator RMS models and 21 loads; its demand and generation
are at 90% of the source case to avoid an overloaded branch. IEEE39 has ten
generator RMS models and 21 loads; its demand and generation are at 95% and
generator voltage targets have been adjusted to meet the voltage limit.
The synchronous model construction uses the same `genrow1` generator family
as Kundur, with fixed `D=100` on IEEE30 and IEEE39. The IEEE9 inverter cases
passed power flow, SSA, and a load-step transient after saving and reloading.
The existing detailed STAMP GFOR/GFOL example was assessed but remains
unstable in this evaluator, so it is not in the stable pool.

Use `additional_grid_evaluator()` for these grids. It supplies controller,
PVD1 droop and response-time, and dispatch tuning without Kundur-specific line
corridors or shunt locations. `ALL_STABLE_GRID_FILES` lists all five saved IEEE
cases alongside the Kundur pool.

## Unstable control case

`grids/kundur_rms_unstable.veragrid` is a fourth Kundur operating point made
from the verified 65% dispatch case. `make_unstable_grid(seed=2026)` randomizes
each generator's `Kp` and `Ki` and accepts a case only when power flow
passes but SSA finds an unstable physical mode. This saved case has a positive
real eigenvalue near `0.234 s⁻¹`. Its evaluator uses
`UNSTABLE_CONTROL_RANGES`, which permits participants to reduce the excessive
controller gain without changing machine properties.

For this grid, generator RMS models expose `X1`, rather than separate `Xd` and
`Xq`. These internal reactances affect the dynamic equations and modes; the
reported static power-flow losses at fixed dispatch come from network flows
and shunt conductance, not directly from `X1`.
