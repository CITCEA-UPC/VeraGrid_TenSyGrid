# Basic RMS tuning set

**60 `.veragrid` cases on ten AC networks**: 50 initially dynamically unstable
controller-tuning cases and 10 fixed-controller network-action cases. The original
collection has five controller, operating-point and topology variants per network. `initialize_grid()` exposes the
existing `TunableGrid` interface, with case-specific evaluation budgets.

| Network | Buses | Controller variants |
| --- | ---: | ---: |
| Lynn | 5 | 5 |
| Six-bus ACOPF | 6 | 5 |
| IEEE 9 | 9 | 5 |
| Kundur two-area | 11 | 5 |
| IEEE 14 | 14 | 5 |
| IEEE RTS | 24 | 5 |
| IEEE 30 | 30 | 5 |
| IEEE 39 | 39 | 5 |
| IEEE 57 | 57 | 5 |
| IEEE 118 | 118 | 5 |

Each family contains:

| File suffix | Dynamic setup | Tunable controls |
| --- | --- | --- |
| `_01` | Synchronous machines | Genrow speed-feedback `Kp` |
| `_02` | Synchronous + GFL | Current PI, power PI, PLL PI |
| `_03` | Synchronous + GFM | Voltage/current PI, P/f and Q/V droop, measurement filters |
| `_04` | Synchronous + GFL + GFM | Both converter catalogs, lower converter share |
| `_05` | Synchronous + GFL + GFM | Both converter catalogs, higher converter share |

Thus **40 cases contain converters**, including 20 mixed GFM/GFL cases. In
converter cases, synchronous controls start stable and are not offered for
tuning. Validation requires converter participation in a growing baseline mode;
the converter models are part of the instability, not decorative additions.

## Use a case

Use an interpreter configured to import **this checkout's** `src` and repository
root, as for the parent hackathon examples. Verify `VeraGridEngine.__file__` if
multiple checkouts are installed. From the repository root:

```python
from VeraGridEngine.api import open_file
from hackathon_files.basic import case_files, initialize_grid, reference_tuning

path = case_files()[1]  # First GFL exercise
case = initialize_grid(open_file(str(path)))

# Implement your tuning algorithm using this case's own handles and bounds.
# The supplied reference is an example proving stabilization is possible.
proposal = reference_tuning(case)
metrics = case.test_tuning(proposal)
print(metrics)
```

The basic set uses a 2.5% minimum damping ratio. The shared evaluator now accepts
`minimum_damping_ratio`; its default remains 5% for existing exercises.

The ten `_02` cases (one per network) are **controller-only exercises**: no
redispatch, line opening/closing, reinforcement, or shunt actions are permitted.
Their physical-action catalogs are empty and their action budgets are zero.
Existing equipment stays in the grid, but contestants cannot switch or add it.
The other 40 cases expose controller parameters **and physical grid actions**:

| Catalog / proposal field | Meaning | Basic budget |
| --- | --- | ---: |
| `openable_lines` / `open_lines` | Open an existing active, non-bridge line | 1 |
| `closable_lines` / `close_lines` | Close an existing out-of-service circuit | 1 |
| `line_candidates` / `add_lines` | Build a new reinforcement circuit | 1 |
| `shunt_options` / `add_shunts` | Add a capacitor or reactor at a load bus | 1 |
| `dispatch_options` / `dispatch_mw` | Redistribute scheduled MW within bounds | Balanced total |

Each saved case contains an existing open line; in controller-only `_02` cases
it stays fixed and is not offered as a closing action. The `_01` variants use a real
screened line outage where a connected, voltage/thermal-feasible operating point
can be found. Other variants include a normally-open standby circuit on an
existing corridor. If no suitable outage is found, `_01` also uses a standby.
The reference closes a line named `Restore outage: ...` when present, as well as
retuning controllers. Converter-case references leave optional standbys open.
The guarantee is a successful **allowed-action combination**, rather than a
parameter-only solution for every outage topology.

Opening catalogs omit bridges of the initial active network. A radial outage
case can therefore have no opening options. Closing permissions apply to existing
physical circuits; a line candidate is a distinct new circuit. The same line
cannot be both opened and closed in one proposal. Each action type has its own
budget, so a permitted circuit transfer can contain one opening and one closing.

Redispatch intervals use the generator's existing P limits and a half-width of
`max(5% * abs(initial MW), 0.5% * Sbase)`. This also allows a small positive
schedule for a zero-scheduled slack generator. Units already outside their
static P limits are not offered. At least two usable options are required.
All changes are atomic and must preserve the total scheduled MW.

The reinforcement candidate follows an existing physical corridor, with its
R/X/B and rating. Shunt candidates have zero conductance and susceptance
`±2% * Sbase` MVAr. These are permitted alternatives, not a promise that every
combination will pass: each proposal is rechecked with PF, RMS, SSA, voltage,
thermal, and transient criteria.

### Example physical proposals

```python
from hackathon_files import RmsTuning

reference = reference_tuning(case)

# Close a real existing line; retain any outage restoration required by the reference.
closing = reference.close_lines or (case.closable_lines[0],)
proposal = RmsTuning(parameters=reference.parameters, close_lines=closing)
metrics = case.test_tuning(proposal)

# Add a new physical circuit, with an RMS model as well as its PF data.
proposal = RmsTuning(parameters=reference.parameters,
                     close_lines=reference.close_lines,
                     add_lines=(case.line_candidates[0],))

# Or add a capacitor/reactor using the actual case-local handle.
proposal = RmsTuning(parameters=reference.parameters,
                     close_lines=reference.close_lines,
                     add_shunts=(case.shunt_options[1],))
```

Opening uses `open_lines=(case.openable_lines[0],)` when that catalog is nonempty.
Never reuse another scenario's handles. Only calls to `case.test_tuning()` spend
trials; creating proposals does not. New circuits and shunts receive native RMS
models. Opening/closing synchronizes PF active state and the native RMS status,
and the evaluator retains the exact organizer catalogs on its copied candidate.

## Controllers and bounds

The GFM wrapper uses the existing `build_gfm_converter_model()` core, including
its P/f and Q/V droop, voltage/current PI loops, measurement filters, electrical
filter dynamics, and limiters. The GFL wrapper uses the existing
`build_gfl_converter_model()` in P/Q control mode, including its PLL, outer
power PI loops, inner current PI loops, current limits, and electrical dynamics.
Neither is an ideal voltage source or a static P/Q injection. The DC side is
represented as a stiff supply; no explicit DC network is added.

| Parameter | Family | Inclusive bounds | Reference value |
| --- | --- | --- | ---: |
| `Kp` | Genrow, `_01` only | -1000 to 300 | -400 |
| `Kp_icl` | GFM/GFL | 0.001 to 5 | 0.5 (six-bus: 3) |
| `Ki_icl` | GFM/GFL | 1 to 500 | 5 |
| `Kp_pol`, `Ki_pol` | GFL | 0.01 to 1; 0.01 to 5 | 0.2; 0.2 |
| `Kp_pll`, `Ki_pll` | GFL | 0.001 to 0.5; 0.01 to 1 | 0.05; 0.1 |
| `Kdp`, `Kdq` | GFM | 0.001 to 1 | 0.05 |
| `tau_P`, `tau_Q` | GFM | 0.001 to 1 s | 0.01 s |

Converter current PI gains initially have low proportional action (`0.001`)
and aggressive integral action (`500`), producing growing converter modes.
The reference reduces integral gain and increases proportional action. Other
converter parameters remain available for exploring damping and interactions;
the reference need not change every offered handle. Each saved file contains
the unstable parameters; references are separate.

Genrow uses `tm = tm0 + Kp*(omega-omega_ref) + Ki*et`. Its signed `Kp` therefore
needs negative feedback for the synchronous exercise. `Ki` stays zero.

## Acceptance and evidence

For every saved case, the builder reopens the file and checks:

- Converged AC power flow and stationary differential/algebraic residuals at
  most `1e-6` before tuning.
- The complete finite baseline spectrum, with a real part above `+1e-4`.
- For converter cases, at least one growing mode with at least 10% of its
  participation assigned to converter-owned variables.
- An allowed reference using the current case's parameter and restoration handles.
- Tuned physical-mode real parts below `-1e-5`, at most one neutral mode, and
  at least 2.5% damping for oscillatory modes above 0.05 Hz.
- Voltages within 0.95–1.05 pu and branch loading at most 100%, using the source
  network's existing ratings.
- A converged 3 s Backward Euler simulation with a 0.02 s step and a 2% increase
  in the first load's active power at 0.2 s.
- A separately initialized tuned equilibrium with residuals at most `1e-6`.

Converter models use descriptor equations. Infinite eigenvalues represent
algebraic constraints and are recorded separately, not treated as unstable
physical modes. Every finite mode is considered; NaNs are rejected.

`validation.json` stores baseline/tuned measurements, converter ratings,
controller and physical-action catalogs, initial open circuits, exact reference
values/restoration actions, and SHA-256 hashes of saved grids
and source files. Rebuild and validate with:

```bash
python -m hackathon_files.basic.build
```

Repeat the saved-grid checks and public trial-session regressions with:

```bash
python -m pytest -q hackathon_files/basic/test_basic.py hackathon_files/basic/test_actions.py
```

These are public basic exercises with a known reference, not a hidden competition
suite. Checks establish local small-signal stability and convergence for the
specified load step, not arbitrary-disturbance stability or protection behavior.

## Provenance and fixed preparation

These are synthesized educational RMS scenarios on repository AC benchmarks,
not validated reproductions of those networks' original dynamic machine data.
`network_specs()` identifies each exact local source and preparation setting.
No source files or engine implementation files are edited.

All synchronous machines use Genrow, with resistance zero and system-base
reactance 1/30 pu. Inertia, damping, dispatch, and demand vary reproducibly.
Loads use constant-power RMS models, shunts constant-admittance models, and
branches the existing RMS templates. Original offline branches and generators
are omitted during base preparation; scenario outages and standby circuits are
then explicitly retained, including their inactive RMS status.
Missing nominal bus voltages receive a 110 kV base. The six-bus source gets a
modeled generator at its otherwise unmodeled slack bus.

Source generation/demand factors are 0.8 for Lynn, 0.45 for six-bus, 0.38 for
IEEE57, and 0.4 for IEEE118. Every family then uses variant factors 0.84–0.94,
with reactive-demand factors 0.96–1.04. Generator voltage targets are 1.02 pu,
except 1.00 pu for IEEE57/118. IEEE57 retains 30% of original tap deviations from
unity; IEEE118 uses unity taps. Both have fixed taps and zero phase shifts.
Original thermal ratings and acceptance thresholds are retained.

Converter cases use an existing source as a fixed-voltage/frequency grid
reference. Other synchronous sources retain stable gains. A selected source's
scheduled injection is split between its synchronous part and added converters
at the same bus, preserving total scheduled MW. Lower-share cases transfer
40% of its remaining dispatch per converter; `_05` transfers 65% per converter
(87.75% of the selected source in total for its two sequential transfers).
These are shares of one selected source, not overall network penetration.

Converter powers are converted between system and converter MVA bases at the
wrapper boundary. Aggregate converter ratings are twice the solved apparent
injection, with minimum ratings of 100 MVA for GFL and 400 MVA for GFM on the
100 MVA source networks (800 MVA GFM minimum for six-bus and IEEE RTS24). This fixed sizing keeps the native current limit
inactive at equilibrium and provides stabilizable GFM filter/grid interactions
in the small networks. Ratings and physical filter parameters are not tunable.

## Controller inspection

See the [controller helper examples](../README.md#inspect-controllers-from-a-tuner)
for discovering PLL, governor, outer and inner loops, finding permitted handles,
and associating full-system modes with those loops. A runnable mixed-converter
example is available as `python -m hackathon_files.intro.examples.controller_examples`.

## Evaluation budgets

The basic collection has **30 unlimited cases and 30 cases capped at 20 full
`case.test_tuning(proposal)` trials**. Budgets alternate by network and variant,
so each variant has five unlimited and five capped cases.
Read `case.max_test_tunings`: `math.inf` means unlimited. Intro scenarios retain
their five-trial allowance. Invalid
proposals and failed solves consume a trial. Final scoring is separate and does
not consume the trial allowance. A fresh session is created for every case.
Modal/controller diagnostics, sensitivity helpers, `run_powerflow`, and
`get_losses` do not spend full RMS trials. The outer
`tuner.test_tuning(raw_grid, evaluator)` helper separately defaults to ten calls
per tuner instance; do not call it inside `tune()`.


## Fixed-controller network-action cases

Ten additional `basic_switching_<network>_01.veragrid` cases (one per network)
expose **no tunable controller parameters and no redispatch**. Their catalogs
permit line opening/closing, one new reinforcement line, and one added shunt.
There is at most one action of each type, for up to four structural changes;
radial initial networks may have no opening option and therefore a maximum of
three. Opening catalogs continue to exclude bridges, avoiding islanding.

These are operating-limit repair challenges. The fixed controllers can already
be dynamically stable, but doing nothing leaves a thermal overload and fails
acceptance. An existing open relief circuit provides a verified switching repair;
equipment additions are alternative permitted strategies. Ratings are explicit
educational limits rather than original utility ratings. Every saved reference
passes PF, equilibrium, full SSA, voltage/thermal limits, and the same load-step
simulation. `reference_tuning()` returns a closing proposal with an empty parameter
mapping for these cases. Five have 20 trials and five have unlimited trials.

`case_files()` returns the original 50 cases followed by the 10 additions.
Run `python -m hackathon_files.basic.switching` with this checkout on `PYTHONPATH`
to build/checkpoint the additional cases. Their validation evidence is stored in
`switching_validation.json`; `validation.json` continues to describe the original
50 dynamically unstable cases.


## Limited set and case marks

`case_files("limited")` returns ten small cases: all six Lynn 5-bus exercises
and variants 01–04 of the six-bus family. `case_files()` or `case_files("all")`
returns all 60 basic cases. Trial budgets and permissions stay unchanged.

Run the shared evaluator with either selection:

```bash
PYTHONPATH=src:. python -m hackathon_files.intro.evaluator --cases limited
PYTHONPATH=src:. python -m hackathon_files.intro.evaluator --cases all
```

Limited runs ten basic grids. All runs eleven intro scenarios plus 60 basic
grids. Both print per-case marks out of 100 and an overall mean, using the
35% damping / 25% loss reduction / 25% robustness / 15% changes weights.
Errors score zero and remain in the average. See the
[scoring conventions](../intro/README.md#understand-the-results).

The higher-share mixed cases for Lynn, IEEE 9, and IEEE 14 use a GFL
converter that regulates AC voltage. Its outer-loop tuning handles include
`Kp_vac` and `Ki_vac`. The other GFL cases use reactive-power regulation.
