# Hackathon RMS tuning

```text
hackathon_files/
├── src/          # Shared tuning, diagnostics, controllers, and grid helpers
├── intro/        # Participant tuner, combined evaluator, and API examples
├── basic/        # 50-case collection and its initialization/action helpers
├── tests/        # Shared-library regression tests
├── scripts/      # Benchmark generation utilities
├── grids/        # Shared grid data
└── stable_rms_grid.veragrid
```

The public API remains `from hackathon_files import RmsTuner, RmsTuning, TunableGrid`.
For individual modules, use `hackathon_files.src`, for example
`from hackathon_files.src.rms_controllers import ControllerBlock`.

From the repository root with your environment activated:

```bash
python hackathon_files/intro/evaluator.py
PYTHONPATH=src:. python -m hackathon_files.intro.examples.controller_examples
PYTHONPATH=src:. python -m hackathon_files.scripts.build_intro_benchmark
python -m pytest hackathon_files/tests
```

The benchmark generation command rebuilds the intro grid; it is not needed to
run the evaluator. See [intro](intro/README.md) and [basic](basic/README.md) for
the participant workflow and case-specific rules. The [contestant API examples](intro/examples/README.md)
cover every public `RmsTuner` helper, trial evaluation, and allowed tuning actions.

## Kundur RMS tuning case

`stable_rms_grid.veragrid` is a copy of
`Grids_and_profiles/grids/kundur_rms_sss.veragrid`.

`KUNDUR_CONTROL_RANGES` in `src/rms_tuning.py` selects the allowed controls and
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
from `openable_lines`, close an existing out-of-service circuit using
`close_lines` handles from `closable_lines`, build a new corridor from `line_candidates`, or shunt
placements from `shunt_options`. Generator setpoints are listed in
`dispatch_options`; use `dispatch_mw={option: new_mw}` to move generation
between units. Each value must stay within its generator limits and within
the published per-case interval, and total scheduled generation must remain fixed.
The generic factory uses 10% intervals; the general basic collection has its own
explicit intervals. Opening and closing the same line in one proposal is invalid.
`BasicRmsTuner` proposes no changes. Machine parameters `M`, `D`, `R1`, and
`X1` are fixed.

`TunableGrid` groups an initialized `RmsProblem` with its permitted actions.
Its default limits are two opened lines, one closed line, one new line, one new shunt, and five
candidate attempts per scenario. The frozen participant view exposes read-only
budget properties; the evaluator owns an independent counter for each scenario. `RmsTuning.is_accepted()` checks these limits.
Inside `tune()`, participants may call `tunable_grid.test_tuning(proposal)` to
compare proposals. Each call, including an invalid proposal, consumes one test; exceeding the budget raises
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
Scenario robustness and the minimal-grid-change score require separate evaluation.
The latter replaces code quality in the brief and retains its 15% weight:
`S_changes = 1 - n / n_max` per scenario, averaged over the evaluation
suite. Each scenario defines `n_max`, the maximum number of changes possible
in that scenario, shared by all teams. By construction, `0 <= n <= n_max`.
Zero actions gives full marks; reaching `n_max` gives zero. For `n_max = 0`,
no changes are possible and the score is defined as 1.
Each line connection/disconnection and model addition/removal counts as one
equally weighted action relative to the scenario's initial grid. Variables and
equations introduced or removed by that action are not counted separately.
Replacing a model counts as two actions; adjusting existing control parameters
or generation setpoints does not count.

Run `python -m hackathon_files.intro.examples.control_response_experiment` with `PYTHONPATH=src:.`
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

## Inspect modes and parameter sensitivities

The helpers live on `RmsTuner` and return plain dictionaries and NumPy arrays.
Inside `tune(tunable_grid)`:

```python
analysis = self.get_modal_analysis(tunable_grid)
A = self.get_state_matrix(analysis)
modes = self.get_modes_from_unstable_eigenvalues(analysis, top_n=3)
for mode in modes:
    print(mode["eigenvalue"], mode["frequency_hz"], mode["damping_ratio"])
    for entry in mode["participants"]:
        variable = entry["variable"]
        print(variable["device_name"], variable["name"], entry["participation"])
        print(variable["tunable_parameters"])

kp = next(p for p in tunable_grid.tunable_parameters if p.name == "Kp")
print(self.get_equilibrium_residual(tunable_grid))
A = self.get_state_matrix(tunable_grid)
dA = self.get_state_matrix_derivative(tunable_grid, kp)
sensitivity = self.get_eigenvalue_sensitivities(tunable_grid, values={kp: 20.0})
print(sensitivity["eigenvalues"], sensitivity["derivatives"])
```

### Modal diagnostics

`get_modal_analysis` accepts a raw grid (copied before power flow), a `TunableGrid`,
or a previous analysis dictionary. Reuse the dictionary to avoid another eigensolve.
It contains `state_matrix`, `mass_matrix`, `eigenvalues`, `participation_factors`,
and `variables` in matrix-row order. Variables include UID, device ID/name, block
path and the owning block's editable parameter handles (available for TunableGrid).

Growing modes are sorted by decreasing real part. The default `real_tolerance=1e-8`
excludes neutral modes. Conjugate pairs appear once; `include_conjugates=True`
returns both. `top_n` limits participants per mode; `None` returns all participants.
Participation identifies modal involvement, not causation or parameter sensitivity.
An empty tuple means no growing modes under this criterion, not that all tuning
acceptance criteria pass. Arrays are read-only; dictionaries are ordinary snapshots.

For ordinary systems `delta_x_dot = A @ delta_x`. Descriptor analyses also supply
`mass_matrix`, with `E @ delta_z_dot = A @ delta_z`; labels include algebraic rows.
Infinite eigenvalues are excluded from unstable-mode reports. The dense eigensolve
uses original variable coordinates and is intended for exercise-sized grids.

### Parameterized matrices

The matrix and sensitivity helpers accept `tunable_grid` directly. Symbolic blocks
are constructed internally from its current initialized problem on each call, so
there is no caller-managed model or stale implicit cache. Parameter overrides are
local to the call. The operating point is still fixed within that calculation.
`evaluate_state_matrix(tunable_grid, ...)` is also available as an explicit alias
for the parameterized evaluation path.

For inspection only,
`get_parameterized_state_matrix` takes a `TunableGrid` and returns a dictionary
containing `parameters`, `parameter_symbols`, `default_values`, `operating_point`,
`state_variables`, `algebraic_variables` and `symbolic_blocks` (`fx`, `fy`, `gx`, `gy`).
Underscore-prefixed entries hold internal evaluation data. The numeric reduction
is `A = fx - fy @ solve(gy, gx)`; no symbolic inverse is expanded.

Pass a mapping of parameter handles to values as the second argument of
`get_state_matrix(tunable_grid, ...)`, or as `values=...` to the derivative/sensitivity helpers.
Omitted parameters retain their saved runtime values. Handles and bounds are
validated. All helpers preserve the source grid and tuning-test budgets.

`get_state_matrix_derivative` analytically differentiates the blocks and algebraic
elimination. `get_eigenvalue_sensitivities` returns a dictionary with `eigenvalues`,
`parameters`, `derivatives`, `valid_modes` and `equilibrium_residual`. Derivative rows
are modes and columns are parameters, in actual parameter units. Negative real
sensitivity means increasing the parameter locally reduces that mode's growth rate.
Repeated or ill-conditioned modes are invalid and have NaN derivatives. Verify
finite changes with `test_tuning()`.

These are fixed-operating-point partial derivatives: they exclude equilibrium
movement and recomputation of other held initialization parameters. Reinitialize the problem after changing its equilibrium; each helper reads that
initialized problem anew. The default equilibrium check requires
a residual <= `1e-6`; the bundled unstable case has about `7.7e-4` and fails it.
Explicitly pass `require_equilibrium=False` to inspect its approximate-point
Jacobian, or choose an appropriate `equilibrium_tolerance`. The sensitivity report
includes the residual. Parameterized models containing explicit derivative variables
are currently rejected; their sensitivities need both `A(p)` and `E(p)` derivatives.

### Invalid actions in proposals

The evaluator sanitizes trial and final `RmsTuning` proposals, warning about
skipped actions. Valid controls and network actions are retained in submitted
order; duplicates and actions beyond their limits are skipped. Values are not
clamped. Invalid or unbalanced redispatch is omitted as a whole. Wrong proposal
return types still raise an error, and electrical checks remain unchanged.
`is_accepted()` remains a strict diagnostic; sanitized trials still cost one
attempt, including when every action is skipped.

## Inspect controllers from a tuner

`RmsTuner` (and `BasicRmsTuner`) provides three related helpers:

```python
controllers = tuner.get_controller_blocks(device)
parameters = tuner.get_controller_parameters(tunable_grid, controllers[0])
modes = tuner.get_controller_modes(tunable_grid, controllers[0])
```

Parameter lookup needs the `TunableGrid` because a parameter appearing in a
controller equation does **not** automatically make it editable. Returned objects
are the original `TunableParameter` handles from that snapshot, with their
`initial_value`, `minimum`, `maximum`, `device_idtag`, and `name`. No new tuning
permission is created. A stable synchronous governor can therefore be discovered
in a converter case while its editable parameter tuple is empty.

### Example: inspect a PLL or current loop

```python
from VeraGridEngine.api import open_file
from hackathon_files import BasicRmsTuner, ControllerRole
from hackathon_files.basic import case_files, initialize_grid

# Fifth case of IEEE9: synchronous machines, GFL, and GFM.
path = next(path for path in case_files() if path.name == "basic_ieee9_05.veragrid")
grid = initialize_grid(open_file(str(path)))
tuner = BasicRmsTuner()

# Cache one full-system eigensolve for all controller queries.
analysis = tuner.get_modal_analysis(grid)
for device in grid.problem.grid.generators:
    for controller in tuner.get_controller_blocks(device):
        if controller.role in (ControllerRole.PLL, ControllerRole.INNER_CURRENT):
            print(device.name, controller.role.name, controller.path)
            for parameter in tuner.get_controller_parameters(grid, controller):
                print(parameter.name, parameter.initial_value,
                      parameter.minimum, parameter.maximum)
            for equation in controller.state_equations:
                print("state:", equation)
            for equation in controller.algebraic_equations:
                print("connected algebraic:", equation)
            for mode in tuner.get_controller_modes(analysis, controller):
                print(mode.eigenvalue, mode.frequency_hz, mode.damping_ratio,
                      mode.participation, mode.scope.name)
```

The same pattern selects `ControllerRole.GOVERNOR`, `OUTER_POWER`,
`OUTER_VOLTAGE`, or `DROOP`. A runnable version is included:

```bash
python -m hackathon_files.intro.examples.controller_examples
```

Run using this checkout's engine and repository root on the Python import path.
The example inspects a real saved mixed-converter case and consumes zero trials.

### Example: propose changes using discovered handles

For this particular IEEE9 case, both converters can be stabilized through their
inner-loop handles. This is an example reference, not a universal tuning rule:

```python
from hackathon_files import ControllerRole, RmsTuning

values = dict()
for device in grid.problem.grid.generators:
    for controller in tuner.get_controller_blocks(device):
        if controller.role == ControllerRole.INNER_CURRENT:
            for parameter in tuner.get_controller_parameters(grid, controller):
                if parameter.name == "Kp_icl":
                    values[parameter] = 0.5
                elif parameter.name == "Ki_icl":
                    values[parameter] = 5.0
                else:
                    pass
        else:
            pass

proposal = RmsTuning(parameters=values)
metrics = grid.test_tuning(proposal)  # This evaluation consumes one trial.
```

### Interpret the results

A `ControllerBlock` contains `role`, `path`, `device_idtag`, `variables`,
`state_equations`, `algebraic_equations`, and `affected_states`.
`template_view=True` identifies an adapter view of a flattened model. Equation
views include directly connected algebraic equations; different views may overlap
and should not be interpreted as independent isolated subsystems. In particular,
the native GFM voltage and current loops share `Kp_icl`/`Ki_icl`: editing that
handle affects both loops, even when obtained through just one view.

Discovery has explicit adapters for the native Genrow, flattened GFL P/Q,
and GFM template contracts, including the saved basic cases. Other physical
blocks are returned as `UNCLASSIFIED`; arbitrary display names are not guessed
into controller roles. For a custom model, explicitly construct a `ControllerBlock`
with its device, actual block, role, diagnostic path, owned coordinate names,
and parameter names, then pass that view to the parameter/mode helpers.

Modes are **full-system modes associated with the selected coordinates**, not
poles of an isolated controller. `ControllerMode` provides `index`, `eigenvalue`,
`frequency_hz`, `damping_ratio`, `participation`, `scope`, and `variable_uids`.
Participation is the selected coordinates' share of the total for that mode,
matched by device identity and symbolic UID. Views can overlap, so participation
fractions should not be summed across controllers. Infinite descriptor constraints
and neutral modes are omitted; conjugate duplicates are omitted by default.
Use `include_conjugates=True` to retain both and `minimum_participation=0.05`
to require at least 5% participation. This filter is not a damping requirement.

An algebraic governor has no independent dynamic state. Its results explicitly
use `ParticipationScope.AFFECTED` for downstream rotor states; dynamic controller
coordinates use `ParticipationScope.OWNED`. Both ordinary state matrices and
converter descriptor snapshots are supported. Diagnostic calls consume no tuning
trials and never invoke the tuner's `tune()` method.

## General basic grid actions

The [50-case basic collection](basic/README.md) includes existing open circuits,
line outage/restoration cases, non-islanding opening options, new-line
reinforcements, capacitor/reactor candidates, and balanced redispatch, alongside
synchronous and GFM/GFL controller tuning. Its minimum damping remains 2.5%.
The validation report records exact action catalogs and reference restoration
actions. The evaluator keeps these organizer-specific permissions on every trial
copy and gives added lines/shunts their native RMS models.
