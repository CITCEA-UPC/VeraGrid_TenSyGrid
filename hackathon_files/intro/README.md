# RMS tuning challenge

Implement `FinalTuner.tune()` in **rms_tunner.py**, then run the evaluator.

```text
intro/
├── README.md
├── rms_tunner.py   # Your algorithm and submission
├── evaluator.py    # Sample evaluation; leave unchanged
├── examples/       # Runnable contestant API examples and function index
└── testgrids/      # Bundled IEEE 9-bus grid with synchronous, GFL and dynamic GFM models
```

Use Python 3.12+ with this checkout's VeraGrid dependencies installed. From the
repository root:

```bash
PYTHONPATH=src:. python -m hackathon_files.intro.evaluator
```

This evaluates your tuner on eleven intro operating points, including changes in active
and reactive demand, followed by all 50 cases in `../basic/testgrids`. The basic
cases use their own controller bounds, physical-action catalogs, and 2.5% minimum
damping; intro scenarios retain the 5% minimum. Separate collection totals and
a combined 61-case total are printed. The full run can take substantially longer
than the intro scenarios alone. Every case checks power flow, stability, operating limits,
and the response to a load increase, and reports losses and damping.
Each scenario starts with a fresh tuner
and grid copy. Exit status is 0 if all cases pass, 1 if any fails or raises an
error. The unchanged baseline is valid but may fail the electrical criteria.
This folder requires the parent `hackathon_files` package and compatible VeraGrid
engine; it is not a standalone installation.

## Write your tuner

The evaluator solves AC power flow and initializes RMS before calling
`tune(tunable_grid)`. Return `RmsTuning()` for no changes, or populate its
`parameters`, `dispatch_mw`, `open_lines`, `add_lines`, and `add_shunts` fields.
Use handles from the current grid's catalogs:

- `tunable_parameters`: controller names, initial values and permitted bounds.
- `control_tensor`: read-only `[initial_value, minimum, maximum]` rows.
- `dispatch_options`: permitted generator MW ranges; preserve total scheduled MW.
- `openable_lines`, `line_candidates`, `shunt_options`: allowed network actions.

Keep `FinalTuner()` constructible without arguments. Do not mutate the input
grid, reuse another scenario's handles, or change the evaluator or test grids.
Invalid actions are skipped with warnings; an invalid redispatch is skipped as
a group. Returning anything other than `RmsTuning` is an error.

Before choosing a search strategy, use these inherited helpers:

- `self.get_controllers(tunable_grid, role=ControllerRole.PLL)` returns installed
  controller views together with their permitted parameter handles. Omit `role`
  for all views, including unclassified blocks.
- `self.get_analysis_capabilities(tunable_grid)` reports which analyses the
  current model structure supports, with reasons for limitations.
- `self.validate_proposal(tunable_grid, proposal)` returns `accepted`, `issues`,
  and the exact `effective_proposal` that evaluator sanitization would retain.

These helpers consume no trials. Validation checks permissions and action limits;
full trials still determine electrical feasibility. The short
[participant helper walkthrough](examples/participant_helpers.py) demonstrates
all three without navigating internal problem fields.

Useful inherited helpers include `self.get_modal_analysis(tunable_grid)`,
`self.get_modes_from_unstable_eigenvalues(tunable_grid)`, and
`self.get_eigenvalue_sensitivities(tunable_grid)`. Diagnostics consume no trials.
Use `tunable_grid.test_tuning(proposal)` for full candidate validation: five
attempts per intro scenario; basic cases allow 20 or unlimited trials, including invalid proposals and failed solves. Final
scoring is separate. Do not call `self.test_tuning()` inside `tune()`; it would
invoke your tuner recursively. See the [parent API guide](../README.md) for more.

## Understand the results

PASS requires converged power flow, physical-mode real parts below `-1e-5`, at
most one neutral mode (`abs(lambda) <= 1e-8`), at least 5% damping for modes above
0.05 Hz, voltages within 0.95–1.05 pu, and branch loading at most 100%. Infinite
descriptor eigenvalues are algebraic constraints and are excluded. No oscillatory
modes means damping is `n/a`.

The evaluator also checks transient convergence after a 2% load increase at 0.2 s
(3 s simulation, 0.02 s step). It reports active losses, peak frequency deviation, and structural changes as
`n/n_max`, with `S_changes = 1 - n/n_max` (1 when `n_max=0`). Each permitted line
opening/closing or line/shunt addition counts once; existing controller tuning
and balanced redispatch count zero. Counts use the sanitized proposal and each
scenario's own catalogs and action limits. Peak voltage deviation has been
removed as a metric; voltage and thermal limits still determine operating
compliance. The PDF assigns the change component 15% of the composite score;
this runner reports its unweighted value and mean, not the full composite score.
These sample operating points do not represent the hidden evaluation suite.

## Contestant API examples

See [examples/README.md](examples/README.md) for runnable examples of every public
`RmsTuner` method, the tuning catalogs, all proposal fields, and trial budgets.
Start with modal/controller inspection, then sensitivities and trial search:

```bash
PYTHONPATH=src:. python -m hackathon_files.intro.examples.modal_examples
PYTHONPATH=src:. python -m hackathon_files.intro.examples.controller_examples
PYTHONPATH=src:. python -m hackathon_files.intro.examples.sensitivity_examples
PYTHONPATH=src:. python -m hackathon_files.intro.examples.powerflow_examples
PYTHONPATH=src:. python -m hackathon_files.intro.examples.tuning_examples
PYTHONPATH=src:. python -m hackathon_files.intro.examples.control_response_experiment
```

Sensitivity examples use a compatible synchronous-machine basic case; analytic
sensitivities are not currently supported for the intro descriptor GFM model.
The examples leave `FinalTuner` and saved grids unchanged.

Basic-case trial budgets are split evenly: 25 unlimited and 25 capped at 20.
Always inspect `tunable_grid.max_test_tunings`; unlimited is `math.inf`.
