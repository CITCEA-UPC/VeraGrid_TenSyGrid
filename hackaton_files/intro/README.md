# RMS tuning challenge

Implement `FinalTuner.tune()` in **rms_tunner.py**, then run the evaluator.

```text
intro/
├── README.md
├── rms_tunner.py   # Your algorithm and submission
├── evaluator.py    # Sample evaluation; leave unchanged
└── testgrids/      # Bundled IEEE 9-bus grid with synchronous, GFL and dynamic GFM models
```

Use Python 3.12+ with this checkout's VeraGrid dependencies installed. From the
repository root:

```bash
PYTHONPATH=src:. python -m hackaton_files.intro.evaluator
```

This evaluates your tuner on eleven operating points, including changes in active
and reactive demand. Every case checks power flow, stability, operating limits,
and the response to a load increase, and reports losses and damping.
Each scenario starts with a fresh tuner
and grid copy. Exit status is 0 if all cases pass, 1 if any fails or raises an
error. The unchanged baseline is valid but may fail the electrical criteria.
This folder requires the parent `hackaton_files` package and compatible VeraGrid
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

Useful inherited helpers include `self.get_modal_analysis(tunable_grid)`,
`self.get_modes_from_unstable_eigenvalues(tunable_grid)`, and
`self.get_eigenvalue_sensitivities(tunable_grid)`. Diagnostics consume no trials.
Use `tunable_grid.test_tuning(proposal)` for full candidate validation: five
attempts per scenario, including invalid proposals and failed solves. Final
scoring is separate. Do not call `self.test_tuning()` inside `tune()`; it would
invoke your tuner recursively. See the [parent API guide](../README.md) for more.

## Understand the results

PASS requires converged power flow, physical-mode real parts below `-1e-5`, at
most one neutral mode (`abs(lambda) <= 1e-8`), at least 5% damping for modes above
0.05 Hz, voltages within 0.95–1.05 pu, and branch loading at most 100%. Infinite
descriptor eigenvalues are algebraic constraints and are excluded. No oscillatory
modes means damping is `n/a`.

The evaluator also checks transient convergence after a 2% load increase at 0.2 s
(3 s simulation, 0.02 s step). It reports active losses and peak frequency and
voltage deviations. No weighted competition score is defined here.
These sample operating points do not represent the hidden evaluation suite.
