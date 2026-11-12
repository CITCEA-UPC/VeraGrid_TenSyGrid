# RMS tuning: start here

This introduction uses the existing `hackaton_files` submission API. Implement
`StarterTuner.tune()` once; both validation scripts evaluate that class on fresh
copies of the same benchmark at different operating points.

## Run the examples

Use Python 3.12+ in an environment with this checkout's VeraGrid dependencies
installed (see the repository installation instructions). From the repository root:

```bash
export PYTHONPATH=src:.
python -m hackaton_files.intro.starter_script
python -m hackaton_files.intro.evaluate_quick
python -m hackaton_files.intro.evaluate_local
```

These are module commands: keep the `-m` and run from the repository root. This
folder depends on the sibling RMS tuning helpers and the engine; copying only
`intro/` into a new repository is not sufficient. For a team starter repository,
include `hackaton_files` and this compatible engine checkout/environment.
Grid paths resolve relative to the Python file, not the working directory.

| File | Purpose |
| --- | --- |
| `benchmark_grid.veragrid` | Bundled mixed IEEE 9-bus RMS example |
| `starter_script.py` | Editable tuner, AC power flow, descriptor matrices and generalized eigenvalues |
| `evaluate_quick.py` | Five deterministic PF/modal screening cases |
| `evaluate_local.py` | Eleven cases with loss/damping KPIs and full load-step validation |
| `scenarios.py` | Shared scenario definitions and isolated grid copies |
| `settings.py` | Allowed controls, including GFM gains and time constants |
| `build_benchmark.py` | Reproduce the benchmark using the existing dynamic GFM core |

## What is in the benchmark?

The grid starts from `../grids/ieee9_rms_mixed_ibr.veragrid`, but replaces its
ideal source with VeraGrid's **dynamic RMS grid-forming converter core** from
`Templates/Rms/vsc_gfm.py`. It retains the GENROW1 synchronous machine (`gen 0`)
and PVD1 grid-following converter (`gen 2`). The GFM at `gen 1` includes droop,
voltage/current control, filter equations and current limits.

`build_benchmark.py` connects the existing `build_gfm_converter_model` core to
an AC generator injection. It adapts the terminal power sign convention and
external mappings; the converter's equations are unchanged. This represents
an AC-connected converter without an explicit DC network. Rebuild with:

```bash
python -m hackaton_files.intro.build_benchmark
```

`settings.py` exposes GFM droop gains `Kdp`, `Kdq`, current-loop gains `Kp_icl`,
`Ki_icl`, and measurement time constants `tau_P`, `tau_Q`, alongside the existing
machine and GFL controls. The search bounds are introductory choices, not
stability guarantees. Filter hardware and current limits are not tunable.

## Write your tuner

`StarterTuner` subclasses `RmsTuner`. The evaluator solves AC power flow,
initializes the RMS model, and calls `tune(tunable_grid)`. Return an `RmsTuning`
proposal. The provided implementation returns no changes, making its results a
reproducible baseline. The commented example shows bounded controller changes.

Use handles from the supplied catalog rather than hard-coded device indices:

- `tunable_parameters`: device identity, control name, initial value and bounds.
- `control_tensor`: read-only rows `[initial_value, minimum, maximum]` in the
  same order as `tunable_parameters`.
- `dispatch_options`: permitted generator MW ranges. Redispatch must preserve
  total scheduled generation; the AC slack balances losses.
- `openable_lines`, `line_candidates`, `shunt_options`: permitted network actions.
  This example supplies no new-line or shunt candidates.
- `RmsTuning(parameters=..., dispatch_mw=..., open_lines=..., add_lines=...,
  add_shunts=...)`: the proposal container. Omitted actions leave devices alone.
- `proposal.is_accepted(tunable_grid)`: checks handles, bounds and action budgets;
  acceptance alone does not imply electrical feasibility or stability.

Do not modify `tunable_grid.problem.grid` directly. The full evaluator applies
returned actions on a copy and recomputes the operating point when necessary.
Each scenario gets a fresh tuner, so keep learned reusable data explicit if your
algorithm needs it. Local validation uses the public evaluator. It now excludes infinite descriptor
eigenvalues (algebraic constraints) before computing physical-mode KPIs; NaNs
and an empty finite spectrum still raise an error.

## Diagnostic functions available on RmsTuner

| Function | What to use it for |
| --- | --- |
| `get_modal_analysis(grid)` | Eigenvalues, state/mass matrices, participation factors and variable ownership; accepts a raw grid, initialized `TunableGrid`, or previous analysis dictionary |
| `get_modes_from_unstable_eigenvalues(grid)` | Growing modes ranked by real part, with participating variables and device labels |
| `get_state_matrix(grid)` | Numerical state matrix; inspect the modal snapshot's mass matrix when using a descriptor system |
| `get_parameterized_state_matrix(tunable_grid)` | Symbolic matrix blocks with allowed controls kept editable |
| `evaluate_state_matrix(tunable_grid, values)` | Evaluate the matrix at candidate control values |
| `get_state_matrix_derivative(tunable_grid, parameter)` | Analytic matrix derivative for an allowed control |
| `get_eigenvalue_sensitivities(tunable_grid)` | Local modal sensitivities to guide a search |
| `get_equilibrium_residual(tunable_grid, values)` | Check whether changed controls still satisfy the initialized equilibrium |
| `test_tuning(grid, evaluator)` | Full evaluation of this tuner's final proposal on one grid |

Inside `tune()`, use `tunable_grid.test_tuning(proposal)` to compare trial
proposals without changing the baseline. It runs full validation and consumes
the per-scenario budget (default 10). Diagnostic helpers do not consume that
budget. Sensitivities are local, fixed-operating-point guidance; validate the
chosen proposal with the full evaluator. See the parent README for the detailed
sensitivity result fields and examples.

The starter explicitly assembles `E dz/dt = A z` and solves `A v = lambda E v`.
A singular `E` is expected for algebraic constraints; never form its inverse.
Ignore infinite descriptor eigenvalues when interpreting physical modes. The
convenience diagnostic helper may instead eliminate algebraic variables and
return a reduced matrix with `mass_matrix=None`.

## Interpret validation results

Quick screening uses nominal, ±5% demand, and ±10% reactive-demand cases. Local
validation adds combined corners and intermediate demand cases. Active generation
scales with active load, and the slack supplies the residual balance. These are
sample operating points, not contingencies or the final hidden scenario suite.

Both scripts print every case and exit with code 1 if any case fails or raises an
error (0 if all pass). Exceptions do not silently remove a scenario from the pass
count. The local mean loss only includes successfully evaluated cases and reports
its denominator; any missing case still fails validation.

The criteria follow the existing evaluator: physical-mode real parts below
`-1e-5`, at most one neutral mode (`|lambda| <= 1e-8`), and damping at least 5%
for oscillatory modes above 0.05 Hz. No oscillatory modes means damping is `n/a`,
not zero. Voltage magnitudes must be within 0.95–1.05 pu and branch loading at
most 100%. Damping is `-Re(lambda)/|lambda|`, printed as a percentage.

Local validation additionally runs the evaluator's 2% load increase at 0.2 s,
with a 3 s simulation and 0.02 s step. It reports transient convergence through
PASS/FAIL, peak frequency/voltage deviations, branch plus shunt active losses
(MW), and damping. Peak deviations are reported KPIs; the evaluator does not
impose additional peak thresholds. No invented weighted competition score is
used. Quick screening skips this transient test, so a quick PASS is provisional.

## Runtime expectations

The quick script reports wall time including grid loading and RMS initialization.
The dynamic converter adds states and derivative-dependent equations. This
implementation does **not** meet the proposed sub-second target. Symbolic
initialization dominates even on this small grid; skipping transients alone is
insufficient. Sub-second iterative feedback would require a separate
prepared-model/cache workflow with careful invalidation for dispatch/topology
changes. Trial evaluations inside a custom tuner can add more time.

The unchanged tuner is a starting point, not a guaranteed passing solution.
The dynamic GFM benchmark must be assessed on its own results; the earlier ideal
source case's passing scores do not apply to it. A damping failure is useful
feedback for the tuning exercise, even when every finite physical mode decays.

## Final submission entry point

When ready, put your algorithm in `FinalTuner.tune()` in `evaluator.py`.
`FinalTuner` directly inherits from `RmsTuner`, takes no required constructor
arguments, and returns an `RmsTuning` built from the current scenario's handles.
The file documents the expected interface, allowed actions, diagnostic helpers,
trial evaluations, and import requirements alongside the implementation.

Run the final class on the five basic scenarios from the repository root:

```bash
PYTHONPATH=src:. python -m hackaton_files.intro.evaluator
```

This command tests **FinalTuner**, while the earlier quick/local commands still
test **StarterTuner**. It checks power flow, modes and operating limits without
a disturbance simulation. It exits with status 1 if any case fails or raises an
error. Importing `FinalTuner` alone does not run the scenarios. The shipped final
class is an unchanged baseline; replace its method with your team's algorithm.

See [the FinalTuner evaluation guide](EVALUATOR.md) for the complete submission
contract, environment setup, scenario definitions, pass/fail criteria, baseline
results, and optional full evaluation of the final class.
