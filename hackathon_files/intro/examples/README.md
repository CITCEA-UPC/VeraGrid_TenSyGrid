# Contestant API examples

Start with the power-flow example, then learn how to inspect dynamics and test
control changes. Each script is a standalone walkthrough with a `main()` entry
point. Run commands from the **repository root**, with your Python environment
activated and the checkout's engine on `PYTHONPATH`.

## Suggested reading order

| Step | Script | What to look for | Cost |
| --- | --- | --- | --- |
| 1 | [Power flow](powerflow_examples.py) | Compare baseline and physical actions; read MW losses, voltage range and loading | Power flows only |
| 2 | [Modal analysis](modal_examples.py) | Read eigenvalues, identify growing modes and inspect their leading participants | Modal diagnostics |
| 3 | [Controllers](controller_examples.py) | Find controller blocks, permitted parameter handles, bounds and associated modes | Modal diagnostics |
| 4 | [Sensitivities](sensitivity_examples.py) | Check equilibrium before using local matrix and eigenvalue derivatives | Symbolic diagnostics; may take time |
| 5 | [Tuning](tuning_examples.py) | Implement `tune()`, compare two candidates within budget and return a proposal | Up to two full trials, plus final evaluation |
| 6 | [Load response](control_response_experiment.py) | Compare gain settings under the same disturbance | Five full simulations |

Run one example at a time:

```bash
# 1. Static checks: no full RMS trials.
PYTHONPATH=src:. python -m hackathon_files.intro.examples.powerflow_examples
# 2. Understand the modes before choosing controls.
PYTHONPATH=src:. python -m hackathon_files.intro.examples.modal_examples
# 3. Discover the controls you are allowed to tune.
PYTHONPATH=src:. python -m hackathon_files.intro.examples.controller_examples
# 4. Explore local derivatives on a supported model.
PYTHONPATH=src:. python -m hackathon_files.intro.examples.sensitivity_examples
# 5. Run a small candidate search.
PYTHONPATH=src:. python -m hackathon_files.intro.examples.tuning_examples
# 6. Compare full disturbance responses.
PYTHONPATH=src:. python -m hackathon_files.intro.examples.control_response_experiment
```

These examples do not change your submission or saved grids. They use different
cases deliberately: modal analysis demonstrates the intro descriptor benchmark;
controller discovery uses the mixed IEEE9 case; power flow, sensitivities and
tuning use the synchronous IEEE9 case; load response uses the saved Kundur grid.
Results from different scripts therefore describe different operating points.

## Reading the results

- **Power flow:** voltages should be within 0.95--1.05 pu and loading at most
  100%. Lower MW losses help, but convergence alone does not establish feasibility
  or dynamic stability.
- **Modes:** a positive eigenvalue real part indicates growth. Damping is shown
  as a percentage; frequency is in Hz. Participation helps locate relevant states
  and controls, without proving that changing a control will improve that mode.
- **Sensitivities:** derivatives describe a local change near the checked
  equilibrium. Invalid mode derivatives are excluded from the reported direction.
- **Trials:** `tests_used / max_test_tunings` shows the full-trial budget.
  Diagnostics and power-flow checks do not consume it; failed full trials do.
- **Tuning:** the example's ranking prefers stability, fewer violations, a smaller
  leading eigenvalue real part and then losses. This is a teaching search, not the
  official composite score or a generally optimal controller setting.

## API lookup

| Contestant API | Example | What it demonstrates |
| --- | --- | --- |
| `get_controllers(grid, role=None)` | `participant_helpers.py`, `controller_examples.py` | Discover installed controller views and their permitted handles |
| `get_analysis_capabilities(grid)` | `participant_helpers.py` | Inspect supported analyses and limitations without solving |
| `validate_proposal(grid, proposal)` | `participant_helpers.py`, `tuning_examples.py` | Explain omissions and preview the exact effective proposal |
| `run_powerflow(proposal, grid=grid, options=None)` | `powerflow_examples.py` | Solve a candidate on a copy; inspect convergence, voltages, and loading |
| `get_losses(proposal, grid=grid, options=None)` | `powerflow_examples.py` | Independently solve the proposal and return active MW losses including shunt conductance |
| `RmsTuner.tune(grid)` | `tuning_examples.py` | Return a proposal without mutating the input |
| `RmsTuner.test_tuning(raw_grid, evaluator)` | `control_response_experiment.py` | Evaluate an entire tuner from outside `tune()` |
| `get_modal_analysis(grid_or_analysis)` | `modal_examples.py`, `controller_examples.py` | Eigenvalues, participation, and reuse of analysis snapshots |
| `get_state_matrix(grid_or_analysis, values)` | `modal_examples.py`, `sensitivity_examples.py` | Descriptor matrices and parameter overrides on supported grids |
| `get_modes_from_unstable_eigenvalues(...)` | `modal_examples.py` | Rank growing modes and inspect their participating devices |
| `get_controller_blocks(device)` | `controller_examples.py` | Discover native controller blocks and roles |
| `get_controller_parameters(grid, controller)` | `controller_examples.py` | Obtain permitted parameter handles for a controller |
| `get_controller_modes(analysis, controller)` | `controller_examples.py` | Connect system modes to controller participation |
| `get_symbolic_state_matrices(grid)` | `sensitivity_examples.py` | Native symbolic A(p), E and variable order, retaining algebraic equations |
| `get_parameterized_state_matrix(grid)` | `sensitivity_examples.py` | Inspect symbolic Jacobian blocks and editable symbols |
| `evaluate_state_matrix(grid, values)` | `sensitivity_examples.py` | Evaluate a matrix at proposed control settings |
| `get_state_matrix_derivative(grid, parameter, values)` | `sensitivity_examples.py` | Analytic local matrix derivative |
| `get_eigenvalue_sensitivities(grid, values)` | `sensitivity_examples.py` | Mode derivatives, honoring the valid-mode mask |
| `get_equilibrium_residual(grid, values)` | `sensitivity_examples.py` | Check whether control changes preserve the initialized equilibrium |
| `TunableGrid.control_tensor` and parameter catalogs | `tuning_examples.py` | Align numeric control rows with permitted handles |
| `TunableGrid.test_tuning(proposal)` | `tuning_examples.py` | Run candidates inside `tune()` and respect the current case budget |
| `TunableGrid.tests_used`, `max_test_tunings` | All inspection examples, `tuning_examples.py` | Diagnostics consume no trials; full trials do |
| `RmsTuning.is_accepted(grid)` | `tuning_examples.py` | Check handles/bounds before electrical evaluation |
| All `RmsTuning` proposal fields | `tuning_examples.py` | Controls, balanced redispatch, line opening/closing, reinforcement, shunts |

## Model support and trial rules

The intro benchmark contains a descriptor GFM model. Modal diagnostics handle
this using the mass matrix and generalized eigenvalues. The analytic sensitivity
helpers currently require a semi-explicit DAE and raise `NotImplementedError`
for descriptor systems; their example uses `basic_ieee9_01` instead. Fixed-point
derivatives do not replace reinitialization or full electrical validation.

The tuning example builds optional physical-action proposals independently; it
does not claim that every action is beneficial. It tests only two control
candidates. Basic cases need their own initializer to supply physical-action
catalogs; the example follows the combined evaluator's initialization path.

Inside `tune()`, call `grid.test_tuning(proposal)`. Call
`tuner.test_tuning(raw_grid, evaluator)` only from outside `tune()`, because it
invokes the tuner. Invalid or failed full trials count against the budget.

## Power-flow-only proposal checks

Inside `tune(tunable_grid)`, pass the current grid explicitly:

```python
proposal = RmsTuning(parameters={handle: value})
pf = self.run_powerflow(proposal, grid=tunable_grid)
if pf.converged:
    losses_mw = self.get_losses(proposal, grid=tunable_grid)
```

Both methods accept optional `PowerFlowOptions` through `options=` and apply the
same action sanitization as final scoring. Each invocation solves an independent
copy and consumes no full RMS trials. `run_powerflow` returns `PowerFlowResults`,
including `converged`, `voltage`, `loading`, and branch `losses`. `get_losses`
returns a float in MW and raises `RuntimeError` if power flow fails to converge.
It includes branch losses and active shunt consumption, matching final scoring.

These checks do not run RMS initialization, SSA, or the load-step simulation.
Controller gain changes generally leave the static operating point unchanged;
use full tuning trials to assess their dynamic effect.

## Symbolic versus numerical matrices

`get_state_matrix(grid)` and `evaluate_state_matrix(grid, values)` return
**numerical NumPy arrays**. `get_state_matrix_derivative()` returns an evaluated
numerical derivative computed analytically from symbolic expressions.

`get_parameterized_state_matrix(grid)` returns symbolic Jacobian blocks in
`model['symbolic_blocks']`: `fx`, `fy`, `gx`, and `gy`.
`get_symbolic_state_matrices(grid)` assembles these into **native VeraGrid
expression arrays** (not SymPy matrices):

```python
pencil = self.get_symbolic_state_matrices(grid)
A = pencil['state_matrix']       # [[fx, fy], [gx, gy]]
E = pencil['mass_matrix']        # diag(I, 0)
variables = pencil['variables']  # states first, then algebraic variables
handle = grid.tunable_parameters[0]
symbol = pencil['parameter_symbols'][handle]
expression = A[0, 0]
print(expression)
print(expression.diff(symbol).simplify())
bindings = {
    symbol.uid: pencil['default_values'][parameter]
    for parameter, symbol in pencil['parameter_symbols'].items()
}
print(expression.eval_uid(bindings))
```

These represent `E @ dz/dt = A(p) @ z` at the initialized operating point.
Only editable controls remain symbolic; state values and other parameters are
frozen. These are local linearized models, not the full nonlinear equations.
The returned variable order labels both columns of A and entries of z; rows
follow the state-equation and algebraic-equation order, respectively.

This pencil is larger than the reduced numerical state matrix
`fx - fy @ solve(gy, gx)`. E is singular when algebraic variables exist, and the
original angle reference is retained. Numerical eigenanalysis requires handling
that reference and the algebraic constraints; use `get_modal_analysis()` for
that analysis. Ordinary `eig(A)` is inappropriate for this pencil. Both symbolic
helpers currently support semi-explicit DAEs; native descriptor models raise
`NotImplementedError`. Inspection consumes no full trials. Changed controls may
shift equilibrium: check `get_equilibrium_residual(grid, values)` before using
a proposed operating point.

## Three helpers to use before tuning

Run the short walkthrough first:

```bash
PYTHONPATH=src:. python -m hackathon_files.intro.examples.participant_helpers
```

### Find installed controllers and permitted parameters

Inside `tune(grid)`, use the current snapshot directly:

```python
from hackathon_files import ControllerRole

for info in self.get_controllers(grid, role=ControllerRole.PLL):
    controller = info.controller
    print(controller.device_name, controller.path)
    for handle in info.parameters:
        print(handle.name, handle.initial_value, handle.minimum, handle.maximum)
```

Omit `role` to include all discovered views, including `UNCLASSIFIED` blocks.
No matches returns an empty tuple. A view may have no editable parameters; its
symbolic equations do not grant tuning permission. Views may overlap. You can
pass `info.controller` to `get_controller_modes(analysis, controller)` and reuse
one modal-analysis snapshot for all controllers.

This discovers controllers already installed in the scenario.
`get_pll_transform_rms()` constructs a new symbolic template; it does not find or
modify the installed PLL. Recognized roles depend on supported native template
signatures; custom blocks remain available as unclassified views.

### Check support before choosing an analysis

```python
capabilities = self.get_analysis_capabilities(grid)
capability = capabilities['analytic_sensitivities']
if capability.supported:
    sensitivities = self.get_eigenvalue_sensitivities(grid)
else:
    print(capability.reason)
```

The returned keys are `modal_analysis`, `state_matrix`,
`symbolic_state_matrices`, `parameter_overrides`, `analytic_sensitivities`, and
`full_trials`. Each value is an `AnalysisCapability(supported, reason)`.
Inspection does not assemble symbolic matrices or solve the model. Support means
the API handles that model structure, not that the operating point is feasible,
that matrix reduction will succeed, or that individual mode derivatives are valid.
`full_trials` reports whether a session is attached; always check the remaining
budget separately, even when that capability is supported.

On a native descriptor model, `get_state_matrix(grid)` returns numerical A in
`E dz/dt = A z`. Obtain E from `get_modal_analysis(grid)['mass_matrix']`; this A
is not a reduced ordinary state matrix. Parameter overrides and symbolic helpers
currently require a semi-explicit model.

### Explain invalid proposals before spending a trial

```python
report = self.validate_proposal(grid, proposal)
if not report.accepted:
    for issue in report.issues:
        print(issue)
    # Fix the proposal, or explicitly choose the sanitized alternative:
    print(report.effective_proposal)
elif grid.tests_used < grid.max_test_tunings:
    metrics = grid.test_tuning(proposal)
```

`ProposalValidationReport` contains:

| Field | Meaning |
| --- | --- |
| `accepted` | The entire proposal passes catalog checks without omissions |
| `issues` | Ordered messages explaining invalid handles, numeric bounds, malformed containers, duplicate/conflicting actions, action limits, or redispatch imbalance |
| `effective_proposal` | Exactly the proposal the evaluator's sanitization retains |

Validation uses the same sanitization logic as scoring, returns messages instead
of warnings, and consumes no trials. Valid independent actions survive; an
invalid redispatch is dropped as a group. Values are never clamped. A wrong
proposal type returns a failed report; the evaluator still rejects such a
submission with an error. Validation checks permissions and action budgets, not
electrical feasibility, remaining trial allowance, or a guaranteed improvement.
