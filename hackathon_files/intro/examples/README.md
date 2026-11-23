# Contestant API examples

Run these from the repository root with your environment activated. Each command
uses the engine in this checkout:

```bash
PYTHONPATH=src:. python -m hackathon_files.intro.examples.modal_examples
PYTHONPATH=src:. python -m hackathon_files.intro.examples.controller_examples
PYTHONPATH=src:. python -m hackathon_files.intro.examples.sensitivity_examples
PYTHONPATH=src:. python -m hackathon_files.intro.examples.tuning_examples
PYTHONPATH=src:. python -m hackathon_files.intro.examples.control_response_experiment
```

These are reference implementations for writing `FinalTuner.tune()`. They do not
change your submission or saved grids. The last two run full simulations and
can take longer. Each script has a `main()` entry point.

| Contestant API | Example | What it demonstrates |
| --- | --- | --- |
| `RmsTuner.tune(grid)` | `tuning_examples.py` | Return a proposal without mutating the input |
| `RmsTuner.test_tuning(raw_grid, evaluator)` | `control_response_experiment.py` | Evaluate an entire tuner from outside `tune()` |
| `get_modal_analysis(grid_or_analysis)` | `modal_examples.py`, `controller_examples.py` | Eigenvalues, participation, and reuse of analysis snapshots |
| `get_state_matrix(grid_or_analysis, values)` | `modal_examples.py`, `sensitivity_examples.py` | Descriptor matrices and parameter overrides on supported grids |
| `get_modes_from_unstable_eigenvalues(...)` | `modal_examples.py` | Rank growing modes and inspect their participating devices |
| `get_controller_blocks(device)` | `controller_examples.py` | Discover native controller blocks and roles |
| `get_controller_parameters(grid, controller)` | `controller_examples.py` | Obtain permitted parameter handles for a controller |
| `get_controller_modes(analysis, controller)` | `controller_examples.py` | Connect system modes to controller participation |
| `get_parameterized_state_matrix(grid)` | `sensitivity_examples.py` | Inspect symbolic Jacobian blocks and editable symbols |
| `evaluate_state_matrix(grid, values)` | `sensitivity_examples.py` | Evaluate a matrix at proposed control settings |
| `get_state_matrix_derivative(grid, parameter, values)` | `sensitivity_examples.py` | Analytic local matrix derivative |
| `get_eigenvalue_sensitivities(grid, values)` | `sensitivity_examples.py` | Mode derivatives, honoring the valid-mode mask |
| `get_equilibrium_residual(grid, values)` | `sensitivity_examples.py` | Check whether control changes preserve the initialized equilibrium |
| `TunableGrid.control_tensor` and parameter catalogs | `tuning_examples.py` | Align numeric control rows with permitted handles |
| `TunableGrid.test_tuning(proposal)` | `tuning_examples.py` | Run candidates inside `tune()` and respect the five-trial budget |
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
