# FinalTuner submission and evaluation guide

Implement your final algorithm in **`FinalTuner` in `evaluator.py`**. This class
inherits from `RmsTuner`. The same file provides a runnable check against five
basic scenarios, so you can verify that your submission imports, returns valid
actions, and meets the sample electrical criteria.

## Expected submission

The required interface is:

```python
from hackaton_files import RmsTuner, RmsTuning, TunableGrid


class FinalTuner(RmsTuner):
    def tune(self, tunable_grid: TunableGrid) -> RmsTuning:
        # Replace this unchanged baseline with your algorithm.
        return RmsTuning()
```

The evaluator constructs the class as `FinalTuner()`, without arguments, and
calls `tune(tunable_grid)` for each scenario. If you add a constructor, keep it
callable without arguments and initialize the base class with `super().__init__()`.

Return an **`RmsTuning` proposal**, not a modified grid, a plain dictionary or an
evaluation result. Use handles from the supplied scenario. Do not mutate the
input grid or change the benchmark, scenario definitions, or acceptance criteria
to improve your reported result.

The shipped implementation returns `RmsTuning()`, meaning “make no changes.”
It is a valid submission interface, but it does not perform optimization or
guarantee that the scenario passes.

## From the starter to the final class

Explore the API in `starter_script.py`, then move your chosen algorithm into
`FinalTuner.tune()` in `evaluator.py`.

| Command module | Class it evaluates | Purpose |
| --- | --- | --- |
| `hackaton_files.intro.starter_script` | `StarterTuner` | Demonstrate loading, power flow, descriptor matrices and eigenvalues |
| `hackaton_files.intro.evaluate_quick` | `StarterTuner` | Five basic PF/modal screening scenarios |
| `hackaton_files.intro.evaluate_local` | `StarterTuner` | Eleven sample scenarios with the existing full evaluator |
| `hackaton_files.intro.evaluator` | **`FinalTuner`** | Check your final class on the five basic scenarios |

Editing `StarterTuner` does not automatically update `FinalTuner`. The final
check explicitly uses `FinalTuner`; keep your intended submission there.

## Environment and imports

Use Python 3.12+ with the dependencies required by this VeraGrid checkout.
Run the following commands **from the repository root**, where `src/` and
`hackaton_files/` are located:

```bash
# Check that the submission imports and can be constructed.
PYTHONPATH=src:. python -c "from hackaton_files.intro.evaluator import FinalTuner; print(FinalTuner())"

# Evaluate the final class on all five basic scenarios.
PYTHONPATH=src:. python -m hackaton_files.intro.evaluator
```

Use your environment's Python executable in place of `python` if necessary.
`PYTHONPATH=src:.` exposes both the engine source and the hackathon package.
Use `-m` rather than running `evaluator.py` directly, because the example uses
package-relative imports. The benchmark path is resolved relative to the source
file, so it does not need a manually supplied file path.

Importing `FinalTuner` does not start scenario evaluation. Keep optimization and
other expensive work inside methods, rather than at module import time.

This folder relies on the parent `hackaton_files` helpers and the compatible
VeraGrid engine. Copying `evaluator.py` alone into an unrelated environment is
not enough. A missing dependency error should be resolved in the Python
environment; it is not a reason to alter the evaluation criteria.

## What your tuner receives

`TunableGrid` contains an RMS problem initialized from a converged AC power flow,
plus catalogs of permitted actions:

| Attribute | Meaning |
| --- | --- |
| `tunable_parameters` | Controller handles with device identity, parameter name, initial value, minimum and maximum |
| `control_tensor` | Read-only rows `[initial_value, minimum, maximum]`, aligned with `tunable_parameters` |
| `dispatch_options` | Generator active-power handles and permitted MW ranges |
| `openable_lines` | Existing lines that may be opened |
| `line_candidates` | Allowed new-line candidates, if provided |
| `shunt_options` | Allowed shunt candidates, if provided |
| `max_test_tunings` / `tests_used` | Trial evaluation budget and consumption |

Return your chosen actions through `RmsTuning(parameters=..., dispatch_mw=...,
open_lines=..., add_lines=..., add_shunts=...)`. Omitted actions leave the
corresponding settings unchanged. Empty catalogs mean that action is unavailable.
Redispatch must preserve total scheduled active generation; AC power flow lets
the slack supply the residual mismatch and losses.

For example, this illustrates how to build a bounded control proposal; it is
not a recommended tuning strategy or a guarantee of improved damping:

```python
parameters = {
    parameter: float(max(parameter.minimum,
                         min(parameter.maximum, 0.95 * parameter.initial_value)))
    for parameter in tunable_grid.tunable_parameters
    if parameter.name == "Kp"
}
return RmsTuning(parameters=parameters)
```

Always select handles from the current scenario instead of assuming numeric
device positions or retaining handles from a previous case.

## Diagnostics and trial evaluations

Inherited helpers can guide your algorithm:

- `self.get_modal_analysis(tunable_grid)` returns eigenvalues, matrices,
  participation factors and variable ownership.
- `self.get_modes_from_unstable_eigenvalues(tunable_grid)` identifies growing
  modes and their participating variables. A mode can have insufficient damping
  while still decaying, so this helper alone is not a complete pass/fail check.
- `self.get_parameterized_state_matrix(tunable_grid)` exposes symbolic matrix
  blocks with editable controls.
- `self.evaluate_state_matrix(tunable_grid, values)` and
  `self.get_state_matrix_derivative(tunable_grid, parameter)` support matrix
  evaluation and analytic derivatives.
- `self.get_eigenvalue_sensitivities(tunable_grid)` provides local modal
  sensitivities, while `self.get_equilibrium_residual(tunable_grid, values)`
  checks consistency with the initialized operating point.

Diagnostics do not consume the trial budget. Sensitivities describe a local,
fixed-operating-point approximation; they do not replace evaluation of the
chosen proposal.

Inside `tune()`, evaluate a candidate with:

```python
proposal = RmsTuning(parameters=parameters)
metrics = tunable_grid.test_tuning(proposal)
```

This uses the attached **full evaluator**, including its load-step simulation,
and consumes one trial (default budget: 10 per scenario). Therefore, using trial
evaluations can add transient simulations and runtime even when the outer command
is the basic screen. Do not call `self.test_tuning(...)` inside `tune()`: that
entry point invokes the tuner again and would recurse.

## How the basic final check works

`run_basic_scenarios()` loads the benchmark and, for each scenario:

1. Creates a fresh grid copy and adjusts load and generation.
2. Constructs a fresh `FinalTuner()` so algorithm state does not leak across cases.
3. Runs AC power flow and initializes the RMS problem.
4. Calls `tune()` and checks that the returned proposal uses permitted handles,
   bounds and action budgets.
5. Applies the proposal and reruns power flow if dispatch or network actions
   require it, then rebuilds the RMS problem and computes its modes.
6. Prints the maximum physical-mode real part, minimum oscillatory damping,
   and PASS or FAIL.

The five scenarios are defined in `scenarios.py`:

| Scenario | Active load and scheduled generation | Reactive load |
| --- | --- | --- |
| nominal | 100% | 100% |
| light | 95% | 95% |
| heavy | 105% | 105% |
| low-reactive | 100% | 90% |
| high-reactive | 100% | 110% |

These are operating-point samples, not a comprehensive contingency suite or a
claim about hidden final scenarios. The basic check itself performs **no
load-step simulation** and does not print a loss KPI or weighted competition
score. It uses the shared quick evaluator.

## PASS, FAIL and ERROR

A basic scenario passes only when all these conditions hold:

| Criterion | Requirement |
| --- | --- |
| AC power flow | Converges |
| Proposal | Uses allowed actions and values within their bounds |
| Physical-mode real parts | Maximum is below `-1e-5` |
| Neutral modes | At most one mode with `abs(lambda) <= 1e-8` |
| Oscillatory damping | At least 5% for every mode above 0.05 Hz |
| Bus voltage magnitude | Within 0.95–1.05 pu |
| Branch loading | At most 100% |

Infinite descriptor eigenvalues represent algebraic constraints and are excluded
from the physical-mode screen. Damping is `-Re(lambda) / abs(lambda)` and is
printed as a percentage. If there are no oscillatory modes above the frequency
cutoff, the damping result is `n/a`.

A negative real part means a mode decays, but does not guarantee sufficient
damping: a slowly decaying oscillation can still fail the 5% criterion.

An exception is printed as ERROR and counts as a failed scenario; the runner
continues to report the remaining cases. Exit status is **0** when all scenarios
pass and **1** if any fails or raises an error. A failure does not necessarily
mean an import or implementation error—it may be a valid proposal with
insufficient electrical performance.

## Current unchanged baseline

The checked `FinalTuner()` imports successfully and executes all five scenarios.
With the benchmark present at the time of the check, the unchanged proposal
produced approximately **0.66–0.67% minimum oscillatory damping**. All five cases
therefore failed the **5%** damping requirement despite negative maximum
physical-mode real parts. The run reported **0/5 passed**, in approximately
**6.1 seconds**, excluding Python import startup.

These are measured baseline results, not hard-coded expectations or a runtime
guarantee. Rerun the command after changing your tuner or benchmark. The intended
next step is to implement and evaluate a tuning strategy, not weaken the criteria.

## Optional full evaluation of FinalTuner

To obtain the existing evaluator's loss, damping and transient KPIs for one case,
use the following in a Python session started with the same `PYTHONPATH`:

```python
from hackaton_files.intro.evaluator import FinalTuner
from hackaton_files.intro.scenarios import load_benchmark
from hackaton_files.intro.settings import make_evaluator

metrics = FinalTuner().test_tuning(load_benchmark(), make_evaluator())
print(metrics)
```

The existing full evaluator includes a 2% active-demand increase at the first
load at 0.2 seconds, simulated for 3 seconds with a 0.02-second step. This is an
additional time-domain check, not part of the basic final screening command.
It reports branch plus shunt active losses, damping, operating-limit violations,
transient convergence, and peak frequency and voltage deviations. No weighted
final competition score is defined by these examples.

For more scenarios, reuse `LOCAL_SCENARIOS` with `make_scenario()` and a fresh
`FinalTuner()` per case. The `evaluate_local` command itself continues to use
`StarterTuner`, so it should not be mistaken for testing your final class.
