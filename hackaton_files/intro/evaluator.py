"""Final submission template and basic scenario check.

WHAT TO SUBMIT
--------------
Implement the class named FinalTuner below. It must inherit from RmsTuner, be
constructible as FinalTuner(), and implement:

    tune(self, tunable_grid: TunableGrid) -> RmsTuning

Move your algorithm from StarterTuner.tune() into this method when ready. Return
an RmsTuning proposal using the handles supplied by the current TunableGrid;
do not return a modified grid, a metrics object, or a dictionary by itself.
An empty RmsTuning() is a valid unchanged baseline, not an optimization algorithm.
Do not change the benchmark or evaluator to improve your reported results.

WHERE TO TRY IT
--------------
Use the same Python environment as VeraGrid. From the repository root:

    PYTHONPATH=src:. python -m hackaton_files.intro.evaluator

This imports FinalTuner and runs it on the five basic scenarios: nominal, light,
heavy, low-reactive and high-reactive. Each case gets a fresh tuner and grid copy.
The check validates the proposal and evaluates AC power flow and modal damping.
It does not run a disturbance simulation. PASS means this sample screen passed,
not that an algorithm is optimal or will pass every unseen scenario.

To verify the submission import independently (without running scenarios):

    PYTHONPATH=src:. python -c "from hackaton_files.intro.evaluator import FinalTuner; print(FinalTuner())"

To try one full evaluation interactively, including the existing load-step test:

    from hackaton_files.intro.evaluator import FinalTuner
    from hackaton_files.intro.scenarios import load_benchmark
    from hackaton_files.intro.settings import make_evaluator
    metrics = FinalTuner().test_tuning(load_benchmark(), make_evaluator())

The older evaluate_quick/evaluate_local commands still use StarterTuner. This
file's command explicitly tests FinalTuner; editing StarterTuner alone does not
change the final submission. Keep expensive computation inside tune(), so that
importing this module never starts an evaluation.
"""
from time import perf_counter

from hackaton_files import RmsTuner, RmsTuning, TunableGrid
from .evaluate_quick import evaluate_quick
from .scenarios import QUICK_SCENARIOS, load_benchmark, make_scenario


class FinalTuner(RmsTuner):
    """Participant entry point: replace the unchanged baseline with your strategy."""

    def tune(self, tunable_grid: TunableGrid) -> RmsTuning:
        """Choose allowed actions for this scenario and return one final proposal.

        tunable_parameters contains control handles with initial_value, minimum
        and maximum. control_tensor provides the same data as numerical rows.
        For example, collect proposed values in a mapping keyed by those handles
        and return RmsTuning(parameters=your_mapping).

        You may also use dispatch_options and the permitted line/shunt catalogs.
        Respect their bounds and budgets; total scheduled MW must be preserved
        by redispatch. Do not retain handles from a different scenario.

        self.get_modal_analysis(tunable_grid) inspects the current modes without
        consuming a trial. tunable_grid.test_tuning(proposal) evaluates a trial
        using the attached full evaluator (including its transient test), up to
        the supplied trial budget. Do not call self.test_tuning() here: that would
        invoke tune() again recursively. Return your chosen proposal instead of
        directly modifying tunable_grid.problem or its grid.
        """
        return RmsTuning()


def run_basic_scenarios(tuner_class=FinalTuner) -> int:
    """Run all five sample cases; return 0 on success, 1 on any failure/error.

    Passing a tuner class is convenient in notebooks or automated checks:
        run_basic_scenarios(FinalTuner)
    Each class must inherit from RmsTuner and support construction without args.
    """
    if not isinstance(tuner_class, type) or not issubclass(tuner_class, RmsTuner):
        raise TypeError("The submission must be an RmsTuner subclass")

    start = perf_counter()
    benchmark = load_benchmark()
    failures = 0
    print(f"Testing {tuner_class.__name__}: AC power flow + modal screening")
    for scenario in QUICK_SCENARIOS:
        try:
            # The shared quick evaluator initializes the problem, calls tune(),
            # checks the returned proposal, applies it and recalculates modes.
            max_real, damping, passed = evaluate_quick(
                tuner_class(), make_scenario(benchmark, scenario))
            damping_text = "n/a" if damping is None else f"{100 * damping:.2f}%"
            print(f"{scenario.name:18s} max Re(lambda)={max_real:+.5f}  "
                  f"damping={damping_text:>8s}  {'PASS' if passed else 'FAIL'}")
            failures += int(not passed)
        except Exception as error:
            # Report failures and continue: a broken case must not disappear
            # from the final denominator or prevent other cases being checked.
            failures += 1
            print(f"{scenario.name:18s} ERROR: {type(error).__name__}: {error}")

    count = len(QUICK_SCENARIOS)
    print(f"FinalTuner basic check: {count - failures}/{count} passed; "
          f"{perf_counter() - start:.3f} s.")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(run_basic_scenarios())
