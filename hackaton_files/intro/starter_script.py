"""Start here: load a grid, solve AC power flow, and inspect its RMS modes.

Run from the repository root:
    PYTHONPATH=src:. python -m hackaton_files.intro.starter_script

Edit StarterTuner.tune() to implement your team's algorithm. Both evaluation
scripts import this same class, so there is only one submission to maintain.
"""
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals
from VeraGridEngine.api import open_file
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import SmallSignalStabilityRmsDriver
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_options import RmsSmallSignalStabilityOptions
from hackaton_files import RmsTuner, RmsTuning, TunableGrid
from .settings import make_evaluator

BENCHMARK_PATH = Path(__file__).with_name("benchmark_grid.veragrid")


class StarterTuner(RmsTuner):
    """A deliberately unchanged baseline; replace tune() with your strategy."""

    def tune(self, tunable_grid: TunableGrid) -> RmsTuning:
        # Each handle identifies an allowed control, its current value and bounds.
        # Returning an empty proposal is valid and preserves the baseline.
        # Example (uncomment to try a small proportional-gain change):
        # parameters = {
        #     p: float(np.clip(0.95 * p.initial_value, p.minimum, p.maximum))
        #     for p in tunable_grid.tunable_parameters if p.name == "Kp"
        # }
        # return RmsTuning(parameters=parameters)
        #
        # Useful tools for a more advanced algorithm:
        # analysis = self.get_modal_analysis(tunable_grid)
        # modes = self.get_modes_from_unstable_eigenvalues(analysis)
        # sensitivities = self.get_eigenvalue_sensitivities(tunable_grid)
        # metrics = tunable_grid.test_tuning(RmsTuning(parameters=parameters))
        # The last call consumes the per-scenario trial budget and runs the full
        # evaluator. Return your chosen proposal; do not mutate the input grid.
        return RmsTuning()


def initialize_grid(grid):
    """Solve the operating point and assemble an initialized RMS problem."""
    power_flow = PowerFlowDriver(grid)
    power_flow.run()
    if not bool(power_flow.results.converged):
        raise RuntimeError("AC power flow did not converge")

    # Initialization uses the solved voltages and generator injections.
    driver = SmallSignalStabilityRmsDriver(
        grid=grid, pf_results=power_flow.results,
        sss_options=RmsSmallSignalStabilityOptions(k=0, ss_assessment_time=0.0),
    )
    evaluator = make_evaluator()  # No Kundur-specific line corridors.
    tunable = TunableGrid.from_problem(
        driver.problem, evaluator.control_ranges,
        shunt_bus_names=(), line_corridors=(), evaluator=evaluator,
    )
    return tunable


def main():
    grid = open_file(str(BENCHMARK_PATH))
    tunable = initialize_grid(grid)
    tuner = StarterTuner()

    # Assemble the full descriptor system E dz/dt = A z explicitly. The vector
    # z contains differential states followed by algebraic variables. VeraGrid
    # stores derivative coefficients with the opposite sign, hence the minus.
    problem = tunable.problem
    x0 = problem.get_x0().copy()
    dx0 = np.zeros_like(x0)
    A = problem.get_static_state_matrix(x0, dx0).toarray()
    E = -problem.get_E_matrix(x0, dx0)
    # For routine diagnostics, get_modal_analysis(tunable) also supplies modes
    # and participation factors; it uses a reduced matrix when appropriate.
    # Solve A v = lambda E v. Do not invert E: it can be singular. Infinite
    # eigenvalues represent algebraic constraints, not unstable physical modes.
    eigenvalues = eigvals(A, b=E)
    finite = eigenvalues[np.isfinite(eigenvalues)]
    print(f"Loaded {grid.name}: {len(grid.buses)} buses")
    print(f"AC power flow converged; A shape: {A.shape}")
    print(f"E shape: {None if E is None else E.shape}")
    print(f"Finite eigenvalues: {len(finite)} / {len(eigenvalues)}")
    for value in sorted(finite, key=lambda value: value.real, reverse=True)[:8]:
        print(f"  {value.real:+.6f} {value.imag:+.6f}j")
    print("\nAllowed controls (value, minimum, maximum):")
    for parameter, row in zip(tunable.tunable_parameters, tunable.control_tensor):
        print(f"  {parameter.key}: {row}")
    proposal = tuner.tune(tunable)
    print(f"\nStarter proposal accepted: {proposal.is_accepted(tunable)}")
    print("Run evaluate_quick or evaluate_local to score the proposed tuning.")


if __name__ == "__main__":
    main()
