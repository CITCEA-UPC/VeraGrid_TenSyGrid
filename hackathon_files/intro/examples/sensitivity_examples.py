"""Evaluate fixed-point matrices and analytic control/mode sensitivities.

These helpers support semi-explicit DAEs; the intro GFM descriptor benchmark
is intentionally not used here. Derivatives are local, not a stability guarantee.
"""
from pathlib import Path
import numpy as np
from VeraGridEngine.api import open_file
from hackathon_files import BasicRmsTuner
from hackathon_files.basic import initialize_grid


def main() -> int:
    path = Path(__file__).resolve().parents[2] / 'basic/testgrids/basic_ieee9_01.veragrid'
    grid = initialize_grid(open_file(str(path)))
    tuner = BasicRmsTuner()
    parameter = grid.tunable_parameters[0]
    model = tuner.get_parameterized_state_matrix(grid)
    print('Editable symbols:', model['parameter_symbols'])
    print('Symbolic Jacobian blocks:', {k: v.shape for k, v in model['symbolic_blocks'].items()})
    # Override one permitted handle; unspecified controls retain their defaults.
    value = float(np.clip(parameter.initial_value - 1, parameter.minimum, parameter.maximum))
    values = {parameter: value}
    residual = tuner.get_equilibrium_residual(grid, values)
    print('Frozen-point equilibrium residual:', residual)
    if residual > 1e-6:
        # Keep the default equilibrium guard. A full trial reinitializes after
        # control changes; a local Jacobian at a non-equilibrium is insufficient.
        print('Proposal shifts equilibrium; using baseline values for derivatives.')
        values = None
    matrix = tuner.evaluate_state_matrix(grid, values)
    np.testing.assert_allclose(tuner.get_state_matrix(grid, values), matrix)
    derivative = tuner.get_state_matrix_derivative(grid, parameter, values)
    sensitivity = tuner.get_eigenvalue_sensitivities(grid, values)
    print('A shape:', matrix.shape, 'dA/dp norm:', np.linalg.norm(derivative))
    # Near-repeated or ill-conditioned modes have invalid derivatives (NaN).
    valid = np.flatnonzero(sensitivity['valid_modes'])
    if len(valid):
        worst = valid[np.argmax(sensitivity['eigenvalues'][valid].real)]
        column = sensitivity['parameters'].index(parameter)
        print('Mode:', sensitivity['eigenvalues'][worst],
              'd lambda/dp:', sensitivity['derivatives'][worst, column])
    print(f'Trials consumed: {grid.tests_used}/{grid.max_test_tunings}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
