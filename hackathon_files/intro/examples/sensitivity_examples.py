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
    # 1. Use a supported semi-explicit model and select a permitted handle.
    path = Path(__file__).resolve().parents[2] / 'basic/testgrids/basic_ieee9_01.veragrid'
    grid = initialize_grid(open_file(str(path)))
    tuner = BasicRmsTuner()
    parameter = grid.tunable_parameters[0]
    model = tuner.get_parameterized_state_matrix(grid)
    print(f'1. Selected parameter: {parameter.name}')
    print(f'   Initial value: {parameter.initial_value:g}')
    print(f'   Allowed range: [{parameter.minimum:g}, {parameter.maximum:g}]')
    print('   Editable symbols:', model['parameter_symbols'])
    print('Symbolic Jacobian blocks:', {k: v.shape for k, v in model['symbolic_blocks'].items()})
    # The helper returns native symbolic expressions, not just numeric arrays.
    # Keep algebraic variables in E dz/dt = A(p) z to avoid a symbolic inverse.
    pencil = tuner.get_symbolic_state_matrices(grid)
    print(f"   Symbolic A(p): {pencil['state_matrix'].shape}")
    print(f"   Symbolic E: {pencil['mass_matrix'].shape}")
    symbol = pencil['parameter_symbols'][parameter]
    # Print only three entries involving this control; full matrices are large.
    shown = 0
    for (row, column), expression in np.ndenumerate(pencil['state_matrix']):
        if symbol in expression.get_vars():
            print(f'   A[{row}, {column}] = {expression}')
            print(f'     Partial derivative: {expression.diff(symbol).simplify()}')
            shown += 1
            if shown == 3:
                break
    if not shown:
        print('   This control has no symbolic dependence in A at this point.')

    # 2. Override one permitted handle; unspecified controls retain their defaults.
    value = float(np.clip(parameter.initial_value - 1, parameter.minimum, parameter.maximum))
    values = {parameter: value}
    residual = tuner.get_equilibrium_residual(grid, values)
    print(f'\n2. Proposed value: {value:g}; equilibrium residual: {residual:.6g}')
    if residual > 1e-6:
        # Keep the default equilibrium guard. A full trial reinitializes after
        # control changes; a local Jacobian at a non-equilibrium is insufficient.
        print('Proposal shifts equilibrium; using baseline values for derivatives.')
        values = None
    # 3. Evaluate A and its derivative at the checked operating point.
    matrix = tuner.evaluate_state_matrix(grid, values)
    np.testing.assert_allclose(tuner.get_state_matrix(grid, values), matrix)
    derivative = tuner.get_state_matrix_derivative(grid, parameter, values)
    sensitivity = tuner.get_eigenvalue_sensitivities(grid, values)
    print(f'\n3. State matrix A: {matrix.shape}')
    print(f'   Norm of dA/dp: {np.linalg.norm(derivative):.6g}')
    # Near-repeated or ill-conditioned modes have invalid derivatives (NaN).
    valid = np.flatnonzero(sensitivity['valid_modes'])
    if len(valid):
        worst = valid[np.argmax(sensitivity['eigenvalues'][valid].real)]
        column = sensitivity['parameters'].index(parameter)
        print(f"   Valid mode with largest real part: {sensitivity['eigenvalues'][worst]:+.6g}")
        print(f"   d(lambda)/d({parameter.name}): {sensitivity['derivatives'][worst, column]:+.6g}")
    else:
        print('   No valid mode derivatives; no sensitivity direction to report.')
    print(f'Trials consumed: {grid.tests_used}/{grid.max_test_tunings}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
