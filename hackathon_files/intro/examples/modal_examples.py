"""Inspect physical modes, including the intro benchmark's descriptor system."""
import numpy as np
from hackathon_files import BasicRmsTuner
from hackathon_files.intro.evaluator import initialize_grid, load_benchmark


def main() -> int:
    # 1. Initialize the benchmark, then compute one reusable analysis.
    grid = initialize_grid(load_benchmark())
    tuner = BasicRmsTuner()
    analysis = tuner.get_modal_analysis(grid)
    # A cached analysis can be reused without another eigensolve.
    assert tuner.get_modal_analysis(analysis) is analysis
    matrix = tuner.get_state_matrix(analysis)
    print('1. Matrix dimensions')
    print(f'   State matrix A: {matrix.shape}')
    mass_matrix = analysis['mass_matrix']
    print(f'   Mass matrix E: {None if mass_matrix is None else mass_matrix.shape}')

    # 2. A descriptor system has E dz/dt = A z. Use the supplied generalized
    # eigenvalues; ordinary eig(A) would ignore E and give incorrect modes.
    eigenvalues = analysis['eigenvalues']
    finite_eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    print(f'\n2. Finite eigenvalues: {len(finite_eigenvalues)}')
    print('   Largest real parts first (up to ten):')
    for eigenvalue in sorted(finite_eigenvalues, key=lambda value: value.real, reverse=True)[:10]:
        print(f'   {eigenvalue:+.6g}')
    print('   Positive real parts indicate growth; zero may be a reference mode.')

    # 3. Inspect growing modes and the states that participate most strongly.
    print('\n3. Growing modes and their leading participants')
    growing_modes = tuner.get_modes_from_unstable_eigenvalues(
        analysis, real_tolerance=1e-8, top_n=3, include_conjugates=False,
    )
    if not growing_modes:
        print('   No growing modes found at the selected tolerance.')
    for mode in growing_modes:
        print(f"Growing mode {mode['eigenvalue']}: "
              f"{mode['frequency_hz']:.3f} Hz, damping {mode['damping_ratio']:.2%}")
        for participant in mode['participants'][:3]:
            variable = participant['variable']
            print(
                f"  Device: {variable['device_name']} | block: {variable['block_path']} | "
                f"participation: {participant['participation']:.2%}"
            )
            print(f"    Tunable handles: {variable['tunable_parameters']}")
    print(f"Trials consumed: {grid.tests_used}/{grid.max_test_tunings}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
