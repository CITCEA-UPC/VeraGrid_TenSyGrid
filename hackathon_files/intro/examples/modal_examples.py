"""Inspect physical modes, including the intro benchmark's descriptor system."""
import numpy as np
from hackathon_files import BasicRmsTuner
from hackathon_files.intro.evaluator import initialize_grid, load_benchmark


def main() -> int:
    grid = initialize_grid(load_benchmark())
    tuner = BasicRmsTuner()
    analysis = tuner.get_modal_analysis(grid)
    # A cached analysis can be reused without another eigensolve.
    assert tuner.get_modal_analysis(analysis) is analysis
    matrix = tuner.get_state_matrix(analysis)
    print(f"State matrix: {matrix.shape}; mass matrix: "
          f"{None if analysis['mass_matrix'] is None else analysis['mass_matrix'].shape}")
    # A descriptor system has E dz/dt = A z. Use the supplied generalized
    # eigenvalues; ordinary eig(A) would ignore E and give incorrect modes.
    physical = analysis['eigenvalues'][np.isfinite(analysis['eigenvalues'])]
    print(f"Physical eigenvalues: {physical}")
    for mode in tuner.get_modes_from_unstable_eigenvalues(
        analysis, real_tolerance=1e-8, top_n=3, include_conjugates=False,
    ):
        print(f"Growing mode {mode['eigenvalue']}: "
              f"{mode['frequency_hz']:.3f} Hz, damping {mode['damping_ratio']:.2%}")
        for participant in mode['participants'][:3]:
            variable = participant['variable']
            print(variable['device_name'], variable['block_path'],
                  participant['participation'], variable['tunable_parameters'])
    print(f"Trials consumed: {grid.tests_used}/{grid.max_test_tunings}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
