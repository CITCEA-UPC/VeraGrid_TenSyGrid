"""Verify the saved polar STAMP grid against the reference RMS spectrum."""
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from VeraGridEngine.api import open_file
from hackathon_files.basic import initialize_grid
from hackathon_files.basic.validation import equilibrium_error
from hackathon_files.src.rms_diagnostics import analyze_grid


def test_saved_stamp_grid_initializes_at_the_reference_equilibrium():
    data = Path(__file__).resolve().parents[1] / 'data' / 'stamp_wscc'
    snapshot = initialize_grid(open_file(str(data / 'wscc_rms.veragrid')))
    assert equilibrium_error(snapshot.problem) < 1e-6

    modes = analyze_grid(snapshot)['eigenvalues']
    reference = np.loadtxt(data / 'stamp_rms_eigenvalues.csv', delimiter=',',
                           skiprows=1, usecols=(1, 2))
    expected = reference[:, 0] + 1j * reference[:, 1]
    assert len(modes) == len(expected) == 88
    assert np.all(np.isfinite(modes))
    distances = abs(modes[:, None] - expected[None, :])
    rows, columns = linear_sum_assignment(distances)
    assert max(distances[rows, columns]) < 1e-5
