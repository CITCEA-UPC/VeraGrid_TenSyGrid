from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import csc_matrix

from VeraGridEngine.Simulations.ShortCircuitStudies.short_circuit import short_circuit_unbalance
from VeraGridEngine.basic_structures import Logger
from VeraGridEngine.enumerations import FaultType


def test_unbalanced_short_circuit_singular_matrix_logs_error_without_warning() -> None:
    """
    Singular sequence matrices must be reported in the logger, not as SciPy warnings.

    :return: None.
    """
    n: int = 2
    logger: Logger = Logger()
    singular_ybus: csc_matrix = csc_matrix((n, n), dtype=complex)
    vbus: np.ndarray = np.ones(n, dtype=complex)
    vnom: np.ndarray = np.ones(n, dtype=float)
    fault_impedance: np.ndarray = np.zeros(n, dtype=complex)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        v0, v1, v2, scc, icc = short_circuit_unbalance(
            bus_idx=0,
            Y0=singular_ybus,
            Y1=singular_ybus,
            Y2=singular_ybus,
            Vbus=vbus,
            Vnom=vnom,
            Zf=fault_impedance,
            fault_type=FaultType.LG,
            baseMVA=100.0,
            logger=logger,
        )

    assert len(recorded_warnings) == 0
    assert logger.has_errors()
    assert np.all(np.isfinite(v0))
    assert np.all(np.isfinite(v1))
    assert np.all(np.isfinite(v2))
    assert np.all(np.isfinite(scc))
    assert np.all(np.isfinite(icc))
