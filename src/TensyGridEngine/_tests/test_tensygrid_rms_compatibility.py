"""The RMS dashboard should explain unsupported saved model coordinates."""

import pytest
import numpy as np

from TensyGridEngine.small_signal_analysis_main import (
    load_grid,
    set_models,
    validate_phasor_rms_models,
    run_small_signal_analysis,
)


def test_stamp_grid_reports_polar_rms_model_mismatch():
    grid = load_grid("ieee9_stamp_gfor_gfol.veragrid")
    set_models(grid)

    with pytest.raises(ValueError, match="requires phasor RMS bus models with Vr and Vi") as error:
        validate_phasor_rms_models(grid)
    assert "Select EMT Floquet" in str(error.value)


def test_generated_phasor_rms_models_are_accepted():
    grid = load_grid("IEEE 9 Bus.gridcal")
    set_models(grid)
    validate_phasor_rms_models(grid)


def test_saved_stamp_phasor_rms_grid_runs_dashboard_ssa():
    grid = load_grid("ieee9_stamp_gfor_gfol_phasor_rms.veragrid")
    set_models(grid)
    assert sum(len(line.rms_model.state_vars) for line in grid.lines) == 12
    assert sum(len(load.rms_model.state_vars) for load in grid.loads) == 18
    assert {gen.name: len(gen.rms_model.state_vars) for gen in grid.generators} == {
        "STAMP SG1": 17,
        "STAMP GFOR1": 21,
        "STAMP GFOL2": 20,
    }

    results = run_small_signal_analysis(grid)
    assert results["problem"].get_states_number() == 88
    assert len(results["eigenvalues"]) == 86
    assert np.all(np.isfinite(results["eigenvalues"]))
