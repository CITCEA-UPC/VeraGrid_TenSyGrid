"""Saved IEEE cases must remain usable by the hackathon RMS evaluator."""

import pytest

from VeraGridEngine.api import open_file

from hackaton_files import ADDITIONAL_STABLE_GRID_FILES, verify_additional_grid


@pytest.mark.parametrize("path", ADDITIONAL_STABLE_GRID_FILES)
def test_saved_ieee_case_is_stable(path) -> None:
    grid = open_file(str(path))
    assert verify_additional_grid(grid).stable
    if "ieee9" in path.name:
        assert len(grid.transformers2w) == 3
    if "mixed_ibr" in path.name:
        assert any("Ideal grid-forming source" in generator.rms_model.name
                   for generator in grid.generators)
        assert any("PVD1" in generator.rms_model.name
                   for generator in grid.generators)
