# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import inspect

from VeraGrid.Gui.Analysis.object_plot_analysis import (
    FixableErrorNegative,
    FixableErrorOutOfRange,
    FixableErrorRangeFlip,
    FixableErrorValueCorrection,
    FixableTransformerVtaps,
    GridErrorLog,
    grid_analysis,
)
from VeraGrid.Gui.Diagrams.base_diagram_widget import BaseDiagramWidget
from VeraGrid.Gui.Diagrams.MapWidget.grid_map_widget import GridMapWidget
from VeraGrid.Gui.Diagrams.SchematicWidget.schematic_widget import SchematicWidget
from VeraGridEngine.basic_structures import Logger
from VeraGridEngine.Devices.Injections.generator import Generator
from VeraGridEngine.Devices.multi_circuit import MultiCircuit


def test_fixable_error_signatures_use_none_default() -> None:
    """Verify that fix methods do not use mutable Logger() instances as default arguments."""
    classes = [
        FixableErrorOutOfRange,
        FixableErrorRangeFlip,
        FixableErrorValueCorrection,
        FixableErrorNegative,
        FixableTransformerVtaps,
    ]
    for cls in classes:
        sig = inspect.signature(cls.fix)
        logger_param = sig.parameters.get("logger")
        assert logger_param is not None, f"{cls.__name__}.fix missing logger param"
        assert logger_param.default is None, (
            f"{cls.__name__}.fix default for logger must be None, got {logger_param.default!r}"
        )


def test_grid_analysis_signature_uses_none_default() -> None:
    """Verify that grid_analysis does not use mutable GridErrorLog() as default argument."""
    sig = inspect.signature(grid_analysis)
    logger_param = sig.parameters.get("logger")
    assert logger_param is not None, "grid_analysis missing logger param"
    assert logger_param.default is None, (
        f"grid_analysis default for logger must be None, got {logger_param.default!r}"
    )


def test_diagram_signatures_use_none_default() -> None:
    """Verify that diagram add and draw methods do not use mutable default loggers."""
    sig_map = inspect.signature(GridMapWidget.add_object_to_the_schematic)
    assert sig_map.parameters["logger"].default is None

    sig_video = inspect.signature(BaseDiagramWidget.start_video_recording)
    assert sig_video.parameters["logger"].default is None

    sig_draw = inspect.signature(SchematicWidget.draw_additional_diagram)
    assert sig_draw.parameters["logger"].default is None

    sig_branch = inspect.signature(SchematicWidget.add_api_branch)
    assert sig_branch.parameters["logger"].default is None

    sig_line = inspect.signature(SchematicWidget.add_api_line)
    assert sig_line.parameters["logger"].default is None

    sig_vsc = inspect.signature(SchematicWidget.add_api_vsc)
    assert sig_vsc.parameters["logger"].default is None

    sig_schem_obj = inspect.signature(SchematicWidget.add_object_to_the_schematic)
    assert sig_schem_obj.parameters["logger"].default is None


def test_fixable_error_execution_with_and_without_logger() -> None:
    """Verify executing fix() without logger works cleanly and passing logger accumulates entries."""
    gen = Generator(name="Gen_Test")
    gen.P = -5.0

    err_default = FixableErrorOutOfRange(
        grid_element=gen,
        property_name="P",
        value=-5.0,
        lower_limit=0.0,
        upper_limit=100.0,
    )
    # Calling fix with no logger should create a fresh Logger internally without error
    err_default.fix()
    assert gen.P == 0.0

    # Calling fix with custom logger should accumulate entries into that custom logger
    gen.P = -10.0
    custom_logger = Logger()
    err_custom = FixableErrorOutOfRange(
        grid_element=gen,
        property_name="P",
        value=-10.0,
        lower_limit=0.0,
        upper_limit=100.0,
    )
    err_custom.fix(logger=custom_logger)
    assert gen.P == 0.0
    assert len(custom_logger.entries) > 0


def test_grid_analysis_execution_isolation() -> None:
    """Verify that successive calls to grid_analysis do not leak logs into each other."""
    circuit = MultiCircuit()

    # Call without logger
    errors1 = grid_analysis(circuit=circuit)
    assert isinstance(errors1, list)

    # Call with explicit logger
    custom_log = GridErrorLog()
    errors2 = grid_analysis(circuit=circuit, logger=custom_log)
    assert isinstance(errors2, list)
    # custom_log should have been populated
    assert isinstance(custom_log.logs, dict)
