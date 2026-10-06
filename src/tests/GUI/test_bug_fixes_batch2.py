# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
import typing
from typing import List, Tuple
import numpy as np
import pandas as pd
from PySide6.QtCore import QRectF
import PySide6.QtWidgets as QtWidgets

from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Devices.Substation.bus import Bus
from VeraGridEngine.Devices.Branches.overhead_line_type import OverheadLineType
from VeraGridEngine.enumerations import DeviceType

from VeraGrid.Gui.Diagrams.SchematicWidget.schematic_widget import SchematicWidget
from VeraGrid.Gui.Diagrams.SchematicWidget.Fluid.fluid_node_graphics import FluidNodeGraphicItem
from VeraGrid.Gui.Main.SubClasses.Model.diagrams import DiagramsMain
from VeraGrid.Gui.Diagrams.SchematicWidget.Injections.injections_template_graphics import InjectionTemplateGraphicItem
from VeraGrid.Gui.DeviceEditors.TowerBuilder.LineBuilderDialogue import TowerBuilderGUI
from VeraGrid.Gui.FileDialogues.CoordinatesInput.coordinates_dialogue import CoordinatesInputGUI
import VeraGrid.Gui.FileDialogues.CoordinatesInput.coordinates_dialogue as coordinates_dialogue_module
from VeraGrid.Gui.FileDialogues.ProfilesInput.profile_dialogue import ProfileInputGUI, StringSubstitutions


class CenterNodesSceneStub:
    """
    Scene stub for testing center_nodes limits and fallback behavior.
    """
    __slots__ = ("_rect", "set_rect_calls")

    def __init__(self, rect: QRectF) -> None:
        """
        Initialize stub with bounding rect.

        :param rect: Initial scene bounding rect
        """
        self._rect: QRectF = rect
        self.set_rect_calls: List[QRectF] = list()

    def itemsBoundingRect(self) -> QRectF:
        """
        Return the stub bounding rect.

        :return: Scene bounding rect
        """
        return QRectF(self._rect)

    def setSceneRect(self, rect: QRectF) -> None:
        """
        Record setSceneRect call.

        :param rect: Scene rect to set
        """
        self.set_rect_calls.append(QRectF(rect))

    def items(self) -> List[object]:
        """
        Return empty items list.

        :return: Empty items list
        """
        return list()


class CenterNodesViewStub:
    """
    Graphics view stub for recording fit and scale calls.
    """
    __slots__ = ("fit_calls", "scale_calls")

    def __init__(self) -> None:
        """
        Initialize view stub.
        """
        self.fit_calls: List[QRectF] = list()
        self.scale_calls: List[Tuple[float, float]] = list()

    def fitInView(self, rect: QRectF, _mode: object) -> None:
        """
        Record fitInView call.

        :param rect: Target rect
        :param _mode: Aspect ratio mode
        """
        self.fit_calls.append(QRectF(rect))

    def scale(self, sx: float, sy: float) -> None:
        """
        Record scale call.

        :param sx: X scale factor
        :param sy: Y scale factor
        """
        self.scale_calls.append((sx, sy))


class CenterNodesWidgetStub:
    """
    Schematic widget stub wrapping scene and view stubs.
    """
    __slots__ = ("diagram_scene", "editor_graphics_view")

    def __init__(self, rect: QRectF) -> None:
        """
        Initialize widget stub.

        :param rect: Initial scene bounding rect
        """
        self.diagram_scene: CenterNodesSceneStub = CenterNodesSceneStub(rect=rect)
        self.editor_graphics_view: CenterNodesViewStub = CenterNodesViewStub()

    def center_nodes(self, margin_factor: float = 0.1, elements: list[object] | None = None) -> None:
        """
        Delegate to SchematicWidget.center_nodes.

        :param margin_factor: Margin factor
        :param elements: Elements to center
        """
        SchematicWidget.center_nodes(self, margin_factor=margin_factor, elements=elements)


class ErrorMessageRecorder:
    """
    Helper to record error messages without blocking UI.
    """
    __slots__ = ("messages",)

    def __init__(self) -> None:
        """
        Initialize error message recorder.
        """
        self.messages: List[str] = list()

    def record(self, msg: str, parent: object = None) -> None:
        """
        Record an error message.

        :param msg: Error message text
        :param parent: Optional Qt parent widget
        """
        self.messages.append(msg)


def test_bug_07_center_nodes_unmatched_elements() -> None:
    """
    Test that center_nodes falls back to full scene bounds when elements do not match scene items.
    """
    initial_rect: QRectF = QRectF(0.0, 0.0, 100.0, 100.0)
    stub: CenterNodesWidgetStub = CenterNodesWidgetStub(rect=initial_rect)

    # Calling with elements that are not in the scene must not overflow scene rect
    unmatched_elements: List[object] = ["non_existent_bus_element"]
    stub.center_nodes(margin_factor=0.1, elements=unmatched_elements)

    assert len(stub.diagram_scene.set_rect_calls) == 1
    rect: QRectF = stub.diagram_scene.set_rect_calls[0]
    assert rect.width() > 0.0
    assert rect.height() > 0.0
    assert rect.width() == 120.0
    assert rect.height() == 120.0


def test_bug_08_tower_builder_show_matrix_empty_and_custom_circuits(qt_app: object) -> None:
    """
    Test that TowerBuilderGUI show_matrix handles empty circuits and custom circuit indices safely.
    """
    tower: OverheadLineType = OverheadLineType()
    dialogue: TowerBuilderGUI = TowerBuilderGUI(tower=tower)

    # Case 1: Empty circuits with None z_seq and y_seq
    dialogue.ui.matrixViewComboBox.setCurrentIndex(2)
    dialogue.show_matrix()
    assert dialogue.ui.matrixTableView.model() is None

    dialogue.ui.matrixViewComboBox.setCurrentIndex(5)
    dialogue.show_matrix()
    assert dialogue.ui.matrixTableView.model() is None

    # Case 2: Populated 3x3 z_seq with single circuit
    tower._z_seq = np.zeros((3, 3), dtype=complex)
    dialogue.ui.matrixViewComboBox.setCurrentIndex(2)
    dialogue.show_matrix()
    assert dialogue.ui.matrixTableView.model() is not None
    assert dialogue.ui.matrixTableView.model().rowCount() == 3
    assert dialogue.ui.matrixTableView.model().columnCount() == 3

    # Case 3: Populated 6x6 y_seq with two circuits
    tower._y_seq = np.zeros((6, 6), dtype=complex)
    dialogue.ui.matrixViewComboBox.setCurrentIndex(5)
    dialogue.show_matrix()
    assert dialogue.ui.matrixTableView.model() is not None
    assert dialogue.ui.matrixTableView.model().rowCount() == 6
    assert dialogue.ui.matrixTableView.model().columnCount() == 6


def test_bug_10_type_annotations_runtime_eval() -> None:
    """
    Test that type hints evaluate at runtime without missing Any or QGraphicsLineItem.
    """
    th_fluid: dict[str, object] = typing.get_type_hints(FluidNodeGraphicItem.auto_assign_branch_slots)
    assert "return" in th_fluid

    th_diagrams: dict[str, object] = typing.get_type_hints(DiagramsMain._on_diagram_tree_item_dropped)
    assert th_diagrams.get("item") is typing.Any

    th_injections: dict[str, object] = typing.get_type_hints(InjectionTemplateGraphicItem.get_associated_widgets)
    assert "return" in th_injections


def test_bug_11_coordinates_input_case_insensitive_and_unsupported(qt_app: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that CoordinatesInputGUI handles uppercase extensions and unsupported formats properly.
    """
    grid: MultiCircuit = MultiCircuit()
    dialogue: CoordinatesInputGUI = CoordinatesInputGUI(grid=grid)

    with tempfile.TemporaryDirectory() as td:
        # Uppercase CSV extension
        csv_file: Path = Path(td) / "coords.CSV"
        csv_file.write_text("Name,Code,X,Y\nBus1,1,10.0,20.0\nBus2,2,30.0,40.0\n")
        dialogue.open_file_now(str(csv_file))

        assert dialogue.original_data_frame is not None
        assert list(dialogue.original_data_frame.columns) == ["Name", "Code", "X", "Y"]
        assert len(dialogue.original_data_frame) == 2

        # Unsupported extension
        txt_file: Path = Path(td) / "coords.txt"
        txt_file.write_text("irrelevant content")
        recorder: ErrorMessageRecorder = ErrorMessageRecorder()
        monkeypatch.setattr(coordinates_dialogue_module, "error_msg", recorder.record)

        dialogue2: CoordinatesInputGUI = CoordinatesInputGUI(grid=grid)
        dialogue2.open_file_now(str(txt_file))

        assert dialogue2.original_data_frame is None
        assert len(recorder.messages) == 1
        assert "Unsupported file format" in recorder.messages[0]


def test_bug_14_profile_dialogue_transform_names(qt_app: object) -> None:
    """
    Test that ProfileInputGUI transform_names transforms column names for each substitution mode.
    """
    grid: MultiCircuit = MultiCircuit()
    bus: Bus = Bus(name="SubstationBus")
    grid.add_bus(bus)
    dialogue: ProfileInputGUI = ProfileInputGUI(
        parent=None,
        circuit=grid,
        dev_type=DeviceType.BusDevice,
        objects=[bus],
        magnitude="Vmin",
    )

    # 1. PSSeBranchName substitution
    dialogue.profile_names = ["FROM_NAME1_100_TO_NAME2_200_1"]
    dialogue.original_data_frame = pd.DataFrame(columns=dialogue.profile_names)
    dialogue.ui.nameTransformationComboBox.setCurrentIndex(
        dialogue.ui.nameTransformationComboBox.findData(StringSubstitutions.PSSeBranchName)
    )
    dialogue.transform_names()
    assert dialogue.profile_names == ["FROM_TO_1"]
    assert list(dialogue.original_data_frame.columns) == ["FROM_TO_1"]

    # 2. PSSeBusGenerator substitution
    dialogue.profile_names = ["BUS1_GEN_1"]
    dialogue.original_data_frame = pd.DataFrame(columns=dialogue.profile_names)
    dialogue.ui.nameTransformationComboBox.setCurrentIndex(
        dialogue.ui.nameTransformationComboBox.findData(StringSubstitutions.PSSeBusGenerator)
    )
    dialogue.transform_names()
    assert dialogue.profile_names == ["BUS1_1"]
    assert list(dialogue.original_data_frame.columns) == ["BUS1_1"]

    # 3. PSSeBusLoad substitution
    dialogue.profile_names = ["LOAD1"]
    dialogue.original_data_frame = pd.DataFrame(columns=dialogue.profile_names)
    dialogue.ui.nameTransformationComboBox.setCurrentIndex(
        dialogue.ui.nameTransformationComboBox.findData(StringSubstitutions.PSSeBusLoad)
    )
    dialogue.transform_names()
    assert dialogue.profile_names == ["LOAD1_1"]
    assert list(dialogue.original_data_frame.columns) == ["LOAD1_1"]
