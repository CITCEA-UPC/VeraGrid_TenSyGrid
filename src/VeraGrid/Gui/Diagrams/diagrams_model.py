# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import json
from typing import List, Union, Optional, Any, Dict
from PySide6 import QtCore, QtGui, QtWidgets

from VeraGridEngine.Devices.Diagrams.base_diagram import BaseDiagram
from VeraGridEngine.Devices.Diagrams.diagram_tree import DiagramFolder, DiagramTree
from VeraGridEngine.Devices.Diagrams.schematic_diagram import SchematicDiagram
from VeraGridEngine.Devices.Diagrams.map_diagram import MapDiagram
from VeraGrid.Gui.Diagrams.SchematicWidget.schematic_widget import SchematicWidget
from VeraGrid.Gui.Diagrams.MapWidget.grid_map_widget import GridMapWidget

DIAGRAM_WIDGET_TYPES = Union[SchematicWidget, GridMapWidget]


class DiagramsTreeModel(QtCore.QAbstractItemModel):
    """
    Hierarchical tree model adapter for DiagramTree and diagram widgets.
    Supports folders upon folders of diagrams, inline renaming, drag-and-drop,
    and fallback compatibility with flat widget lists.
    """
    MIME_TYPE = "application/x-veragrid-diagram-tree-item"
    item_dropped = QtCore.Signal(object)

    def __init__(self,
                 tree_or_widgets: Optional[Union[DiagramTree, List[DIAGRAM_WIDGET_TYPES]]] = None,
                 widgets_list: Optional[List[DIAGRAM_WIDGET_TYPES]] = None,
                 parent=None,
                 tree: Optional[DiagramTree] = None):
        super().__init__(parent)

        if tree is not None:
            tree_or_widgets = tree
        elif tree_or_widgets is None:
            tree_or_widgets = DiagramTree()

        if isinstance(tree_or_widgets, DiagramTree):
            self._tree: DiagramTree = tree_or_widgets
            self._widgets: List[DIAGRAM_WIDGET_TYPES] = widgets_list if widgets_list is not None else list()
        else:
            # Backward compatibility: list of widgets passed directly
            self._widgets = list(tree_or_widgets)
            self._tree = DiagramTree()
            for widget in self._widgets:
                if hasattr(widget, 'diagram'):
                    self._tree.add_diagram(widget.diagram)

        # For backwards compatibility with code accessing model.items
        self.items = self._widgets

        self.bus_branch_editor_icon = QtGui.QIcon()
        self.bus_branch_editor_icon.addPixmap(QtGui.QPixmap(":/Icons/icons/schematic.png"))

        self.map_editor_icon = QtGui.QIcon()
        self.map_editor_icon.addPixmap(QtGui.QPixmap(":/Icons/icons/map.png"))

        self.folder_icon = QtGui.QIcon()
        if QtWidgets.QApplication.instance():
            self.folder_icon = QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirIcon)
        if self.folder_icon.isNull():
            self.folder_icon.addPixmap(QtGui.QPixmap(":/Icons/icons/tree.png"))

    @property
    def tree(self) -> DiagramTree:
        return self._tree

    @property
    def widgets(self) -> List[DIAGRAM_WIDGET_TYPES]:
        return self._widgets

    def set_widgets(self, widgets: List[DIAGRAM_WIDGET_TYPES]) -> None:
        self._widgets = widgets
        self.items = widgets

    def get_widget_for_diagram(self, diagram: BaseDiagram) -> Optional[DIAGRAM_WIDGET_TYPES]:
        """
        Find the GUI widget corresponding to a BaseDiagram instance.
        """
        for w in self._widgets:
            if getattr(w, 'diagram', None) is diagram:
                return w
            if getattr(w, 'diagram', None) is not None and w.diagram.idtag == diagram.idtag:
                return w
        return None

    def refresh(self) -> None:
        """
        Full model reset when tree structure changes externally.
        """
        self.beginResetModel()
        self.items = self._widgets
        self.endResetModel()

    # -------------------------------------------------------------------------
    # QAbstractItemModel interface
    # -------------------------------------------------------------------------
    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 1

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        if not parent.isValid():
            return len(self._tree.folders) + len(self._tree.diagrams)

        parent_item = parent.internalPointer()
        if isinstance(parent_item, DiagramFolder):
            return len(parent_item.folders) + len(parent_item.diagrams)

        return 0

    def index(self, row: int, column: int, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> QtCore.QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()

        if not parent.isValid():
            if row < len(self._tree.folders):
                child_item = self._tree.folders[row]
            else:
                diag_idx = row - len(self._tree.folders)
                if diag_idx < len(self._tree.diagrams):
                    child_item = self._tree.diagrams[diag_idx]
                else:
                    return QtCore.QModelIndex()
        else:
            parent_item = parent.internalPointer()
            if isinstance(parent_item, DiagramFolder):
                if row < len(parent_item.folders):
                    child_item = parent_item.folders[row]
                else:
                    diag_idx = row - len(parent_item.folders)
                    if diag_idx < len(parent_item.diagrams):
                        child_item = parent_item.diagrams[diag_idx]
                    else:
                        return QtCore.QModelIndex()
            else:
                return QtCore.QModelIndex()

        return self.createIndex(row, column, child_item)

    def parent(self, index: QtCore.QModelIndex) -> QtCore.QModelIndex:
        if not index.isValid():
            return QtCore.QModelIndex()

        item = index.internalPointer()
        if isinstance(item, DiagramFolder):
            parent_folder = item.parent
        elif isinstance(item, BaseDiagram):
            parent_folder = item.group
        else:
            return QtCore.QModelIndex()

        if parent_folder is None:
            return QtCore.QModelIndex()

        grand_parent = parent_folder.parent
        if grand_parent is None:
            try:
                row = self._tree.folders.index(parent_folder)
            except ValueError:
                return QtCore.QModelIndex()
        else:
            try:
                row = grand_parent.folders.index(parent_folder)
            except ValueError:
                return QtCore.QModelIndex()

        return self.createIndex(row, 0, parent_folder)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        item = index.internalPointer()

        if role in (QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole):
            return item.name

        elif role == QtCore.Qt.ItemDataRole.DecorationRole:
            if isinstance(item, DiagramFolder):
                return self.folder_icon
            elif isinstance(item, SchematicDiagram) or getattr(item, 'diagram_type', None) == getattr(SchematicDiagram, 'diagram_type', None):
                return self.bus_branch_editor_icon
            elif isinstance(item, MapDiagram):
                return self.map_editor_icon
            else:
                return self.bus_branch_editor_icon

        return None

    def setData(self, index: QtCore.QModelIndex, value: Any, role: int = QtCore.Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid():
            return False

        new_name = str(value).strip()
        if not new_name:
            return False

        item = index.internalPointer()
        item.name = new_name

        if isinstance(item, BaseDiagram):
            widget = self.get_widget_for_diagram(item)
            if widget is not None:
                widget.name = new_name

        self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole])
        return True

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        """

        :param index:
        :return:
        """
        if not index.isValid():
            return QtCore.Qt.ItemFlag.ItemIsDropEnabled

        return (QtCore.Qt.ItemFlag.ItemIsEnabled |
                QtCore.Qt.ItemFlag.ItemIsSelectable |
                QtCore.Qt.ItemFlag.ItemIsEditable |
                QtCore.Qt.ItemFlag.ItemIsDragEnabled |
                QtCore.Qt.ItemFlag.ItemIsDropEnabled)

    def index_for_item(self, item: Union[DiagramFolder, BaseDiagram]) -> QtCore.QModelIndex:
        """
        Build the QModelIndex that points to a specific DiagramFolder or BaseDiagram.
        """
        if isinstance(item, DiagramFolder):
            parent_folder = item.parent
            if parent_folder is None:
                if item in self._tree.folders:
                    row = self._tree.folders.index(item)
                    return self.createIndex(row, 0, item)
                return QtCore.QModelIndex()
            parent_index = self.index_for_item(parent_folder)
            if not parent_index.isValid():
                return QtCore.QModelIndex()
            if item in parent_folder.folders:
                row = parent_folder.folders.index(item)
                return self.index(row, 0, parent_index)
            return QtCore.QModelIndex()

        elif isinstance(item, BaseDiagram):
            parent_folder = item.group
            if parent_folder is None:
                if item in self._tree.diagrams:
                    row = len(self._tree.folders) + self._tree.diagrams.index(item)
                    return self.createIndex(row, 0, item)
                return QtCore.QModelIndex()
            parent_index = self.index_for_item(parent_folder)
            if not parent_index.isValid():
                return QtCore.QModelIndex()
            if item in parent_folder.diagrams:
                row = len(parent_folder.folders) + parent_folder.diagrams.index(item)
                return self.index(row, 0, parent_index)
            return QtCore.QModelIndex()

        return QtCore.QModelIndex()

    def get_item(self, index: QtCore.QModelIndex) -> Optional[Union[DiagramFolder, BaseDiagram]]:
        """
        Retrieve the underlying DiagramFolder or BaseDiagram from a model index.
        """
        if index.isValid():
            return index.internalPointer()
        return None

    # -------------------------------------------------------------------------
    # Drag and Drop support
    # -------------------------------------------------------------------------
    def supportedDropActions(self) -> QtCore.Qt.DropAction:
        """

        :return:
        """
        return QtCore.Qt.DropAction.MoveAction | QtCore.Qt.DropAction.CopyAction

    def canDropMimeData(self,
                        data: QtCore.QMimeData,
                        action: QtCore.Qt.DropAction,
                        row: int,
                        column: int,
                        parent: QtCore.QModelIndex) -> bool:
        """

        :param data:
        :param action:
        :param row:
        :param column:
        :param parent:
        :return:
        """
        if not data.hasFormat(self.MIME_TYPE):
            return False
        return True

    def mimeTypes(self) -> List[str]:
        """

        :return:
        """
        return [self.MIME_TYPE]

    def mimeData(self, indexes: List[QtCore.QModelIndex]) -> QtCore.QMimeData:
        """

        :param indexes:
        :return:
        """
        mime = QtCore.QMimeData()
        if not indexes:
            return mime

        valid_indexes = [idx for idx in indexes if idx.isValid()]
        if not valid_indexes:
            return mime

        item = valid_indexes[0].internalPointer()
        payload = {
            "type": "folder" if isinstance(item, DiagramFolder) else "diagram",
            "idtag": item.idtag,
        }
        mime.setData(self.MIME_TYPE, json.dumps(payload).encode("utf-8"))
        return mime

    def dropMimeData(self,
                     data: QtCore.QMimeData,
                     action: QtCore.Qt.DropAction,
                     row: int,
                     column: int,
                     parent: QtCore.QModelIndex) -> bool:
        """

        :param data:
        :param action:
        :param row:
        :param column:
        :param parent:
        :return:
        """
        if not data.hasFormat(self.MIME_TYPE):
            return False

        try:
            payload = json.loads(bytes(data.data(self.MIME_TYPE)).decode("utf-8"))
        except Exception:
            return False

        item_type = payload.get("type")
        idtag = payload.get("idtag")

        # Determine target folder
        target_folder: Optional[DiagramFolder] = None
        if parent.isValid():
            parent_item = parent.internalPointer()
            if isinstance(parent_item, DiagramFolder):
                target_folder = parent_item
            elif isinstance(parent_item, BaseDiagram):
                target_folder = parent_item.group
        else:
            target_folder = None

        if item_type == "diagram":
            diagram = None
            for d in self._tree.get_all_diagrams():
                if d.idtag == idtag:
                    diagram = d
                    break
            if diagram is not None:
                self.beginResetModel()
                self._tree.move_diagram(diagram, target_folder=target_folder)
                self.endResetModel()
                self.item_dropped.emit(diagram)
                return True

        elif item_type == "folder":
            folder = self._tree.find_folder(idtag)
            if folder is not None:
                if target_folder is folder:
                    return False
                if target_folder is not None and folder.find_folder(target_folder.idtag) is not None:
                    return False
                try:
                    self.beginResetModel()
                    self._tree.move_folder(folder, target_parent=target_folder)
                    self.endResetModel()
                    self.item_dropped.emit(folder)
                    return True
                except ValueError:
                    return False

        return False
