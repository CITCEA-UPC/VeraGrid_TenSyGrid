# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Circuit-wide saved-model comparison page for RMS and EMT models."""

from __future__ import annotations

from typing import Dict, List

from PySide6 import QtCore, QtGui, QtWidgets

from VeraGrid.Gui.DynamicModelEditor.ModelComparison.dynamic_model_comparison_models import (
    DynamicModelComparisonGroup,
    DynamicModelComparisonTableModel,
    build_model_comparison_groups,
)
from VeraGrid.Gui.DynamicModelEditor.ModelComparison.dynamic_model_comparison_page_ui import (
    Ui_DynamicModelComparisonPage,
)
from VeraGrid.Gui.DynamicModelEditor.Workspace.dynamic_editor_entries import DynamicEditorEntry
from VeraGrid.Gui.messages import yes_no_question
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.enumerations import DeviceType, DynamicSimulationMode


class DynamicModelComparisonPage(QtWidgets.QWidget):
    """Compare and batch-edit parameters of equivalent saved dynamic models."""

    dirtyStateChanged = QtCore.Signal(bool)

    __slots__ = (
        "circuit",
        "mode",
        "ui",
        "_groups",
        "_tree_model",
        "_tree_proxy",
        "_table_models",
        "_current_table_model",
        "_table_menu",
        "_fill_column_action",
        "_context_index",
        "_prepared_to_delete",
        "_dirty",
        "_updating_tree_item",
    )

    def __init__(
            self,
            circuit: MultiCircuit,
            mode: DynamicSimulationMode,
            parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Create one comparison page for a circuit and simulation family.

        :param circuit: Circuit containing the authoritative device models.
        :param mode: RMS or EMT model family.
        :param parent: Optional owning widget.
        :return: None.
        """
        QtWidgets.QWidget.__init__(self, parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.circuit: MultiCircuit = circuit
        self.mode: DynamicSimulationMode = mode
        self.ui: Ui_DynamicModelComparisonPage = Ui_DynamicModelComparisonPage()
        self.ui.setupUi(self)
        self._groups: List[DynamicModelComparisonGroup] = list()
        self._tree_model: QtGui.QStandardItemModel = QtGui.QStandardItemModel(self)
        self._tree_proxy: QtCore.QSortFilterProxyModel = QtCore.QSortFilterProxyModel(self)
        self._table_models: Dict[DynamicModelComparisonGroup, DynamicModelComparisonTableModel] = dict()
        self._current_table_model: DynamicModelComparisonTableModel | None = None
        self._table_menu: QtWidgets.QMenu = QtWidgets.QMenu(self.ui.comparisonTableView)
        self._fill_column_action: QtGui.QAction = self._table_menu.addAction(
            self.tr("Assign value to column")
        )
        self._fill_column_action.setIcon(QtGui.QIcon(":/Icons/icons/copy2down.png"))
        self._context_index: QtCore.QModelIndex = QtCore.QModelIndex()
        self._prepared_to_delete: bool = False
        self._dirty: bool = False
        self._updating_tree_item: bool = False

        # The source model holds typed group objects while the proxy performs
        # recursive, case-insensitive filtering without rebuilding families.
        self._tree_model.setHorizontalHeaderLabels(list((self.tr("Model types"),)))
        self._tree_proxy.setSourceModel(self._tree_model)
        self._tree_proxy.setRecursiveFilteringEnabled(True)
        self._tree_proxy.setFilterCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.ui.modelsTreeView.setModel(self._tree_proxy)

        # Keep the model-family catalog compact so the comparison spreadsheet
        # receives most of the horizontal space and remains column-resizable.
        self.ui.comparisonSplitter.setStretchFactor(0, 0)
        self.ui.comparisonSplitter.setStretchFactor(1, 1)
        self.ui.comparisonSplitter.setSizes(list((260, 740)))
        comparison_header: QtWidgets.QHeaderView = self.ui.comparisonTableView.horizontalHeader()
        comparison_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        comparison_header.setStretchLastSection(False)
        comparison_header.setMinimumSectionSize(70)
        comparison_header.setDefaultSectionSize(110)
        comparison_header.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
        self.ui.comparisonTableView.setWordWrap(False)
        # Replace the generic spreadsheet editing menu with the sole operation
        # that makes sense when comparing one parameter across many devices.
        self.ui.comparisonTableView.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.ui.comparisonTableView.set_fill_handle_visible(visible=False)

        self._connect_signals()
        # The comparison owns a snapshot of the saved models taken at the
        # explicit open request. Open Dynamic Editor documents are independent.
        self._rebuild_from_saved_models()

    @property
    def has_unapplied_changes(self) -> bool:
        """Return whether names or parameter values are staged.

        :return: Page dirty state.
        """
        return self._dirty

    def get_dynamic_editor_entry(self) -> DynamicEditorEntry | None:
        """Return no device entry because the page covers the complete circuit.

        :return: Always ``None``.
        """
        return None

    def get_dynamic_editor_mode(self) -> DynamicSimulationMode:
        """Return the simulation family represented by the page.

        :return: RMS or EMT mode.
        """
        return self.mode

    def get_dynamic_editor_display_title(self) -> str:
        """Return the global comparison tab title.

        :return: RMS or EMT comparison title.
        """
        return self.tr("{mode} comparison").format(mode=self.mode.name)

    def can_close_editor(self, parent: QtWidgets.QWidget | None = None) -> bool:
        """Confirm discarding staged comparison edits before tab destruction.

        :param parent: Widget owning the confirmation dialogue.
        :return: Whether the page may close.
        """
        if not self.has_unapplied_changes:
            return True
        else:
            dialogue_parent: QtWidgets.QWidget = self if parent is None else parent
            return yes_no_question(
                text=self.tr("There are unapplied changes. Do you want to close without applying them?"),
                title=self.tr("Unsaved changes"),
                parent=dialogue_parent,
            )

    def _connect_signals(self) -> None:
        """Connect page controls, model edits and session notifications.

        :return: None.
        """
        self.ui.actionSave.triggered.connect(self.save_changes)
        self.ui.searchButton.clicked.connect(self.apply_search)
        self.ui.searchLineEdit.textChanged.connect(self.apply_search_text)
        selection_model: QtCore.QItemSelectionModel | None = self.ui.modelsTreeView.selectionModel()
        if selection_model is not None:
            selection_model.currentChanged.connect(self._on_tree_selection_changed)
        else:
            pass
        self._tree_model.itemChanged.connect(self._on_tree_item_changed)
        self.ui.comparisonTableView.customContextMenuRequested.connect(self._show_table_context_menu)
        self._fill_column_action.triggered.connect(self._fill_current_column)

    def _disconnect_signals(self) -> None:
        """Disconnect every long-lived signal before native Qt deletion.

        :return: None.
        """
        try:
            self.ui.actionSave.triggered.disconnect(self.save_changes)
        except (RuntimeError, TypeError):
            pass
        try:
            self.ui.searchButton.clicked.disconnect(self.apply_search)
        except (RuntimeError, TypeError):
            pass
        try:
            self.ui.searchLineEdit.textChanged.disconnect(self.apply_search_text)
        except (RuntimeError, TypeError):
            pass
        selection_model: QtCore.QItemSelectionModel | None = self.ui.modelsTreeView.selectionModel()
        if selection_model is not None:
            try:
                selection_model.currentChanged.disconnect(self._on_tree_selection_changed)
            except (RuntimeError, TypeError):
                pass
        else:
            pass
        try:
            self._tree_model.itemChanged.disconnect(self._on_tree_item_changed)
        except (RuntimeError, TypeError):
            pass
        try:
            self.ui.comparisonTableView.customContextMenuRequested.disconnect(self._show_table_context_menu)
        except (RuntimeError, TypeError):
            pass
        try:
            self._fill_column_action.triggered.disconnect(self._fill_current_column)
        except (RuntimeError, TypeError):
            pass

    def _rebuild_from_saved_models(self) -> None:
        """Replace every projection from authoritative saved device models.

        :return: None.
        """
        self._dispose_table_models()
        self._groups = build_model_comparison_groups(circuit=self.circuit, mode=self.mode)
        self._updating_tree_item = True
        self._tree_model.clear()
        self._tree_model.setHorizontalHeaderLabels(list((self.tr("Model types"),)))
        type_items: Dict[DeviceType, QtGui.QStandardItem] = dict()
        group: DynamicModelComparisonGroup
        for group in self._groups:
            type_item: QtGui.QStandardItem | None = type_items.get(group.device_type, None)
            if type_item is None:
                type_item = QtGui.QStandardItem(str(group.device_type.value))
                type_item.setEditable(False)
                type_items[group.device_type] = type_item
                self._tree_model.invisibleRootItem().appendRow(type_item)
            else:
                pass
            group_item: QtGui.QStandardItem = QtGui.QStandardItem(group.get_display_name())
            group_item.setEditable(group.is_name_editable())
            group_item.setData(group, int(QtCore.Qt.ItemDataRole.UserRole) + 1)
            if group.is_name_editable():
                group_item.setToolTip(self.tr("Double-click or press F2 to rename this structural model family"))
            else:
                group_item.setToolTip(self.tr("Devices assigned to the same native template"))
            type_item.appendRow(group_item)
        self._updating_tree_item = False
        self.ui.modelsTreeView.expandAll()
        self.ui.statusLabel.setVisible(False)
        self._set_dirty(False)
        self._select_first_group()

    def _dispose_table_models(self) -> None:
        """Detach and delete every lazily created group table model.

        :return: None.
        """
        self.ui.comparisonTableView.setModel(None)
        table_model: DynamicModelComparisonTableModel
        for table_model in self._table_models.values():
            try:
                table_model.dirtyStateChanged.disconnect(self._on_table_dirty_changed)
            except (RuntimeError, TypeError):
                pass
            table_model.setParent(None)
            table_model.deleteLater()
        self._table_models.clear()
        self._current_table_model = None
        self._context_index = QtCore.QModelIndex()
        self._fill_column_action.setEnabled(False)

    def _select_first_group(self) -> None:
        """Select the first available model family after a rebuild.

        :return: None.
        """
        if self._tree_proxy.rowCount() == 0:
            self.ui.comparisonTableView.setModel(None)
            return
        else:
            first_type: QtCore.QModelIndex = self._tree_proxy.index(0, 0)
        if self._tree_proxy.rowCount(first_type) == 0:
            self.ui.comparisonTableView.setModel(None)
        else:
            first_group: QtCore.QModelIndex = self._tree_proxy.index(0, 0, first_type)
            self.ui.modelsTreeView.setCurrentIndex(first_group)

    @QtCore.Slot()
    def apply_search(self) -> None:
        """Apply the current model-family search text.

        :return: None.
        """
        self.apply_search_text(self.ui.searchLineEdit.text())

    @QtCore.Slot(str)
    def apply_search_text(self, search_text: str) -> None:
        """Filter device types and model families recursively.

        :param search_text: Text entered in the left search field.
        :return: None.
        """
        self._tree_proxy.setFilterFixedString(search_text.strip())
        self.ui.modelsTreeView.expandAll()

    def _group_from_proxy_index(
            self,
            proxy_index: QtCore.QModelIndex,
    ) -> DynamicModelComparisonGroup | None:
        """Resolve one visible tree index to its typed model family.

        :param proxy_index: Index in the recursive filter proxy.
        :return: Model group or ``None`` for device-type rows.
        """
        if not proxy_index.isValid():
            return None
        else:
            source_index: QtCore.QModelIndex = self._tree_proxy.mapToSource(proxy_index)
            item: QtGui.QStandardItem | None = self._tree_model.itemFromIndex(source_index)
        if item is None:
            return None
        else:
            value: object = item.data(int(QtCore.Qt.ItemDataRole.UserRole) + 1)
        if isinstance(value, DynamicModelComparisonGroup):
            return value
        else:
            return None

    @QtCore.Slot(QtCore.QModelIndex, QtCore.QModelIndex)
    def _on_tree_selection_changed(
            self,
            current: QtCore.QModelIndex,
            _previous: QtCore.QModelIndex,
    ) -> None:
        """Show the spreadsheet for the newly selected model family.

        :param current: New proxy-model index.
        :param _previous: Previous proxy-model index supplied by Qt.
        :return: None.
        """
        group: DynamicModelComparisonGroup | None = self._group_from_proxy_index(current)
        if group is None:
            self.ui.comparisonTableView.setModel(None)
            self._current_table_model = None
            return
        else:
            pass
        table_model: DynamicModelComparisonTableModel | None = self._table_models.get(group, None)
        if table_model is None:
            table_model = DynamicModelComparisonTableModel(group=group, parent=self)
            table_model.dirtyStateChanged.connect(self._on_table_dirty_changed)
            self._table_models[group] = table_model
        else:
            pass
        self._current_table_model = table_model
        self.ui.comparisonTableView.setModel(table_model)
        self._fit_initial_table_columns()

    def _fit_initial_table_columns(self) -> None:
        """Give the device column room while retaining readable parameters.

        :return: None.
        """
        table_model: DynamicModelComparisonTableModel | None = self._current_table_model
        if table_model is None:
            return
        else:
            header: QtWidgets.QHeaderView = self.ui.comparisonTableView.horizontalHeader()
        column_index: int
        for column_index in range(table_model.columnCount()):
            if column_index == 0:
                header.resizeSection(column_index, 155)
            else:
                header.resizeSection(column_index, 110)

    @QtCore.Slot(QtGui.QStandardItem)
    def _on_tree_item_changed(self, item: QtGui.QStandardItem) -> None:
        """Validate and stage an inline structural-family rename.

        :param item: Edited source-model item.
        :return: None.
        """
        if self._updating_tree_item:
            return
        else:
            value: object = item.data(int(QtCore.Qt.ItemDataRole.UserRole) + 1)
        if not isinstance(value, DynamicModelComparisonGroup) or not value.is_name_editable():
            return
        else:
            group: DynamicModelComparisonGroup = value
            normalized_name: str = item.text().strip()
        valid_name: bool = normalized_name != "" and not self._has_duplicate_group_name(
            group=group,
            candidate_name=normalized_name,
        )
        if valid_name:
            group.set_pending_name(name=normalized_name)
            if item.text() != normalized_name:
                self._updating_tree_item = True
                item.setText(normalized_name)
                self._updating_tree_item = False
            else:
                pass
            self._refresh_dirty_state()
        else:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Invalid model-family name"),
                self.tr("Model-family names must be non-empty and unique within the device type."),
            )
            self._updating_tree_item = True
            item.setText(group.get_display_name())
            self._updating_tree_item = False

    def _has_duplicate_group_name(
            self,
            group: DynamicModelComparisonGroup,
            candidate_name: str,
    ) -> bool:
        """Check one staged name against sibling model families.

        :param group: Group being renamed.
        :param candidate_name: Normalized proposed label.
        :return: Whether a sibling already uses the label.
        """
        candidate_key: str = candidate_name.casefold()
        sibling: DynamicModelComparisonGroup
        for sibling in self._groups:
            if sibling is group or sibling.device_type != group.device_type:
                pass
            elif sibling.get_display_name().casefold() == candidate_key:
                return True
            else:
                pass
        return False

    @QtCore.Slot(bool)
    def _on_table_dirty_changed(self, _dirty: bool) -> None:
        """Refresh the aggregate page state after one table changes.

        :param _dirty: Sender table dirty state.
        :return: None.
        """
        self._refresh_dirty_state()

    def _refresh_dirty_state(self) -> None:
        """Combine staged family names and all lazily opened tables.

        :return: None.
        """
        dirty: bool = False
        group: DynamicModelComparisonGroup
        for group in self._groups:
            if group.has_name_changes():
                dirty = True
            else:
                pass
        table_model: DynamicModelComparisonTableModel
        for table_model in self._table_models.values():
            if table_model.has_changes():
                dirty = True
            else:
                pass
        self._set_dirty(dirty=dirty)

    def _set_dirty(self, dirty: bool) -> None:
        """Set and emit the page dirty state only when it changes.

        :param dirty: New aggregate dirty state.
        :return: None.
        """
        if dirty != self._dirty:
            self._dirty = dirty
            self.dirtyStateChanged.emit(dirty)
        else:
            pass

    @QtCore.Slot(QtCore.QPoint)
    def _show_table_context_menu(self, position: QtCore.QPoint) -> None:
        """Offer the dataset-style whole-column assignment for one clicked cell.

        :param position: Table-local context-menu position.
        :return: None.
        """
        context_index: QtCore.QModelIndex = self.ui.comparisonTableView.indexAt(position)
        if context_index.isValid():
            # Make the clicked cell the source, matching the dataset table
            # behavior even when it was not part of the previous selection.
            self.ui.comparisonTableView.setCurrentIndex(context_index)
            self._context_index = context_index
            enabled: bool = (
                self._current_table_model is not None
                and context_index.column() > 0
                and bool(context_index.flags() & QtCore.Qt.ItemFlag.ItemIsEditable)
            )
            self._fill_column_action.setEnabled(enabled)
            self._table_menu.exec(self.ui.comparisonTableView.viewport().mapToGlobal(position))
        else:
            self._context_index = QtCore.QModelIndex()

    @QtCore.Slot()
    def _fill_current_column(self) -> None:
        """Copy the current cell value to every device in its parameter column.

        :return: None.
        """
        table_model: DynamicModelComparisonTableModel | None = self._current_table_model
        context_index: QtCore.QModelIndex = self._context_index
        if table_model is not None and context_index.isValid() and context_index.column() > 0:
            table_model.fill_column_from_row(
                column_index=context_index.column(),
                source_row=context_index.row(),
            )
        else:
            pass

    @QtCore.Slot()
    def save_changes(self) -> None:
        """Commit every change staged in this comparison snapshot.

        :return: None.
        """
        if not self.has_unapplied_changes:
            return
        else:
            pass

        # All parsing happened while staging cells, so applying this page does
        # not inspect or mutate any open Dynamic Editor working document.
        group: DynamicModelComparisonGroup
        for group in self._groups:
            group.apply_name()
        table_model: DynamicModelComparisonTableModel
        for table_model in self._table_models.values():
            table_model.apply_changes()
        self._set_dirty(False)
        # A successful save is represented by the clean tab title, matching the
        # Dynamic Editor workflow. Reserve this label for actionable warnings.
        self.ui.statusLabel.clear()
        self.ui.statusLabel.setVisible(False)

    def refresh_from_saved_model(self) -> None:
        """Replace a clean comparison with a new saved-model snapshot.

        The workspace calls this method only for an explicit comparison-button
        request. Normal tab activation and Dynamic Editor saves do not refresh
        or validate the snapshot.

        :return: None.
        """
        if self.has_unapplied_changes:
            pass
        else:
            self._rebuild_from_saved_models()

    def prepare_to_delete(self) -> None:
        """Sever models, proxies and session signals before Qt destroys the tab.

        :return: None.
        """
        if self._prepared_to_delete:
            return
        else:
            self._prepared_to_delete = True
        self._disconnect_signals()
        self.ui.comparisonTableView.closePersistentEditor(self.ui.comparisonTableView.currentIndex())
        self._dispose_table_models()
        self.ui.modelsTreeView.setModel(None)
        self._tree_proxy.setSourceModel(None)
        self._table_menu.clear()
        self._tree_proxy.setParent(None)
        self._tree_proxy.deleteLater()
        self._tree_model.setParent(None)
        self._tree_model.deleteLater()
        self._table_menu.setParent(None)
        self._table_menu.deleteLater()
        self._groups.clear()

    def set_dark_mode(self) -> None:
        """Accept a workspace dark-theme notification.

        :return: None.
        """
        self.ui.comparisonTableView.viewport().update()
        self.ui.modelsTreeView.viewport().update()

    def set_light_mode(self) -> None:
        """Accept a workspace light-theme notification.

        :return: None.
        """
        self.ui.comparisonTableView.viewport().update()
        self.ui.modelsTreeView.viewport().update()

    def set_light_mode(self) -> None:
        """Accept a workspace light-theme notification.

        :return: None.
        """
        self.ui.comparisonTableView.viewport().update()
        self.ui.modelsTreeView.viewport().update()
