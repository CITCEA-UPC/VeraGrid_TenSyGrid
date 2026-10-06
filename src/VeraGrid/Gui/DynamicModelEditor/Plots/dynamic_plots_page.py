# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Global pre-simulation RMS and EMT plot-definition editor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from VeraGrid.Gui.dialog_lifecycle import delete_dialog_safely, exec_dialog_safely
from VeraGrid.Gui.DynamicModelEditor.Plots.dynamic_plots_page_ui import Ui_DynamicPlotsPage
from VeraGrid.Gui.DynamicModelEditor.Plots.dynamic_plots_handler import DynamicsResultsHandler
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.enumerations import DynamicPlotMode, DynamicSimulationMode, PlotSimulationType

if TYPE_CHECKING:
    from VeraGrid.Gui.DynamicModelEditor.Workspace.dynamic_editor_entries import DynamicEditorEntry
    from VeraGrid.Gui.DynamicModelEditor.Workspace.dynamic_editor_workspace_session import DynamicEditorWorkspaceSession


def get_plot_simulation_type(mode: DynamicSimulationMode) -> PlotSimulationType:
    """Convert a workspace simulation mode to its persistent plot family.

    :param mode: Workspace RMS or EMT mode.
    :return: Matching persistent plot simulation type.
    """
    if mode == DynamicSimulationMode.RMS:
        return PlotSimulationType.RMS
    else:
        return PlotSimulationType.EMT


class DynamicPlotsPage(QtWidgets.QWidget):
    """Edit circuit-wide persistent plot definitions before simulation."""

    dirtyStateChanged = QtCore.Signal(bool)

    __slots__ = (
        "circuit",
        "handler",
        "mode",
        "session",
        "ui",
        "_prepared_to_delete",
    )

    def __init__(self,
                 circuit: MultiCircuit,
                 mode: DynamicSimulationMode,
                 session: "DynamicEditorWorkspaceSession",
                 parent: QtWidgets.QWidget | None = None) -> None:
        """Create one global plots page for a circuit and simulation family.

        :param circuit: Circuit that owns the persistent plot assets.
        :param mode: RMS or EMT family edited by the page.
        :param session: Shared dynamic workspace session used for synchronization.
        :param parent: Optional Qt parent widget.
        :return: None.
        """
        QtWidgets.QWidget.__init__(self, parent)
        self.circuit: MultiCircuit = circuit
        self.mode: DynamicSimulationMode = mode
        self.session: DynamicEditorWorkspaceSession = session
        self._prepared_to_delete: bool = False
        self.ui: Ui_DynamicPlotsPage = Ui_DynamicPlotsPage()
        self.ui.setupUi(self)

        # The handler is deliberately created without results: its source tree
        # comes from every compatible dynamic model in the complete circuit.
        self.handler: DynamicsResultsHandler = DynamicsResultsHandler(
            results=None,
            circuit=self.circuit,
            simulation_type=get_plot_simulation_type(mode=self.mode),
            dialog_parent=self,
        )
        self._configure_views()
        self._connect_signals()
        self._refresh_empty_state()

    @property
    def has_unapplied_changes(self) -> bool:
        """Return whether the page has buffered changes.

        :return: Always ``False`` because plot assets are updated immediately.
        """
        return False

    def get_dynamic_editor_entry(self) -> "DynamicEditorEntry | None":
        """Return the contextual device entry represented by this page.

        :return: Always ``None`` because plot definitions cover the full circuit.
        """
        return None

    def get_dynamic_editor_mode(self) -> DynamicSimulationMode:
        """Return the RMS or EMT family edited by this page.

        :return: Page simulation mode.
        """
        return self.mode

    def get_dynamic_editor_display_title(self) -> str:
        """Return the global tab title.

        :return: RMS or EMT plots title.
        """
        return self.tr("{mode} plots").format(mode=self.mode.name)

    def can_close_editor(self, parent: QtWidgets.QWidget | None = None) -> bool:
        """Allow the immediately-persisted page to close without a prompt.

        :param parent: Unused parent for compatibility with workspace pages.
        :return: Always ``True``.
        """
        del parent
        return True

    def _configure_views(self) -> None:
        """Attach handler models and configure drag-and-drop behavior.

        :return: None.
        """
        self.ui.availableVariablesTreeView.setModel(self.handler.get_view_model())
        self.ui.availableVariablesTreeView.setDragEnabled(True)
        self.ui.availableVariablesTreeView.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.DragOnly
        )
        self.ui.availableVariablesTreeView.setDefaultDropAction(QtCore.Qt.DropAction.CopyAction)
        self.ui.plotsTreeView.setModel(self.handler.get_plots_model())
        self.ui.plotsTreeView.setAcceptDrops(True)
        self.ui.plotsTreeView.setDropIndicatorShown(True)
        self.ui.plotsTreeView.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DropOnly)
        self.ui.plotsTreeView.setDefaultDropAction(QtCore.Qt.DropAction.CopyAction)
        self.ui.plotsTreeView.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.ui.plotsSplitter.setSizes([430, 470])
        self.ui.plotsTreeView.expandAll()

    def _connect_signals(self) -> None:
        """Connect page, model, and cross-screen synchronization signals.

        :return: None.
        """
        self.ui.addDynamicPlotButton.clicked.connect(self.add_plot_group)
        self.ui.deleteDynamicPlotButton.clicked.connect(self.delete_plot_entry)
        self.ui.searchButton.clicked.connect(self.apply_search)
        self.ui.searchLineEdit.returnPressed.connect(self.apply_search)
        self.ui.plotsTreeView.customContextMenuRequested.connect(self.show_plots_context_menu)
        self.handler.get_plots_model().rowsInserted.connect(self._expand_inserted_parent)
        self.handler.get_plots_model().plotDefinitionsChanged.connect(self._notify_plot_definitions_changed)
        self.session.dynamicPlotsChanged.connect(self._on_external_plot_definitions_changed)

    def _refresh_empty_state(self) -> None:
        """Update the source-tree empty-state tooltip.

        :return: None.
        """
        source_model: QtCore.QAbstractItemModel | None = self.handler.get_view_model()
        has_sources: bool = source_model is not None and source_model.rowCount() > 0
        if has_sources:
            tooltip: str = self.tr("Drag a variable or parameter to a dynamic plot")
        else:
            tooltip = self.tr("No {mode} dynamic model variables are available in this circuit").format(
                mode=self.mode.name
            )
        self.ui.availableVariablesTreeView.setToolTip(tooltip)

    def refresh_from_saved_model(self) -> None:
        """Reload circuit model metadata and persistent plot definitions.

        :return: None.
        """
        self.handler.refresh_pre_simulation_content()
        self.ui.availableVariablesTreeView.setModel(self.handler.get_view_model())
        self.ui.plotsTreeView.setModel(self.handler.get_plots_model())
        self.ui.plotsTreeView.expandAll()
        self._refresh_empty_state()

    @QtCore.Slot()
    def apply_search(self) -> None:
        """Filter the circuit-wide source tree using the search text.

        :return: None.
        """
        search_text: str = self.ui.searchLineEdit.text().strip()
        self.handler.set_search_text(search_text=search_text)
        self.ui.availableVariablesTreeView.expandAll()

    @QtCore.Slot()
    def add_plot_group(self) -> None:
        """Create a persistent time-series or X-Y plot group.

        :return: None.
        """
        suggested_name: str = self.handler.get_next_group_name()
        dialog: QtWidgets.QDialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr("New dynamic plot"))
        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout(dialog)
        name_label: QtWidgets.QLabel = QtWidgets.QLabel(self.tr("Plot name"), dialog)
        name_edit: QtWidgets.QLineEdit = QtWidgets.QLineEdit(dialog)
        mode_label: QtWidgets.QLabel = QtWidgets.QLabel(self.tr("Plot mode"), dialog)
        mode_combo: QtWidgets.QComboBox = QtWidgets.QComboBox(dialog)
        buttons: QtWidgets.QDialogButtonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            dialog,
        )
        name_edit.setText(suggested_name)
        mode_combo.addItem(self.tr("Time Series (Y vs Time)"), DynamicPlotMode.TIME_SERIES)
        mode_combo.addItem(self.tr("X-Y Plot (Y vs X)"), DynamicPlotMode.XY)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(name_label)
        layout.addWidget(name_edit)
        layout.addWidget(mode_label)
        layout.addWidget(mode_combo)
        layout.addWidget(buttons)
        try:
            accepted: bool = exec_dialog_safely(dialog=dialog) == QtWidgets.QDialog.DialogCode.Accepted
            if accepted:
                selected_mode_data: object = mode_combo.currentData()
                selected_mode: DynamicPlotMode = DynamicPlotMode.TIME_SERIES
                if isinstance(selected_mode_data, DynamicPlotMode):
                    selected_mode = selected_mode_data
                else:
                    pass
                created: bool = self.handler.create_plot_group(
                    name=name_edit.text(),
                    mode=selected_mode,
                )
                if created:
                    self.ui.plotsTreeView.expandAll()
                    self._notify_plot_definitions_changed()
                else:
                    self._show_warning(self.tr("The plot group name is empty or already exists."))
            else:
                pass
        finally:
            delete_dialog_safely(dialog=dialog)

    @QtCore.Slot()
    def delete_plot_entry(self) -> None:
        """Delete the selected persistent plot group or variable.

        :return: None.
        """
        selected_indexes: list[QtCore.QModelIndex] = self.ui.plotsTreeView.selectedIndexes()
        if len(selected_indexes) > 0:
            deleted: bool = self.handler.delete_plot_entry_from_index(index=selected_indexes[0])
            if deleted:
                self.ui.plotsTreeView.update()
                self._notify_plot_definitions_changed()
            else:
                self._show_warning(self.tr("The selected dynamic plot entry could not be deleted."))
        else:
            self._show_warning(self.tr("Select a plot group or variable first."))

    @QtCore.Slot(QtCore.QPoint)
    def show_plots_context_menu(self, position: QtCore.QPoint) -> None:
        """Show the appropriate rename action for the selected plot row.

        :param position: Plot-tree viewport position requested by Qt.
        :return: None.
        """
        index: QtCore.QModelIndex = self.ui.plotsTreeView.indexAt(position)
        if index.isValid():
            item: QtGui.QStandardItem | None = self.handler.get_plots_model().itemFromIndex(index)
            if item is not None:
                menu: QtWidgets.QMenu = QtWidgets.QMenu(parent=self.ui.plotsTreeView)
                if item.parent() is None:
                    rename_action: QtGui.QAction = menu.addAction(self.tr("Rename group"))
                    selected_action: QtGui.QAction | None = menu.exec(
                        self.ui.plotsTreeView.viewport().mapToGlobal(position)
                    )
                    if selected_action == rename_action:
                        self.rename_plot_group(index=index)
                    else:
                        pass
                else:
                    rename_action = menu.addAction(self.tr("Rename variable"))
                    selected_action = menu.exec(self.ui.plotsTreeView.viewport().mapToGlobal(position))
                    if selected_action == rename_action:
                        self.rename_plot_variable(index=index)
                    else:
                        pass
            else:
                pass
        else:
            pass

    def rename_plot_group(self, index: QtCore.QModelIndex) -> None:
        """Rename the persistent plot group represented by an index.

        :param index: Selected top-level plot-group index.
        :return: None.
        """
        old_name: str | None = self.handler.get_plot_group_name_from_index(index=index)
        if old_name is not None:
            new_name: str
            accepted: bool
            new_name, accepted = QtWidgets.QInputDialog.getText(
                self,
                self.tr("Rename dynamic plot"),
                self.tr("Plot name"),
                text=old_name,
            )
            if accepted:
                renamed: bool = self.handler.rename_plot_group(old_name=old_name, new_name=new_name)
                if renamed:
                    self._notify_plot_definitions_changed()
                else:
                    self._show_warning(self.tr("The plot group name is empty or already exists."))
            else:
                pass
        else:
            self._show_warning(self.tr("Select a plot group first."))

    def rename_plot_variable(self, index: QtCore.QModelIndex) -> None:
        """Rename the persistent plot entry represented by an index.

        :param index: Selected child plot-entry index.
        :return: None.
        """
        item: QtGui.QStandardItem | None = self.handler.get_plots_model().itemFromIndex(index)
        if item is not None:
            current_name: str = item.text()
            suffix: str
            for suffix in (" [missing]", " [pending]"):
                if current_name.endswith(suffix):
                    current_name = current_name[:-len(suffix)]
                else:
                    pass
            new_name: str
            accepted: bool
            new_name, accepted = QtWidgets.QInputDialog.getText(
                self,
                self.tr("Rename dynamic variable"),
                self.tr("Variable name"),
                text=current_name,
            )
            if accepted:
                renamed: bool = self.handler.rename_plot_variable_from_index(index=index, new_name=new_name)
                if renamed:
                    self._notify_plot_definitions_changed()
                else:
                    self._show_warning(self.tr("The variable name is empty or could not be changed."))
            else:
                pass
        else:
            self._show_warning(self.tr("Select a variable first."))

    @QtCore.Slot(QtCore.QModelIndex, int, int)
    def _expand_inserted_parent(self, parent: QtCore.QModelIndex, first: int, last: int) -> None:
        """Keep the destination group expanded after a drag-and-drop operation.

        :param parent: Parent index receiving inserted rows.
        :param first: First inserted row.
        :param last: Last inserted row.
        :return: None.
        """
        del first
        del last
        if parent.isValid():
            self.ui.plotsTreeView.setExpanded(parent, True)
        else:
            pass

    @QtCore.Slot()
    def _notify_plot_definitions_changed(self) -> None:
        """Notify other GUI projections after this page changes plot assets.

        :return: None.
        """
        self.session.notify_dynamic_plots_changed(mode=self.mode, source=self)

    @QtCore.Slot(object, object)
    def _on_external_plot_definitions_changed(self, mode: object, source: object) -> None:
        """Reload assets changed by Results or another plot page.

        :param mode: Simulation family whose definitions changed.
        :param source: GUI object that originated the change.
        :return: None.
        """
        if source is not self and mode == self.mode:
            self.handler.refresh_plot_definitions()
            self.ui.plotsTreeView.update()
        else:
            pass

    def _show_warning(self, message: str) -> None:
        """Display a page-owned validation warning.

        :param message: User-facing warning text.
        :return: None.
        """
        QtWidgets.QMessageBox.warning(self, self.tr("Dynamic plots"), message)

    def prepare_to_delete(self) -> None:
        """Disconnect long-lived cross-window signals before page deletion.

        :return: None.
        """
        if self._prepared_to_delete:
            return
        else:
            self._prepared_to_delete = True
        self.handler.close_plot_dialogs()
        try:
            self.session.dynamicPlotsChanged.disconnect(self._on_external_plot_definitions_changed)
        except (RuntimeError, TypeError):
            pass

    def set_dark_mode(self) -> None:
        """Accept the workspace dark-theme notification.

        :return: None.
        """
        # Standard Qt widgets inherit the workspace palette automatically.
        pass

    def set_light_mode(self) -> None:
        """Accept the workspace light-theme notification.

        :return: None.
        """
        # Standard Qt widgets inherit the workspace palette automatically.
        pass
