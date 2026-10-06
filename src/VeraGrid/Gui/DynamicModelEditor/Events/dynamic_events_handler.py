# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import List, Protocol

from PySide6 import QtCore, QtGui, QtWidgets

from VeraGrid.Gui.DynamicModelEditor.Events.dynamic_events_support import collect_block_runtime_event_parameters
from VeraGrid.Gui.DynamicModelEditor.Workspace.dynamic_editor_entries import DynamicEditorEntry
from VeraGrid.Gui.DynamicModelEditor.Workspace.dynamic_editor_entries import entry_supports_dynamic_events
from VeraGrid.Gui.DynamicModelEditor.Workspace.dynamic_editor_entries import get_block_for_entry
from VeraGrid.Gui.DynamicModelEditor.Workspace.dynamic_editor_entries import iter_dynamic_editor_entries
from VeraGridEngine.Devices.Events.emt_event import EmtEvent
from VeraGridEngine.Devices.Events.emt_events_group import EmtEventsGroup
from VeraGridEngine.Devices.Events.rms_event import RmsEvent
from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup
from VeraGridEngine.Devices.Parents.editable_device import EditableDevice
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Var
from VeraGridEngine.enumerations import DeviceType, DynamicEventTransitionType, DynamicSimulationMode


class DynamicEventParameterCandidate:
    """Describe one device parameter that can receive a dynamic event."""

    __slots__ = (
        "device",
        "parameter",
        "is_mode_parameter",
    )

    def __init__(
            self,
            device: EditableDevice,
            parameter: Var,
            is_mode_parameter: bool,
    ) -> None:
        """Build one stable parameter candidate.

        :param device: Circuit device owning the dynamic model.
        :param parameter: Runtime event or mode parameter.
        :param is_mode_parameter: Whether step alignment is recommended.
        :return: None.
        """
        self.device: EditableDevice = device
        self.parameter: Var = parameter
        self.is_mode_parameter: bool = is_mode_parameter

    def get_device_label(self) -> str:
        """Return a disambiguated device label.

        :return: Device type and device name.
        """
        return f"{self.device.device_type.value}: {self.device.name}"


def _dynamic_event_candidate_sort_key(
        candidate: DynamicEventParameterCandidate,
) -> tuple[str, str, str, str, int]:
    """Return the stable presentation order for one parameter candidate.

    :param candidate: Candidate being ordered.
    :return: Device-type, device and parameter sorting key.
    """
    return (
        str(candidate.device.device_type.value).casefold(),
        str(candidate.device.name).casefold(),
        str(candidate.device.idtag),
        str(candidate.parameter.name).casefold(),
        int(candidate.parameter.uid),
    )


def build_dynamic_event_parameter_candidates(
        circuit: MultiCircuit,
        mode: DynamicSimulationMode,
) -> List[DynamicEventParameterCandidate]:
    """Collect every event-capable device parameter for one simulation mode.

    :param circuit: Circuit whose saved dynamic models are scanned.
    :param mode: RMS or EMT model family.
    :return: Deterministically ordered parameter candidates.
    """
    candidates: List[DynamicEventParameterCandidate] = list()
    entry: DynamicEditorEntry

    # Scan the same canonical device universe used by the workspace tree so
    # templates and unsupported circuit objects cannot become event targets.
    for entry in iter_dynamic_editor_entries(circuit=circuit):
        supports_mode: bool = mode in entry.available_modes
        supports_events: bool = entry_supports_dynamic_events(entry=entry)
        if supports_mode and supports_events and isinstance(entry.api_object, EditableDevice):
            block: Block = get_block_for_entry(entry=entry, mode=mode)
            parameters: List[Var]
            mode_parameter_uids: set[int]
            parameters, mode_parameter_uids = collect_block_runtime_event_parameters(block=block)
            parameter: Var
            for parameter in parameters:
                candidates.append(
                    DynamicEventParameterCandidate(
                        device=entry.api_object,
                        parameter=parameter,
                        is_mode_parameter=parameter.uid in mode_parameter_uids,
                    )
                )
        else:
            pass

    # A stable order makes the toolbar '+' default predictable and keeps the
    # visual hierarchy from moving between refreshes.
    candidates.sort(key=_dynamic_event_candidate_sort_key)
    return candidates


class _DynamicEventsDropHandler(Protocol):
    """Typed interface used by the events tree during a parameter drop."""

    def candidate_from_mime_data(self, mime_data: QtCore.QMimeData) -> DynamicEventParameterCandidate | None:
        """Resolve one parameter candidate from drag data.

        :param mime_data: Drag payload created by the parameters model.
        :return: Matching candidate or ``None``.
        """
        ...

    def first_candidate_for_device(self, device: EditableDevice) -> DynamicEventParameterCandidate | None:
        """Return the first candidate owned by one device.

        :param device: Device whose candidate is requested.
        :return: First matching candidate or ``None``.
        """
        ...

    def candidate_for_device_parameter(
            self,
            device: EditableDevice | None,
            parameter: Var,
    ) -> DynamicEventParameterCandidate | None:
        """Resolve one exact device and parameter pair.

        :param device: Candidate device.
        :param parameter: Candidate parameter.
        :return: Matching candidate or ``None``.
        """
        ...

    def add_candidate_to_group(
            self,
            group: RmsEventsGroup | EmtEventsGroup,
            candidate: DynamicEventParameterCandidate,
    ) -> bool:
        """Create an event for a dropped candidate.

        :param group: Drop target group.
        :param candidate: Dropped parameter candidate.
        :return: Whether an event was created.
        """
        ...


class DynamicEventParametersTreeModel(QtGui.QStandardItemModel):
    """Expose event parameters as device type, device and parameter nodes."""

    __slots__ = (
        "_candidates",
        "_mode",
    )

    CANDIDATE_ROLE: int = int(QtCore.Qt.ItemDataRole.UserRole) + 1
    MIME_TYPE: str = "application/x-veragrid-dynamic-event-parameter"

    def __init__(
            self,
            mode: DynamicSimulationMode,
            candidates: List[DynamicEventParameterCandidate],
            parent: QtCore.QObject | None = None,
    ) -> None:
        """Build the parameter source model.

        :param mode: RMS or EMT family represented by the model.
        :param candidates: Event-capable parameter candidates.
        :param parent: Optional Qt owner.
        :return: None.
        """
        QtGui.QStandardItemModel.__init__(self, parent)
        self._mode: DynamicSimulationMode = mode
        self._candidates: List[DynamicEventParameterCandidate] = list(candidates)
        self.rebuild(candidates=candidates)

    def rebuild(self, candidates: List[DynamicEventParameterCandidate]) -> None:
        """Rebuild the complete source hierarchy.

        :param candidates: Current event-capable parameter candidates.
        :return: None.
        """
        self._candidates = list(candidates)
        self.clear()
        self.setHorizontalHeaderLabels([self.tr("Parameters")])
        device_type: DeviceType
        type_candidates: List[DynamicEventParameterCandidate]

        # Group with explicit list scans to avoid introducing a string-keyed
        # dictionary for presentation-only state.
        grouped_types: List[tuple[DeviceType, List[DynamicEventParameterCandidate]]] = list()
        candidate: DynamicEventParameterCandidate
        for candidate in self._candidates:
            matching_type_candidates: List[DynamicEventParameterCandidate] | None = None
            grouped_type: tuple[DeviceType, List[DynamicEventParameterCandidate]]
            for grouped_type in grouped_types:
                if grouped_type[0] == candidate.device.device_type:
                    matching_type_candidates = grouped_type[1]
                else:
                    pass
            if matching_type_candidates is None:
                matching_type_candidates = list()
                grouped_types.append((candidate.device.device_type, matching_type_candidates))
            else:
                pass
            matching_type_candidates.append(candidate)

        for device_type, type_candidates in grouped_types:
            type_item: QtGui.QStandardItem = QtGui.QStandardItem(str(device_type.value))
            type_item.setEditable(False)
            self.appendRow(type_item)
            grouped_devices: List[tuple[EditableDevice, List[DynamicEventParameterCandidate]]] = list()

            for candidate in type_candidates:
                matching_device_candidates: List[DynamicEventParameterCandidate] | None = None
                grouped_device: tuple[EditableDevice, List[DynamicEventParameterCandidate]]
                for grouped_device in grouped_devices:
                    if grouped_device[0] is candidate.device:
                        matching_device_candidates = grouped_device[1]
                    else:
                        pass
                if matching_device_candidates is None:
                    matching_device_candidates = list()
                    grouped_devices.append((candidate.device, matching_device_candidates))
                else:
                    pass
                matching_device_candidates.append(candidate)

            device: EditableDevice
            device_candidates: List[DynamicEventParameterCandidate]
            for device, device_candidates in grouped_devices:
                device_item: QtGui.QStandardItem = QtGui.QStandardItem(str(device.name))
                device_item.setEditable(False)
                type_item.appendRow(device_item)
                for candidate in device_candidates:
                    parameter_item: QtGui.QStandardItem = QtGui.QStandardItem(str(candidate.parameter.name))
                    parameter_item.setEditable(False)
                    parameter_item.setData(candidate, self.CANDIDATE_ROLE)
                    device_item.appendRow(parameter_item)

    def candidate_from_index(self, index: QtCore.QModelIndex) -> DynamicEventParameterCandidate | None:
        """Return the candidate stored by one source-model index.

        :param index: Source-model index.
        :return: Stored parameter candidate or ``None``.
        """
        if index.isValid():
            candidate: object = index.data(self.CANDIDATE_ROLE)
            if isinstance(candidate, DynamicEventParameterCandidate):
                return candidate
            else:
                return None
        else:
            return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        """Allow dragging only real parameter leaves.

        :param index: Candidate source-model index.
        :return: Qt item flags.
        """
        base_flags: QtCore.Qt.ItemFlag = QtGui.QStandardItemModel.flags(self, index)
        if self.candidate_from_index(index=index) is not None:
            return base_flags | QtCore.Qt.ItemFlag.ItemIsDragEnabled
        else:
            return base_flags & ~QtCore.Qt.ItemFlag.ItemIsDragEnabled

    def mimeTypes(self) -> List[str]:
        """Return the private MIME type used for event parameters.

        :return: Supported MIME type list.
        """
        return [self.MIME_TYPE]

    def supportedDragActions(self) -> QtCore.Qt.DropAction:
        """Return the copy-only drag contract.

        :return: Copy action.
        """
        return QtCore.Qt.DropAction.CopyAction

    def mimeData(self, indexes: List[QtCore.QModelIndex]) -> QtCore.QMimeData:
        """Encode the first dragged parameter leaf.

        :param indexes: Selected source-model indexes.
        :return: MIME payload containing mode, device identity and parameter UID.
        """
        mime_data: QtCore.QMimeData = QtCore.QMimeData()
        candidate: DynamicEventParameterCandidate | None = None
        index: QtCore.QModelIndex
        for index in indexes:
            if candidate is None:
                candidate = self.candidate_from_index(index=index)
            else:
                pass

        if candidate is not None:
            payload: QtCore.QByteArray = QtCore.QByteArray()
            stream: QtCore.QDataStream = QtCore.QDataStream(payload, QtCore.QIODevice.OpenModeFlag.WriteOnly)
            stream.writeQString(self._mode.name)
            stream.writeQString(str(candidate.device.idtag))
            # Symbolic variable UIDs can use the full UUID-sized integer range,
            # which does not fit in QDataStream's signed 64-bit integer fields.
            # Preserve the complete identifier by transporting its decimal text.
            stream.writeQString(str(candidate.parameter.uid))
            mime_data.setData(self.MIME_TYPE, payload)
        else:
            pass
        return mime_data


class DynamicEventsTreeModel(QtGui.QStandardItemModel):
    """Present real event groups and their events as one editable tree."""

    __slots__ = (
        "_circuit",
        "_drop_handler",
        "_mode",
    )

    GROUP_ROLE: int = int(QtCore.Qt.ItemDataRole.UserRole) + 1
    EVENT_ROLE: int = int(QtCore.Qt.ItemDataRole.UserRole) + 2
    DEVICE_COLUMN: int = 0
    PARAMETER_COLUMN: int = 1
    TIME_COLUMN: int = 2
    END_TIME_COLUMN: int = 3
    VALUE_COLUMN: int = 4
    FORCE_STEP_COLUMN: int = 5
    TRANSITION_COLUMN: int = 6
    COLUMN_COUNT: int = 7

    def __init__(
            self,
            circuit: MultiCircuit,
            mode: DynamicSimulationMode,
            drop_handler: _DynamicEventsDropHandler,
            parent: QtCore.QObject | None = None,
    ) -> None:
        """Build the global events tree model.

        :param circuit: Circuit owning the real groups and events.
        :param mode: RMS or EMT family represented by the model.
        :param drop_handler: Typed object resolving parameter drops.
        :param parent: Optional Qt owner.
        :return: None.
        """
        QtGui.QStandardItemModel.__init__(self, parent)
        self._circuit: MultiCircuit = circuit
        self._mode: DynamicSimulationMode = mode
        self._drop_handler: _DynamicEventsDropHandler = drop_handler
        self.rebuild()

    def get_groups(self) -> List[RmsEventsGroup | EmtEventsGroup]:
        """Return the real groups for this simulation family.

        :return: Ordered event groups.
        """
        if self._mode == DynamicSimulationMode.RMS:
            return list(self._circuit.rms_events_groups)
        else:
            return list(self._circuit.emt_events_groups)

    def get_events(self) -> List[RmsEvent | EmtEvent]:
        """Return the real events for this simulation family.

        :return: Ordered events.
        """
        if self._mode == DynamicSimulationMode.RMS:
            return list(self._circuit.rms_events)
        else:
            return list(self._circuit.emt_events)

    def get_group_events(self, group: RmsEventsGroup | EmtEventsGroup) -> List[RmsEvent | EmtEvent]:
        """Return every event belonging to one group.

        :param group: Real event group.
        :return: Events assigned to the group.
        """
        events: List[RmsEvent | EmtEvent] = list()
        event: RmsEvent | EmtEvent
        for event in self.get_events():
            if event.group is group:
                events.append(event)
            else:
                pass
        return events

    def rebuild(self) -> None:
        """Rebuild every group and child event row from circuit assets.

        :return: None.
        """
        self.clear()
        self.setHorizontalHeaderLabels([
            self.tr("Device"),
            self.tr("Parameter"),
            self.tr("Time"),
            self.tr("End time"),
            self.tr("Value"),
            self.tr("Force step"),
            self.tr("Transition type"),
        ])
        group: RmsEventsGroup | EmtEventsGroup
        for group in self.get_groups():
            group_row: List[QtGui.QStandardItem] = list()
            column_index: int
            for column_index in range(self.COLUMN_COUNT):
                group_item: QtGui.QStandardItem = QtGui.QStandardItem()
                group_item.setData(group, self.GROUP_ROLE)
                group_row.append(group_item)
            self.appendRow(group_row)
            event: RmsEvent | EmtEvent
            for event in self.get_group_events(group=group):
                event_row: List[QtGui.QStandardItem] = list()
                for column_index in range(self.COLUMN_COUNT):
                    event_item: QtGui.QStandardItem = QtGui.QStandardItem()
                    event_item.setData(event, self.EVENT_ROLE)
                    event_row.append(event_item)
                group_row[0].appendRow(event_row)

    def group_from_index(self, index: QtCore.QModelIndex) -> RmsEventsGroup | EmtEventsGroup | None:
        """Return a group stored directly on one row.

        :param index: Source-model index.
        :return: Group or ``None`` for event rows.
        """
        if index.isValid():
            group: object = index.data(self.GROUP_ROLE)
            if isinstance(group, (RmsEventsGroup, EmtEventsGroup)):
                return group
            else:
                return None
        else:
            return None

    def event_from_index(self, index: QtCore.QModelIndex) -> RmsEvent | EmtEvent | None:
        """Return an event stored on one child row.

        :param index: Source-model index.
        :return: Event or ``None`` for group rows.
        """
        if index.isValid():
            event: object = index.data(self.EVENT_ROLE)
            if isinstance(event, (RmsEvent, EmtEvent)):
                return event
            else:
                return None
        else:
            return None

    def index_for_group(self, group: RmsEventsGroup | EmtEventsGroup) -> QtCore.QModelIndex:
        """Locate a top-level group by object identity.

        :param group: Group to locate.
        :return: Source-model index or an invalid index.
        """
        row_index: int
        for row_index in range(self.rowCount()):
            index: QtCore.QModelIndex = self.index(row_index, self.DEVICE_COLUMN)
            if self.group_from_index(index=index) is group:
                return index
            else:
                pass
        return QtCore.QModelIndex()

    def index_for_event(self, event: RmsEvent | EmtEvent) -> QtCore.QModelIndex:
        """Locate a child event by object identity.

        :param event: Event to locate.
        :return: Source-model index or an invalid index.
        """
        group_row_index: int
        for group_row_index in range(self.rowCount()):
            group_index: QtCore.QModelIndex = self.index(group_row_index, self.DEVICE_COLUMN)
            event_row_index: int
            for event_row_index in range(self.rowCount(group_index)):
                event_index: QtCore.QModelIndex = self.index(
                    event_row_index,
                    self.DEVICE_COLUMN,
                    group_index,
                )
                if self.event_from_index(index=event_index) is event:
                    return event_index
                else:
                    pass
        return QtCore.QModelIndex()

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        """Expose group checks, event editing and group drop targets.

        :param index: Source-model index.
        :return: Qt item flags.
        """
        if not index.isValid():
            return QtCore.Qt.ItemFlag.ItemIsDropEnabled
        else:
            pass
        group: RmsEventsGroup | EmtEventsGroup | None = self.group_from_index(index=index)
        event: RmsEvent | EmtEvent | None = self.event_from_index(index=index)
        if group is not None:
            group_flags: QtCore.Qt.ItemFlag = (
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsDropEnabled
            )
            if index.column() == self.DEVICE_COLUMN:
                group_flags = group_flags | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            else:
                pass
            return group_flags
        elif event is not None:
            event_flags: QtCore.Qt.ItemFlag = (
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
            )
            if index.column() == self.FORCE_STEP_COLUMN:
                return event_flags | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            else:
                return event_flags | QtCore.Qt.ItemFlag.ItemIsEditable
        else:
            return QtCore.Qt.ItemFlag.ItemIsEnabled

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> object:
        """Read group and event values directly from persistent assets.

        :param index: Source-model index.
        :param role: Requested Qt data role.
        :return: Role-specific display or edit value.
        """
        if role in (self.GROUP_ROLE, self.EVENT_ROLE):
            return QtGui.QStandardItemModel.data(self, index, role)
        else:
            pass
        group: RmsEventsGroup | EmtEventsGroup | None = self.group_from_index(index=index)
        event: RmsEvent | EmtEvent | None = self.event_from_index(index=index)
        if group is not None:
            if role == QtCore.Qt.ItemDataRole.DisplayRole and index.column() == self.DEVICE_COLUMN:
                return str(group.name)
            elif role == QtCore.Qt.ItemDataRole.CheckStateRole and index.column() == self.DEVICE_COLUMN:
                if group.active:
                    return QtCore.Qt.CheckState.Checked
                else:
                    return QtCore.Qt.CheckState.Unchecked
            else:
                return None
        elif event is not None:
            if role == QtCore.Qt.ItemDataRole.CheckStateRole and index.column() == self.FORCE_STEP_COLUMN:
                if event.force_step_alignment:
                    return QtCore.Qt.CheckState.Checked
                else:
                    return QtCore.Qt.CheckState.Unchecked
            elif role == QtCore.Qt.ItemDataRole.EditRole:
                return self._event_edit_value(event=event, column=index.column())
            elif role == QtCore.Qt.ItemDataRole.DisplayRole:
                return self._event_display_value(event=event, column=index.column())
            else:
                return None
        else:
            return QtGui.QStandardItemModel.data(self, index, role)

    def _event_edit_value(self, event: RmsEvent | EmtEvent, column: int) -> object:
        """Return the raw value edited in one event column.

        :param event: Real event represented by the row.
        :param column: Event column index.
        :return: Raw editable value.
        """
        if column == self.DEVICE_COLUMN:
            return event.device
        elif column == self.PARAMETER_COLUMN:
            return event.parameter
        elif column == self.TIME_COLUMN:
            return float(event.time)
        elif column == self.END_TIME_COLUMN:
            return event.end_time
        elif column == self.VALUE_COLUMN:
            return float(event.value)
        elif column == self.FORCE_STEP_COLUMN:
            return bool(event.force_step_alignment)
        elif column == self.TRANSITION_COLUMN:
            return event.transition_type
        else:
            return None

    def _event_display_value(self, event: RmsEvent | EmtEvent, column: int) -> str:
        """Return the user-facing text for one event column.

        :param event: Real event represented by the row.
        :param column: Event column index.
        :return: Display text.
        """
        if column == self.DEVICE_COLUMN:
            if isinstance(event.device, EditableDevice):
                return str(event.device.name)
            else:
                return self.tr("Missing device")
        elif column == self.PARAMETER_COLUMN:
            if isinstance(event.parameter, Var):
                return str(event.parameter.name)
            else:
                return self.tr("Missing parameter")
        elif column == self.TIME_COLUMN:
            return str(event.time)
        elif column == self.END_TIME_COLUMN:
            return "" if event.end_time is None else str(event.end_time)
        elif column == self.VALUE_COLUMN:
            return str(event.value)
        elif column == self.FORCE_STEP_COLUMN:
            return ""
        elif column == self.TRANSITION_COLUMN:
            return str(event.transition_type.value)
        else:
            return ""

    def setData(
            self,
            index: QtCore.QModelIndex,
            value: object,
            role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        """Apply one group checkbox or event-cell edit immediately.

        :param index: Edited source-model index.
        :param value: New typed value.
        :param role: Qt edit or check-state role.
        :return: Whether the edit was accepted.
        """
        group: RmsEventsGroup | EmtEventsGroup | None = self.group_from_index(index=index)
        event: RmsEvent | EmtEvent | None = self.event_from_index(index=index)
        accepted: bool = False
        if group is not None and index.column() == self.DEVICE_COLUMN and role == QtCore.Qt.ItemDataRole.CheckStateRole:
            group.active = (
                value == QtCore.Qt.CheckState.Checked
                or value == int(QtCore.Qt.CheckState.Checked.value)
            )
            accepted = True
        elif event is not None:
            accepted = self._set_event_data(event=event, column=index.column(), value=value, role=role)
        else:
            pass

        if accepted:
            parent_index: QtCore.QModelIndex = index.parent()
            left_index: QtCore.QModelIndex = self.index(index.row(), 0, parent_index)
            right_index: QtCore.QModelIndex = self.index(index.row(), self.COLUMN_COUNT - 1, parent_index)
            self.dataChanged.emit(
                left_index,
                right_index,
                [
                    QtCore.Qt.ItemDataRole.DisplayRole,
                    QtCore.Qt.ItemDataRole.EditRole,
                    QtCore.Qt.ItemDataRole.CheckStateRole,
                ],
            )
            return True
        else:
            return False

    def _set_event_data(self, event: RmsEvent | EmtEvent, column: int, value: object, role: int) -> bool:
        """Validate and apply one typed event property.

        :param event: Event being edited.
        :param column: Edited event column.
        :param value: New typed value.
        :param role: Qt edit or check-state role.
        :return: Whether the edit was accepted.
        """
        if column == self.DEVICE_COLUMN and role == QtCore.Qt.ItemDataRole.EditRole and isinstance(value, EditableDevice):
            candidate: DynamicEventParameterCandidate | None = self._drop_handler.first_candidate_for_device(device=value)
            if candidate is not None:
                event.set_device(elm=value)
                event.parameter = candidate.parameter
                event.force_step_alignment = candidate.is_mode_parameter
                return True
            else:
                return False
        elif column == self.PARAMETER_COLUMN and role == QtCore.Qt.ItemDataRole.EditRole and isinstance(value, Var):
            candidate = self._drop_handler.candidate_for_device_parameter(device=event.device, parameter=value)
            if candidate is not None:
                event.parameter = candidate.parameter
                event.force_step_alignment = candidate.is_mode_parameter
                return True
            else:
                return False
        elif column in (self.TIME_COLUMN, self.END_TIME_COLUMN, self.VALUE_COLUMN) and role == QtCore.Qt.ItemDataRole.EditRole:
            try:
                numeric_value: float = float(value)
            except (TypeError, ValueError):
                return False
            if column == self.TIME_COLUMN:
                event.time = numeric_value
            elif column == self.END_TIME_COLUMN:
                event.end_time = numeric_value
            else:
                event.value = numeric_value
            return True
        elif column == self.FORCE_STEP_COLUMN and role == QtCore.Qt.ItemDataRole.CheckStateRole:
            event.force_step_alignment = (
                value == QtCore.Qt.CheckState.Checked
                or value == int(QtCore.Qt.CheckState.Checked.value)
            )
            return True
        elif column == self.TRANSITION_COLUMN and role == QtCore.Qt.ItemDataRole.EditRole:
            if isinstance(value, DynamicEventTransitionType):
                event.transition_type = value
                return True
            else:
                return False
        else:
            return False

    def mimeTypes(self) -> List[str]:
        """Return the parameter MIME type accepted by event groups.

        :return: Supported MIME type list.
        """
        return [DynamicEventParametersTreeModel.MIME_TYPE]

    def supportedDropActions(self) -> QtCore.Qt.DropAction:
        """Return the copy-only drop contract.

        :return: Copy action.
        """
        return QtCore.Qt.DropAction.CopyAction

    def canDropMimeData(
            self,
            data: QtCore.QMimeData,
            action: QtCore.Qt.DropAction,
            row: int,
            column: int,
            parent: QtCore.QModelIndex,
    ) -> bool:
        """Accept a parameter only when the target resolves to a group.

        :param data: Incoming parameter MIME data.
        :param action: Requested drop action.
        :param row: Proposed child row.
        :param column: Proposed target column.
        :param parent: Proposed target parent.
        :return: Whether the drop can be accepted.
        """
        del column
        if action == QtCore.Qt.DropAction.IgnoreAction:
            return True
        else:
            pass
        group: RmsEventsGroup | EmtEventsGroup | None = self._drop_group(parent=parent, row=row)
        candidate: DynamicEventParameterCandidate | None = self._drop_handler.candidate_from_mime_data(mime_data=data)
        return group is not None and candidate is not None

    def dropMimeData(
            self,
            data: QtCore.QMimeData,
            action: QtCore.Qt.DropAction,
            row: int,
            column: int,
            parent: QtCore.QModelIndex,
    ) -> bool:
        """Create one event from a dropped parameter.

        :param data: Incoming parameter MIME data.
        :param action: Requested drop action.
        :param row: Proposed child row.
        :param column: Proposed target column.
        :param parent: Proposed target parent.
        :return: Whether an event was created.
        """
        del column
        if action == QtCore.Qt.DropAction.IgnoreAction:
            return True
        else:
            pass
        group: RmsEventsGroup | EmtEventsGroup | None = self._drop_group(parent=parent, row=row)
        candidate: DynamicEventParameterCandidate | None = self._drop_handler.candidate_from_mime_data(mime_data=data)
        if group is not None and candidate is not None:
            return self._drop_handler.add_candidate_to_group(group=group, candidate=candidate)
        else:
            return False

    def _drop_group(
            self,
            parent: QtCore.QModelIndex,
            row: int,
    ) -> RmsEventsGroup | EmtEventsGroup | None:
        """Resolve a drop position to one top-level group.

        :param parent: Proposed drop parent.
        :param row: Proposed row when dropping between top-level items.
        :return: Target group or ``None``.
        """
        group: RmsEventsGroup | EmtEventsGroup | None = self.group_from_index(index=parent)
        if group is not None:
            return group
        elif not parent.isValid() and 0 <= row < self.rowCount():
            return self.group_from_index(index=self.index(row, self.DEVICE_COLUMN))
        else:
            return None


class DynamicEventsHandler(QtCore.QObject):
    """Coordinate global event candidates, models and immediate mutations."""

    eventAdded = QtCore.Signal(object)
    modelAboutToBeRebuilt = QtCore.Signal()
    modelRebuilt = QtCore.Signal()

    __slots__ = (
        "circuit",
        "mode",
        "candidates",
        "parameters_model",
        "parameters_proxy",
        "events_model",
        "events_proxy",
    )

    def __init__(
            self,
            circuit: MultiCircuit,
            mode: DynamicSimulationMode,
            parent: QtCore.QObject | None = None,
    ) -> None:
        """Build models for one circuit-wide RMS or EMT events page.

        :param circuit: Circuit owning event assets and saved dynamic models.
        :param mode: RMS or EMT family.
        :param parent: Optional Qt owner.
        :return: None.
        """
        QtCore.QObject.__init__(self, parent)
        self.circuit: MultiCircuit = circuit
        self.mode: DynamicSimulationMode = mode
        self.candidates: List[DynamicEventParameterCandidate] = build_dynamic_event_parameter_candidates(
            circuit=circuit,
            mode=mode,
        )
        self.parameters_model: DynamicEventParametersTreeModel = DynamicEventParametersTreeModel(
            mode=mode,
            candidates=self.candidates,
            parent=self,
        )
        self.parameters_proxy: QtCore.QSortFilterProxyModel = QtCore.QSortFilterProxyModel(self)
        self.parameters_proxy.setSourceModel(self.parameters_model)
        self.parameters_proxy.setRecursiveFilteringEnabled(True)
        self.parameters_proxy.setFilterCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.parameters_proxy.setFilterKeyColumn(-1)
        self.events_model: DynamicEventsTreeModel = DynamicEventsTreeModel(
            circuit=circuit,
            mode=mode,
            drop_handler=self,
            parent=self,
        )
        self.events_proxy: QtCore.QSortFilterProxyModel = QtCore.QSortFilterProxyModel(self)
        self.events_proxy.setSourceModel(self.events_model)
        self.events_proxy.setRecursiveFilteringEnabled(True)
        self.events_proxy.setAutoAcceptChildRows(True)
        self.events_proxy.setFilterCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.events_proxy.setFilterKeyColumn(-1)

    def refresh(self) -> None:
        """Rescan saved models and rebuild both trees.

        :return: None.
        """
        self.candidates = build_dynamic_event_parameter_candidates(circuit=self.circuit, mode=self.mode)
        self.parameters_model.rebuild(candidates=self.candidates)
        self.rebuild_events_model()

    def rebuild_events_model(self) -> None:
        """Rebuild the events projection with preservation lifecycle signals.

        Views use the pre-reset signal to retain presentation state that Qt
        discards when the standard-item model is cleared.

        :return: None.
        """
        self.modelAboutToBeRebuilt.emit()
        self.events_model.rebuild()
        self.modelRebuilt.emit()

    def set_parameter_search_text(self, search_text: str) -> None:
        """Apply the independent parameters-tree filter.

        :param search_text: Case-insensitive filter expression.
        :return: None.
        """
        self.parameters_proxy.setFilterFixedString(search_text)

    def set_events_search_text(self, search_text: str) -> None:
        """Apply the independent events-tree filter.

        :param search_text: Case-insensitive filter expression.
        :return: None.
        """
        self.events_proxy.setFilterFixedString(search_text)

    def candidates_for_device(self, device: EditableDevice | None) -> List[DynamicEventParameterCandidate]:
        """Return candidates owned by one device.

        :param device: Device whose parameters are requested.
        :return: Ordered parameter candidates.
        """
        matches: List[DynamicEventParameterCandidate] = list()
        candidate: DynamicEventParameterCandidate
        for candidate in self.candidates:
            if candidate.device is device:
                matches.append(candidate)
            else:
                pass
        return matches

    def eligible_devices(self) -> List[EditableDevice]:
        """Return unique devices that currently expose event parameters.

        :return: Deterministically ordered eligible devices.
        """
        devices: List[EditableDevice] = list()
        candidate: DynamicEventParameterCandidate
        for candidate in self.candidates:
            if candidate.device not in devices:
                devices.append(candidate.device)
            else:
                pass
        return devices

    def first_candidate(self) -> DynamicEventParameterCandidate | None:
        """Return the deterministic default event target.

        :return: First candidate or ``None``.
        """
        if len(self.candidates) > 0:
            return self.candidates[0]
        else:
            return None

    def first_candidate_for_device(self, device: EditableDevice) -> DynamicEventParameterCandidate | None:
        """Return the first parameter belonging to one device.

        :param device: Device whose first parameter is requested.
        :return: First matching candidate or ``None``.
        """
        candidate: DynamicEventParameterCandidate
        for candidate in self.candidates:
            if candidate.device is device:
                return candidate
            else:
                pass
        return None

    def candidate_for_device_parameter(
            self,
            device: EditableDevice | None,
            parameter: Var,
    ) -> DynamicEventParameterCandidate | None:
        """Resolve one exact device and parameter pair.

        :param device: Candidate event device.
        :param parameter: Candidate runtime parameter.
        :return: Matching candidate or ``None``.
        """
        candidate: DynamicEventParameterCandidate
        for candidate in self.candidates:
            same_parameter: bool = (
                candidate.parameter is parameter
                or int(candidate.parameter.uid) == int(parameter.uid)
            )
            if candidate.device is device and same_parameter:
                return candidate
            else:
                pass
        return None

    def candidate_from_mime_data(self, mime_data: QtCore.QMimeData) -> DynamicEventParameterCandidate | None:
        """Resolve one source-tree parameter payload.

        :param mime_data: Drag payload created by the parameters model.
        :return: Matching live candidate or ``None``.
        """
        if not mime_data.hasFormat(DynamicEventParametersTreeModel.MIME_TYPE):
            return None
        else:
            pass
        payload: QtCore.QByteArray = mime_data.data(DynamicEventParametersTreeModel.MIME_TYPE)
        stream: QtCore.QDataStream = QtCore.QDataStream(payload, QtCore.QIODevice.OpenModeFlag.ReadOnly)
        mode_name: str = stream.readQString()
        device_idtag: str = stream.readQString()
        parameter_uid_text: str = stream.readQString()
        if mode_name != self.mode.name:
            return None
        else:
            pass
        candidate: DynamicEventParameterCandidate
        for candidate in self.candidates:
            same_device: bool = str(candidate.device.idtag) == device_idtag
            same_parameter: bool = str(candidate.parameter.uid) == parameter_uid_text
            if same_device and same_parameter:
                return candidate
            else:
                pass
        return None

    def add_candidate_to_group(
            self,
            group: RmsEventsGroup | EmtEventsGroup,
            candidate: DynamicEventParameterCandidate,
    ) -> bool:
        """Create and select-ready event data for one group and parameter.

        :param group: Target RMS or EMT event group.
        :param candidate: Device parameter receiving the event.
        :return: Whether the event family matched and creation succeeded.
        """
        event: RmsEvent | EmtEvent | None = None
        if self.mode == DynamicSimulationMode.RMS and isinstance(group, RmsEventsGroup):
            event = RmsEvent(
                name=f"RMS event {len(self.circuit.rms_events)}",
                device=candidate.device,
                parameter=candidate.parameter,
                group=group,
                force_step_alignment=candidate.is_mode_parameter,
            )
            self.circuit.add_rms_event(obj=event)
        elif self.mode == DynamicSimulationMode.EMT and isinstance(group, EmtEventsGroup):
            event = EmtEvent(
                name=f"EMT event {len(self.circuit.emt_events)}",
                device=candidate.device,
                parameter=candidate.parameter,
                group=group,
                force_step_alignment=candidate.is_mode_parameter,
            )
            self.circuit.add_emt_event(obj=event)
        else:
            pass
        if event is not None:
            self.rebuild_events_model()
            self.eventAdded.emit(event)
            return True
        else:
            return False

    def create_group(self, name: str) -> RmsEventsGroup | EmtEventsGroup | None:
        """Create one uniquely named real event group.

        :param name: Validated group name.
        :return: Created group or ``None`` when the name already exists.
        """
        normalized_name: str = name.strip()
        existing_group: RmsEventsGroup | EmtEventsGroup
        for existing_group in self.events_model.get_groups():
            if existing_group.name.casefold() == normalized_name.casefold():
                return None
            else:
                pass
        if self.mode == DynamicSimulationMode.RMS:
            created_group: RmsEventsGroup | EmtEventsGroup = RmsEventsGroup(name=normalized_name)
            self.circuit.add_rms_events_group(obj=created_group)
        else:
            created_group = EmtEventsGroup(name=normalized_name)
            self.circuit.add_emt_events_group(obj=created_group)
        self.rebuild_events_model()
        return created_group

    def delete_event(self, event: RmsEvent | EmtEvent) -> None:
        """Delete one real event and rebuild the projection.

        :param event: Event to delete.
        :return: None.
        """
        if isinstance(event, RmsEvent):
            self.circuit.delete_rms_event(obj=event)
        else:
            self.circuit.delete_emt_event(obj=event)
        self.rebuild_events_model()

    def delete_group(self, group: RmsEventsGroup | EmtEventsGroup) -> None:
        """Delete one group and its real child events.

        :param group: Group to delete.
        :return: None.
        """
        if isinstance(group, RmsEventsGroup):
            self.circuit.delete_rms_events_group(obj=group)
        else:
            self.circuit.delete_emt_events_group(obj=group)
        self.rebuild_events_model()

    def source_events_index(self, proxy_index: QtCore.QModelIndex) -> QtCore.QModelIndex:
        """Map a right-tree proxy index to its source model.

        :param proxy_index: Events proxy-model index.
        :return: Source-model index.
        """
        return self.events_proxy.mapToSource(proxy_index)

    def group_from_proxy_index(self, proxy_index: QtCore.QModelIndex) -> RmsEventsGroup | EmtEventsGroup | None:
        """Resolve a directly selected group row.

        :param proxy_index: Events proxy-model index.
        :return: Selected group or ``None``.
        """
        return self.events_model.group_from_index(index=self.source_events_index(proxy_index=proxy_index))

    def event_from_proxy_index(self, proxy_index: QtCore.QModelIndex) -> RmsEvent | EmtEvent | None:
        """Resolve a selected child event row.

        :param proxy_index: Events proxy-model index.
        :return: Selected event or ``None``.
        """
        return self.events_model.event_from_index(index=self.source_events_index(proxy_index=proxy_index))

    def proxy_index_for_group(self, group: RmsEventsGroup | EmtEventsGroup) -> QtCore.QModelIndex:
        """Locate a group in the filtered right tree.

        :param group: Group to locate.
        :return: Proxy-model index.
        """
        return self.events_proxy.mapFromSource(self.events_model.index_for_group(group=group))

    def proxy_index_for_event(self, event: RmsEvent | EmtEvent) -> QtCore.QModelIndex:
        """Locate an event in the filtered right tree.

        :param event: Event to locate.
        :return: Proxy-model index.
        """
        return self.events_proxy.mapFromSource(self.events_model.index_for_event(event=event))


class DynamicEventsItemDelegate(QtWidgets.QStyledItemDelegate):
    """Provide typed editors for the heterogeneous event columns."""

    __slots__ = ("_handler",)

    def __init__(self, handler: DynamicEventsHandler, parent: QtCore.QObject | None = None) -> None:
        """Build the event-row delegate.

        :param handler: Candidate catalog used by dependent selectors.
        :param parent: Optional Qt owner.
        :return: None.
        """
        QtWidgets.QStyledItemDelegate.__init__(self, parent)
        self._handler: DynamicEventsHandler = handler

    def createEditor(
            self,
            parent: QtWidgets.QWidget,
            option: QtWidgets.QStyleOptionViewItem,
            index: QtCore.QModelIndex,
    ) -> QtWidgets.QWidget:
        """Create the correct editor for one event column.

        :param parent: Parent view widget.
        :param option: Qt style option.
        :param index: Edited proxy-model index.
        :return: Typed editor widget.
        """
        event: object = index.data(DynamicEventsTreeModel.EVENT_ROLE)
        if not isinstance(event, (RmsEvent, EmtEvent)):
            return QtWidgets.QWidget(parent)
        else:
            pass
        if index.column() == DynamicEventsTreeModel.DEVICE_COLUMN:
            device_combo: QtWidgets.QComboBox = QtWidgets.QComboBox(parent)
            device: EditableDevice
            for device in self._handler.eligible_devices():
                device_combo.addItem(f"{device.device_type.value}: {device.name}", device)
            return device_combo
        elif index.column() == DynamicEventsTreeModel.PARAMETER_COLUMN:
            parameter_combo: QtWidgets.QComboBox = QtWidgets.QComboBox(parent)
            candidate: DynamicEventParameterCandidate
            for candidate in self._handler.candidates_for_device(device=event.device):
                parameter_combo.addItem(str(candidate.parameter.name), candidate.parameter)
            return parameter_combo
        elif index.column() in (
            DynamicEventsTreeModel.TIME_COLUMN,
            DynamicEventsTreeModel.END_TIME_COLUMN,
            DynamicEventsTreeModel.VALUE_COLUMN,
        ):
            numeric_editor: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox(parent)
            numeric_editor.setDecimals(9)
            numeric_editor.setRange(-1.0e12, 1.0e12)
            numeric_editor.setSingleStep(0.01)
            return numeric_editor
        elif index.column() == DynamicEventsTreeModel.TRANSITION_COLUMN:
            transition_combo: QtWidgets.QComboBox = QtWidgets.QComboBox(parent)
            transition_type: DynamicEventTransitionType
            for transition_type in DynamicEventTransitionType:
                transition_combo.addItem(str(transition_type.value), transition_type)
            return transition_combo
        else:
            return QtWidgets.QStyledItemDelegate.createEditor(self, parent, option, index)

    def setEditorData(self, editor: QtWidgets.QWidget, index: QtCore.QModelIndex) -> None:
        """Populate a typed editor from the current event value.

        :param editor: Editor widget created for the cell.
        :param index: Edited proxy-model index.
        :return: None.
        """
        current_value: object = index.data(QtCore.Qt.ItemDataRole.EditRole)
        if isinstance(editor, QtWidgets.QComboBox):
            item_index: int
            matched: bool = False
            for item_index in range(editor.count()):
                if editor.itemData(item_index) is current_value or editor.itemData(item_index) == current_value:
                    editor.setCurrentIndex(item_index)
                    matched = True
                else:
                    pass
            if not matched and editor.count() > 0:
                editor.setCurrentIndex(0)
            else:
                pass
        elif isinstance(editor, QtWidgets.QDoubleSpinBox):
            if current_value is None:
                editor.setValue(0.0)
            else:
                editor.setValue(float(current_value))
        else:
            QtWidgets.QStyledItemDelegate.setEditorData(self, editor, index)

    def setModelData(
            self,
            editor: QtWidgets.QWidget,
            model: QtCore.QAbstractItemModel,
            index: QtCore.QModelIndex,
    ) -> None:
        """Commit the typed editor value through the events model.

        :param editor: Active editor widget.
        :param model: Proxy model receiving the edit.
        :param index: Edited proxy-model index.
        :return: None.
        """
        if isinstance(editor, QtWidgets.QComboBox):
            model.setData(index, editor.currentData(), QtCore.Qt.ItemDataRole.EditRole)
        elif isinstance(editor, QtWidgets.QDoubleSpinBox):
            model.setData(index, float(editor.value()), QtCore.Qt.ItemDataRole.EditRole)
        else:
            QtWidgets.QStyledItemDelegate.setModelData(self, editor, model, index)
