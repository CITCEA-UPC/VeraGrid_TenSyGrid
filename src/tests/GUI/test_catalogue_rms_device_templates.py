"""Contracts for catalogue actions derived from the dynamic Library."""

from __future__ import annotations

from PySide6 import QtWidgets

from VeraGrid.Gui.CatalogueElementsDialogue.catalogue_actions import CatalogueAction
from VeraGrid.Gui.CatalogueElementsDialogue.catalogue_elements_dialogue import (
    CatalogueElementsSelectionDialogue,
)
from VeraGrid.Gui.DynamicModelEditor.Editor.DynamicLibrary.dynamic_editor_library import (
    LibraryDeviceTemplateSpec,
    get_dynamic_library_device_specs,
)
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.enumerations import DynamicSimulationMode


def test_rms_catalogue_matches_library_devices(qt_app: QtWidgets.QApplication) -> None:
    """Require a one-to-one mapping between RMS Devices and catalogue actions.

    :param qt_app: Shared Qt application required to construct the dialog.
    :return: None.
    """
    application: QtWidgets.QApplication = qt_app
    circuit: MultiCircuit = MultiCircuit()
    dialog: CatalogueElementsSelectionDialogue = CatalogueElementsSelectionDialogue(
        parent=None,
        circuit=circuit,
    )
    actions: list[CatalogueAction] = dialog.build_rms_actions()
    specs: list[LibraryDeviceTemplateSpec] = get_dynamic_library_device_specs(
        mode=DynamicSimulationMode.RMS,
    )

    assert list(action.unique_key for action in actions) == list(
        spec.unique_key for spec in specs
    )
    assert list(action.name for action in actions) == list(spec.label for spec in specs)
    assert all(action.voltage_text == "" for action in actions)
    assert list(action.description_text for action in actions) == list(
        spec.description for spec in specs
    )
    assert len(set(action.unique_key for action in actions)) == len(actions)
    assert any(spec.unique_key.startswith("rms:international:") for spec in specs)
    assert all("governor" not in action.unique_key.lower() for action in actions)
    assert all("exciter" not in action.unique_key.lower() for action in actions)
    assert all("stabilizer" not in action.unique_key.lower() for action in actions)

    line_action: CatalogueAction | None = None
    action: CatalogueAction
    for action in actions:
        if action.unique_key == "rms:get_line_rms_template":
            line_action = action
        else:
            pass
    assert line_action is not None
    line_action.execute(circuit=circuit)
    assert circuit.rms_models[-1].code == line_action.unique_key

    dialog.close()
    application.processEvents()


def test_emt_catalogue_matches_library_devices(qt_app: QtWidgets.QApplication) -> None:
    """Require a one-to-one mapping between EMT Devices and catalogue actions.

    :param qt_app: Shared Qt application required to construct the dialog.
    :return: None.
    """
    application: QtWidgets.QApplication = qt_app
    circuit: MultiCircuit = MultiCircuit()
    dialog: CatalogueElementsSelectionDialogue = CatalogueElementsSelectionDialogue(
        parent=None,
        circuit=circuit,
    )
    actions: list[CatalogueAction] = dialog.build_emt_actions()
    specs: list[LibraryDeviceTemplateSpec] = get_dynamic_library_device_specs(
        mode=DynamicSimulationMode.EMT,
    )

    assert list(action.unique_key for action in actions) == list(
        spec.unique_key for spec in specs
    )
    assert list(action.name for action in actions) == list(spec.label for spec in specs)
    assert all(action.voltage_text == "" for action in actions)
    assert list(action.description_text for action in actions) == list(
        spec.description for spec in specs
    )
    assert len(set(action.unique_key for action in actions)) == len(actions)
    assert all("governor" not in action.unique_key.lower() for action in actions)
    assert all("exciter" not in action.unique_key.lower() for action in actions)
    assert all("stabilizer" not in action.unique_key.lower() for action in actions)

    load_action: CatalogueAction | None = None
    action: CatalogueAction
    for action in actions:
        if action.unique_key == "emt:get_dc_load_emt_template":
            load_action = action
        else:
            pass
    assert load_action is not None
    load_action.execute(circuit=circuit)
    assert circuit.emt_models[-1].code == load_action.unique_key

    dialog.close()
    application.processEvents()
