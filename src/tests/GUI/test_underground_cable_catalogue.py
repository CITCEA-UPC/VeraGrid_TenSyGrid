from PySide6 import QtCore, QtWidgets

from VeraGrid.Gui.DeviceEditors.UndergroundCableBuilder.table_models import UndergroundCableCatalogueModel
from VeraGrid.Gui.object_model import ObjectsModel
from VeraGridEngine.Devices.Branches.underground_cable_type import UndergroundCableType
from VeraGridEngine.Devices.Parents.editable_device import GCProp


def test_cable_layer_fields_editable_in_both_tables(qt_app: object) -> None:
    """Keep the builder and database table aligned with the scalar cable API.

    :param qt_app: Shared Qt application fixture.
    :return: None.
    """
    _application: object = qt_app
    cable: UndergroundCableType = UndergroundCableType(name='Manual')
    builder_model: UndergroundCableCatalogueModel = UndergroundCableCatalogueModel(cables=list((cable,)))
    property_names: tuple[str, ...] = tuple(prop.name for prop in builder_model.properties)

    # Each supported layer has its own numeric field, not a shared vector cell.
    field_name: str
    for field_name in ('nominal_voltage', 'core_diameter', 'core_internal_diameter',
                       'cable_diameter', 'sheath_thickness', 'main_insulation_thickness',
                       'core_resistivity', 'sheath_resistivity',
                       'core_filling_factor', 'sheath_filling_factor',
                       'core_relative_permeability', 'sheath_relative_permeability',
                       'main_insulation_permittivity', 'outer_insulation_permittivity',
                       'main_insulation_loss_tangent', 'outer_insulation_loss_tangent',
                       'skin_effect_factor'):
        prop: GCProp = cable.registered_properties[field_name]
        column: int = property_names.index(field_name)
        index: QtCore.QModelIndex = builder_model.index(0, column)
        assert prop.tpe is float
        assert builder_model.flags(index) & QtCore.Qt.ItemFlag.ItemIsEditable
        assert builder_model.data(index, QtCore.Qt.ItemDataRole.ToolTipRole) == prop.definition
        assert builder_model.headerData(column, QtCore.Qt.Orientation.Horizontal,
                                        QtCore.Qt.ItemDataRole.ToolTipRole) == prop.definition
        label: str = field_name.replace('_', ' ')
        expected_header: str = f'{label} ({prop.units})' if prop.units else label
        assert builder_model.headerData(column, QtCore.Qt.Orientation.Horizontal) == expected_header

    core_index: QtCore.QModelIndex = builder_model.index(0, property_names.index('core_resistivity'))
    original_sheath: float = cable.sheath_resistivity
    assert builder_model.setData(core_index, '2.0')
    assert cable.core_resistivity == 2.0
    assert cable.sheath_resistivity == original_sheath
    assert not builder_model.setData(core_index, 'invalid')
    assert cable.core_resistivity == 2.0

    # The generic database editor needs no special cable widget to edit scalars.
    view: QtWidgets.QTableView = QtWidgets.QTableView()
    database_model: ObjectsModel = ObjectsModel(
        objects=list((cable,)),
        property_list=list(cable.property_list),
        time_index=None,
        parent=view,
        editable=True,
    )
    sheath_index: QtCore.QModelIndex = database_model.index(0, database_model.attributes.index('sheath_resistivity'))
    assert database_model.flags(sheath_index) & QtCore.Qt.ItemFlag.ItemIsEditable
    assert database_model.setData(sheath_index, 2.8, QtCore.Qt.ItemDataRole.EditRole)
    assert cable.sheath_resistivity == 2.8
    assert cable.core_resistivity == 2.0

    # Informational PF inputs must not suggest that editing them changes Z or Y.
    for field_name in ('core_dc_resistance', 'proximity_effect_factor'):
        index = builder_model.index(0, property_names.index(field_name))
        assert not builder_model.flags(index) & QtCore.Qt.ItemFlag.ItemIsEditable
        assert not builder_model.setData(index, 123.0)
        database_index: QtCore.QModelIndex = database_model.index(0, database_model.attributes.index(field_name))
        assert not database_model.flags(database_index) & QtCore.Qt.ItemFlag.ItemIsEditable

    view.deleteLater()
