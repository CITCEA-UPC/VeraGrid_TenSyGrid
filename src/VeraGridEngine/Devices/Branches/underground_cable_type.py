# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import Tuple

from VeraGridEngine.Devices.Parents.editable_device import EditableDevice, GCProp
from VeraGridEngine.enumerations import DeviceType, PrpCat


class UndergroundCableType(EditableDevice):
    """Single-core cable construction with a core and a metallic sheath."""

    __slots__ = (
        '_nominal_voltage',
        '_core_dc_resistance',
        '_core_diameter',
        '_core_internal_diameter',
        '_cable_diameter',
        '_sheath_thickness',
        '_main_insulation_thickness',
        '_core_resistivity',
        '_sheath_resistivity',
        '_core_filling_factor',
        '_sheath_filling_factor',
        '_main_insulation_permittivity',
        '_outer_insulation_permittivity',
        '_main_insulation_loss_tangent',
        '_outer_insulation_loss_tangent',
        '_core_relative_permeability',
        '_sheath_relative_permeability',
        '_skin_effect_factor',
        '_proximity_effect_factor',
    )

    # Descriptive property names also supply GUI headers, as for Wire.
    # Keep the previous names as file aliases so existing catalogues still load.
    LOCAL_PROPERTY_DECLARATIONS: Tuple[GCProp, ...] = (
        GCProp(prop_name='nominal_voltage', old_names=('Vnom',), units='kV', tpe=float,
               definition='Rated cable voltage (PowerFactory uline)',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='core_dc_resistance', old_names=('rpha',), units='Ohm/km', tpe=float,
               definition='Imported core DC resistance at 20 degrees Celsius; informational, not used by the geometric calculation',
               editable=False, cat=(PrpCat.TP,)),
        GCProp(prop_name='core_diameter', old_names=('diaCon',), units='mm', tpe=float,
               definition='Core outer diameter',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='core_internal_diameter', old_names=('diaTube',), units='mm', tpe=float,
               definition='Core inner diameter',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='cable_diameter', old_names=('diaCab',), units='mm', tpe=float,
               definition='Overall cable diameter, including outer insulation',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='sheath_thickness', old_names=('thSht',), units='mm', tpe=float,
               definition='Metallic sheath thickness',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='main_insulation_thickness', old_names=('thIns',), units='mm', tpe=float,
               definition='Core-to-sheath insulation thickness (PowerFactory thIns[0]); outer insulation is derived from cable_diameter',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='core_resistivity', old_names=('crho_core',), units='uOhm*cm', tpe=float,
               definition='Core resistivity at 20 degrees Celsius (PowerFactory crho[0])',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='sheath_resistivity', old_names=('crho_sheath',), units='uOhm*cm', tpe=float,
               definition='Sheath resistivity at 20 degrees Celsius (PowerFactory crho[1])',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='core_filling_factor', old_names=('Cf_core',), units='%', tpe=float,
               definition='Core conducting filling factor (PowerFactory Cf[0])',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='sheath_filling_factor', old_names=('Cf_sheath',), units='%', tpe=float,
               definition='Sheath conducting filling factor (PowerFactory Cf[1])',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='main_insulation_permittivity', old_names=('cepsr_core_sheath',), units='', tpe=float,
               definition='Core-to-sheath insulation relative permittivity (PowerFactory cepsr[0])',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='outer_insulation_permittivity', old_names=('cepsr_outer',), units='', tpe=float,
               definition='Outer insulation relative permittivity (PowerFactory cepsr[1])',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='main_insulation_loss_tangent', old_names=('ctand_core_sheath',), units='', tpe=float,
               definition='Core-to-sheath insulation dielectric loss factor (PowerFactory ctand[0])',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='outer_insulation_loss_tangent', old_names=('ctand_outer',), units='', tpe=float,
               definition='Outer insulation dielectric loss factor (PowerFactory ctand[1])',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='core_relative_permeability', old_names=('my_core',), units='', tpe=float,
               definition='Core relative magnetic permeability (PowerFactory my[0])',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='sheath_relative_permeability', old_names=('my_sheath',), units='', tpe=float,
               definition='Sheath relative magnetic permeability (PowerFactory my[1])',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='skin_effect_factor', old_names=('ks',), units='', tpe=float,
               definition='Core skin-effect factor',
               cat=(PrpCat.TP,)),
        GCProp(prop_name='proximity_effect_factor', old_names=('kp',), units='', tpe=float,
               definition='Imported core proximity-effect factor; informational, not used by this calculation',
               editable=False, cat=(PrpCat.TP,)),
    )

    def __init__(self,
                 name: str = 'Underground cable',
                 idtag: str | None = None,
                 code: str = '',
                 nominal_voltage: float = 1.0,
                 core_dc_resistance: float = 0.0,
                 core_diameter: float = 10.0,
                 core_internal_diameter: float = 0.0,
                 cable_diameter: float = 30.0,
                 sheath_thickness: float = 1.0,
                 main_insulation_thickness: float = 5.0,
                 core_resistivity: float = 1.7241,
                 sheath_resistivity: float = 1.7241,
                 core_filling_factor: float = 100.0,
                 sheath_filling_factor: float = 100.0,
                 main_insulation_permittivity: float = 2.3,
                 outer_insulation_permittivity: float = 2.5,
                 main_insulation_loss_tangent: float = 0.0,
                 outer_insulation_loss_tangent: float = 0.0,
                 core_relative_permeability: float = 1.0,
                 sheath_relative_permeability: float = 1.0,
                 skin_effect_factor: float = 1.0,
                 proximity_effect_factor: float = 1.0) -> None:
        """Create one physical cable using scalar inputs for its two conducting layers.

        :param name: Cable catalogue name.
        :param idtag: Persistent VeraGrid identifier.
        :param code: Optional user code.
        :param nominal_voltage: Rated cable voltage (PowerFactory uline) in kV.
        :param core_dc_resistance: Imported core DC resistance at 20 degrees Celsius; informational, not used by the geometric calculation in Ohm/km.
        :param core_diameter: Core outer diameter in mm.
        :param core_internal_diameter: Core inner diameter in mm.
        :param cable_diameter: Overall cable diameter, including outer insulation in mm.
        :param sheath_thickness: Metallic sheath thickness in mm.
        :param main_insulation_thickness: Core-to-sheath insulation thickness (PowerFactory thIns[0]); outer insulation is derived from cable_diameter in mm.
        :param core_resistivity: Core resistivity at 20 degrees Celsius (PowerFactory crho[0]) in uOhm*cm.
        :param sheath_resistivity: Sheath resistivity at 20 degrees Celsius (PowerFactory crho[1]) in uOhm*cm.
        :param core_filling_factor: Core conducting filling factor (PowerFactory Cf[0]) in %.
        :param sheath_filling_factor: Sheath conducting filling factor (PowerFactory Cf[1]) in %.
        :param main_insulation_permittivity: Core-to-sheath insulation relative permittivity (PowerFactory cepsr[0]).
        :param outer_insulation_permittivity: Outer insulation relative permittivity (PowerFactory cepsr[1]).
        :param main_insulation_loss_tangent: Core-to-sheath insulation dielectric loss factor (PowerFactory ctand[0]).
        :param outer_insulation_loss_tangent: Outer insulation dielectric loss factor (PowerFactory ctand[1]).
        :param core_relative_permeability: Core relative magnetic permeability (PowerFactory my[0]).
        :param sheath_relative_permeability: Sheath relative magnetic permeability (PowerFactory my[1]).
        :param skin_effect_factor: Core skin-effect factor.
        :param proximity_effect_factor: Imported core proximity-effect factor; informational, not used by this calculation.
        :return: None.
        """
        EditableDevice.__init__(
            self,
            name=name,
            idtag=idtag,
            code=code,
            device_type=DeviceType.UndergroundCableTypeDevice,
        )

        # Store only the supported layers; the DGS parser maps the PowerFactory vectors.
        self.nominal_voltage = nominal_voltage
        self.core_dc_resistance = core_dc_resistance
        self.core_diameter = core_diameter
        self.core_internal_diameter = core_internal_diameter
        self.cable_diameter = cable_diameter
        self.sheath_thickness = sheath_thickness
        self.main_insulation_thickness = main_insulation_thickness
        self.core_resistivity = core_resistivity
        self.sheath_resistivity = sheath_resistivity
        self.core_filling_factor = core_filling_factor
        self.sheath_filling_factor = sheath_filling_factor
        self.main_insulation_permittivity = main_insulation_permittivity
        self.outer_insulation_permittivity = outer_insulation_permittivity
        self.main_insulation_loss_tangent = main_insulation_loss_tangent
        self.outer_insulation_loss_tangent = outer_insulation_loss_tangent
        self.core_relative_permeability = core_relative_permeability
        self.sheath_relative_permeability = sheath_relative_permeability
        self.skin_effect_factor = skin_effect_factor
        self.proximity_effect_factor = proximity_effect_factor

    @property
    def nominal_voltage(self) -> float:
        """Read the nominal_voltage input.

        :return: Rated cable voltage (PowerFactory uline) in kV.
        """
        return self._nominal_voltage

    @nominal_voltage.setter
    def nominal_voltage(self, value: float) -> None:
        """Store the nominal_voltage input.

        :param value: Rated cable voltage (PowerFactory uline) in kV.
        :return: None.
        """
        self._nominal_voltage = float(value)

    @property
    def core_dc_resistance(self) -> float:
        """Read the core_dc_resistance input.

        :return: Imported core DC resistance at 20 degrees Celsius; informational, not used by the geometric calculation in Ohm/km.
        """
        return self._core_dc_resistance

    @core_dc_resistance.setter
    def core_dc_resistance(self, value: float) -> None:
        """Store the core_dc_resistance input.

        :param value: Imported core DC resistance at 20 degrees Celsius; informational, not used by the geometric calculation in Ohm/km.
        :return: None.
        """
        self._core_dc_resistance = float(value)

    @property
    def core_diameter(self) -> float:
        """Read the core_diameter input.

        :return: Core outer diameter in mm.
        """
        return self._core_diameter

    @core_diameter.setter
    def core_diameter(self, value: float) -> None:
        """Store the core_diameter input.

        :param value: Core outer diameter in mm.
        :return: None.
        """
        self._core_diameter = float(value)

    @property
    def core_internal_diameter(self) -> float:
        """Read the core_internal_diameter input.

        :return: Core inner diameter in mm.
        """
        return self._core_internal_diameter

    @core_internal_diameter.setter
    def core_internal_diameter(self, value: float) -> None:
        """Store the core_internal_diameter input.

        :param value: Core inner diameter in mm.
        :return: None.
        """
        self._core_internal_diameter = float(value)

    @property
    def cable_diameter(self) -> float:
        """Read the cable_diameter input.

        :return: Overall cable diameter, including outer insulation in mm.
        """
        return self._cable_diameter

    @cable_diameter.setter
    def cable_diameter(self, value: float) -> None:
        """Store the cable_diameter input.

        :param value: Overall cable diameter, including outer insulation in mm.
        :return: None.
        """
        self._cable_diameter = float(value)

    @property
    def sheath_thickness(self) -> float:
        """Read the sheath_thickness input.

        :return: Metallic sheath thickness in mm.
        """
        return self._sheath_thickness

    @sheath_thickness.setter
    def sheath_thickness(self, value: float) -> None:
        """Store the sheath_thickness input.

        :param value: Metallic sheath thickness in mm.
        :return: None.
        """
        self._sheath_thickness = float(value)

    @property
    def main_insulation_thickness(self) -> float:
        """Read the main_insulation_thickness input.

        :return: Core-to-sheath insulation thickness (PowerFactory thIns[0]); outer insulation is derived from cable_diameter in mm.
        """
        return self._main_insulation_thickness

    @main_insulation_thickness.setter
    def main_insulation_thickness(self, value: float) -> None:
        """Store the main_insulation_thickness input.

        :param value: Core-to-sheath insulation thickness (PowerFactory thIns[0]); outer insulation is derived from cable_diameter in mm.
        :return: None.
        """
        self._main_insulation_thickness = float(value)

    @property
    def core_resistivity(self) -> float:
        """Read the core_resistivity input.

        :return: Core resistivity at 20 degrees Celsius (PowerFactory crho[0]) in uOhm*cm.
        """
        return self._core_resistivity

    @core_resistivity.setter
    def core_resistivity(self, value: float) -> None:
        """Store the core_resistivity input.

        :param value: Core resistivity at 20 degrees Celsius (PowerFactory crho[0]) in uOhm*cm.
        :return: None.
        """
        self._core_resistivity = float(value)

    @property
    def sheath_resistivity(self) -> float:
        """Read the sheath_resistivity input.

        :return: Sheath resistivity at 20 degrees Celsius (PowerFactory crho[1]) in uOhm*cm.
        """
        return self._sheath_resistivity

    @sheath_resistivity.setter
    def sheath_resistivity(self, value: float) -> None:
        """Store the sheath_resistivity input.

        :param value: Sheath resistivity at 20 degrees Celsius (PowerFactory crho[1]) in uOhm*cm.
        :return: None.
        """
        self._sheath_resistivity = float(value)

    @property
    def core_filling_factor(self) -> float:
        """Read the core_filling_factor input.

        :return: Core conducting filling factor (PowerFactory Cf[0]) in %.
        """
        return self._core_filling_factor

    @core_filling_factor.setter
    def core_filling_factor(self, value: float) -> None:
        """Store the core_filling_factor input.

        :param value: Core conducting filling factor (PowerFactory Cf[0]) in %.
        :return: None.
        """
        self._core_filling_factor = float(value)

    @property
    def sheath_filling_factor(self) -> float:
        """Read the sheath_filling_factor input.

        :return: Sheath conducting filling factor (PowerFactory Cf[1]) in %.
        """
        return self._sheath_filling_factor

    @sheath_filling_factor.setter
    def sheath_filling_factor(self, value: float) -> None:
        """Store the sheath_filling_factor input.

        :param value: Sheath conducting filling factor (PowerFactory Cf[1]) in %.
        :return: None.
        """
        self._sheath_filling_factor = float(value)

    @property
    def main_insulation_permittivity(self) -> float:
        """Read the main_insulation_permittivity input.

        :return: Core-to-sheath insulation relative permittivity (PowerFactory cepsr[0]).
        """
        return self._main_insulation_permittivity

    @main_insulation_permittivity.setter
    def main_insulation_permittivity(self, value: float) -> None:
        """Store the main_insulation_permittivity input.

        :param value: Core-to-sheath insulation relative permittivity (PowerFactory cepsr[0]).
        :return: None.
        """
        self._main_insulation_permittivity = float(value)

    @property
    def outer_insulation_permittivity(self) -> float:
        """Read the outer_insulation_permittivity input.

        :return: Outer insulation relative permittivity (PowerFactory cepsr[1]).
        """
        return self._outer_insulation_permittivity

    @outer_insulation_permittivity.setter
    def outer_insulation_permittivity(self, value: float) -> None:
        """Store the outer_insulation_permittivity input.

        :param value: Outer insulation relative permittivity (PowerFactory cepsr[1]).
        :return: None.
        """
        self._outer_insulation_permittivity = float(value)

    @property
    def main_insulation_loss_tangent(self) -> float:
        """Read the main_insulation_loss_tangent input.

        :return: Core-to-sheath insulation dielectric loss factor (PowerFactory ctand[0]).
        """
        return self._main_insulation_loss_tangent

    @main_insulation_loss_tangent.setter
    def main_insulation_loss_tangent(self, value: float) -> None:
        """Store the main_insulation_loss_tangent input.

        :param value: Core-to-sheath insulation dielectric loss factor (PowerFactory ctand[0]).
        :return: None.
        """
        self._main_insulation_loss_tangent = float(value)

    @property
    def outer_insulation_loss_tangent(self) -> float:
        """Read the outer_insulation_loss_tangent input.

        :return: Outer insulation dielectric loss factor (PowerFactory ctand[1]).
        """
        return self._outer_insulation_loss_tangent

    @outer_insulation_loss_tangent.setter
    def outer_insulation_loss_tangent(self, value: float) -> None:
        """Store the outer_insulation_loss_tangent input.

        :param value: Outer insulation dielectric loss factor (PowerFactory ctand[1]).
        :return: None.
        """
        self._outer_insulation_loss_tangent = float(value)

    @property
    def core_relative_permeability(self) -> float:
        """Read the core_relative_permeability input.

        :return: Core relative magnetic permeability (PowerFactory my[0]).
        """
        return self._core_relative_permeability

    @core_relative_permeability.setter
    def core_relative_permeability(self, value: float) -> None:
        """Store the core_relative_permeability input.

        :param value: Core relative magnetic permeability (PowerFactory my[0]).
        :return: None.
        """
        self._core_relative_permeability = float(value)

    @property
    def sheath_relative_permeability(self) -> float:
        """Read the sheath_relative_permeability input.

        :return: Sheath relative magnetic permeability (PowerFactory my[1]).
        """
        return self._sheath_relative_permeability

    @sheath_relative_permeability.setter
    def sheath_relative_permeability(self, value: float) -> None:
        """Store the sheath_relative_permeability input.

        :param value: Sheath relative magnetic permeability (PowerFactory my[1]).
        :return: None.
        """
        self._sheath_relative_permeability = float(value)

    @property
    def skin_effect_factor(self) -> float:
        """Read the skin_effect_factor input.

        :return: Core skin-effect factor.
        """
        return self._skin_effect_factor

    @skin_effect_factor.setter
    def skin_effect_factor(self, value: float) -> None:
        """Store the skin_effect_factor input.

        :param value: Core skin-effect factor.
        :return: None.
        """
        self._skin_effect_factor = float(value)

    @property
    def proximity_effect_factor(self) -> float:
        """Read the proximity_effect_factor input.

        :return: Imported core proximity-effect factor; informational, not used by this calculation.
        """
        return self._proximity_effect_factor

    @proximity_effect_factor.setter
    def proximity_effect_factor(self, value: float) -> None:
        """Store the proximity_effect_factor input.

        :param value: Imported core proximity-effect factor; informational, not used by this calculation.
        :return: None.
        """
        self._proximity_effect_factor = float(value)
