# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import List, TypeAlias

from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.acdc_converter import ACDCConverter
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.acdc_converterdc_terminal import ACDCConverterDCTerminal
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.cs_converter import CsConverter
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_base_terminal import DCBaseTerminal
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_breaker import DCBreaker
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_busbar import DCBusbar
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_chopper import DCChopper
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_conducting_equipment import DCConductingEquipment
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_converter_unit import DCConverterUnit
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_disconnector import DCDisconnector
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_equipment_container import DCEquipmentContainer
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_ground import DCGround
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_line import DCLine
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_line_segment import DCLineSegment
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_node import DCNode
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_series_device import DCSeriesDevice
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_shunt import DCShunt
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_switch import DCSwitch
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_terminal import DCTerminal
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.full_model import FullModel
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.cgm_region import CGMRegion
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.merging_agent import MergingAgent
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.modelling_authority import ModellingAuthority
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.per_lengthdc_line_parameter import PerLengthDCLineParameter
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.vs_capability_curve import VsCapabilityCurve
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.vs_converter import VsConverter
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.bus_name_marker import BusNameMarker
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.analog_control import AnalogControl
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.control import Control
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.limit import Limit
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.limit_set import LimitSet
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.measurement import Measurement
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.measurement_value import MeasurementValue
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.quality61850 import Quality61850
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.energy_scheduling_type import EnergySchedulingType
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.energy_source import EnergySource
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.fossil_fuel import FossilFuel
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.generating_unit import GeneratingUnit
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.hydro_generating_unit import HydroGeneratingUnit
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.hydro_power_plant import HydroPowerPlant
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.hydro_pump import HydroPump
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.nuclear_generating_unit import NuclearGeneratingUnit
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.solar_generating_unit import SolarGeneratingUnit
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.thermal_generating_unit import ThermalGeneratingUnit
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.wind_generating_unit import WindGeneratingUnit
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.acdc_terminal import ACDCTerminal
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.base_voltage import BaseVoltage
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.basic_interval_schedule import BasicIntervalSchedule
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.bay import Bay
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.conducting_equipment import ConductingEquipment
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.connectivity_node import ConnectivityNode
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.connectivity_node_container import ConnectivityNodeContainer
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.curve import Curve
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.curve_data import CurveData
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.equipment import Equipment
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.equipment_container import EquipmentContainer
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.geographical_region import GeographicalRegion
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.identified_object import IdentifiedObject
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.power_system_resource import PowerSystemResource
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.regular_interval_schedule import RegularIntervalSchedule
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.reporting_group import ReportingGroup
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.sub_geographical_region import SubGeographicalRegion
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.substation import Substation
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.terminal import Terminal
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.voltage_level import VoltageLevel
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.active_power_limit import ActivePowerLimit
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.apparent_power_limit import ApparentPowerLimit
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.current_limit import CurrentLimit
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.operational_limit import OperationalLimit
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.operational_limit_set import OperationalLimitSet
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.operational_limit_type import OperationalLimitType
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.voltage_limit import VoltageLimit
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.ac_line_segment import ACLineSegment
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.asynchronous_machine import AsynchronousMachine
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.breaker import Breaker
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.busbar_section import BusbarSection
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.conductor import Conductor
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.connector import Connector
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.disconnector import Disconnector
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.earth_fault_compensator import EarthFaultCompensator
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.energy_consumer import EnergyConsumer
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.external_network_injection import ExternalNetworkInjection
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.ground import Ground
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.ground_disconnector import GroundDisconnector
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.grounding_impedance import GroundingImpedance
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.junction import Junction
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.line import Line
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.linear_shunt_compensator import LinearShuntCompensator
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.load_break_switch import LoadBreakSwitch
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.mutual_coupling import MutualCoupling
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.nonlinear_shunt_compensator import NonlinearShuntCompensator
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.nonlinear_shunt_compensator_point import \
    NonlinearShuntCompensatorPoint
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.petersen_coil import PetersenCoil
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.phase_tap_changer import PhaseTapChanger
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.phase_tap_changer_asymmetrical import \
    PhaseTapChangerAsymmetrical
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.phase_tap_changer_linear import PhaseTapChangerLinear
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.phase_tap_changer_non_linear import PhaseTapChangerNonLinear
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.phase_tap_changer_symmetrical import \
    PhaseTapChangerSymmetrical
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.phase_tap_changer_table import PhaseTapChangerTable
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.phase_tap_changer_table_point import PhaseTapChangerTablePoint
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.phase_tap_changer_tabular import PhaseTapChangerTabular
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.power_transformer import PowerTransformer
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.power_transformer_end import PowerTransformerEnd
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.protected_switch import ProtectedSwitch
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.ratio_tap_changer import RatioTapChanger
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.ratio_tap_changer_table import RatioTapChangerTable
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.ratio_tap_changer_table_point import RatioTapChangerTablePoint
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.reactive_capability_curve import ReactiveCapabilityCurve
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.regulating_cond_eq import RegulatingCondEq
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.regulating_control import RegulatingControl
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.rotating_machine import RotatingMachine
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.series_compensator import SeriesCompensator
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.shunt_compensator import ShuntCompensator
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.static_var_compensator import StaticVarCompensator
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.switch import Switch
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.synchronous_machine import SynchronousMachine
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.tap_changer import TapChanger
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.tap_changer_control import TapChangerControl
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.tap_changer_table_point import TapChangerTablePoint
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.transformer_end import TransformerEnd
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.conform_load import ConformLoad
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.conform_load_group import ConformLoadGroup
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.energy_area import EnergyArea
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.load_area import LoadArea
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.load_group import LoadGroup
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.load_response_characteristic import LoadResponseCharacteristic
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.non_conform_load import NonConformLoad
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.non_conform_load_group import NonConformLoadGroup
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.season_day_type_schedule import SeasonDayTypeSchedule
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.sub_load_area import SubLoadArea
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.equivalent_branch import EquivalentBranch
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.equivalent_equipment import EquivalentEquipment
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.equivalent_injection import EquivalentInjection
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.equivalent_network import EquivalentNetwork
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.equivalent_shunt import EquivalentShunt
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.control_area import ControlArea
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.control_area_generating_unit import ControlAreaGeneratingUnit
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.tie_flow import TieFlow
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_topological_island import DCTopologicalIsland
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.sv_status import SvStatus
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.sv_injection import SvInjection
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.sv_power_flow import SvPowerFlow
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.sv_shunt_compensator_sections import \
    SvShuntCompensatorSections
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.sv_tap_step import SvTapStep
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.sv_voltage import SvVoltage
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.dc_topological_node import DCTopologicalNode
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.topological_node import TopologicalNode
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.topological_island import TopologicalIsland
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.coordinate_system import CoordinateSystem
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.location import Location
from VeraGridEngine.IO.cim.cgmes.cgmes_v2_4_15.devices.position_point import PositionPoint

CGMES_2_4_15_ASSETS: TypeAlias = (
        ACDCConverter |
        ACDCConverterDCTerminal |
        CsConverter |
        DCBaseTerminal |
        DCBreaker |
        DCBusbar |
        DCChopper |
        DCConductingEquipment |
        DCConverterUnit |
        DCDisconnector |
        DCEquipmentContainer |
        DCGround |
        DCLine |
        DCLineSegment |
        DCNode |
        DCSeriesDevice |
        DCShunt |
        DCSwitch |
        DCTerminal |
        CGMRegion |
        MergingAgent |
        ModellingAuthority |
        FullModel |
        PerLengthDCLineParameter |
        VsCapabilityCurve |
        VsConverter |
        BusNameMarker |
        AnalogControl |
        Control |
        Limit |
        LimitSet |
        Measurement |
        MeasurementValue |
        Quality61850 |
        EnergySchedulingType |
        EnergySource |
        FossilFuel |
        GeneratingUnit |
        HydroGeneratingUnit |
        HydroPowerPlant |
        HydroPump |
        NuclearGeneratingUnit |
        SolarGeneratingUnit |
        ThermalGeneratingUnit |
        WindGeneratingUnit |
        ACDCTerminal |
        BaseVoltage |
        BasicIntervalSchedule |
        Bay |
        ConductingEquipment |
        ConnectivityNode |
        ConnectivityNodeContainer |
        Curve |
        CurveData |
        Equipment |
        EquipmentContainer |
        GeographicalRegion |
        IdentifiedObject |
        PowerSystemResource |
        RegularIntervalSchedule |
        ReportingGroup |
        SubGeographicalRegion |
        Substation |
        Terminal |
        VoltageLevel |
        ActivePowerLimit |
        ApparentPowerLimit |
        CurrentLimit |
        OperationalLimit |
        OperationalLimitSet |
        OperationalLimitType |
        VoltageLimit |
        ACLineSegment |
        AsynchronousMachine |
        Breaker |
        BusbarSection |
        Conductor |
        Connector |
        Disconnector |
        EarthFaultCompensator |
        EnergyConsumer |
        ExternalNetworkInjection |
        Ground |
        GroundDisconnector |
        GroundingImpedance |
        Junction |
        Line |
        LinearShuntCompensator |
        LoadBreakSwitch |
        MutualCoupling |
        NonlinearShuntCompensator |
        NonlinearShuntCompensatorPoint |
        PetersenCoil |
        PhaseTapChanger |
        PhaseTapChangerAsymmetrical |
        PhaseTapChangerLinear |
        PhaseTapChangerNonLinear |
        PhaseTapChangerSymmetrical |
        PhaseTapChangerTable |
        PhaseTapChangerTablePoint |
        PhaseTapChangerTabular |
        PowerTransformer |
        PowerTransformerEnd |
        ProtectedSwitch |
        RatioTapChanger |
        RatioTapChangerTable |
        RatioTapChangerTablePoint |
        ReactiveCapabilityCurve |
        RegulatingCondEq |
        RegulatingControl |
        RotatingMachine |
        SeriesCompensator |
        ShuntCompensator |
        StaticVarCompensator |
        Switch |
        SynchronousMachine |
        TapChanger |
        TapChangerControl |
        TapChangerTablePoint |
        TransformerEnd |
        ConformLoad |
        ConformLoadGroup |
        EnergyArea |
        LoadArea |
        LoadGroup |
        LoadResponseCharacteristic |
        NonConformLoad |
        NonConformLoadGroup |
        SeasonDayTypeSchedule |
        SubLoadArea |
        EquivalentBranch |
        EquivalentEquipment |
        EquivalentInjection |
        EquivalentNetwork |
        EquivalentShunt |
        ControlArea |
        ControlAreaGeneratingUnit |
        TieFlow |
        DCTopologicalIsland |
        SvStatus |
        SvInjection |
        SvPowerFlow |
        SvShuntCompensatorSections |
        SvTapStep |
        SvVoltage |
        DCTopologicalNode |
        TopologicalNode |
        TopologicalIsland |
        CoordinateSystem |
        Location |
        PositionPoint
)


class Cgmes_2_4_15_Assets:
    __slots__ = (
        "ACDCConverterAction_list",
        "ACDCConverterController_list",
        "ACDCConverterDCTerminal_list",
        "ACDCConverterRegularSchedule_list",
        "ACDCConverter_list",
        "ACDCTerminal_list",
        "ACDCTimePoint_list",
        "ACEmulationControlFunction_list",
        "ACLineSegment_list",
        "ACPointOfCommonCoupling_list",
        "ACTieCorridor_list",
        "ActivePowerControlFunction_list",
        "ActivePowerLimitSchedule_list",
        "ActivePowerLimitTimePoint_list",
        "ActivePowerLimit_list",
        "Activity_list",
        "Agent_list",
        "AmbientTemperatureDependencyCurve_list",
        "AnalogControl_list",
        "ApparentPowerLimitSchedule_list",
        "ApparentPowerLimitTimePoint_list",
        "ApparentPowerLimit_list",
        "AreaBorderTerminal_list",
        "AreaDispatchableUnit_list",
        "AreaInterchangeController_list",
        "AssessedElementRegularSchedule_list",
        "AssessedElementRegularTimePoint_list",
        "AssessedElementSchedule_list",
        "AssessedElementTimePoint_list",
        "AssessedElementWithContingency_list",
        "AssessedElementWithRemedialAction_list",
        "AssessedElement_list",
        "Assessment_list",
        "AsynchronousMachineRegularSchedule_list",
        "AsynchronousMachineSchedule_list",
        "AsynchronousMachineTimePoint_list",
        "AsynchronousMachine_list",
        "AutomationBlockGroup_list",
        "AutomationFunction_list",
        "AvailabilityContainer_list",
        "AvailabilityEnabled_list",
        "AvailabilityEquipment_list",
        "AvailabilityExceptionalLimit_list",
        "AvailabilityGroup_list",
        "AvailabilityPowerSystemFunction_list",
        "AvailabilityRemedialActionScheme_list",
        "AvailabilityRemedialAction_list",
        "AvailabilitySchedule_list",
        "AvailabilityTimePoint_list",
        "BaseCaseCurrentLimitSchedule_list",
        "BaseCaseCurrentLimitTimePoint_list",
        "BaseCaseCurrentLimit_list",
        "BaseCasePowerFlowResult_list",
        "BaseIrregularTimeSeries_list",
        "BaseOverloadLimitCurve_list",
        "BaseRegularIntervalSchedule_list",
        "BaseTimeSeries_list",
        "BaseVoltage_list",
        "BasicIntervalSchedule_list",
        "BatteryUnitAction_list",
        "BatteryUnitSchedule_list",
        "BatteryUnitTimePoint_list",
        "Bay_list",
        "BiddingZoneAction_list",
        "BiddingZoneBorder_list",
        "BiddingZone_list",
        "BipolarDCSystem_list",
        "BoundaryFrame_list",
        "Breaker_list",
        "BusNameMarker_list",
        "BusbarSection_list",
        "CGMRegion_list",
        "CalculationBasedImpactAssessmentMatrix_list",
        "CapacityCalculationRegion_list",
        "CircuitShare_list",
        "Circuit_list",
        "ClosedDistributionSystemOperator_list",
        "Collection_list",
        "CompensatorController_list",
        "ConceptScheme_list",
        "ConductingEquipment_list",
        "Conductor_list",
        "ConformLoadGroup_list",
        "ConformLoad_list",
        "ConnectingImpactAssessmentMatrix_list",
        "ConnectivityNodeContainer_list",
        "ConnectivityNode_list",
        "Connector_list",
        "ContingencyArea_list",
        "ContingencyElement_list",
        "ContingencyEquipment_list",
        "ContingencyPowerFlowResult_list",
        "ContingencySchedule_list",
        "ContingencyTimePoint_list",
        "ContingencyWithRemedialAction_list",
        "Contingency_list",
        "ControlAreaGeneratingUnit_list",
        "ControlAreaRegularSchedule_list",
        "ControlAreaSchedule_list",
        "ControlAreaTimePoint_list",
        "ControlArea_list",
        "ControlFunctionBlockAction_list",
        "ControlFunctionBlock_list",
        "Control_list",
        "ControllableQuantity_list",
        "CoordinateSystem_list",
        "CoordinatedCapacityCalculator_list",
        "CountertradeRemedialAction_list",
        "CountertradeScheduleAction_list",
        "CrossBorderRelevance_list",
        "CsConverterRegularSchedule_list",
        "CsConverterSchedule_list",
        "CsConverterTimePoint_list",
        "CsConverter_list",
        "CurrentControlFunction_list",
        "CurrentDroopControlFunction_list",
        "CurrentDroopOverride_list",
        "CurrentLimitSchedule_list",
        "CurrentLimitTimePoint_list",
        "CurrentLimit_list",
        "CurveData_list",
        "Curve_list",
        "DCBaseTerminal_list",
        "DCBiPole_list",
        "DCBreaker_list",
        "DCBusbar_list",
        "DCBypassSwitch_list",
        "DCChopper_list",
        "DCCommutationSwitch_list",
        "DCConductingEquipment_list",
        "DCConverterParallelingSwitch_list",
        "DCConverterUnit_list",
        "DCCurrentControlFunction_list",
        "DCDisconnector_list",
        "DCEarthReturnTransferSwitch_list",
        "DCEquipmentContainer_list",
        "DCGround_list",
        "DCHarmonicFilter_list",
        "DCHighSpeedSwitch_list",
        "DCLineParallelingSwitch_list",
        "DCLineSegment_list",
        "DCLine_list",
        "DCMetalicReturnSwitch_list",
        "DCNeutralBusGroundingSwitch_list",
        "DCNeutralBusSwitch_list",
        "DCNode_list",
        "DCPointOfCommonCoupling_list",
        "DCPole_list",
        "DCSeriesDevice_list",
        "DCShunt_list",
        "DCSmoothingReactorArrester_list",
        "DCSmoothingReactor_list",
        "DCSubstationBipole_list",
        "DCSubstationPole_list",
        "DCSubstation_list",
        "DCSwitch_list",
        "DCSystem_list",
        "DCTerminal_list",
        "DCTieCorridor_list",
        "DCTopologicalIsland_list",
        "DCTopologicalNode_list",
        "DCVoltageControlFunction_list",
        "Dataset_list",
        "DifferenceModel_list",
        "DirectCurrentBipoleController_list",
        "DirectCurrentCircuit_list",
        "DirectCurrentEquipmentController_list",
        "DirectCurrentMasterController_list",
        "DirectCurrentPoleController_list",
        "DirectCurrentSubstationBipoleController_list",
        "DirectCurrentSubstationController_list",
        "DirectCurrentSubstationPoleController_list",
        "DirectCurrentSystemOperator_list",
        "Disconnector_list",
        "DistributionSystemOperator_list",
        "DurationOverloadLimitCurve_list",
        "EarthFaultCompensator_list",
        "EdgeControlArea_list",
        "EdgeSchedulingArea_list",
        "ElectricalChargingUnit_list",
        "EnablingTimePoint_list",
        "EnergyAlignmentCoordinator_list",
        "EnergyArea_list",
        "EnergyBlockComponent_list",
        "EnergyBlockOrder_list",
        "EnergyComponent_list",
        "EnergyConnectionRegularSchedule_list",
        "EnergyConnectionSchedule_list",
        "EnergyConnectionTimePoint_list",
        "EnergyConsumer_list",
        "EnergyCoordinationRegion_list",
        "EnergyExchangePoint_list",
        "EnergyGroup_list",
        "EnergySchedulingType_list",
        "EnergySourceModification_list",
        "EnergySourceReference_list",
        "EnergySource_list",
        "EnergyType_list",
        "Entity_list",
        "EquipmentContainer_list",
        "EquipmentControllerAction_list",
        "EquipmentController_list",
        "Equipment_list",
        "EquivalentBranch_list",
        "EquivalentEquipment_list",
        "EquivalentGeneratingUnit_list",
        "EquivalentInjectionAction_list",
        "EquivalentInjectionRegularSchedule_list",
        "EquivalentInjectionSchedule_list",
        "EquivalentInjectionTimePoint_list",
        "EquivalentInjection_list",
        "EquivalentNetwork_list",
        "EquivalentPowerElectronicsUnit_list",
        "EquivalentPowerPlant_list",
        "EquivalentShunt_list",
        "EventSchedule_list",
        "EventTimePoint_list",
        "ExceptionalContingency_list",
        "ExceptionalPowerTransferCorridor_list",
        "ExternalNetworkInjectionAction_list",
        "ExternalNetworkInjectionRegularSchedule_list",
        "ExternalNetworkInjectionSchedule_list",
        "ExternalNetworkInjectionTimePoint_list",
        "ExternalNetworkInjection_list",
        "FACTSEquipment_list",
        "FaultCause_list",
        "FaultOutage_list",
        "FlexibleEnergyUnit_list",
        "FossilFuel_list",
        "Frame_list",
        "FrequencyControlFunction_list",
        "FrequencyMonitoringTerminal_list",
        "FuelStorageRegularSchedule_list",
        "FuelStorageSchedule_list",
        "FuelStorageTimePoint_list",
        "FuelStorage_list",
        "FullModel_list",
        "FunctionBlock_list",
        "FunctionInputVariable_list",
        "FunctionOutputVariable_list",
        "GateInputPin_list",
        "Gate_list",
        "GeneratingUnitSchedule_list",
        "GeneratingUnitTimePoint_list",
        "GeneratingUnit_list",
        "GenericAvailableSchedule_list",
        "GenericEnablingSchedule_list",
        "GenericSequenceSchedule_list",
        "GenericValueSchedule_list",
        "GenericValueTimePoint_list",
        "GeographicalRegion_list",
        "Geometry_list",
        "GeothermalGeneratingUnit_list",
        "GridConnectionPoint_list",
        "GridDisturbance_list",
        "GridStateAlterationCollection_list",
        "GridStateAlterationRemedialAction_list",
        "GridStateAlterationSchedule_list",
        "GridStateAlterationTimePoint_list",
        "GridStateAlteration_list",
        "GridStateIntensitySchedule_list",
        "GroundDisconnector_list",
        "Ground_list",
        "GroundingImpedance_list",
        "HourPattern_list",
        "HourPeriod_list",
        "HydroGeneratingUnit_list",
        "HydroPowerPlant_list",
        "HydroPump_list",
        "IdentifiedObject_list",
        "ImpactAssessmentMatrix_list",
        "ImpedanceControlFunction_list",
        "InServiceAction_list",
        "InServiceRegularSchedule_list",
        "InServiceSchedule_list",
        "InServiceTimePoint_list",
        "InfeedLimitSchedule_list",
        "InfeedLimitTimePoint_list",
        "InfeedLimit_list",
        "InfeedTerminal_list",
        "InfluenceArea_list",
        "InjectionController_list",
        "Installation_list",
        "Interruption_list",
        "IntertemporalPropertyRange_list",
        "Junction_list",
        "LicenseDocument_list",
        "LimitDependencyCurve_list",
        "LimitSet_list",
        "Limit_list",
        "LineCircuit_list",
        "Line_list",
        "LinearShuntCompensator_list",
        "ListBasedImpactAssessmentMatrix_list",
        "LoadAction_list",
        "LoadArea_list",
        "LoadBreakSwitch_list",
        "LoadFrequencyControlArea_list",
        "LoadFrequencyControlBlock_list",
        "LoadFrequencyControlOperator_list",
        "LoadGroup_list",
        "LoadResponseCharacteristic_list",
        "Location_list",
        "LossCurve_list",
        "MeasurementCalculatorInput_list",
        "MeasurementCalculator_list",
        "MeasurementValue_list",
        "Measurement_list",
        "MergingAgent_list",
        "MobileElectricalUnit_list",
        "Model_list",
        "ModelingAuthority_list",
        "Modeling_list",
        "ModellingAuthority_list",
        "ModularStaticSynchronousSeriesCompensator_list",
        "MonitoringArea_list",
        "MonopolarDCSystem_list",
        "MustRunSchedule_list",
        "MustRunTimePoint_list",
        "MutualCoupling_list",
        "NameType_list",
        "Name_list",
        "NamingAuthority_list",
        "NonConformLoadGroup_list",
        "NonConformLoad_list",
        "NonlinearShuntCompensatorPoint_list",
        "NonlinearShuntCompensator_list",
        "NuclearGeneratingUnit_list",
        "ObjectType_list",
        "ObservabilityArea_list",
        "ObservableQuantity_list",
        "OperationalLimitSet_list",
        "OperationalLimitType_list",
        "OperationalLimit_list",
        "OrdinaryContingency_list",
        "OrdinaryPowerTransferCorridor_list",
        "OrganisationRole_list",
        "Organisation_list",
        "OutOfRangeContingency_list",
        "OutageCoordinationRegion_list",
        "OutageCoordinator_list",
        "OutagePlanningAgent_list",
        "OutcomeValue_list",
        "OverlappingZone_list",
        "OwnerRemedialActionAssessment_list",
        "PTCActivePowerSupport_list",
        "ParticipationFactorTimePoint_list",
        "PerLengthDCLineParameter_list",
        "PetersenCoil_list",
        "PhaseControlFunction_list",
        "PhaseTapChangerAsymmetrical_list",
        "PhaseTapChangerLinear_list",
        "PhaseTapChangerNonLinear_list",
        "PhaseTapChangerSymmetrical_list",
        "PhaseTapChangerTablePoint_list",
        "PhaseTapChangerTable_list",
        "PhaseTapChangerTabular_list",
        "PhaseTapChanger_list",
        "PinContingency_list",
        "PinDCTerminal_list",
        "PinEquipmentTripping_list",
        "PinEquipment_list",
        "PinGate_list",
        "PinMeasurement_list",
        "PinOperationalLimit_list",
        "PinPowerTransferCorridor_list",
        "PinTerminal_list",
        "Plant_list",
        "PointOfCommonCoupling_list",
        "PositionPoint_list",
        "PowerBidDependency_list",
        "PowerBidScheduleTimePoint_list",
        "PowerBidSchedule_list",
        "PowerCapacity_list",
        "PowerElectricalChemicalUnit_list",
        "PowerElectronicsConnectionAction_list",
        "PowerElectronicsConnectionController_list",
        "PowerElectronicsMarineUnit_list",
        "PowerElectronicsUnitController_list",
        "PowerFactorControlFunction_list",
        "PowerFlowResult_list",
        "PowerFrequencyController_list",
        "PowerPlantController_list",
        "PowerRemedialActionSchedule_list",
        "PowerRemedialActionTimePoint_list",
        "PowerRemedialAction_list",
        "PowerScheduleAction_list",
        "PowerSchedule_list",
        "PowerShiftKeyDistribution_list",
        "PowerShiftKeySchedule_list",
        "PowerShiftKeyStrategy_list",
        "PowerSystemOrganisationRole_list",
        "PowerSystemProjectGroup_list",
        "PowerSystemProject_list",
        "PowerSystemResource_list",
        "PowerTimePoint_list",
        "PowerTransferCorridor_list",
        "PowerTransformerCircuit_list",
        "PowerTransformerEnd_list",
        "PowerTransformer_list",
        "PropertyReference_list",
        "ProportionalEnergyComponent_list",
        "ProposingRemedialActionScheduleShare_list",
        "ProtectedSwitch_list",
        "QualitativeRemedialActionImpact_list",
        "Quality61850_list",
        "QuantitativeRemedialActionImpact_list",
        "QuantityValue_list",
        "RangeConstraint_list",
        "RatioTapChangerTablePoint_list",
        "RatioTapChangerTable_list",
        "RatioTapChanger_list",
        "ReactiveCapabilityCurve_list",
        "ReactivePowerControlFunction_list",
        "RecoveryOverloadLimitCurve_list",
        "RedispatchRemedialAction_list",
        "RedispatchScheduleAction_list",
        "Region_list",
        "RegularIntervalSchedule_list",
        "RegulatingCondEq_list",
        "RegulatingControlAction_list",
        "RegulatingControlRegularSchedule_list",
        "RegulatingControlSchedule_list",
        "RegulatingControlTimePoint_list",
        "RegulatingControl_list",
        "RemedialActionApplied_list",
        "RemedialActionCost_list",
        "RemedialActionDependency_list",
        "RemedialActionGroupSchedule_list",
        "RemedialActionGroupTimePoint_list",
        "RemedialActionGroup_list",
        "RemedialActionImpact_list",
        "RemedialActionOutcomeValue_list",
        "RemedialActionScheduleDependency_list",
        "RemedialActionScheduleGroup_list",
        "RemedialActionScheduleOutcomeValue_list",
        "RemedialActionScheduleResponse_list",
        "RemedialActionSchedule_list",
        "RemedialActionSchemeSchedule_list",
        "RemedialActionSchemeTimePoint_list",
        "RemedialActionScheme_list",
        "RemedialAction_list",
        "ReportingGroup_list",
        "ReservoirRegularSchedule_list",
        "ReservoirSchedule_list",
        "ReservoirTimePoint_list",
        "RightsStatement_list",
        "RotatingMachineAction_list",
        "RotatingMachineController_list",
        "RotatingMachine_list",
        "SSSCController_list",
        "SSSCSimulationSettings_list",
        "ScheduleResourceController_list",
        "ScheduleResource_list",
        "SchedulingAreaExchangePoint_list",
        "SchedulingArea_list",
        "SchemeRemedialAction_list",
        "SeasonDayTypeSchedule_list",
        "SecondarySubstation_list",
        "SecurityCoordinator_list",
        "SemanticAsset_list",
        "SensitivityArea_list",
        "SensitivityFactor_list",
        "SensitivityMatrix_list",
        "SequenceTimePoint_list",
        "SeriesCompensator_list",
        "SetPointAction_list",
        "ShuntCompensatorModification_list",
        "ShuntCompensatorSchedule_list",
        "ShuntCompensatorTimePoint_list",
        "ShuntCompensator_list",
        "SolarGeneratingUnit_list",
        "SolarRadiationDependencyCurve_list",
        "StageTriggerSchedule_list",
        "StageTriggerTimePoint_list",
        "StageTrigger_list",
        "Stage_list",
        "StaticPropertyRange_list",
        "StaticSynchronousCompensator_list",
        "StaticSynchronousSeriesCompensator_list",
        "StaticVarCompensatorAction_list",
        "StaticVarCompensatorSchedule_list",
        "StaticVarCompensatorTimePoint_list",
        "StaticVarCompensator_list",
        "SubControlArea_list",
        "SubGeographicalRegion_list",
        "SubLoadArea_list",
        "SubSchedulingArea_list",
        "SubstationController_list",
        "Substation_list",
        "SupportedProfileCollection_list",
        "SvInjection_list",
        "SvPowerFlow_list",
        "SvShuntCompensatorSections_list",
        "SvStatus_list",
        "SvTapStep_list",
        "SvVoltage_list",
        "SwitchRegularSchedule_list",
        "SwitchSchedule_list",
        "SwitchTimePoint_list",
        "Switch_list",
        "SynchronousArea_list",
        "SynchronousMachineRegularSchedule_list",
        "SynchronousMachineSchedule_list",
        "SynchronousMachineTimePoint_list",
        "SynchronousMachine_list",
        "SystemControl_list",
        "SystemOperationCoordinator_list",
        "SystemOperator_list",
        "TCSCCompensationPoint_list",
        "TCSCController_list",
        "TapChangerControlRegularSchedule_list",
        "TapChangerControlSchedule_list",
        "TapChangerControl_list",
        "TapChangerController_list",
        "TapChangerTablePoint_list",
        "TapChanger_list",
        "TapPositionAction_list",
        "TapRegularSchedule_list",
        "TapScheduleTimePoint_list",
        "TapSchedule_list",
        "Terminal_list",
        "ThermalGeneratingUnit_list",
        "ThyristorControlledSeriesCompensator_list",
        "TieCorridor_list",
        "TieFlow_list",
        "TopologicalIsland_list",
        "TopologicalNode_list",
        "TopologyAction_list",
        "TransformerEnd_list",
        "TransmissionSystemOperator_list",
        "TriggerCondition_list",
        "UnifiedPowerFlowController_list",
        "UnitCostSchedule_list",
        "UnitCostTimePoint_list",
        "Verification_list",
        "VoltageAngleLimit_list",
        "VoltageAngleSchedule_list",
        "VoltageAngleTimePoint_list",
        "VoltageControlFunction_list",
        "VoltageInjectionControlFunction_list",
        "VoltageLevel_list",
        "VoltageLimitSchedule_list",
        "VoltageLimitTimePoint_list",
        "VoltageLimit_list",
        "VsCapabilityCurve_list",
        "VsConverterRegularSchedule_list",
        "VsConverterSchedule_list",
        "VsConverterTimePoint_list",
        "VsConverter_list",
        "WindGeneratingUnit_list",
        "association_inverse_dict",
        "class_dict",
    )

    def __init__(self):
        self.class_dict = {
            'ACDCConverter': ACDCConverter,
            'ACDCConverterDCTerminal': ACDCConverterDCTerminal,
            'CsConverter': CsConverter,
            'DCBaseTerminal': DCBaseTerminal,
            'DCBreaker': DCBreaker,
            'DCBusbar': DCBusbar,
            'DCChopper': DCChopper,
            'DCConductingEquipment': DCConductingEquipment,
            'DCConverterUnit': DCConverterUnit,
            'DCDisconnector': DCDisconnector,
            'DCEquipmentContainer': DCEquipmentContainer,
            'DCGround': DCGround,
            'DCLine': DCLine,
            'DCLineSegment': DCLineSegment,
            'DCNode': DCNode,
            'DCSeriesDevice': DCSeriesDevice,
            'DCShunt': DCShunt,
            'DCSwitch': DCSwitch,
            'DCTerminal': DCTerminal,
            'CGMRegion': CGMRegion,
            'MergingAgent': MergingAgent,
            'ModellingAuthority': ModellingAuthority,
            'ModelingAuthority': ModellingAuthority,
            'PerLengthDCLineParameter': PerLengthDCLineParameter,
            'VsCapabilityCurve': VsCapabilityCurve,
            'VsConverter': VsConverter,
            'BusNameMarker': BusNameMarker,
            'AnalogControl': AnalogControl,
            'Control': Control,
            'Limit': Limit,
            'LimitSet': LimitSet,
            'Measurement': Measurement,
            'MeasurementValue': MeasurementValue,
            'Quality61850': Quality61850,
            'EnergySchedulingType': EnergySchedulingType,
            'EnergySource': EnergySource,
            'FossilFuel': FossilFuel,
            'GeneratingUnit': GeneratingUnit,
            'HydroGeneratingUnit': HydroGeneratingUnit,
            'HydroPowerPlant': HydroPowerPlant,
            'HydroPump': HydroPump,
            'NuclearGeneratingUnit': NuclearGeneratingUnit,
            'SolarGeneratingUnit': SolarGeneratingUnit,
            'ThermalGeneratingUnit': ThermalGeneratingUnit,
            'WindGeneratingUnit': WindGeneratingUnit,
            'ACDCTerminal': ACDCTerminal,
            'BaseVoltage': BaseVoltage,
            'BasicIntervalSchedule': BasicIntervalSchedule,
            'Bay': Bay,
            'ConductingEquipment': ConductingEquipment,
            'ConnectivityNode': ConnectivityNode,
            'ConnectivityNodeContainer': ConnectivityNodeContainer,
            'Curve': Curve,
            'CurveData': CurveData,
            'Equipment': Equipment,
            'EquipmentContainer': EquipmentContainer,
            'GeographicalRegion': GeographicalRegion,
            'IdentifiedObject': IdentifiedObject,
            'PowerSystemResource': PowerSystemResource,
            'RegularIntervalSchedule': RegularIntervalSchedule,
            'ReportingGroup': ReportingGroup,
            'SubGeographicalRegion': SubGeographicalRegion,
            'Substation': Substation,
            'Terminal': Terminal,
            'VoltageLevel': VoltageLevel,
            'ActivePowerLimit': ActivePowerLimit,
            'ApparentPowerLimit': ApparentPowerLimit,
            'CurrentLimit': CurrentLimit,
            'OperationalLimit': OperationalLimit,
            'OperationalLimitSet': OperationalLimitSet,
            'OperationalLimitType': OperationalLimitType,
            'VoltageLimit': VoltageLimit,
            'ACLineSegment': ACLineSegment,
            'AsynchronousMachine': AsynchronousMachine,
            'Breaker': Breaker,
            'BusbarSection': BusbarSection,
            'Conductor': Conductor,
            'Connector': Connector,
            'Disconnector': Disconnector,
            'EarthFaultCompensator': EarthFaultCompensator,
            'EnergyConsumer': EnergyConsumer,
            'ExternalNetworkInjection': ExternalNetworkInjection,
            'Ground': Ground,
            'GroundDisconnector': GroundDisconnector,
            'GroundingImpedance': GroundingImpedance,
            'Junction': Junction,
            'Line': Line,
            'LinearShuntCompensator': LinearShuntCompensator,
            'LoadBreakSwitch': LoadBreakSwitch,
            'MutualCoupling': MutualCoupling,
            'NonlinearShuntCompensator': NonlinearShuntCompensator,
            'NonlinearShuntCompensatorPoint': NonlinearShuntCompensatorPoint,
            'PetersenCoil': PetersenCoil,
            'PhaseTapChanger': PhaseTapChanger,
            'PhaseTapChangerAsymmetrical': PhaseTapChangerAsymmetrical,
            'PhaseTapChangerLinear': PhaseTapChangerLinear,
            'PhaseTapChangerNonLinear': PhaseTapChangerNonLinear,
            'PhaseTapChangerSymmetrical': PhaseTapChangerSymmetrical,
            'PhaseTapChangerTable': PhaseTapChangerTable,
            'PhaseTapChangerTablePoint': PhaseTapChangerTablePoint,
            'PhaseTapChangerTabular': PhaseTapChangerTabular,
            'PowerTransformer': PowerTransformer,
            'PowerTransformerEnd': PowerTransformerEnd,
            'ProtectedSwitch': ProtectedSwitch,
            'RatioTapChanger': RatioTapChanger,
            'RatioTapChangerTable': RatioTapChangerTable,
            'RatioTapChangerTablePoint': RatioTapChangerTablePoint,
            'ReactiveCapabilityCurve': ReactiveCapabilityCurve,
            'RegulatingCondEq': RegulatingCondEq,
            'RegulatingControl': RegulatingControl,
            'RotatingMachine': RotatingMachine,
            'SeriesCompensator': SeriesCompensator,
            'ShuntCompensator': ShuntCompensator,
            'StaticVarCompensator': StaticVarCompensator,
            'Switch': Switch,
            'SynchronousMachine': SynchronousMachine,
            'TapChanger': TapChanger,
            'TapChangerControl': TapChangerControl,
            'TapChangerTablePoint': TapChangerTablePoint,
            'TransformerEnd': TransformerEnd,
            'ConformLoad': ConformLoad,
            'ConformLoadGroup': ConformLoadGroup,
            'EnergyArea': EnergyArea,
            'LoadArea': LoadArea,
            'LoadGroup': LoadGroup,
            'LoadResponseCharacteristic': LoadResponseCharacteristic,
            'NonConformLoad': NonConformLoad,
            'NonConformLoadGroup': NonConformLoadGroup,
            'SeasonDayTypeSchedule': SeasonDayTypeSchedule,
            'SubLoadArea': SubLoadArea,
            'EquivalentBranch': EquivalentBranch,
            'EquivalentEquipment': EquivalentEquipment,
            'EquivalentInjection': EquivalentInjection,
            'EquivalentNetwork': EquivalentNetwork,
            'EquivalentShunt': EquivalentShunt,
            'ControlArea': ControlArea,
            'ControlAreaGeneratingUnit': ControlAreaGeneratingUnit,
            'TieFlow': TieFlow,
            'DCTopologicalIsland': DCTopologicalIsland,
            'SvStatus': SvStatus,
            'SvInjection': SvInjection,
            'SvPowerFlow': SvPowerFlow,
            'SvShuntCompensatorSections': SvShuntCompensatorSections,
            'SvTapStep': SvTapStep,
            'SvVoltage': SvVoltage,
            'DCTopologicalNode': DCTopologicalNode,
            'TopologicalNode': TopologicalNode,
            'TopologicalIsland': TopologicalIsland,
            'CoordinateSystem': CoordinateSystem,
            'Location': Location,
            'PositionPoint': PositionPoint,
            'FullModel': FullModel,
        }

        self.association_inverse_dict = {
            ('EnergySource', 'EnergySchedulingType'): 'EnergySource',
            ('EnergySource', 'BaseVoltage'): 'ConductingEquipment',
            ('EnergySource', 'EquipmentContainer'): 'Equipments',
            ('RegulatingControl', 'Terminal'): 'RegulatingControl',
            ('ControlAreaGeneratingUnit', 'ControlArea'): 'ControlAreaGeneratingUnit',
            ('ControlAreaGeneratingUnit', 'GeneratingUnit'): 'ControlAreaGeneratingUnit',
            ('NuclearGeneratingUnit', 'EquipmentContainer'): 'Equipments',
            ('Line', 'Region'): 'Lines',
            ('ApparentPowerLimit', 'OperationalLimitSet'): 'OperationalLimitValue',
            ('ApparentPowerLimit', 'OperationalLimitType'): 'OperationalLimit',
            ('ControlArea', 'EnergyArea'): 'ControlArea',
            ('RatioTapChangerTablePoint', 'RatioTapChangerTable'): 'RatioTapChangerTablePoint',
            ('SubGeographicalRegion', 'Region'): 'Regions',
            ('PowerTransformer', 'BaseVoltage'): 'ConductingEquipment',
            ('PowerTransformer', 'EquipmentContainer'): 'Equipments',
            ('Terminal', 'ConductingEquipment'): 'Terminals',
            ('Terminal', 'ConnectivityNode'): 'Terminals',
            ('Terminal', 'BusNameMarker'): 'Terminal',
            ('Terminal', 'TopologicalNode'): 'Terminal',
            ('EquivalentBranch', 'EquivalentNetwork'): 'EquivalentEquipments',
            ('EquivalentBranch', 'BaseVoltage'): 'ConductingEquipment',
            ('EquivalentBranch', 'EquipmentContainer'): 'Equipments',
            ('ConformLoadGroup', 'SubLoadArea'): 'LoadGroups',
            ('Disconnector', 'BaseVoltage'): 'ConductingEquipment',
            ('Disconnector', 'EquipmentContainer'): 'Equipments',
            ('BusNameMarker', 'ReportingGroup'): 'BusNameMarker',
            ('ThermalGeneratingUnit', 'EquipmentContainer'): 'Equipments',
            ('DCConverterUnit', 'Substation'): 'DCConverterUnit',
            ('LoadBreakSwitch', 'BaseVoltage'): 'ConductingEquipment',
            ('LoadBreakSwitch', 'EquipmentContainer'): 'Equipments',
            ('FossilFuel', 'ThermalGeneratingUnit'): 'FossilFuels',
            ('PowerTransformerEnd', 'PowerTransformer'): 'PowerTransformerEnd',
            ('PowerTransformerEnd', 'BaseVoltage'): 'TransformerEnds',
            ('PowerTransformerEnd', 'Terminal'): 'TransformerEnd',
            ('TapChangerControl', 'Terminal'): 'RegulatingControl',
            ('RatioTapChanger', 'TransformerEnd'): 'RatioTapChanger',
            ('RatioTapChanger', 'RatioTapChangerTable'): 'RatioTapChanger',
            ('RatioTapChanger', 'TapChangerControl'): 'TapChanger',
            ('DCSeriesDevice', 'EquipmentContainer'): 'Equipments',
            ('DCShunt', 'EquipmentContainer'): 'Equipments',
            ('ConformLoad', 'LoadGroup'): 'EnergyConsumers',
            ('ConformLoad', 'LoadResponse'): 'EnergyConsumer',
            ('ConformLoad', 'BaseVoltage'): 'ConductingEquipment',
            ('ConformLoad', 'EquipmentContainer'): 'Equipments',
            ('DCLine', 'Region'): 'DCLines',
            ('BusbarSection', 'BaseVoltage'): 'ConductingEquipment',
            ('BusbarSection', 'EquipmentContainer'): 'Equipments',
            ('HydroGeneratingUnit', 'HydroPowerPlant'): 'HydroGeneratingUnits',
            ('HydroGeneratingUnit', 'EquipmentContainer'): 'Equipments',
            ('SynchronousMachine', 'InitialReactiveCapabilityCurve'): 'InitiallyUsedBySynchronousMachines',
            ('SynchronousMachine', 'GeneratingUnit'): 'RotatingMachine',
            ('SynchronousMachine', 'RegulatingControl'): 'RegulatingCondEq',
            ('SynchronousMachine', 'BaseVoltage'): 'ConductingEquipment',
            ('SynchronousMachine', 'EquipmentContainer'): 'Equipments',
            ('OperationalLimitSet', 'Equipment'): 'OperationalLimitSet',
            ('OperationalLimitSet', 'Terminal'): 'OperationalLimitSet',
            ('VoltageLimit', 'OperationalLimitSet'): 'OperationalLimitValue',
            ('VoltageLimit', 'OperationalLimitType'): 'OperationalLimit',
            ('VsConverter', 'CapabilityCurve'): 'VsConverterDCSides',
            ('VsConverter', 'PccTerminal'): 'ConverterDCSides',
            ('VsConverter', 'BaseVoltage'): 'ConductingEquipment',
            ('VsConverter', 'EquipmentContainer'): 'Equipments',
            ('HydroPump', 'RotatingMachine'): 'HydroPump',
            ('HydroPump', 'HydroPowerPlant'): 'HydroPumps',
            ('HydroPump', 'EquipmentContainer'): 'Equipments',
            ('GroundingImpedance', 'BaseVoltage'): 'ConductingEquipment',
            ('GroundingImpedance', 'EquipmentContainer'): 'Equipments',
            ('Ground', 'BaseVoltage'): 'ConductingEquipment',
            ('Ground', 'EquipmentContainer'): 'Equipments',
            ('PetersenCoil', 'BaseVoltage'): 'ConductingEquipment',
            ('PetersenCoil', 'EquipmentContainer'): 'Equipments',
            ('AsynchronousMachine', 'GeneratingUnit'): 'RotatingMachine',
            ('AsynchronousMachine', 'RegulatingControl'): 'RegulatingCondEq',
            ('AsynchronousMachine', 'BaseVoltage'): 'ConductingEquipment',
            ('AsynchronousMachine', 'EquipmentContainer'): 'Equipments',
            ('DCDisconnector', 'EquipmentContainer'): 'Equipments',
            ('Breaker', 'BaseVoltage'): 'ConductingEquipment',
            ('Breaker', 'EquipmentContainer'): 'Equipments',
            ('WindGeneratingUnit', 'EquipmentContainer'): 'Equipments',
            ('EnergyConsumer', 'LoadResponse'): 'EnergyConsumer',
            ('EnergyConsumer', 'BaseVoltage'): 'ConductingEquipment',
            ('EnergyConsumer', 'EquipmentContainer'): 'Equipments',
            ('PhaseTapChangerAsymmetrical', 'TransformerEnd'): 'PhaseTapChanger',
            ('PhaseTapChangerAsymmetrical', 'TapChangerControl'): 'TapChanger',
            ('LinearShuntCompensator', 'RegulatingControl'): 'RegulatingCondEq',
            ('LinearShuntCompensator', 'BaseVoltage'): 'ConductingEquipment',
            ('LinearShuntCompensator', 'EquipmentContainer'): 'Equipments',
            ('SeriesCompensator', 'BaseVoltage'): 'ConductingEquipment',
            ('SeriesCompensator', 'EquipmentContainer'): 'Equipments',
            ('VoltageLevel', 'Substation'): 'VoltageLevels',
            ('VoltageLevel', 'BaseVoltage'): 'VoltageLevel',
            ('SubLoadArea', 'LoadArea'): 'SubLoadAreas',
            ('ExternalNetworkInjection', 'RegulatingControl'): 'RegulatingCondEq',
            ('ExternalNetworkInjection', 'BaseVoltage'): 'ConductingEquipment',
            ('ExternalNetworkInjection', 'EquipmentContainer'): 'Equipments',
            ('GeneratingUnit', 'EquipmentContainer'): 'Equipments',
            ('GroundDisconnector', 'BaseVoltage'): 'ConductingEquipment',
            ('GroundDisconnector', 'EquipmentContainer'): 'Equipments',
            ('DCBusbar', 'EquipmentContainer'): 'Equipments',
            ('PhaseTapChangerSymmetrical', 'TransformerEnd'): 'PhaseTapChanger',
            ('PhaseTapChangerSymmetrical', 'TapChangerControl'): 'TapChanger',
            ('NonlinearShuntCompensator', 'RegulatingControl'): 'RegulatingCondEq',
            ('NonlinearShuntCompensator', 'BaseVoltage'): 'ConductingEquipment',
            ('NonlinearShuntCompensator', 'EquipmentContainer'): 'Equipments',
            ('Substation', 'Region'): 'Substations',
            ('ConnectivityNode', 'ConnectivityNodeContainer'): 'ConnectivityNodes',
            ('ConnectivityNode', 'TopologicalNode'): 'ConnectivityNodes',
            ('TieFlow', 'ControlArea'): 'TieFlow',
            ('TieFlow', 'Terminal'): 'TieFlow',
            ('DCGround', 'EquipmentContainer'): 'Equipments',
            ('DCLineSegment', 'PerLengthParameter'): 'DCLineSegments',
            ('DCLineSegment', 'EquipmentContainer'): 'Equipments',
            ('Switch', 'BaseVoltage'): 'ConductingEquipment',
            ('Switch', 'EquipmentContainer'): 'Equipments',
            ('DCTerminal', 'DCConductingEquipment'): 'DCTerminals',
            ('DCTerminal', 'DCNode'): 'DCTerminals',
            ('DCTerminal', 'BusNameMarker'): 'Terminal',
            ('DCTerminal', 'DCTopologicalNode'): 'DCTerminals',
            ('ACLineSegment', 'BaseVoltage'): 'ConductingEquipment',
            ('ACLineSegment', 'EquipmentContainer'): 'Equipments',
            ('StaticVarCompensator', 'RegulatingControl'): 'RegulatingCondEq',
            ('StaticVarCompensator', 'BaseVoltage'): 'ConductingEquipment',
            ('StaticVarCompensator', 'EquipmentContainer'): 'Equipments',
            ('ActivePowerLimit', 'OperationalLimitSet'): 'OperationalLimitValue',
            ('ActivePowerLimit', 'OperationalLimitType'): 'OperationalLimit',
            ('SolarGeneratingUnit', 'EquipmentContainer'): 'Equipments',
            ('DCNode', 'DCEquipmentContainer'): 'DCNodes',
            ('DCNode', 'DCTopologicalNode'): 'DCNodes',
            ('PhaseTapChangerLinear', 'TransformerEnd'): 'PhaseTapChanger',
            ('PhaseTapChangerLinear', 'TapChangerControl'): 'TapChanger',
            ('MutualCoupling', 'Second_Terminal'): 'HasSecondMutualCoupling',
            ('MutualCoupling', 'First_Terminal'): 'HasFirstMutualCoupling',
            ('CsConverter', 'PccTerminal'): 'ConverterDCSides',
            ('CsConverter', 'BaseVoltage'): 'ConductingEquipment',
            ('CsConverter', 'EquipmentContainer'): 'Equipments',
            ('DCSwitch', 'EquipmentContainer'): 'Equipments',
            ('DCBreaker', 'EquipmentContainer'): 'Equipments',
            ('NonlinearShuntCompensatorPoint', 'NonlinearShuntCompensator'): 'NonlinearShuntCompensatorPoints',
            ('Junction', 'BaseVoltage'): 'ConductingEquipment',
            ('Junction', 'EquipmentContainer'): 'Equipments',
            ('DCChopper', 'EquipmentContainer'): 'Equipments',
            ('PhaseTapChangerTablePoint', 'PhaseTapChangerTable'): 'PhaseTapChangerTablePoint',
            ('CurveData', 'Curve'): 'CurveDatas',
            ('NonConformLoadGroup', 'SubLoadArea'): 'LoadGroups',
            ('EquivalentShunt', 'EquivalentNetwork'): 'EquivalentEquipments',
            ('EquivalentShunt', 'BaseVoltage'): 'ConductingEquipment',
            ('EquivalentShunt', 'EquipmentContainer'): 'Equipments',
            ('CurrentLimit', 'OperationalLimitSet'): 'OperationalLimitValue',
            ('CurrentLimit', 'OperationalLimitType'): 'OperationalLimit',
            ('ACDCConverterDCTerminal', 'DCConductingEquipment'): 'DCTerminals',
            ('ACDCConverterDCTerminal', 'DCNode'): 'DCTerminals',
            ('ACDCConverterDCTerminal', 'BusNameMarker'): 'Terminal',
            ('ACDCConverterDCTerminal', 'DCTopologicalNode'): 'DCTerminals',
            ('Bay', 'VoltageLevel'): 'Bays',
            ('PhaseTapChangerTabular', 'PhaseTapChangerTable'): 'PhaseTapChangerTabular',
            ('PhaseTapChangerTabular', 'TransformerEnd'): 'PhaseTapChanger',
            ('PhaseTapChangerTabular', 'TapChangerControl'): 'TapChanger',
            ('EquivalentInjection', 'ReactiveCapabilityCurve'): 'EquivalentInjection',
            ('EquivalentInjection', 'EquivalentNetwork'): 'EquivalentEquipments',
            ('EquivalentInjection', 'BaseVoltage'): 'ConductingEquipment',
            ('EquivalentInjection', 'EquipmentContainer'): 'Equipments',
            ('NonConformLoad', 'LoadGroup'): 'EnergyConsumers',
            ('NonConformLoad', 'LoadResponse'): 'EnergyConsumer',
            ('NonConformLoad', 'BaseVoltage'): 'ConductingEquipment',
            ('NonConformLoad', 'EquipmentContainer'): 'Equipments',
            ('PositionPoint', 'Location'): 'PositionPoints',
            ('Location', 'PowerSystemResources'): 'Location',
            ('Location', 'CoordinateSystem'): 'Location',
            ('DCTopologicalIsland', 'DCTopologicalNodes'): 'DCTopologicalIsland',
            ('SvShuntCompensatorSections', 'ShuntCompensator'): 'SvShuntCompensatorSections',
            ('SvInjection', 'TopologicalNode'): 'SvInjection',
            ('SvTapStep', 'TapChanger'): 'SvTapStep',
            ('TopologicalIsland', 'AngleRefTopologicalNode'): 'AngleRefTopologicalIsland',
            ('TopologicalIsland', 'TopologicalNodes'): 'TopologicalIsland',
            ('SvStatus', 'ConductingEquipment'): 'SvStatus',
            ('SvVoltage', 'TopologicalNode'): 'SvVoltage',
            ('SvPowerFlow', 'Terminal'): 'SvPowerFlow',
            ('TopologicalNode', 'BaseVoltage'): 'TopologicalNode',
            ('TopologicalNode', 'ConnectivityNodeContainer'): 'TopologicalNode',
            ('TopologicalNode', 'ReportingGroup'): 'TopologicalNode',
            ('DCTopologicalNode', 'DCEquipmentContainer'): 'DCTopologicalNode',
        }

        self.ACDCConverter_list: List[ACDCConverter] = list()
        self.ACDCConverterDCTerminal_list: List[ACDCConverterDCTerminal] = list()
        self.CsConverter_list: List[CsConverter] = list()
        self.DCBaseTerminal_list: List[DCBaseTerminal] = list()
        self.DCBreaker_list: List[DCBreaker] = list()
        self.DCBusbar_list: List[DCBusbar] = list()
        self.DCChopper_list: List[DCChopper] = list()
        self.DCConductingEquipment_list: List[DCConductingEquipment] = list()
        self.DCConverterUnit_list: List[DCConverterUnit] = list()
        self.DCDisconnector_list: List[DCDisconnector] = list()
        self.DCEquipmentContainer_list: List[DCEquipmentContainer] = list()
        self.DCGround_list: List[DCGround] = list()
        self.DCLine_list: List[DCLine] = list()
        self.DCLineSegment_list: List[DCLineSegment] = list()
        self.DCNode_list: List[DCNode] = list()
        self.DCSeriesDevice_list: List[DCSeriesDevice] = list()
        self.DCShunt_list: List[DCShunt] = list()
        self.DCSwitch_list: List[DCSwitch] = list()
        self.DCTerminal_list: List[DCTerminal] = list()
        self.CGMRegion_list: List[CGMRegion] = list()
        self.MergingAgent_list: List[MergingAgent] = list()
        self.ModellingAuthority_list: List[ModellingAuthority] = list()
        self.ModelingAuthority_list: List[ModellingAuthority] = list()
        self.PerLengthDCLineParameter_list: List[PerLengthDCLineParameter] = list()
        self.VsCapabilityCurve_list: List[VsCapabilityCurve] = list()
        self.VsConverter_list: List[VsConverter] = list()
        self.BusNameMarker_list: List[BusNameMarker] = list()
        self.AnalogControl_list: List[AnalogControl] = list()
        self.Control_list: List[Control] = list()
        self.Limit_list: List[Limit] = list()
        self.LimitSet_list: List[LimitSet] = list()
        self.Measurement_list: List[Measurement] = list()
        self.MeasurementValue_list: List[MeasurementValue] = list()
        self.Quality61850_list: List[Quality61850] = list()
        self.EnergySchedulingType_list: List[EnergySchedulingType] = list()
        self.EnergySource_list: List[EnergySource] = list()
        self.FossilFuel_list: List[FossilFuel] = list()
        self.GeneratingUnit_list: List[GeneratingUnit] = list()
        self.HydroGeneratingUnit_list: List[HydroGeneratingUnit] = list()
        self.HydroPowerPlant_list: List[HydroPowerPlant] = list()
        self.HydroPump_list: List[HydroPump] = list()
        self.NuclearGeneratingUnit_list: List[NuclearGeneratingUnit] = list()
        self.SolarGeneratingUnit_list: List[SolarGeneratingUnit] = list()
        self.ThermalGeneratingUnit_list: List[ThermalGeneratingUnit] = list()
        self.WindGeneratingUnit_list: List[WindGeneratingUnit] = list()
        self.ACDCTerminal_list: List[ACDCTerminal] = list()
        self.BaseVoltage_list: List[BaseVoltage] = list()
        self.BasicIntervalSchedule_list: List[BasicIntervalSchedule] = list()
        self.Bay_list: List[Bay] = list()
        self.ConductingEquipment_list: List[ConductingEquipment] = list()
        self.ConnectivityNode_list: List[ConnectivityNode] = list()
        self.ConnectivityNodeContainer_list: List[ConnectivityNodeContainer] = list()
        self.Curve_list: List[Curve] = list()
        self.CurveData_list: List[CurveData] = list()
        self.Equipment_list: List[Equipment] = list()
        self.EquipmentContainer_list: List[EquipmentContainer] = list()
        self.GeographicalRegion_list: List[GeographicalRegion] = list()
        self.IdentifiedObject_list: List[IdentifiedObject] = list()
        self.PowerSystemResource_list: List[PowerSystemResource] = list()
        self.RegularIntervalSchedule_list: List[RegularIntervalSchedule] = list()
        self.ReportingGroup_list: List[ReportingGroup] = list()
        self.SubGeographicalRegion_list: List[SubGeographicalRegion] = list()
        self.Substation_list: List[Substation] = list()
        self.Terminal_list: List[Terminal] = list()
        self.VoltageLevel_list: List[VoltageLevel] = list()
        self.ActivePowerLimit_list: List[ActivePowerLimit] = list()
        self.ApparentPowerLimit_list: List[ApparentPowerLimit] = list()
        self.CurrentLimit_list: List[CurrentLimit] = list()
        self.OperationalLimit_list: List[OperationalLimit] = list()
        self.OperationalLimitSet_list: List[OperationalLimitSet] = list()
        self.OperationalLimitType_list: List[OperationalLimitType] = list()
        self.VoltageLimit_list: List[VoltageLimit] = list()
        self.ACLineSegment_list: List[ACLineSegment] = list()
        self.AsynchronousMachine_list: List[AsynchronousMachine] = list()
        self.Breaker_list: List[Breaker] = list()
        self.BusbarSection_list: List[BusbarSection] = list()
        self.Conductor_list: List[Conductor] = list()
        self.Connector_list: List[Connector] = list()
        self.Disconnector_list: List[Disconnector] = list()
        self.EarthFaultCompensator_list: List[EarthFaultCompensator] = list()
        self.EnergyConsumer_list: List[EnergyConsumer] = list()
        self.ExternalNetworkInjection_list: List[ExternalNetworkInjection] = list()
        self.Ground_list: List[Ground] = list()
        self.GroundDisconnector_list: List[GroundDisconnector] = list()
        self.GroundingImpedance_list: List[GroundingImpedance] = list()
        self.Junction_list: List[Junction] = list()
        self.Line_list: List[Line] = list()
        self.LinearShuntCompensator_list: List[LinearShuntCompensator] = list()
        self.LoadBreakSwitch_list: List[LoadBreakSwitch] = list()
        self.MutualCoupling_list: List[MutualCoupling] = list()
        self.NonlinearShuntCompensator_list: List[NonlinearShuntCompensator] = list()
        self.NonlinearShuntCompensatorPoint_list: List[NonlinearShuntCompensatorPoint] = list()
        self.PetersenCoil_list: List[PetersenCoil] = list()
        self.PhaseTapChanger_list: List[PhaseTapChanger] = list()
        self.PhaseTapChangerAsymmetrical_list: List[PhaseTapChangerAsymmetrical] = list()
        self.PhaseTapChangerLinear_list: List[PhaseTapChangerLinear] = list()
        self.PhaseTapChangerNonLinear_list: List[PhaseTapChangerNonLinear] = list()
        self.PhaseTapChangerSymmetrical_list: List[PhaseTapChangerSymmetrical] = list()
        self.PhaseTapChangerTable_list: List[PhaseTapChangerTable] = list()
        self.PhaseTapChangerTablePoint_list: List[PhaseTapChangerTablePoint] = list()
        self.PhaseTapChangerTabular_list: List[PhaseTapChangerTabular] = list()
        self.PowerTransformer_list: List[PowerTransformer] = list()
        self.PowerTransformerEnd_list: List[PowerTransformerEnd] = list()
        self.ProtectedSwitch_list: List[ProtectedSwitch] = list()
        self.RatioTapChanger_list: List[RatioTapChanger] = list()
        self.RatioTapChangerTable_list: List[RatioTapChangerTable] = list()
        self.RatioTapChangerTablePoint_list: List[RatioTapChangerTablePoint] = list()
        self.ReactiveCapabilityCurve_list: List[ReactiveCapabilityCurve] = list()
        self.RegulatingCondEq_list: List[RegulatingCondEq] = list()
        self.RegulatingControl_list: List[RegulatingControl] = list()
        self.RotatingMachine_list: List[RotatingMachine] = list()
        self.SeriesCompensator_list: List[SeriesCompensator] = list()
        self.ShuntCompensator_list: List[ShuntCompensator] = list()
        self.StaticVarCompensator_list: List[StaticVarCompensator] = list()
        self.Switch_list: List[Switch] = list()
        self.SynchronousMachine_list: List[SynchronousMachine] = list()
        self.TapChanger_list: List[TapChanger] = list()
        self.TapChangerControl_list: List[TapChangerControl] = list()
        self.TapChangerTablePoint_list: List[TapChangerTablePoint] = list()
        self.TransformerEnd_list: List[TransformerEnd] = list()
        self.ConformLoad_list: List[ConformLoad] = list()
        self.ConformLoadGroup_list: List[ConformLoadGroup] = list()
        self.EnergyArea_list: List[EnergyArea] = list()
        self.LoadArea_list: List[LoadArea] = list()
        self.LoadGroup_list: List[LoadGroup] = list()
        self.LoadResponseCharacteristic_list: List[LoadResponseCharacteristic] = list()
        self.NonConformLoad_list: List[NonConformLoad] = list()
        self.NonConformLoadGroup_list: List[NonConformLoadGroup] = list()
        self.SeasonDayTypeSchedule_list: List[SeasonDayTypeSchedule] = list()
        self.SubLoadArea_list: List[SubLoadArea] = list()
        self.EquivalentBranch_list: List[EquivalentBranch] = list()
        self.EquivalentEquipment_list: List[EquivalentEquipment] = list()
        self.EquivalentInjection_list: List[EquivalentInjection] = list()
        self.EquivalentNetwork_list: List[EquivalentNetwork] = list()
        self.EquivalentShunt_list: List[EquivalentShunt] = list()
        self.ControlArea_list: List[ControlArea] = list()
        self.ControlAreaGeneratingUnit_list: List[ControlAreaGeneratingUnit] = list()
        self.TieFlow_list: List[TieFlow] = list()
        self.DCTopologicalIsland_list: List[DCTopologicalIsland] = list()
        self.SvStatus_list: List[SvStatus] = list()
        self.SvInjection_list: List[SvInjection] = list()
        self.SvPowerFlow_list: List[SvPowerFlow] = list()
        self.SvShuntCompensatorSections_list: List[SvShuntCompensatorSections] = list()
        self.SvTapStep_list: List[SvTapStep] = list()
        self.SvVoltage_list: List[SvVoltage] = list()
        self.DCTopologicalNode_list: List[DCTopologicalNode] = list()
        self.TopologicalNode_list: List[TopologicalNode] = list()
        self.TopologicalIsland_list: List[TopologicalIsland] = list()
        self.CoordinateSystem_list: List[CoordinateSystem] = list()
        self.Location_list: List[Location] = list()
        self.PositionPoint_list: List[PositionPoint] = list()
        self.FullModel_list: List[FullModel] = list()
