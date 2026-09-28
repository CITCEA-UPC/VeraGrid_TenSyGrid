# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path

import numpy as np

import VeraGridEngine.Devices as dev
from VeraGridEngine.IO.file_open import FileOpen
from VeraGridEngine.IO.file_save import FileSave


def build_pf_606_cable() -> dev.UndergroundCableType:
    """Build the full-precision ``250 AA CN`` PowerFactory construction.

    :return: Physical cable input used by the PowerFactory 606 system.
    """
    return dev.UndergroundCableType(
        name='250 AA CN',
        nominal_voltage=15.0,
        core_dc_resistance=0.222551,
        core_diameter=12.716179847717285,
        core_internal_diameter=0.0,
        cable_diameter=30.54450798034668,
        sheath_thickness=0.15925799310207367,
        main_insulation_thickness=7.103906154632568,
        core_resistivity=2.8264000415802,
        sheath_resistivity=1.7240999937057495,
        core_filling_factor=100.0,
        sheath_filling_factor=99.44011688232422,
        main_insulation_permittivity=2.299999952316284,
        outer_insulation_permittivity=2.5,
        main_insulation_loss_tangent=0.0,
        outer_insulation_loss_tangent=0.0,
        core_relative_permeability=1.0,
        sheath_relative_permeability=0.999993622303009,
        skin_effect_factor=1.0,
        proximity_effect_factor=1.0,
    )


def build_pf_606_system(cable: dev.UndergroundCableType) -> dev.UndergroundLineType:
    """Build and calculate the full-precision PowerFactory 606 geometry.

    :param cable: Shared physical cable construction.
    :return: Calculated three-phase cable system.
    """
    system: dev.UndergroundLineType = dev.UndergroundLineType(
        name='606',
        Imax=0.26,
        Vnom=15.0,
        freq=60.0,
        earth_resistivity=100.0,
    )
    x_positions: np.ndarray = np.array([-0.1524, 0.0, 0.1524], dtype=float)
    phase_index: int
    for phase_index in range(3):
        system.add_cable_relationship(
            cable=cable,
            xpos=float(x_positions[phase_index]),
            ypos=0.609599980492801,
            phase=phase_index + 1,
        )
    assert system.compute()
    return system


def test_pf_606_primitive_matrices_match_powerfactory_2024() -> None:
    """Match full-precision primitive values read from PowerFactory 2024.

    :return: None.
    """
    system: dev.UndergroundLineType = build_pf_606_system(cable=build_pf_606_cable())
    expected_impedance: np.ndarray = np.array(
        [
            0.2822073966824634 + 0.9089254735303836j,
            1.338651047334841 + 0.8329460885648026j,
            0.05912471567796385 + 0.65056887441533j,
            0.059124687236848154 + 0.5983068106852449j,
            0.05912476191650118 + 0.8330938788964809j,
        ],
        dtype=complex,
    )
    actual_impedance: np.ndarray = np.array(
        [
            system.z_primitive[0, 0],
            system.z_primitive[3, 3],
            system.z_primitive[0, 1],
            system.z_primitive[0, 2],
            system.z_primitive[0, 3],
        ],
        dtype=complex,
    )
    expected_admittance: np.ndarray = np.array(
        [64.3048474379772e-6j, 522.6045601338822e-6j],
        dtype=complex,
    )
    actual_admittance: np.ndarray = np.array(
        [system.y_primitive[0, 0], system.y_primitive[3, 3]],
        dtype=complex,
    )
    np.testing.assert_allclose(actual_impedance, expected_impedance, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(actual_admittance, expected_admittance, rtol=0.0, atol=1.0e-15)


def test_physical_cable_system_survives_veragrid_roundtrip(tmp_path: Path) -> None:
    """Preserve physical inputs and catalogue references in native files.

    :param tmp_path: Isolated pytest output directory.
    :return: None.
    """
    cable: dev.UndergroundCableType = build_pf_606_cable()
    system: dev.UndergroundLineType = build_pf_606_system(cable=cable)
    circuit: dev.MultiCircuit = dev.MultiCircuit()
    circuit.add_underground_cable(obj=cable)
    circuit.add_underground_line(obj=system)
    output_path: Path = tmp_path / 'physical_underground_cable.veragrid'
    FileSave(circuit=circuit, file_name=str(output_path)).save()

    restored: dev.MultiCircuit = FileOpen(file_name=str(output_path)).open()
    assert len(restored.underground_cable_constructions) == 1
    assert len(restored.underground_cable_types) == 1
    restored_cable: dev.UndergroundCableType = restored.underground_cable_constructions[0]
    restored_system: dev.UndergroundLineType = restored.underground_cable_types[0]
    assert restored_system.cables_in_system.data[0].cable is restored_cable
    assert restored_system.compute()
    np.testing.assert_allclose(restored_system.z_primitive, system.z_primitive, rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(restored_system.y_primitive, system.y_primitive)
    assert restored_cable.sheath_filling_factor == cable.sheath_filling_factor
    assert restored_cable.outer_insulation_permittivity == cable.outer_insulation_permittivity


def test_powerfactory_v7_dgs_imports_physical_cables() -> None:
    """Import the tracked V7.00 cable project fixture end to end.

    :return: None.
    """
    dgs_path: Path = (
        Path(__file__).resolve().parents[1]
        / 'data'
        / 'grids'
        / 'DGS'
        / 'UnbalancedNetworkModel_Cables_606_v7.dgs'
    )
    circuit: dev.MultiCircuit = FileOpen(file_name=str(dgs_path)).open()
    assert len(circuit.underground_cable_constructions) == 3
    assert len(circuit.underground_cable_types) == 4

    system_606: dev.UndergroundLineType = next(
        system for system in circuit.underground_cable_types if system.name == '606'
    )
    line_606: dev.Line = next(line for line in circuit.lines if line.name == 'LC692-675')
    line_607: dev.Line = next(line for line in circuit.lines if line.name == 'LC684-652')
    assert line_606.template is system_606
    assert isinstance(line_607.template, dev.UndergroundLineType)
    assert line_607.template.z_nabc.shape == (1, 1)
    # PF exposes this core admittance when the eliminated concentric-neutral sheath is grounded.
    np.testing.assert_allclose(
        line_607.template.y_nabc,
        np.array([[52.11094270079967e-6j]], dtype=complex),
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        system_606.z_primitive[0, 0],
        0.2822073966824634 + 0.9089254735303836j,
        rtol=0.0,
        atol=1.0e-7,
    )
    assert np.isclose(system_606.cables_in_system.data[0].cable.sheath_filling_factor, 99.44012)
