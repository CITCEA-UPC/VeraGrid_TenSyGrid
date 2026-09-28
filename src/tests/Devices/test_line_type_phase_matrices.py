from typing import Any

import numpy as np

from VeraGridEngine.basic_structures import Logger
from VeraGridEngine.Compilers.circuit_to_data import compile_numerical_circuit_at
from VeraGridEngine.Devices.Branches.overhead_line_type import OverheadLineType
from VeraGridEngine.Devices.Branches.line import Line
from VeraGridEngine.Devices.Branches.sequence_line_type import SequenceLineType
from VeraGridEngine.Devices.Branches.underground_line_type import UndergroundLineType
from VeraGridEngine.Devices.Branches.wire import Wire
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Devices.Substation.bus import Bus


def _expected_phase_matrix(positive_sequence_value: complex,
                           zero_sequence_value: complex) -> np.ndarray:
    diagonal_value = (2.0 * positive_sequence_value + zero_sequence_value) / 3.0
    off_diagonal_value = (zero_sequence_value - positive_sequence_value) / 3.0
    matrix = np.full((3, 3), off_diagonal_value, dtype=np.complex128)
    np.fill_diagonal(matrix, diagonal_value)
    return matrix


def _assert_template_matrix_contract(template, expected_z: np.ndarray, expected_y: np.ndarray) -> None:
    np.testing.assert_allclose(template.z_nabc, expected_z)
    np.testing.assert_allclose(template.y_nabc, expected_y)
    np.testing.assert_array_equal(template.z_phases_nabc, np.array([1, 2, 3]))
    np.testing.assert_array_equal(template.y_phases_nabc, np.array([1, 2, 3]))

    line = Line(template=template, length=10.0)
    angular_frequency = 2.0 * np.pi * 50.0
    expected_tau = line.length * np.sqrt(np.imag(expected_z[0, 0]) * np.imag(expected_y[0, 0])) / angular_frequency
    assert np.isclose(line.get_tau(w=angular_frequency), expected_tau)


def test_sequence_line_type_exposes_physical_phase_matrices() -> None:
    template = SequenceLineType(R=0.1, X=0.4, B=12.0, G=2.0,
                                R0=0.3, X0=1.2, B0=6.0, G0=1.0)
    expected_z = _expected_phase_matrix(0.1 + 0.4j, 0.3 + 1.2j)
    expected_y = _expected_phase_matrix((2.0 + 12.0j) * 1.0e-6,
                                        (1.0 + 6.0j) * 1.0e-6)

    _assert_template_matrix_contract(template, expected_z, expected_y)


def test_sequence_line_type_builds_shunt_matrix_from_capacitance() -> None:
    template = SequenceLineType(CnF=12.0, CnF0=6.0, G=2.0, G0=1.0,
                                use_conductance=True)
    expected_y = _expected_phase_matrix(
        2.0e-6 + 2j * np.pi * 50.0 * 12.0e-9,
        1.0e-6 + 2j * np.pi * 50.0 * 6.0e-9,
    )

    np.testing.assert_allclose(template.y_nabc, expected_y)


def test_underground_line_type_exposes_physical_phase_matrices() -> None:
    template = UndergroundLineType(R=0.1, X=0.4, B=12.0,
                                   R0=0.3, X0=1.2, B0=6.0)
    expected_z = _expected_phase_matrix(0.1 + 0.4j, 0.3 + 1.2j)
    expected_y = _expected_phase_matrix(12.0e-6j, 6.0e-6j)

    _assert_template_matrix_contract(template, expected_z, expected_y)


def test_underground_line_type_builds_shunt_matrix_from_capacitance() -> None:
    template = UndergroundLineType(C=0.012, C0=0.006, freq=60.0)
    expected_y = _expected_phase_matrix(
        2j * np.pi * 60.0 * 0.012e-6,
        2j * np.pi * 60.0 * 0.006e-6,
    )

    np.testing.assert_allclose(template.y_nabc, expected_y)


def test_physical_line_templates_with_zero_sequence_scalars_are_not_reduced() -> None:
    """Keep matrix-defined tower and cable lines in the three-phase network.

    :return: None.
    """
    grid: MultiCircuit = MultiCircuit()
    bus_from: Bus = grid.add_bus(Bus(name="From"))
    bus_to: Bus = grid.add_bus(Bus(name="To"))
    templates: tuple[OverheadLineType, UndergroundLineType] = (
        OverheadLineType(name="Single-phase tower"),
        UndergroundLineType(name="Single-phase cable"),
    )

    template: OverheadLineType | UndergroundLineType
    for template in templates:
        line: Line = Line(
            name=template.name,
            bus_from=bus_from,
            bus_to=bus_to,
            r=0.0,
            x=0.0,
            template=template,
        )
        grid.add_line(line)

    numerical_circuit: Any = compile_numerical_circuit_at(circuit=grid, fill_three_phase=True)

    np.testing.assert_array_equal(
        numerical_circuit.passive_branch_data.reducible,
        np.array([False, False]),
    )


def test_single_phase_overhead_line_without_neutral_applies_physical_matrix() -> None:
    """Apply a phase-C tower that has no neutral conductor.

    :return: None.
    """
    # Model the physical conductor exported by PowerFactory for LineConfig605.
    conductor: Wire = Wire(
        name="LineConfig605 conductor",
        r=0.6959,
        max_current=0.23,
        diameter=8.8392,
    )
    template: OverheadLineType = OverheadLineType(
        name="LineConfig605",
        Vnom=4.16,
        earth_resistivity=100.0,
        frequency=60.0,
    )
    template.add_wire_relationship(
        wire=conductor,
        xpos=0.1524,
        ypos=7.3152,
        phase=3,
    )

    # Applying the template must retain the physical matrix on phase C.
    bus_from: Bus = Bus(name="684", Vnom=4.16)
    bus_to: Bus = Bus(name="611", Vnom=4.16)
    line: Line = Line(
        name="LOHL684-611",
        bus_from=bus_from,
        bus_to=bus_to,
        length=0.0914,
    )
    logger: Logger = Logger()
    line.apply_template(obj=template, Sbase=100.0, freq=60.0, logger=logger)

    assert not logger.has_errors()
    assert line.template is template
    np.testing.assert_array_equal(template.z_phases_nabc, np.array([3]))
    np.testing.assert_array_equal(template.y_phases_nabc, np.array([3]))
    assert line.ys.phC
    assert not line.ys.phN
    assert line.ys.values[3, 3] != 0.0j
    assert line.ysh.values[3, 3] != 0.0j
