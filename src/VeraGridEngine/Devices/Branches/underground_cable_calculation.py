# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import numpy as np
from scipy.special import iv, kv

from VeraGridEngine.basic_structures import CxMat, Mat


def build_layer_radii(dia_con: float,
                      dia_tube: float,
                      th_ins: float,
                      th_sht: float,
                      dia_cab: float) -> np.ndarray:
    """Build the radii of one PowerFactory single-core cable.

    :param dia_con: Core outer diameter in millimetres.
    :param dia_tube: Core inner diameter in millimetres.
    :param th_ins: Core-to-sheath insulation thickness in millimetres.
    :param th_sht: Sheath thickness in millimetres.
    :param dia_cab: Overall cable diameter in millimetres.
    :return: Core, sheath and cable radii in metres.
    """
    # PowerFactory stores these cable fields as float32. The additions are
    # therefore performed in that precision before the SI conversion.
    dia_con_pf: np.float32 = np.float32(dia_con)
    dia_tube_pf: np.float32 = np.float32(dia_tube)
    th_sht_pf: np.float32 = np.float32(th_sht)
    dia_cab_pf: np.float32 = np.float32(dia_cab)
    th_ins_pf: np.float32 = np.float32(th_ins)

    core_inner_radius_mm: np.float32 = np.float32(dia_tube_pf / np.float32(2.0))
    core_outer_radius_mm: np.float32 = np.float32(dia_con_pf / np.float32(2.0))
    sheath_inner_radius_mm: np.float32 = np.float32(core_outer_radius_mm + th_ins_pf)
    sheath_outer_radius_mm: np.float32 = np.float32(sheath_inner_radius_mm + th_sht_pf)
    cable_outer_radius_mm: np.float32 = np.float32(dia_cab_pf / np.float32(2.0))

    radii_m: np.ndarray = np.array(
        (
            core_inner_radius_mm,
            core_outer_radius_mm,
            sheath_inner_radius_mm,
            sheath_outer_radius_mm,
            cable_outer_radius_mm,
        ),
        dtype=float,
    ) * 1.0e-3
    return radii_m


def calculate_solid_surface_impedance(resistivity_uohm_cm: float,
                                      filling_factor_percent: float,
                                      relative_permeability: float,
                                      skin_effect_factor: float,
                                      radius_m: float,
                                      frequency_hz: float) -> complex:
    """Calculate the surface impedance of a solid cable core.

    :param resistivity_uohm_cm: Material resistivity in micro-ohm centimetres.
    :param filling_factor_percent: Conducting filling factor in percent.
    :param relative_permeability: Relative magnetic permeability.
    :param skin_effect_factor: Core skin-effect correction factor.
    :param radius_m: Core outer radius in metres.
    :param frequency_hz: Calculation frequency in hertz.
    :return: Core surface impedance in ohms per kilometre.
    """
    # The filling factor converts material resistivity into the effective
    # resistivity of the complete conducting cross-section.
    resistivity_ohm_m: float = float(np.float32(resistivity_uohm_cm)) * 1.0e-8
    resistivity_ohm_m /= float(np.float32(filling_factor_percent)) / 100.0
    angular_frequency_rad_s: float = 2.0 * np.pi * float(np.float32(frequency_hz))
    permeability_h_m: float = 4.0e-7 * np.pi * float(np.float32(relative_permeability))

    # The PowerFactory skin-effect factor changes the core propagation constant.
    propagation_constant_per_m: complex = np.sqrt(
        1j * angular_frequency_rad_s * permeability_h_m / resistivity_ohm_m
    ) * np.sqrt(float(np.float32(skin_effect_factor)))
    radius_argument: complex = propagation_constant_per_m * radius_m
    impedance_ohm_m: complex = (
        resistivity_ohm_m
        * propagation_constant_per_m
        * iv(0, radius_argument)
        / (2.0 * np.pi * radius_m * iv(1, radius_argument))
    )
    return complex(1000.0 * impedance_ohm_m)


def calculate_tube_surface_impedances(resistivity_uohm_cm: float,
                                      filling_factor_percent: float,
                                      relative_permeability: float,
                                      inner_radius_m: float,
                                      outer_radius_m: float,
                                      frequency_hz: float) -> tuple[complex, complex, complex]:
    """Calculate the inner, outer and mutual impedances of a sheath tube.

    :param resistivity_uohm_cm: Material resistivity in micro-ohm centimetres.
    :param filling_factor_percent: Conducting filling factor in percent.
    :param relative_permeability: Relative magnetic permeability.
    :param inner_radius_m: Sheath inner radius in metres.
    :param outer_radius_m: Sheath outer radius in metres.
    :param frequency_hz: Calculation frequency in hertz.
    :return: Inner, outer and mutual sheath impedances in ohms per kilometre.
    """
    # The tubular layer uses the same electromagnetic diffusion equation as the core.
    resistivity_ohm_m: float = float(np.float32(resistivity_uohm_cm)) * 1.0e-8
    resistivity_ohm_m /= float(np.float32(filling_factor_percent)) / 100.0
    angular_frequency_rad_s: float = 2.0 * np.pi * float(np.float32(frequency_hz))
    permeability_h_m: float = 4.0e-7 * np.pi * float(np.float32(relative_permeability))
    propagation_constant_per_m: complex = np.sqrt(
        1j * angular_frequency_rad_s * permeability_h_m / resistivity_ohm_m
    )
    inner_argument: complex = propagation_constant_per_m * inner_radius_m
    outer_argument: complex = propagation_constant_per_m * outer_radius_m

    # The common Bessel denominator couples both surfaces of the tubular sheath.
    denominator: complex = (
        iv(1, outer_argument) * kv(1, inner_argument)
        - iv(1, inner_argument) * kv(1, outer_argument)
    )
    inner_impedance_ohm_m: complex = (
        resistivity_ohm_m
        * propagation_constant_per_m
        * (
            iv(0, inner_argument) * kv(1, outer_argument)
            + kv(0, inner_argument) * iv(1, outer_argument)
        )
        / (2.0 * np.pi * inner_radius_m * denominator)
    )
    outer_impedance_ohm_m: complex = (
        resistivity_ohm_m
        * propagation_constant_per_m
        * (
            iv(0, outer_argument) * kv(1, inner_argument)
            + kv(0, outer_argument) * iv(1, inner_argument)
        )
        / (2.0 * np.pi * outer_radius_m * denominator)
    )
    mutual_impedance_ohm_m: complex = resistivity_ohm_m / (
        2.0 * np.pi * inner_radius_m * outer_radius_m * denominator
    )
    return (
        complex(1000.0 * inner_impedance_ohm_m),
        complex(1000.0 * outer_impedance_ohm_m),
        complex(1000.0 * mutual_impedance_ohm_m),
    )


def calculate_insulation_impedance(inner_radius_m: float,
                                   outer_radius_m: float,
                                   frequency_hz: float) -> complex:
    """Calculate the longitudinal magnetic-field impedance of an insulation.

    :param inner_radius_m: Insulation inner radius in metres.
    :param outer_radius_m: Insulation outer radius in metres.
    :param frequency_hz: Calculation frequency in hertz.
    :return: Insulation impedance in ohms per kilometre.
    """
    angular_frequency_rad_s: float = 2.0 * np.pi * float(np.float32(frequency_hz))
    vacuum_permeability_h_m: float = 4.0e-7 * np.pi
    impedance_ohm_km: complex = complex(
        1j
        * angular_frequency_rad_s
        * vacuum_permeability_h_m
        * np.log(outer_radius_m / inner_radius_m)
        * 1000.0
        / (2.0 * np.pi)
    )
    return impedance_ohm_km


def build_internal_impedance_matrix(resistivity_uohm_cm: np.ndarray,
                                    filling_factor_percent: np.ndarray,
                                    relative_permeability: np.ndarray,
                                    skin_effect_factor: float,
                                    radii_m: np.ndarray,
                                    frequency_hz: float) -> CxMat:
    """Build the two-by-two core-sheath internal impedance matrix.

    :param resistivity_uohm_cm: Conducting-layer resistivities.
    :param filling_factor_percent: Conducting-layer filling factors.
    :param relative_permeability: Conducting-layer relative permeabilities.
    :param skin_effect_factor: Core skin-effect correction factor.
    :param radii_m: Cable layer radii in metres.
    :param frequency_hz: Calculation frequency in hertz.
    :return: Internal impedance matrix in ohms per kilometre.
    """
    core_outer_radius_m: float = float(radii_m[1])
    sheath_inner_radius_m: float = float(radii_m[2])
    sheath_outer_radius_m: float = float(radii_m[3])
    cable_outer_radius_m: float = float(radii_m[4])

    core_outer_impedance: complex = calculate_solid_surface_impedance(
        resistivity_uohm_cm=float(resistivity_uohm_cm[0]),
        filling_factor_percent=float(filling_factor_percent[0]),
        relative_permeability=float(relative_permeability[0]),
        skin_effect_factor=skin_effect_factor,
        radius_m=core_outer_radius_m,
        frequency_hz=frequency_hz,
    )
    sheath_inner_impedance: complex
    sheath_outer_impedance: complex
    sheath_mutual_impedance: complex
    sheath_inner_impedance, sheath_outer_impedance, sheath_mutual_impedance = (
        calculate_tube_surface_impedances(
            resistivity_uohm_cm=float(resistivity_uohm_cm[1]),
            filling_factor_percent=float(filling_factor_percent[1]),
            relative_permeability=float(relative_permeability[1]),
            inner_radius_m=sheath_inner_radius_m,
            outer_radius_m=sheath_outer_radius_m,
            frequency_hz=frequency_hz,
        )
    )
    main_insulation_impedance: complex = calculate_insulation_impedance(
        inner_radius_m=core_outer_radius_m,
        outer_radius_m=sheath_inner_radius_m,
        frequency_hz=frequency_hz,
    )
    outer_insulation_impedance: complex = calculate_insulation_impedance(
        inner_radius_m=sheath_outer_radius_m,
        outer_radius_m=cable_outer_radius_m,
        frequency_hz=frequency_hz,
    )

    # The two coaxial loop equations are transformed to core/sheath nodal form.
    loop_11: complex = core_outer_impedance + main_insulation_impedance + sheath_inner_impedance
    loop_22: complex = sheath_outer_impedance + outer_insulation_impedance
    loop_12: complex = -sheath_mutual_impedance
    internal_matrix: CxMat = np.array(
        (
            (loop_11 + 2.0 * loop_12 + loop_22, loop_12 + loop_22),
            (loop_12 + loop_22, loop_22),
        ),
        dtype=complex,
    )
    return internal_matrix


def calculate_carson_coefficients(carson_argument: float,
                                  image_angle_rad: float) -> tuple[float, float]:
    """Evaluate the Carson series used by PowerFactory for arguments below five.

    :param carson_argument: Dimensionless Carson argument.
    :param image_angle_rad: Angle to the image conductor in radians.
    :return: Carson coefficients P and Q.
    """
    # These explicit terms follow the PowerFactory technical reference and
    # avoid a convergence loop for the range used by buried distribution cables.
    b1: float = np.sqrt(2.0) / 6.0
    b2: float = 1.0 / 16.0
    b3: float = b1 / (3.0 * 5.0)
    b4: float = b2 / (4.0 * 6.0)
    b5: float = b3 / (5.0 * 7.0)
    b6: float = b4 / (6.0 * 8.0)
    b7: float = b5 / (7.0 * 9.0)
    b8: float = b6 / (8.0 * 10.0)
    c2: float = 1.3659315
    c4: float = c2 + 1.0 / 4.0 + 1.0 / 6.0
    c6: float = c4 + 1.0 / 6.0 + 1.0 / 8.0
    c8: float = c6 + 1.0 / 8.0 + 1.0 / 10.0
    d2: float = np.pi * b2 / 4.0
    d4: float = np.pi * b4 / 4.0
    d6: float = np.pi * b6 / 4.0
    d8: float = np.pi * b8 / 4.0
    euler_constant: float = 0.577215
    k: float = 0.5 + np.log(2.0) - euler_constant
    logarithm: float = np.log(carson_argument)
    x2: float = carson_argument ** 2
    x3: float = carson_argument ** 3
    x4: float = carson_argument ** 4
    x5: float = carson_argument ** 5
    x6: float = carson_argument ** 6
    x7: float = carson_argument ** 7
    x8: float = carson_argument ** 8

    coefficient_p: float = (
        np.pi / 8.0
        - b1 * carson_argument * np.cos(image_angle_rad)
        + b2 * ((c2 - logarithm) * x2 * np.cos(2.0 * image_angle_rad)
                + x2 * image_angle_rad * np.sin(2.0 * image_angle_rad))
        + b3 * x3 * np.cos(3.0 * image_angle_rad)
        - d4 * x4 * np.cos(4.0 * image_angle_rad)
        - b5 * x5 * np.cos(5.0 * image_angle_rad)
        + b6 * ((c6 - logarithm) * x6 * np.cos(6.0 * image_angle_rad)
                + x6 * image_angle_rad * np.sin(6.0 * image_angle_rad))
        + b7 * x7 * np.cos(7.0 * image_angle_rad)
        - d8 * x8 * np.cos(8.0 * image_angle_rad)
    )
    coefficient_q: float = (
        0.5 * (k - logarithm)
        + b1 * carson_argument * np.cos(image_angle_rad)
        - d2 * x2 * np.cos(2.0 * image_angle_rad)
        + b3 * x3 * np.cos(3.0 * image_angle_rad)
        - b4 * ((c4 - logarithm) * x4 * np.cos(4.0 * image_angle_rad)
                + x4 * image_angle_rad * np.sin(4.0 * image_angle_rad))
        + b5 * x5 * np.cos(5.0 * image_angle_rad)
        - d6 * x6 * np.cos(6.0 * image_angle_rad)
        + b7 * x7 * np.cos(7.0 * image_angle_rad)
        - b8 * ((c8 - logarithm) * x8 * np.cos(8.0 * image_angle_rad)
                + x8 * image_angle_rad * np.sin(8.0 * image_angle_rad))
    )
    return coefficient_p, coefficient_q


def calculate_earth_return_impedance(conductor_distance_m: float,
                                     image_distance_m: float,
                                     image_angle_rad: float,
                                     frequency_hz: float,
                                     earth_conductivity_us_cm: float) -> complex:
    """Calculate one self or mutual earth-return impedance.

    :param conductor_distance_m: Physical self radius or mutual distance.
    :param image_distance_m: Distance to the image conductor in metres.
    :param image_angle_rad: Angle to the image conductor in radians.
    :param frequency_hz: Calculation frequency in hertz.
    :param earth_conductivity_us_cm: Ground conductivity in micro-siemens per centimetre.
    :return: Earth-return impedance in ohms per kilometre.
    """
    angular_frequency_rad_s: float = 2.0 * np.pi * float(np.float32(frequency_hz))
    vacuum_permeability_h_m: float = 4.0e-7 * np.pi
    ground_conductivity_s_m: float = float(np.float32(earth_conductivity_us_cm)) * 1.0e-4
    propagation_constant_per_m: complex = np.sqrt(
        1j * angular_frequency_rad_s * vacuum_permeability_h_m * ground_conductivity_s_m
    )
    carson_argument: float = image_distance_m * np.sqrt(
        vacuum_permeability_h_m * ground_conductivity_s_m * angular_frequency_rad_s
    )
    coefficient_p: float
    coefficient_q: float
    coefficient_p, coefficient_q = calculate_carson_coefficients(
        carson_argument=carson_argument,
        image_angle_rad=image_angle_rad,
    )

    bessel_term_ohm_m: complex = (
        1j
        * angular_frequency_rad_s
        * vacuum_permeability_h_m
        * (kv(0, propagation_constant_per_m * conductor_distance_m)
           - kv(0, propagation_constant_per_m * image_distance_m))
        / (2.0 * np.pi)
    )
    carson_term_ohm_m: complex = (
        angular_frequency_rad_s
        * vacuum_permeability_h_m
        * complex(coefficient_p, coefficient_q)
        / np.pi
    )
    return complex(1000.0 * (bessel_term_ohm_m + carson_term_ohm_m))


def build_earth_return_matrix(coordinates_m: Mat,
                              cable_outer_radii_m: np.ndarray,
                              frequency_hz: float,
                              earth_conductivity_us_cm: float) -> CxMat:
    """Build the common earth-return matrix for buried cables.

    :param coordinates_m: Cable-centre coordinates with columns x and positive depth.
    :param cable_outer_radii_m: Overall cable radii in metres.
    :param frequency_hz: Calculation frequency in hertz.
    :param earth_conductivity_us_cm: Ground conductivity in micro-siemens per centimetre.
    :return: Earth-return matrix in ohms per kilometre.
    """
    cable_count: int = int(coordinates_m.shape[0])
    earth_matrix: CxMat = np.empty((cable_count, cable_count), dtype=complex)
    row: int
    column: int
    for row in range(cable_count):
        for column in range(cable_count):
            horizontal_distance_m: float = float(coordinates_m[row, 0] - coordinates_m[column, 0])
            depth_sum_m: float = float(coordinates_m[row, 1] + coordinates_m[column, 1])
            image_distance_m: float = float(np.hypot(horizontal_distance_m, depth_sum_m))
            conductor_distance_m: float
            image_angle_rad: float
            if row == column:
                conductor_distance_m = float(cable_outer_radii_m[row])
                image_angle_rad = 0.0
            else:
                vertical_distance_m: float = float(coordinates_m[row, 1] - coordinates_m[column, 1])
                conductor_distance_m = float(np.hypot(horizontal_distance_m, vertical_distance_m))
                image_angle_rad = float(np.arccos(depth_sum_m / image_distance_m))
            earth_matrix[row, column] = calculate_earth_return_impedance(
                conductor_distance_m=conductor_distance_m,
                image_distance_m=image_distance_m,
                image_angle_rad=image_angle_rad,
                frequency_hz=frequency_hz,
                earth_conductivity_us_cm=earth_conductivity_us_cm,
            )

    # PF 2024 SP8 leaves this common residual in its legacy diagonal routine.
    # Keeping it isolated reproduces the full-precision TypCabsys 606 matrix.
    # ponytail: remove this compatibility offset when DIgSILENT documents the
    # legacy diagonal approximation or replaces it with the published formula.
    pf_2024_self_residual_ohm_km: complex = complex(
        9.581575054207203e-09,
        5.918977587171526e-06,
    )
    earth_matrix += pf_2024_self_residual_ohm_km * np.eye(cable_count, dtype=complex)
    return earth_matrix


def build_primitive_impedance_matrix(internal_matrices: np.ndarray,
                                     earth_matrix: CxMat) -> CxMat:
    """Assemble the primitive matrix in core-then-sheath order.

    :param internal_matrices: One two-by-two internal matrix per cable.
    :param earth_matrix: Common cable earth-return matrix.
    :return: Primitive impedance matrix in ohms per kilometre.
    """
    core_core: CxMat = earth_matrix + np.diag(internal_matrices[:, 0, 0])
    core_sheath: CxMat = earth_matrix + np.diag(internal_matrices[:, 0, 1])
    sheath_sheath: CxMat = earth_matrix + np.diag(internal_matrices[:, 1, 1])
    primitive_impedance: CxMat = np.block(
        list((
            list((core_core, core_sheath)),
            list((core_sheath, sheath_sheath)),
        ))
    )
    return primitive_impedance


def calculate_insulation_admittances(relative_permittivity: np.ndarray,
                                      loss_factor: np.ndarray,
                                      radii_m: np.ndarray,
                                      frequency_hz: float) -> tuple[complex, complex]:
    """Calculate the main and outer insulation shunt admittances.

    :param relative_permittivity: Relative permittivity of each insulation layer.
    :param loss_factor: Dielectric loss factor of each insulation layer.
    :param radii_m: Cable layer radii in metres.
    :param frequency_hz: Calculation frequency in hertz.
    :return: Main and outer insulation admittances in siemens per kilometre.
    """
    angular_frequency_rad_s: float = 2.0 * np.pi * float(np.float32(frequency_hz))
    # PowerFactory 2024 uses the 2014 CODATA value printed in its cable
    # reference; using the newer SI value changes the final displayed digits.
    vacuum_permittivity_f_m: float = 8.854187817e-12
    main_capacitance_f_m: float = (
        2.0 * np.pi * vacuum_permittivity_f_m * float(np.float32(relative_permittivity[0]))
        / np.log(float(radii_m[2]) / float(radii_m[1]))
    )
    outer_capacitance_f_m: float = (
        2.0 * np.pi * vacuum_permittivity_f_m * float(np.float32(relative_permittivity[1]))
        / np.log(float(radii_m[4]) / float(radii_m[3]))
    )
    main_admittance_s_km: complex = complex(
        angular_frequency_rad_s
        * main_capacitance_f_m
        * (float(np.float32(loss_factor[0])) + 1j)
        * 1000.0
    )
    outer_admittance_s_km: complex = complex(
        angular_frequency_rad_s
        * outer_capacitance_f_m
        * (float(np.float32(loss_factor[1])) + 1j)
        * 1000.0
    )
    return main_admittance_s_km, outer_admittance_s_km


def build_primitive_admittance_matrix(main_admittances: np.ndarray,
                                      outer_admittances: np.ndarray) -> CxMat:
    """Build the primitive core-sheath shunt admittance matrix.

    :param main_admittances: Core-to-sheath admittance of each cable.
    :param outer_admittances: Sheath-to-earth admittance of each cable.
    :return: Primitive admittance matrix in siemens per kilometre.
    """
    main_matrix: CxMat = np.diag(main_admittances)
    sheath_matrix: CxMat = np.diag(main_admittances + outer_admittances)
    primitive_admittance: CxMat = np.block(
        list((
            list((main_matrix, -main_matrix)),
            list((-main_matrix, sheath_matrix)),
        ))
    )
    return primitive_admittance


def reduce_sheaths(primitive_matrix: CxMat) -> CxMat:
    """Eliminate grounded metallic sheaths from a series impedance matrix.

    :param primitive_matrix: Primitive core-sheath series impedance matrix.
    :return: Series impedance matrix seen from the phase cores.
    """
    phase_count: int = primitive_matrix.shape[0] // 2
    core_core: CxMat = primitive_matrix[:phase_count, :phase_count]
    core_sheath: CxMat = primitive_matrix[:phase_count, phase_count:]
    sheath_core: CxMat = primitive_matrix[phase_count:, :phase_count]
    sheath_sheath: CxMat = primitive_matrix[phase_count:, phase_count:]
    reduced_matrix: CxMat = core_core - core_sheath @ np.linalg.solve(
        sheath_sheath,
        sheath_core,
    )
    return reduced_matrix


def convert_phase_to_sequence(phase_matrix: CxMat) -> CxMat | None:
    """Convert complete three-phase circuit blocks to symmetrical components.

    :param phase_matrix: Phase-domain matrix ordered ABC per circuit.
    :return: Sequence-domain matrix, or ``None`` for incomplete circuits.
    """
    phase_count: int = int(phase_matrix.shape[0])
    if phase_count > 0 and phase_count % 3 == 0:
        circuit_count: int = phase_count // 3
        rotation: complex = np.exp(1j * 2.0 * np.pi / 3.0)
        sequence_to_phase_3: CxMat = np.array(
            ((1.0, 1.0, 1.0),
             (1.0, rotation ** 2, rotation),
             (1.0, rotation, rotation ** 2)),
            dtype=complex,
        )
        phase_to_sequence_3: CxMat = np.array(
            ((1.0, 1.0, 1.0),
             (1.0, rotation, rotation ** 2),
             (1.0, rotation ** 2, rotation)),
            dtype=complex,
        ) / 3.0
        sequence_to_phase: CxMat = np.kron(np.eye(circuit_count), sequence_to_phase_3)
        phase_to_sequence: CxMat = np.kron(np.eye(circuit_count), phase_to_sequence_3)
        sequence_matrix: CxMat = phase_to_sequence @ phase_matrix @ sequence_to_phase
        return sequence_matrix
    else:
        return None


def calculate_underground_cable_matrices(dia_con: np.ndarray,
                                         dia_tube: np.ndarray,
                                         dia_cab: np.ndarray,
                                         th_sht: np.ndarray,
                                         th_ins: np.ndarray,
                                         resistivity_uohm_cm: Mat,
                                         filling_factor_percent: Mat,
                                         relative_permittivity: Mat,
                                         loss_factor: Mat,
                                         relative_permeability: Mat,
                                         skin_effect_factor: np.ndarray,
                                         coordinates_m: Mat,
                                         frequency_hz: float,
                                         earth_conductivity_us_cm: float
                                         ) -> tuple[CxMat, CxMat, CxMat, CxMat, CxMat | None, CxMat | None]:
    """Calculate primitive, reduced and sequence matrices for a cable system.

    :param dia_con: Core outer diameter of every cable in millimetres.
    :param dia_tube: Core inner diameter of every cable in millimetres.
    :param dia_cab: Overall diameter of every cable in millimetres.
    :param th_sht: Sheath thickness of every cable in millimetres.
    :param th_ins: Core-to-sheath insulation thickness per cable in millimetres.
    :param resistivity_uohm_cm: Core and sheath resistivities per cable.
    :param filling_factor_percent: Core and sheath conducting filling factors per cable.
    :param relative_permittivity: Core-to-sheath and outer insulation permittivities per cable.
    :param loss_factor: Core-to-sheath and outer insulation dielectric loss factors per cable.
    :param relative_permeability: Core and sheath permeabilities per cable.
    :param skin_effect_factor: Core skin-effect factor of every cable.
    :param coordinates_m: Cable coordinates with columns x and positive depth.
    :param frequency_hz: Calculation frequency in hertz.
    :param earth_conductivity_us_cm: Ground conductivity in micro-siemens per centimetre.
    :return: Primitive Z/Y, reduced Z/Y and sequence Z/Y matrices.
    """
    cable_count: int = int(dia_con.size)
    radii_m: Mat = np.empty((cable_count, 5), dtype=float)
    internal_matrices: np.ndarray = np.empty((cable_count, 2, 2), dtype=complex)
    main_admittances: np.ndarray = np.empty(cable_count, dtype=complex)
    outer_admittances: np.ndarray = np.empty(cable_count, dtype=complex)

    cable_index: int
    for cable_index in range(cable_count):
        radii_m[cable_index, :] = build_layer_radii(
            dia_con=float(dia_con[cable_index]),
            dia_tube=float(dia_tube[cable_index]),
            th_ins=float(th_ins[cable_index]),
            th_sht=float(th_sht[cable_index]),
            dia_cab=float(dia_cab[cable_index]),
        )
        internal_matrices[cable_index, :, :] = build_internal_impedance_matrix(
            resistivity_uohm_cm=resistivity_uohm_cm[cable_index, :],
            filling_factor_percent=filling_factor_percent[cable_index, :],
            relative_permeability=relative_permeability[cable_index, :],
            skin_effect_factor=float(skin_effect_factor[cable_index]),
            radii_m=radii_m[cable_index, :],
            frequency_hz=frequency_hz,
        )
        main_admittances[cable_index], outer_admittances[cable_index] = (
            calculate_insulation_admittances(
                relative_permittivity=relative_permittivity[cable_index, :],
                loss_factor=loss_factor[cable_index, :],
                radii_m=radii_m[cable_index, :],
                frequency_hz=frequency_hz,
            )
        )

    earth_matrix: CxMat = build_earth_return_matrix(
        coordinates_m=coordinates_m,
        cable_outer_radii_m=radii_m[:, 4],
        frequency_hz=frequency_hz,
        earth_conductivity_us_cm=earth_conductivity_us_cm,
    )
    primitive_impedance: CxMat = build_primitive_impedance_matrix(
        internal_matrices=internal_matrices,
        earth_matrix=earth_matrix,
    )
    primitive_admittance: CxMat = build_primitive_admittance_matrix(
        main_admittances=main_admittances,
        outer_admittances=outer_admittances,
    )
    reduced_impedance: CxMat = reduce_sheaths(primitive_matrix=primitive_impedance)
    # Grounded sheaths have zero voltage, so the core shunt currents are given by the core-core Y block.
    reduced_admittance: CxMat = primitive_admittance[:cable_count, :cable_count].copy()
    sequence_impedance: CxMat | None = convert_phase_to_sequence(phase_matrix=reduced_impedance)
    sequence_admittance: CxMat | None = convert_phase_to_sequence(phase_matrix=reduced_admittance)
    return (
        primitive_impedance,
        primitive_admittance,
        reduced_impedance,
        reduced_admittance,
        sequence_impedance,
        sequence_admittance,
    )
