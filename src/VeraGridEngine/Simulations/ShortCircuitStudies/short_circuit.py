# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import numpy as np
import warnings
from typing import Tuple
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
from VeraGridEngine.enumerations import FaultType
from VeraGridEngine.basic_structures import CxVec, Vec, Logger


def safe_spsolve(system_matrix: csc_matrix,
                 rhs: CxVec,
                 logger: Logger | None,
                 context: str,
                 fault_impedance: complex) -> CxVec | None:
    """
    Solve a sparse linear system and report singular/non-finite states as errors.

    :param system_matrix: Sparse system matrix.
    :param rhs: Right-hand side vector.
    :param logger: Logger where the numerical error must be reported.
    :param context: Text identifying the short-circuit solve stage.
    :param fault_impedance: Fault impedance used for the case.
    :return: Solution vector, or ``None`` when the solve is unusable.
    """
    solution: CxVec | None

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=MatrixRankWarning)

        try:
            solution = np.asarray(spsolve(system_matrix, rhs), dtype=complex)
        except MatrixRankWarning:
            solution = None
        except RuntimeError:
            solution = None
        except ValueError:
            solution = None

    if solution is None:
        if logger is not None:
            logger.add_error(msg="Short-circuit matrix is singular",
                             device=context,
                             value=fault_impedance,
                             expected_value="Non-singular matrix and valid fault impedance")
        else:
            pass
        return None
    else:
        if np.all(np.isfinite(solution)):
            return solution
        else:
            if logger is not None:
                logger.add_error(msg="Short-circuit solve produced non-finite values",
                                 device=context,
                                 value=fault_impedance,
                                 expected_value="Finite solution")
            else:
                pass
            return None


def short_circuit_3p(bus_idx: int,
                     Ybus: csc_matrix,
                     Vbus: CxVec,
                     Vnom: Vec,
                     Zf: CxVec,
                     baseMVA: float,
                     logger: Logger | None = None) -> Tuple[CxVec, CxVec, CxVec]:
    """
    Executes a 3-phase balanced short circuit study
    :param bus_idx: Index of the bus at which the short circuit is being studied
    :param Ybus: Admittance matrix
    :param Vbus: Voltages of the buses in the steady state
    :param Vnom: Nominal Voltages of the buses (kV)
    :param Zf: Fault impedance array
    :param baseMVA: Circuit base power (100 MVA)
    :param logger: Logger object (optional)
    :return: Voltages after the short circuit (p.u.), Short circuit power in MVA
    Computed as V = Vpre + V increment, where Vpre is the power flow solution and
    V increment is the fault contribution
    The short-circuit power is V_i^2 / Z[i,i], it will tend to infinity if the
    generator is ideal (r1 and x1 approach 0)
    """

    n = len(Vbus)

    if abs(Zf[bus_idx]) == 0.0:
        if logger is not None:
            logger.add_error(msg="Short-circuit fault impedance is zero",
                             device="3-phase fault",
                             value=Zf[bus_idx],
                             expected_value="Non-zero fault impedance")
        else:
            pass
        return Vbus.copy(), np.zeros(n, dtype=complex), np.zeros(n, dtype=complex)
    else:
        pass

    tmp = np.zeros(n)
    tmp[bus_idx] = 1
    Zcol = safe_spsolve(system_matrix=Ybus,
                        rhs=tmp,
                        logger=logger,
                        context="3-phase Thevenin impedance",
                        fault_impedance=Zf[bus_idx])
    if Zcol is None:
        return Vbus.copy(), np.zeros(n, dtype=complex), np.zeros(n, dtype=complex)
    else:
        pass

    Zii = Zcol[bus_idx]
    fault_denominator = Zii + Zf[bus_idx]

    if not np.isfinite(fault_denominator) or abs(fault_denominator) == 0.0:
        if logger is not None:
            logger.add_error(msg="Invalid short-circuit fault impedance",
                             device="3-phase fault",
                             value=Zf[bus_idx],
                             expected_value="Non-zero finite Thevenin plus fault impedance")
        else:
            pass
        return Vbus.copy(), np.zeros(n, dtype=complex), np.zeros(n, dtype=complex)
    else:
        pass

    Ifvec = np.zeros(n, dtype=complex)
    Ifvec[bus_idx] = Vbus[bus_idx] / fault_denominator

    Av_col = safe_spsolve(system_matrix=Ybus,
                          rhs=Ifvec,
                          logger=logger,
                          context="3-phase voltage increment",
                          fault_impedance=Zf[bus_idx])
    if Av_col is None:
        return Vbus.copy(), np.zeros(n, dtype=complex), np.zeros(n, dtype=complex)
    else:
        pass

    Av = - Av_col
    V = Vbus + Av

    idx_buses = range(n)
    Ibase = baseMVA / Vnom[idx_buses]
    if not np.isfinite(Zii) or abs(Zii) == 0.0:
        ICC = np.zeros(n, dtype=complex)
        SCC = np.zeros(n, dtype=complex)
        if logger is not None:
            logger.add_error(msg="Invalid short-circuit Thevenin impedance",
                             device="3-phase fault",
                             value=Zii,
                             expected_value="Non-zero finite impedance")
        else:
            pass
    else:
        ICC = Ibase * Vbus[idx_buses] / Zii
        SCC = baseMVA * Vbus[idx_buses] * Vbus[idx_buses] / Zii

    return V, SCC, ICC


def short_circuit_unbalance(bus_idx: int,
                            Y0: csc_matrix,
                            Y1: csc_matrix,
                            Y2: csc_matrix,
                            Vbus: CxVec,
                            Vnom: Vec,
                            Zf: CxVec,
                            fault_type: FaultType,
                            baseMVA: float,
                            logger: Logger | None = None) -> Tuple[CxVec, CxVec, CxVec, CxVec, CxVec]:
    """
    Executes the unbalanced short circuits (LG, LL, LLG types)

    :param bus_idx: bus where the fault is caused
    :param Y0: Admittance matrix for the zero sequence
    :param Y1: Admittance matrix for the positive sequence
    :param Y2: Admittance matrix for the negative sequence
    :param Vbus: pre-fault voltage (positive sequence)
    :param Vnom: nominal voltage (kV)
    :param Zf: fault impedance
    :param fault_type: type of unbalanced fault
    :param baseMVA: base MVA (100 MVA)
    :param logger: Logger (optional)
    :return: V0, V1, V2 for all buses
    """

    n = len(Vbus)

    # solve the fault
    Vpr = Vbus[bus_idx]

    tmp = np.zeros(n)
    tmp[bus_idx] = 1
    Zflt = Zf[bus_idx]
    fallback_v0 = np.zeros(n, dtype=complex)
    fallback_v1 = Vbus.copy()
    fallback_v2 = np.zeros(n, dtype=complex)
    fallback_power = np.zeros_like(Vbus)

    if abs(Zflt) == 0.0:
        if logger is not None:
            logger.add_error(msg="Short-circuit fault impedance is zero",
                             device=fault_type.value,
                             value=Zflt,
                             expected_value="Non-zero fault impedance")
        else:
            pass
        return fallback_v0, fallback_v1, fallback_v2, fallback_power, fallback_power
    else:
        pass

    Zth0_col = safe_spsolve(system_matrix=Y0,
                            rhs=tmp,
                            logger=logger,
                            context="zero-sequence Thevenin impedance",
                            fault_impedance=Zflt)
    Zth1_col = safe_spsolve(system_matrix=Y1,
                            rhs=tmp,
                            logger=logger,
                            context="positive-sequence Thevenin impedance",
                            fault_impedance=Zflt)
    Zth2_col = safe_spsolve(system_matrix=Y2,
                            rhs=tmp,
                            logger=logger,
                            context="negative-sequence Thevenin impedance",
                            fault_impedance=Zflt)

    if Zth0_col is None or Zth1_col is None or Zth2_col is None:
        return fallback_v0, fallback_v1, fallback_v2, fallback_power, fallback_power
    else:
        pass

    Zth0 = Zth0_col[bus_idx]
    Zth1 = Zth1_col[bus_idx]
    Zth2 = Zth2_col[bus_idx]

    if fault_type == FaultType.LG:
        denominator = Zth0 + Zth1 + Zth2 + 3 * Zflt
        if not np.isfinite(denominator) or abs(denominator) == 0.0:
            if logger is not None:
                logger.add_error(msg="Invalid short-circuit fault impedance",
                                 device=fault_type.value,
                                 value=Zflt,
                                 expected_value="Non-zero finite sequence fault denominator")
            else:
                pass
            return fallback_v0, fallback_v1, fallback_v2, fallback_power, fallback_power
        else:
            pass
        I0 = Vpr / denominator
        I1 = I0
        I2 = I0
    elif fault_type == FaultType.LL:  # between phases b and c
        I0 = 0
        denominator = Zth1 + Zth2 + Zflt
        if not np.isfinite(denominator) or abs(denominator) == 0.0:
            if logger is not None:
                logger.add_error(msg="Invalid short-circuit fault impedance",
                                 device=fault_type.value,
                                 value=Zflt,
                                 expected_value="Non-zero finite sequence fault denominator")
            else:
                pass
            return fallback_v0, fallback_v1, fallback_v2, fallback_power, fallback_power
        else:
            pass
        I1 = Vpr / denominator
        I2 = - I1
    elif fault_type == FaultType.LLG:  # between phases b and c
        shunt_denominator = Zth2 + Zth0 + 3 * Zflt
        if not np.isfinite(shunt_denominator) or abs(shunt_denominator) == 0.0:
            if logger is not None:
                logger.add_error(msg="Invalid short-circuit fault impedance",
                                 device=fault_type.value,
                                 value=Zflt,
                                 expected_value="Non-zero finite sequence fault denominator")
            else:
                pass
            return fallback_v0, fallback_v1, fallback_v2, fallback_power, fallback_power
        else:
            pass

        denominator = Zth1 + Zth2 * (Zth0 + 3 * Zflt) / shunt_denominator
        if not np.isfinite(denominator) or abs(denominator) == 0.0:
            if logger is not None:
                logger.add_error(msg="Invalid short-circuit fault impedance",
                                 device=fault_type.value,
                                 value=Zflt,
                                 expected_value="Non-zero finite sequence fault denominator")
            else:
                pass
            return fallback_v0, fallback_v1, fallback_v2, fallback_power, fallback_power
        else:
            pass

        I1 = Vpr / denominator
        I0 = -I1 * Zth2 / shunt_denominator
        I2 = -I1 * (Zth0 + 3 * Zflt) / shunt_denominator
    else:
        raise Exception('Unknown unbalanced fault type')

    # obtain the post fault voltages
    I0_vec = np.zeros(n, dtype=complex)
    I1_vec = np.zeros(n, dtype=complex)
    I2_vec = np.zeros(n, dtype=complex)

    I0_vec[bus_idx] = I0
    I1_vec[bus_idx] = I1
    I2_vec[bus_idx] = I2

    V0_col = safe_spsolve(system_matrix=Y0,
                          rhs=I0_vec,
                          logger=logger,
                          context="zero-sequence post-fault voltage",
                          fault_impedance=Zflt)
    V1_col = safe_spsolve(system_matrix=Y1,
                          rhs=I1_vec,
                          logger=logger,
                          context="positive-sequence post-fault voltage",
                          fault_impedance=Zflt)
    V2_col = safe_spsolve(system_matrix=Y2,
                          rhs=I2_vec,
                          logger=logger,
                          context="negative-sequence post-fault voltage",
                          fault_impedance=Zflt)

    if V0_col is None or V1_col is None or V2_col is None:
        return fallback_v0, fallback_v1, fallback_v2, fallback_power, fallback_power
    else:
        pass

    V0_fin = - V0_col
    V1_fin = Vbus - V1_col
    V2_fin = - V2_col

    SCC = np.zeros_like(Vbus)
    ICC = np.zeros_like(Vbus)
    Ibase = baseMVA / Vnom[bus_idx]
    if not np.isfinite(Zth1) or abs(Zth1) == 0.0:
        if logger is not None:
            logger.add_error(msg="Invalid short-circuit Thevenin impedance",
                             device=fault_type.value,
                             value=Zth1,
                             expected_value="Non-zero finite impedance")
        else:
            pass
    else:
        ICC[bus_idx] = Ibase * V1_fin[bus_idx] / Zth1
        SCC[bus_idx] = baseMVA * V1_fin[bus_idx] * V1_fin[bus_idx] / Zth1

    return V0_fin, V1_fin, V2_fin, SCC, ICC


# def short_circuit_phases():
#
