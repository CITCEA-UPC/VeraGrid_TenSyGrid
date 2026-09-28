# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path

import numpy as np

import VeraGridEngine.api as gce
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_results_3ph import PowerFlowResults3Ph


def test_3ph_underground_cables() -> None:
    """Compare every physical bus-phase voltage magnitude with PowerFactory 2024.

    The reference was captured from UnbalancedNetworkModel_Cables / Study Case
    alongside this V7.00 DGS, using DGS Export Definitions 8 - VeraGrid and
    "All elements of the active project". Values are PF m:u:A/B/C in per unit;
    NaN marks phases absent in PowerFactory, not failed VeraGrid calculations.
    PowerFactory is not required to run this regression test.

    :return: None.
    """
    # Keep the complete export in the repository, including constant-current
    # and constant-impedance loads, so the test does not depend on local files.
    dgs_path: Path = (
        Path(__file__).resolve().parents[1]
        / 'data' / 'grids' / 'DGS' / 'UnbalancedNetworkModel_Cables_all_elements.dgs'
    )
    grid: gce.MultiCircuit = gce.open_file(filename=str(dgs_path))
    options: gce.PowerFlowOptions = gce.PowerFlowOptions(tolerance=1.0e-10, max_iter=1000)
    results: PowerFlowResults3Ph = gce.power_flow3ph(grid=grid, options=options)
    assert results.converged, 'The underground-cable three-phase power flow did not converge'

    # Store PF values independently of VeraGrid results, retaining full precision.
    reference_bus_names: tuple[str, ...] = (
        '611',
        '632',
        '633',
        '634',
        '645',
        '646',
        '650',
        '652',
        '671',
        '675',
        '680',
        '684',
        '692',
        'RG60',
        'Terminal',
        'Terminal(1)',
        'Terminal(2)',
        'Terminal(3)',
        'Terminal(4)',
    )
    reference_voltage: np.ndarray = np.array((
        (np.nan, np.nan, 0.8495612995414918),  # 611
        (0.9390388717054378, 0.9891649359773482, 0.9146344367503207),  # 632
        (0.9356654953174434, 0.9874487992620161, 0.9115345226049181),  # 633
        (0.9095889005732547, 0.9677730034415565, 0.8906173465334425),  # 634
        (np.nan, 0.9795749187742443, 0.9139985000074125),  # 645
        (np.nan, 0.9779396702064247, 0.9120970121382167),  # 646
        (0.9999999999614995, 1.000000000112593, 0.9999999999259078),  # 650
        (0.8866243007116027, np.nan, np.nan),  # 652
        (0.9103226383843847, 1.0094515176496819, 0.8537563592025426),  # 671
        (0.8553309767351582, 1.0404516971899196, 0.8352072581365114),  # 675
        (0.9103227083927946, 1.0094515950486254, 0.8537564253457731),  # 680
        (0.9088537576712543, np.nan, 0.8515172558355142),  # 684
        (0.9103226383825522, 1.0094515176494747, 0.8537563592011114),  # 692
        (0.9830219588781812, 0.9905683323843949, 0.9835440131017165),  # RG60
        (0.934299119637844, 0.9920407896344386, 0.9038973879304337),  # Terminal
        (0.9295386264955702, 0.9951200590104425, 0.893395351399264),  # Terminal(1)
        (0.9247590098835551, 0.9984017484980738, 0.8831292676908911),  # Terminal(2)
        (0.9199619541191407, 1.0018848684389394, 0.8731000888429148),  # Terminal(3)
        (0.9151492175030621, 1.00556844258935, 0.8633087868978496),  # Terminal(4)
    ), dtype=float)

    # Match by bus name instead of assuming that both tools return the same order.
    bus_indices: dict[str, int] = dict((name, index) for index, name in enumerate(results.bus_names))
    assert len(bus_indices) == len(results.bus_names), 'Duplicate VeraGrid bus names'
    assert set(bus_indices) == set(reference_bus_names), 'VeraGrid and PF bus sets differ'
    bus_order: np.ndarray = np.array(tuple(bus_indices[name] for name in reference_bus_names), dtype=int)
    voltage: np.ndarray = np.abs(np.column_stack((
        results.voltage_A, results.voltage_B, results.voltage_C,
    )))[bus_order, :]

    # Check all 50 physical phases with a strictly absolute tolerance. Non-finite
    # calculated voltages also fail; only phases absent in PF are excluded.
    present_phases: np.ndarray = np.isfinite(reference_voltage)
    assert np.count_nonzero(present_phases) == 50
    failed_phases: np.ndarray = np.argwhere(
        present_phases & ~np.isclose(voltage, reference_voltage, rtol=0.0, atol=1.1e-3)
    )
    phase_names: tuple[str, str, str] = ('A', 'B', 'C')
    differences: np.ndarray = np.abs(voltage - reference_voltage)
    failure_details: str = '\n'.join(
        f'{reference_bus_names[bus_index]}-{phase_names[phase_index]}: '
        f'VG={voltage[bus_index, phase_index]:.12f} pu, '
        f'PF={reference_voltage[bus_index, phase_index]:.12f} pu, '
        f'absolute difference={differences[bus_index, phase_index]:.12f} pu'
        for bus_index, phase_index in failed_phases
    )
    assert failed_phases.shape[0] == 0, (
        'Bus-phase voltage differences exceed 1e-3 pu:\n' + failure_details
    )
