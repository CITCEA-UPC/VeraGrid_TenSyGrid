# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path

import numpy as np

from VeraGridEngine.basic_structures import Logger
from VeraGridEngine.Devices.Branches.line import Line
from VeraGridEngine.Devices.Branches.underground_line_type import UndergroundLineType
from VeraGridEngine.Devices.Injections.generator import Generator
from VeraGridEngine.Devices.Injections.load import Load
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.enumerations import SolverType
from VeraGridEngine.IO.file_open import FileOpen
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions


def test_legacy_underground_line() -> None:
    """Load an old scalar cable catalogue and preserve its balanced power flow.

    :return: None.
    """
    # This committed fixture predates the physical cable implementation. Load
    # its native records, rather than manufacturing legacy JSON with today's API.
    file_path: Path = Path(__file__).resolve().parents[1] / 'data' / 'grids' / 'test_line_templates.gridcal'
    opener: FileOpen = FileOpen(file_name=str(file_path))
    grid: MultiCircuit = opener.open()
    assert not opener.logger.has_errors(), str(opener.logger)
    assert len(grid.buses) == 2
    assert len(grid.lines) == 1
    assert len(grid.underground_cable_types) == 1
    assert len(grid.underground_cable_constructions) == 0

    # The old UndergroundLineType keeps its scalar inputs and line reference;
    # it does not require an UndergroundCableType or any geometric composition.
    cable: UndergroundLineType = grid.underground_cable_types[0]
    line: Line = grid.lines[0]
    assert line.template is cable
    assert cable.idtag == '5cce623cf66c4326aa38f65b6f61282e'
    assert not cable.has_physical_data()
    np.testing.assert_array_equal(
        (cable.Imax, cable.Vnom, cable.freq, cable.R, cable.X, cable.B, cable.C,
         cable.R0, cable.X0, cable.B0, cable.C0, cable.n_circuits, cable.capex, cable.opex),
        (25.0, 10.0, 50.0, 0.01, 0.02, 0.03, 0.0,
         0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0),
    )

    # The fixture tests catalogue loading: it has no injections and its saved
    # line B is stale (30000 pu). Reapply the unchanged scalar template, then
    # supply a fixed operating point, identically to the pre-feature baseline.
    logger: Logger = grid.apply_all_branch_types()
    assert not logger.has_errors(), str(logger)
    np.testing.assert_allclose(
        (line.R, line.X, line.B, line.R0, line.X0, line.B0, line.rate),
        (0.01, 0.02, 0.0, 0.0, 0.0, 0.0, 433.0127018922193),
        rtol=0.0,
        atol=1.0e-12,
    )
    grid.buses[0].is_slack = True
    grid.add_generator(bus=grid.buses[0], api_obj=Generator(name='Supply', P=20.0, vset=1.0))
    grid.add_load(bus=grid.buses[1], api_obj=Load(name='Demand', P=20.0, Q=10.0))
    options: PowerFlowOptions = PowerFlowOptions(
        solver_type=SolverType.NR,
        retry_with_other_methods=False,
        tolerance=1.0e-10,
    )
    driver: PowerFlowDriver = PowerFlowDriver(grid=grid, options=options)
    driver.run()
    assert driver.results.converged
    assert not driver.logger.has_errors(), str(driver.logger)

    # Reference computed with this fixture and operating point at cfd0311aa,
    # before underground_cables. Check complex voltages, both-end powers and
    # losses so an unchanged voltage magnitude alone cannot mask a regression.
    np.testing.assert_allclose(
        driver.results.voltage,
        np.array((1.0 + 0.0j, 0.995974797746821 - 0.003j)),
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        driver.results.Sf,
        np.array((20.050404506358177 + 10.100809012715928j,)),
        rtol=0.0,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        driver.results.St,
        np.array((-20.0 - 10.0j,)),
        rtol=0.0,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        driver.results.losses,
        np.array((0.0504045063579959 + 0.10080901271599j,)),
        rtol=0.0,
        atol=1.0e-8,
    )
