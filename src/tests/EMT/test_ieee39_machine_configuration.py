"""Reject the static grid's placeholder impedances as EMT machine data."""

import numpy as np
import pytest

from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Devices.Substation.bus import Bus
from VeraGridEngine.Devices.Injections.generator import Generator
from TensyGridEngine.emt.ieee9_emt_simulation import attach_emt_models
from TensyGridEngine.emt.ieee39_emt_simulation import configure_ieee39_simple_machines


def test_ieee39_machine_reduction_has_valid_inductances_and_field_time_constant():
    grid = MultiCircuit(Sbase=100, fbase=50)
    for index in range(10):
        bus = Bus(name=f'bus{index}', Vnom=345)
        grid.add_bus(bus)
        grid.add_generator(bus, Generator(name=f'gen{index}'))
    original = [(gen.R1, gen.X1, gen.R0, gen.X0) for gen in grid.generators]
    attach_emt_models(grid)
    configure_ieee39_simple_machines(grid)
    expected_td0 = (10.2, 6.56, 5.7, 5.69, 5.4, 7.3, 5.66, 6.7, 4.79, 7.)
    for gen, td0 in zip(grid.generators, expected_td0):
        model = gen.emt_model
        values = {var.name: float(expr.value) for var, expr in
                  list(model.parameters.items()) + list(model.event_dict.items())
                  if var.name in ('Ld', 'La', 'Lmq', 'Lf', 'Rf', 'Ra', 'R0', 'L0', 'H')}
        mutual = values['Ld'] - values['La']
        assert mutual > 0 and values['Lmq'] > 0
        matrix = np.array([[values['Ld'], -mutual],
                           [-mutual, mutual + values['Lf']]])
        assert np.min(np.linalg.eigvalsh(matrix)) > 0
        assert (mutual + values['Lf']) / (2*np.pi*grid.fBase*values['Rf']) == pytest.approx(td0)
        assert values['H'] > 0
        assert values['L0'] == values['La']
        assert values['R0'] == values['Ra']
        for mapped in model.api_obj_mapping.values():
            assert mapped.name not in ('Ld', 'Ra', 'L0', 'R0')
    assert [(gen.R1, gen.X1, gen.R0, gen.X0) for gen in grid.generators] == original
