"""Check actual transformer descriptors against an independent port equivalent."""
import numpy as np
import pytest

from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Devices.Substation.bus import Bus
from VeraGridEngine.Devices.Branches.transformer import Transformer2W
from trunk.benchmarks.transformer_equivalence import audit_transformer, assert_simple_compatible


@pytest.mark.parametrize('tap',[.935,1.,1.069])
def test_transformer_ports_and_passive_termination(tap):
    grid=MultiCircuit(fbase=60.)
    source,destination=Bus(name='source',Vnom=110),Bus(name='destination',Vnom=110)
    grid.add_bus(source);grid.add_bus(destination)
    transformer=Transformer2W(bus_from=source,bus_to=destination,r=.01,x=.1,tap_module=tap)
    grid.add_transformer2w(transformer)
    result=audit_transformer(grid,transformer)
    assert result['linear_core_enabled']
    assert result['coupled_origin_residual'] < 1e-10
    assert result['simple_vs_static_max_abs'] < 1e-12
    assert result['coupled_vs_static_max_abs'] < 1e-3
    assert result['balanced_return_path_coupling_effect'] < 1e-10
    assert result['arbitrary_phase_return_path_coupling_effect'] > 1e-6
    assert min(result['core_shunt_susceptance_eigenvalues']) > 0
    for value in result['minimum_admittance_hermitian_eigenvalue'].values():
        assert value > -1e-10
    for value in result['largest_finite_terminated_pole_real'].values():
        assert value < 0


def test_simple_comparison_rejects_unrepresented_shunt():
    transformer=Transformer2W(bus_from=Bus(),bus_to=Bus(),r=.01,x=.1,g=.01)
    with pytest.raises(ValueError,match='zero shunt'):
        assert_simple_compatible(transformer)
