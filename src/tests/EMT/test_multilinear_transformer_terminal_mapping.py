"""Keep multilinear transformer equation currents connected to network KCL."""

import pytest

from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Devices.Substation.bus import Bus
from VeraGridEngine.Devices.Branches.transformer import Transformer2W
from VeraGridEngine.Templates.Emt.xfmr_emt_multilinear_template import get_xfmr_emt_template_multilinear
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.enumerations import VarPowerFlowReferenceType


@pytest.mark.parametrize('from_side', [True, False])
def test_attachment_preserves_multilinear_transformer_terminal_currents(from_side: bool) -> None:
    """Public attachment must retain the same currents used by device equations."""
    grid = MultiCircuit()
    source, destination = Bus(name='source', Vnom=20), Bus(name='destination', Vnom=10)
    grid.add_bus(source)
    grid.add_bus(destination)
    transformer = Transformer2W(bus_from=source, bus_to=destination, HV=20, LV=10, r=0.01, x=0.1)
    grid.add_transformer2w(transformer)
    get_bus_emt_template(grid, source)
    get_bus_emt_template(grid, destination)
    block = get_xfmr_emt_template_multilinear(grid.var_factory).block
    references = ((VarPowerFlowReferenceType.if_A, VarPowerFlowReferenceType.if_B,
                   VarPowerFlowReferenceType.if_C) if from_side else
                  (VarPowerFlowReferenceType.it_A, VarPowerFlowReferenceType.it_B,
                   VarPowerFlowReferenceType.it_C))
    original = [block.external_mapping[reference] for reference in references]
    set_emt_model(transformer, block, grid.var_factory)
    for reference, current in zip(references, original):
        assert transformer.emt_model.external_mapping[reference] is current
        assert current in transformer.emt_model.out_vars
        assert current.ref == reference
        assert any(current.uid in {var.uid for var in equation.get_vars()}
                   for equation in transformer.emt_model.algebraic_eqs)


@pytest.mark.parametrize('tap', [0.935, 1.0, 1.069])
def test_leakage_dynamics_match_pf_equivalent_at_off_nominal_tap(tap: float) -> None:
    """PF phasors must give the same fundamental leakage-current derivative."""
    import numpy as np
    from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory

    block = get_xfmr_emt_template_multilinear(VarFactory()).block
    resistance, reactance, omega = .01, .1, 2 * np.pi * 60
    voltage_from = 1.03 * np.exp(.15j)
    voltage_to = .98 * np.exp(-.04j)
    # Independent standard PF transformer equivalent (no shunt branch).
    current_from = (voltage_from / tap - voltage_to) / (tap * complex(resistance, reactance))
    bindings = {}
    for equation in block.state_eqs[:3]:
        for var in equation.get_vars():
            bindings[var.uid] = 0.0
    variables = {var.name: var for equation in block.state_eqs[:3] for var in equation.get_vars()}
    for name, value in {'omega_base': omega, 'xfmr_tap_module': tap,
                        'xfmr_sc_voltage_pct': 100 * np.hypot(resistance, reactance),
                        'xfmr_sc_resistance_pct': 100 * resistance}.items():
        bindings[variables[name].uid] = value
    for i in range(3):
        for j in range(3):
            bindings[variables[f'xfmr_cf_{i}_{j}'].uid] = float(i == j)
            bindings[variables[f'xfmr_ct_{i}_{j}'].uid] = float(i == j)
    for i, phase in enumerate('ABC'):
        rotation = np.exp(-2j * np.pi * i / 3)
        bindings[variables[f'vf_{phase}'].uid] = float((voltage_from * rotation).real)
        bindings[variables[f'vt_{phase}'].uid] = float((voltage_to * rotation).real)
        bindings[variables[f'i_leak_{phase}'].uid] = float((current_from * rotation).real)
        expected = float((1j * omega * current_from * rotation).real)
        assert block.state_eqs[i].eval_uid(bindings) == pytest.approx(expected, rel=2e-5, abs=2e-5)
