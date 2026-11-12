"""The optional profile preserves PF power and regulates voltage with the right sign."""
import numpy as np
import pytest
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Devices.Substation.bus import Bus
from VeraGridEngine.Devices.Injections.generator import Generator
from VeraGridEngine.Devices.Injections.load import Load
from TensyGridEngine.emt.ieee9_emt_simulation import attach_emt_models
from TensyGridEngine.emt.ieee39_emt_simulation import configure_ieee39_stability_profile


def make_grid():
    grid=MultiCircuit(Sbase=100,fbase=50)
    bus=Bus(name='bus',Vnom=345)
    grid.add_bus(bus)
    grid.add_generator(bus,Generator(name='gen'))
    grid.add_load(bus,Load(name='load',P=80,Q=20))
    attach_emt_models(grid)
    return grid


def parameters(model):
    return {var.name:(var,expr) for block in model.get_all_blocks()
            for table in (block.event_dict,block.parameters) for var,expr in table.items()}


@pytest.mark.parametrize('voltage',[.95,1.07])
def test_profile_keeps_pf_load_power_and_adds_negative_voltage_feedback(voltage):
    grid=make_grid()
    configure_ieee39_stability_profile(grid,np.array([voltage+0j]))
    load=parameters(grid.loads[0].emt_model)
    value=lambda key:float(load[key][1].value)
    ratio=(np.sqrt(2*voltage**2+value('eps'))+value('eps'))/value('V0')
    assert value('a1')*ratio**2+value('a2')*ratio+value('a3')==pytest.approx(1.)
    assert value('a4')*ratio**2+value('a5')*ratio+value('a6')==pytest.approx(1.)
    assert value('a1')*ratio**2==pytest.approx(.25)
    assert grid.loads[0].P==80 and grid.loads[0].Q==20
    model=grid.generators[0].emt_model
    machine=parameters(model)
    assert machine['D'][1].value/(2*machine['H'][1].value)==pytest.approx(.25)
    names={v.name:v for v in model.state_vars+model.algebraic_vars+model.in_vars}
    names.update({key:entry[0] for key,entry in machine.items()})
    bindings={names[name].uid:value for name,value in
              {'omega_base':2*np.pi*50,'v_f':.02,'Rf':.02,'i_f':1.}.items()}
    for phase,angle in zip('ABC',[0.,-2*np.pi/3,2*np.pi/3]):
        bindings[names['v_'+phase].uid]=np.sqrt(2)*voltage*np.sin(angle)
    row=next(i for i,var in enumerate(model.state_vars) if var.name=='psi_f')
    assert model.state_eqs[row].eval_uid(bindings)==pytest.approx(0.,abs=1e-12)
    for phase in 'ABC':
        bindings[names['v_'+phase].uid]*=1.01
    assert model.state_eqs[row].eval_uid(bindings)==pytest.approx(-2*np.pi*50*.02*2*.01)


@pytest.mark.parametrize('fraction',[-.1,1.1,float('nan')])
def test_profile_rejects_invalid_zip_fractions(fraction):
    grid=make_grid()
    with pytest.raises(ValueError):
        configure_ieee39_stability_profile(grid,np.array([1.+0j]),impedance_fraction=fraction)
