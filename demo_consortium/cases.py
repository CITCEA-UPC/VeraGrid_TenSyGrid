"""Build two public compact benchmark topologies with explicit demo models."""
from pathlib import Path

import numpy as np
import VeraGridEngine.api as gce
from VeraGridEngine.Templates.Rms.bus_phasor_rms_template import initialize_bus_phasor_rms
from VeraGridEngine.Templates.Rms.line_phasor_rms_template import get_line_phasor_rms_template
from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
from VeraGridEngine.Templates.Emt.transformer_emt_template import get_series_transformer_emt_template
from VeraGridEngine.Templates.Emt.load_RLC_emt_template import get_shunt_rlc_combo_emt_template
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model, set_rms_model
from VeraGridEngine.enumerations import ShuntConnectionType
from demo_consortium.models import source_model, impedance_load_rms


def build_case(case: str, emt: bool, limiter: bool = False, command: float | None = None):
    """Return a fresh grid, its PF operating point and the inverter identity."""
    root = Path(__file__).resolve().parents[1]
    paths = {'ieee9': 'hackathon_files/grids/ieee9_rms_transformers.veragrid',
             'kundur11': 'hackathon_files/grids/kundur_rms_60.veragrid'}
    grid = gce.open_file(str(root/paths[case]))
    for device in [*grid.buses, *grid.generators, *grid.loads, *grid.lines, *grid.transformers2w]:
        device.emt_model = Block()
        device.rms_model = Block()
    grid.name = f'Consortium {case} {"EMT" if emt else "RMS"} multilinear SG + averaged GFM'
    pf = gce.power_flow(grid, options=gce.PowerFlowOptions(control_q=False, tolerance=1e-9))
    if not pf.converged:
        raise RuntimeError('Power flow did not converge')
    inverter = grid.generators[-1]
    bus_index = {bus.idtag: i for i, bus in enumerate(grid.buses)}
    for bus in grid.buses:
        if emt:
            get_bus_emt_template(grid, bus)
        else:
            initialize_bus_phasor_rms(bus, grid.var_factory)
    for generator in grid.generators:
        generator.name = ('GFM ' if generator is inverter else 'SG ')+generator.name
        generator.code = 'consortium_averaged_gfm' if generator is inverter else 'consortium_classical_sg'
        index = bus_index[generator.bus.idtag]
        power = complex(pf.Sbus[index])/grid.Sbase
        rating = generator.Snom if generator.Snom < 5000 else max(100., 1.3*abs(power)*grid.Sbase)
        generator.Snom = rating
        generator.is_controlled = True
        model = source_model(grid.var_factory, generator.name.replace(' ', '_'), complex(pf.voltage[index]), power,
                             rating, grid.Sbase, grid.fBase, generator is inverter, emt,
                             limiter and generator is inverter, command if generator is inverter else None)
        if emt:
            set_emt_model(generator, model, grid.var_factory)
        else:
            set_rms_model(generator, model, grid.var_factory)
    for line in grid.lines:
        model = (get_pi_line_emt_template(grid.var_factory, phN=False, phA=True, phB=True, phC=True,
                                        name=line.name).block if emt
                 else get_line_phasor_rms_template(grid.var_factory, name=line.name).block)
        (set_emt_model if emt else set_rms_model)(line, model, grid.var_factory)
    for transformer in grid.transformers2w:
        model = (get_series_transformer_emt_template(grid.var_factory, name=transformer.name).block if emt
                 else get_line_phasor_rms_template(grid.var_factory, name=transformer.name).block)
        (set_emt_model if emt else set_rms_model)(transformer, model, grid.var_factory)
    for load in grid.loads:
        index = bus_index[load.bus.idtag]
        if emt:
            model = get_shunt_rlc_combo_emt_template(grid.var_factory, include_r=True, include_l=load.Q > 0,
                include_c=False, phA=True, phB=True, phC=True, connection_type=ShuntConnectionType.FloatingStar,
                name=load.name).block
            # Initialize inductive currents at the PF sinusoidal operating point.
            for block in model.get_all_blocks():
                for variable in block.state_vars:
                    if variable.name.startswith('iL_'):
                        phase = 'ABC'.index(variable.name[-1])
                        angle = np.angle(pf.voltage[index])-phase*2*np.pi/3
                        block.init_eqs[variable] = grid.var_factory.add_const(np.sqrt(2.)*load.Q/grid.Sbase/abs(pf.voltage[index])*np.cos(angle))
        else:
            admittance = np.conj(complex(load.P, load.Q)/grid.Sbase)/abs(pf.voltage[index])**2
            model = impedance_load_rms(grid.var_factory, load.name.replace(' ', '_'), admittance)
        (set_emt_model if emt else set_rms_model)(load, model, grid.var_factory)
    return grid, pf, inverter
