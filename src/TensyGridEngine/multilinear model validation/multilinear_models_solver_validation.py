# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import cProfile as profile
import numpy as np
from matplotlib import pyplot as plt

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Templates.Rms.generation_tensygrid_ml import get_complete_generator_template
from VeraGridEngine.Templates.Rms.line_rms_template import get_line_rms_template
from VeraGridEngine.Templates.Rms.load_rms_template import get_load_rms_template
from VeraGridEngine.Utils.Symbolic.bus_rms_template import initialize_bus_rms
from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
from VeraGridEngine.Simulations.Rms.problems.rms_problem_tensygrid import RmsProblemTensygrid
from VeraGridEngine.Simulations.Rms.numerical.back_euler_ts import BackEulerImplicitTensygrid

from VeraGridEngine.enumerations import DynamicIntegrationMethod, RmsInitializationMethod
from VeraGridEngine.Devices.Aggregation import RmsEvent
from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup
from VeraGridEngine.enumerations import VarPowerFlowReferenceType
import VeraGridEngine.api as gce

###########################################################################################################################
# Build VeraGrid object
###########################################################################################################################

pr = profile.Profile()
pr.disable()


def find_name_in_block(name: str, block: Block):
    for var in block.algebraic_vars + block.state_vars + list(block.event_dict.keys()):
        if name == var.name:
            return var

    for mdl in block.children:
        res = find_name_in_block(name, mdl)
        if res is not None:
            return res


grid = gce.MultiCircuit(Sbase=100, fbase=50.0)

# Buses
bus0 = gce.Bus(name="Bus0", Vnom=10, is_slack=True)
bus1 = gce.Bus(name="Bus1", Vnom=10)

grid.add_bus(bus0)
grid.add_bus(bus1)

for bus in grid.buses:
    initialize_bus_rms(bus, vf=grid.var_factory)

# Lines
line0 = gce.Line(name="line 0-2", bus_from=bus0, bus_to=bus1, r=0.029585798816568046, x=0.07100591715976332, b=0.03,
                 rate=900.0)

grid.add_line(line0)

# load
load = gce.Load(P=9.999999, Q=0.999999)

grid.add_load(bus=bus1, api_obj=load)

# Generators
gen0 = gce.Generator(name="Gen0", P=10, vset=1.0, Snom=900,
                     x1=0.86138701, r1=0.3, freq=50.0,
                     M=10.0,
                     D=1.0,
                     omega_ref=1.0,
                     Kp=1.0,
                     Ki=10.0,
                     )

grid.add_generator(bus=bus0, api_obj=gen0)

######################################################################################################
# Build Rms models
######################################################################################################



# line
line0_mdl = get_line_rms_template(grid.var_factory).block

# load
load_mdl = get_load_rms_template(grid.var_factory).block

# generator
genqec1_mdl = get_complete_generator_template(grid.var_factory, implicit=True).block

# connection with buses

grid.var_factory.add_connections([genqec1_mdl.in_vars[0]], [bus0.rms_model.out_vars[0]])
grid.var_factory.add_connections([genqec1_mdl.in_vars[1]], [bus0.rms_model.out_vars[1]])

grid.var_factory.add_connections([line0_mdl.in_vars[0]], [bus0.rms_model.out_vars[0]])
grid.var_factory.add_connections([line0_mdl.in_vars[1]], [bus0.rms_model.out_vars[1]])

grid.var_factory.add_connections([line0_mdl.in_vars[2]], [bus1.rms_model.out_vars[0]])
grid.var_factory.add_connections([line0_mdl.in_vars[3]], [bus1.rms_model.out_vars[1]])

u_gov = find_name_in_block("u_gov1", genqec1_mdl)

grid.var_factory.add_connections([load_mdl.in_vars[0]], [bus1.rms_model.out_vars[0]])
grid.var_factory.add_connections([load_mdl.in_vars[1]], [bus1.rms_model.out_vars[1]])

# external mapping
big_gen1 = Block(children=[genqec1_mdl])
big_gen1.external_mapping.update({VarPowerFlowReferenceType.P: genqec1_mdl.out_vars[0]})
big_gen1.external_mapping.update({VarPowerFlowReferenceType.Q: genqec1_mdl.out_vars[1]})

# attach rms models to Veragrid objects
line0.rms_model = line0_mdl
load.rms_model = load_mdl
gen0.rms_model = big_gen1

Pl0 = find_name_in_block('Pl0', load_mdl)
print(Pl0)
events_group = RmsEventsGroup("simulation1")
event = RmsEvent(device=load, parameter=Pl0, time=.01, value=-0.9, group=events_group)
grid.add_rms_event(event)

options = gce.PowerFlowOptions(
    solver_type=gce.SolverType.NR,
    retry_with_other_methods=False,
    verbose=0,
    initialize_with_existing_solution=True,
    tolerance=1e-6,
    max_iter=25,
    control_q=False,
    control_taps_modules=True,
    control_taps_phase=True,
    control_remote_voltage=True,
    orthogonalize_controls=True,
    apply_temperature_correction=True,
    branch_impedance_tolerance_mode=gce.BranchImpedanceMode.Specified,
    distributed_slack=False,
    ignore_single_node_islands=False,
    trust_radius=1.0,
    backtracking_parameter=0.05,
    use_stored_guess=False,
    initialize_angles=False,
    generate_report=False,
)
res = gce.power_flow(grid, options=options)

print(f"Converged: {res.converged}")
print(res.get_bus_df())
print(res.get_branch_df())

params_mapping = dict()

pr.enable()

options = RmsOptions(time_step=0.001,
                     simulation_time=0.1,
                     tolerance=1e-6,
                     integration_method=DynamicIntegrationMethod.DaeBackEuler,
                     initialization_method=RmsInitializationMethod.Explicit,
                     use_init_values=False,
                     max_iter=1000,
                     verbose=0)

problem = RmsProblemTensygrid(grid=grid,
                        options=options,
                        pf_results=res)

problem.set_events_group(events_group)

solver = BackEulerImplicitTensygrid(
    problem=problem,
    t0=0,
    t_end=options.simulation_time,
    h=options.time_step,
    max_iter=options.max_iter
)

t, y, well_initialized, converged = solver.simulate()

pr.disable()

xlim = options.simulation_time

Pg = genqec1_mdl.out_vars[0]
Qg = genqec1_mdl.out_vars[1]
Ir = find_name_in_block("IRPu", genqec1_mdl)
Efd = find_name_in_block("E_fd", genqec1_mdl)
Te = find_name_in_block("Te", genqec1_mdl)
omega = find_name_in_block("omega", genqec1_mdl)

u_gov1 = find_name_in_block("u_gov1", genqec1_mdl)
y_subexciter = find_name_in_block("y_subexciter1", genqec1_mdl)
y_exciter1 = find_name_in_block("y_exciter1", genqec1_mdl)
y_exciter4 = find_name_in_block("y_exciter4", genqec1_mdl)
Efe = find_name_in_block("f_output", genqec1_mdl)
delta = find_name_in_block("delta", genqec1_mdl)
Vm = find_name_in_block("Vm", genqec1_mdl)

# Generator state variables
plt.plot(t, y[:, problem.get_var_idx(omega)], label="omega (pu)")
plt.plot(t, y[:, problem.get_var_idx(delta)], label="delta (pu)")
plt.plot(t, y[:, problem.get_var_idx(Pg)], label="P (pu)")
plt.plot(t, y[:, problem.get_var_idx(Qg)], label="Q (pu)")

plt.legend(loc='upper right', ncol=2)
plt.xlabel("Time (s)")
plt.ylabel("Values (pu)")
plt.xlim([0, xlim])
# plt.ylim([0.85, 1.15])
plt.grid(True)
plt.tight_layout()
plt.show()

pr.dump_stats('profile.pstat')



Eq_prime = find_name_in_block("Eq_prime", genqec1_mdl)
Ed_prime = find_name_in_block("Ed_prime", genqec1_mdl)
Psiq_prime = find_name_in_block("Psiq_prime", genqec1_mdl)
Psid_prime = find_name_in_block("Psid_prime", genqec1_mdl)

Ed1 = find_name_in_block("Ed1", genqec1_mdl)
Eq1 = find_name_in_block("Eq1", genqec1_mdl)


Tm = genqec1_mdl.out_vars[0]
# Generator state variables
u_cos = find_name_in_block("u_cos", genqec1_mdl)
u_sin = find_name_in_block("u_sin", genqec1_mdl)
delta_idx = problem.get_var_idx(find_name_in_block("delta", genqec1_mdl))
vm_idx =  problem.get_var_idx(bus0.rms_model.out_vars[1])
delta_values = y[:, delta_idx]
vm_values = y[:, vm_idx]
cos_x = np.cos(vm_values - delta_values)
sin_x = np.sin(vm_values - delta_values)
# 4. Plot or Print
plt.plot(t, cos_x, label="cos(x)")
plt.plot(t, sin_x, label="sin(x)")
plt.plot(t, y[:, problem.get_var_idx(u_cos)], label="u_cos (pu)")
plt.plot(t, y[:, problem.get_var_idx(u_sin)], label="u_sin (pu)")

plt.legend(loc='upper right', ncol=2)
plt.xlabel("Time (s)")
plt.ylabel("Values (pu)")
plt.xlim([0, xlim])
# plt.ylim([0.85, 1.15])
plt.grid(True)
plt.tight_layout()
plt.show()
