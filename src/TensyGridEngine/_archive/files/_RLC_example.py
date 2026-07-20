# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0

import cProfile as profile
from typing import List

import VeraGridEngine.api as gce

from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Templates.Rms.line_rms_template import get_line_rms_template
from VeraGridEngine.Templates.Rms.load_rms_template import get_load_rms_template
from VeraGridEngine.Utils.Symbolic.bus_rms_template import initialize_bus_rms

from VeraGridEngine.Simulations.Rms.problems.rms_problem_tensygrid import (
    RmsProblemTensygrid,
)
from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions

from VeraGridEngine.enumerations import (
    DynamicIntegrationMethod,
    RmsInitializationMethod,
)

from trunk.tensygrid.files._PolynomialMatrixBuilder_class import (
    PolynomialMatrixBuilder,
)

######################################################################
# Utility
######################################################################

def find_name_in_block(name: str, block: Block):
    for var in block.algebraic_vars + block.state_vars + list(block.event_dict.keys()):
        if name == var.name:
            return var

    for mdl in block.children:
        res = find_name_in_block(name, mdl)
        if res is not None:
            return res


######################################################################
# Build RLC network using ONLY VeraGrid models
######################################################################

pr = profile.Profile()
pr.disable()

# ------------------------------------------------------------
# Grid (acts as circuit container)
# ------------------------------------------------------------
grid = gce.MultiCircuit(Sbase=1.0, fbase=50.0)

# Voltage source (slack bus)
bus_src = gce.Bus(name="Source", Vnom=1.0, is_slack=True)

# RLC node
bus_rlc = gce.Bus(name="RLC_Node", Vnom=1.0)

grid.add_bus(bus_src)
grid.add_bus(bus_rlc)

# Initialize RMS bus models
for bus in grid.buses:
    initialize_bus_rms(bus, vf=grid.var_factory)

# ------------------------------------------------------------
# Physical parameters
# ------------------------------------------------------------
R = 5.0                 # ohm
L = 10e-3               # H
C = 5e-6                # F
f = 50.0                # Hz

import math
w = 2 * math.pi * f

X = w * L               # reactance
B = w * C               # susceptance

# ------------------------------------------------------------
# R + L → transmission line
# ------------------------------------------------------------
line_rl = gce.Line(
    name="RL_branch",
    bus_from=bus_src,
    bus_to=bus_rlc,
    r=R,
    x=X,
    b=0.0,
    rate=100.0,
)

grid.add_line(line_rl)

# ------------------------------------------------------------
# Capacitor → capacitive load
# ------------------------------------------------------------
capacitor = gce.Load(
    P=0.0,
    Q=-B      # capacitive injection
)

grid.add_load(bus=bus_rlc, api_obj=capacitor)

######################################################################
# Build RMS dynamic models (templates generate equations)
######################################################################


# Templates (NO manual equations)
line_mdl = get_line_rms_template(grid.var_factory).block
load_mdl = get_load_rms_template(grid.var_factory).block

# -----------------------------
# Connect RL line to buses
# -----------------------------
grid.var_factory.add_connections(
    [line_mdl.in_vars[0]],
    [bus_src.rms_model.out_vars[0]],
)
grid.var_factory.add_connections(
    [line_mdl.in_vars[1]],
    [bus_src.rms_model.out_vars[1]],
)

grid.var_factory.add_connections(
    [line_mdl.in_vars[2]],
    [bus_rlc.rms_model.out_vars[0]],
)
grid.var_factory.add_connections(
    [line_mdl.in_vars[3]],
    [bus_rlc.rms_model.out_vars[1]],
)

# -----------------------------
# Connect capacitor
# -----------------------------
grid.var_factory.add_connections(
    [load_mdl.in_vars[0]],
    [bus_rlc.rms_model.out_vars[0]],
)
grid.var_factory.add_connections(
    [load_mdl.in_vars[1]],
    [bus_rlc.rms_model.out_vars[1]],
)

# Attach models
line_rl.rms_model = line_mdl
capacitor.rms_model = load_mdl

######################################################################
# Power flow (initial operating point)
######################################################################

pf_options = gce.PowerFlowOptions(
    solver_type=gce.SolverType.NR,
    tolerance=1e-6,
    max_iter=25,
    verbose=0,
)

pf_results = gce.power_flow(grid, options=pf_options)

######################################################################
# RMS Simulation Problem
######################################################################

options = RmsOptions(
    time_step=0.001,
    simulation_time=0.1,
    tolerance=1e-6,
    integration_method=DynamicIntegrationMethod.DaeBackEuler,
    initialization_method=RmsInitializationMethod.Explicit,
    use_init_values=False,
    max_iter=1000,
    verbose=0,
)

pr.enable()

problem = RmsProblemTensygrid(
    grid=grid,
    options=options,
    pf_results=pf_results,
)

######################################################################
# Extract symbolic equations (generated by VeraGrid)
######################################################################

algebraic_eqs: List[str] = [
    str(expression) for expression in problem._algebraic_eqs
]

state_eqs: List[str] = [
    str(expression) for expression in problem._state_eqs
]

algebraic_vars: List[str] = [
    str(expression) for expression in problem._algebraic_vars
]

state_vars: List[str] = [
    str(expression) for expression in problem._state_vars
]

print(f"\nState variables ({len(state_vars)}):")
print(state_vars)

print(f"\nAlgebraic variables ({len(algebraic_vars)}):")
print(algebraic_vars)

eqs = algebraic_eqs + state_eqs

print(f"\nNumber of equations: {len(eqs)}")
[print(f"\nEquation {i}:\n{eq}") for i, eq in enumerate(eqs)]

######################################################################
# Polynomial builder (unchanged)
######################################################################

builder = PolynomialMatrixBuilder(
    eqs=eqs,
    ineqs=[],
    state_vars=state_vars,
    algebraic_vars=algebraic_vars,
    verbose=True,
)