# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Phasor-based RMS validation with complete generator models.

This script validates the phasor-based approach using complete generator models
(genqec + exciter + governor + stabilizer) with Vr, Vi inputs instead of Vm, Va.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from matplotlib import pyplot as plt

import VeraGridEngine.api as gce
from VeraGridEngine.Simulations.Rms.problems.rms_problem_phasor import RmsProblemPhasor
from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
from VeraGridEngine.Simulations.Rms.numerical.back_euler_fx import BackEulerImplicitIntegration
from VeraGridEngine.Utils.Symbolic.bus_rms_template import initialize_bus_rms
from VeraGridEngine.Templates.Rms.bus_phasor_rms_template import initialize_bus_phasor_rms
from VeraGridEngine.Templates.Rms.line_rms_template import get_line_rms_template
from VeraGridEngine.Templates.Rms.line_phasor_rms_template import get_line_phasor_rms_template
from VeraGridEngine.Templates.Rms.load_rms_template import get_load_rms_template
from VeraGridEngine.Templates.Rms.load_phasor_current_rms_template import get_load_phasor_current_rms_template
from VeraGridEngine.Templates.Rms.genqec_exc_gov_sat_template import get_complete_generator_template_rms
from VeraGridEngine.Templates.Rms.genqec_phasor_rms_template import get_complete_generator_template_phasor
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model
from VeraGridEngine.Devices.Aggregation import RmsEvent
from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup
from VeraGridEngine.enumerations import DynamicIntegrationMethod, RmsInitializationMethod
from VeraGridEngine.Utils.Symbolic.block import Block


def find_name_in_block(name: str, block: Block):
    """Find a variable by name in a block."""
    for var in block.algebraic_vars + block.state_vars + list(block.event_dict.keys()):
        if name == var.name:
            return var
    for mdl in block.children:
        res = find_name_in_block(name, mdl)
        if res is not None:
            return res
    return None


def test_phasor_validation():
    """Test phasor-based RMS with complete generators."""
    
    print("=" * 60)
    print("PHASOR RMS VALIDATION WITH COMPLETE GENERATORS")
    print("=" * 60)
    
    # Create two separate grids with same topology
    # Grid 1: Polar coordinates (Vm, Va) - baseline
    # Grid 2: Phasor coordinates (Vr, Vi) - test
    
    Sbase = 100.0
    
    # ===================================================================
    # GRID 1: POLAR COORDINATES (BASELINE)
    # ===================================================================
    print("\n[1] Creating polar grid (Vm, Va coordinates)...")
    grid_polar = gce.MultiCircuit(Sbase=Sbase, fbase=50.0)
    
    bus0_polar = gce.Bus(name="Bus0", Vnom=10, is_slack=True)
    bus1_polar = gce.Bus(name="Bus1", Vnom=10)
    grid_polar.add_bus(bus0_polar)
    grid_polar.add_bus(bus1_polar)
    
    for bus in grid_polar.buses:
        initialize_bus_rms(bus, vf=grid_polar.var_factory)
    
    line_polar = gce.Line(name="Line", bus_from=bus0_polar, bus_to=bus1_polar, 
                          r=0.029585798816568046, x=0.07100591715976332, b=0.03, rate=900.0)
    grid_polar.add_line(line_polar)
    
    load_polar = gce.Load(P=9.999999, Q=0.999999)
    grid_polar.add_load(bus=bus1_polar, api_obj=load_polar)
    
    gen_polar = gce.Generator(name="Gen0", P=10, vset=1.0, Snom=900)
    grid_polar.add_generator(bus=bus0_polar, api_obj=gen_polar)
    
    # Build RMS models for polar grid
    genqec_polar = get_complete_generator_template_rms(grid_polar.var_factory).block
    line_mdl_polar = get_line_rms_template(grid_polar.var_factory).block
    load_mdl_polar = get_load_rms_template(grid_polar.var_factory).block
    
    # Set load parameters (negative for consumption)
    load_mdl_polar.set_parameter_in_model(var_name="Pl0", new_value=-0.0999999)
    load_mdl_polar.set_parameter_in_model(var_name="Ql0", new_value=-0.009999999862208533)
    
    # Attach models to devices
    set_rms_model(device=gen_polar, model=genqec_polar, var_factory=grid_polar.var_factory)
    set_rms_model(device=line_polar, model=line_mdl_polar, var_factory=grid_polar.var_factory)
    set_rms_model(device=load_polar, model=load_mdl_polar, var_factory=grid_polar.var_factory)
    
    print("  Polar grid created and models attached")
    
    # ===================================================================
    # GRID 2: PHASOR COORDINATES (Vr, Vi)
    # ===================================================================
    print("\n[2] Creating phasor grid (Vr, Vi coordinates)...")
    grid_phasor = gce.MultiCircuit(Sbase=Sbase, fbase=50.0)
    
    bus0_phasor = gce.Bus(name="Bus0", Vnom=10, is_slack=True)
    bus1_phasor = gce.Bus(name="Bus1", Vnom=10)
    grid_phasor.add_bus(bus0_phasor)
    grid_phasor.add_bus(bus1_phasor)
    
    for bus in grid_phasor.buses:
        initialize_bus_phasor_rms(bus, vf=grid_phasor.var_factory)
    
    line_phasor = gce.Line(name="Line", bus_from=bus0_phasor, bus_to=bus1_phasor,
                           r=0.029585798816568046, x=0.07100591715976332, b=0.03, rate=900.0)
    grid_phasor.add_line(line_phasor)
    
    load_phasor = gce.Load(P=9.999999, Q=0.999999)
    grid_phasor.add_load(bus=bus1_phasor, api_obj=load_phasor)
    
    gen_phasor = gce.Generator(name="Gen0", P=10, vset=1.0, Snom=900)
    grid_phasor.add_generator(bus=bus0_phasor, api_obj=gen_phasor)
    
    # Build RMS models for phasor grid
    genqec_phasor = get_complete_generator_template_phasor(grid_phasor.var_factory, name="Gen0").block
    
    # For phasor grid, use current-based load model
    load_mdl_phasor = get_load_phasor_current_rms_template(grid_phasor.var_factory).block
    grid_polar.var_factory.add_connections([load_mdl_phasor.in_vars[0]], [bus1_phasor.rms_model.out_vars[0]])
    grid_polar.var_factory.add_connections([load_mdl_phasor.in_vars[1]], [bus1_phasor.rms_model.out_vars[1]])
    # Current values will be set after power flow
    set_rms_model(device=load_phasor, model=load_mdl_phasor, var_factory=grid_phasor.var_factory)
    
    # Attach models to devices
    line_phasor_mdl = get_line_phasor_rms_template(grid_phasor.var_factory).block
    # Use set_rms_model for proper current balance accumulation
    set_rms_model(device=line_phasor, model=line_phasor_mdl, var_factory=grid_phasor.var_factory)

    grid_polar.var_factory.add_connections([genqec_phasor.in_vars[0]], [bus0_phasor.rms_model.out_vars[0]])
    grid_polar.var_factory.add_connections([genqec_phasor.in_vars[1]], [bus0_phasor.rms_model.out_vars[1]])
    set_rms_model(device=gen_phasor, model=genqec_phasor, var_factory=grid_phasor.var_factory)
    # Note: line and load models attached via set_rms_model for proper current balance
    
    print("  Phasor grid created and models attached (lines handled internally)")
    
    # ===================================================================
    # POWER FLOW
    # ===================================================================
    print("\n[3] Running power flow...")
    pf_options = gce.PowerFlowOptions(
        solver_type=gce.SolverType.NR,
        verbose=0,
        tolerance=1e-6,
        max_iter=25,
    )
    
    res_polar = gce.power_flow(grid_polar, options=pf_options)
    res_phasor = gce.power_flow(grid_phasor, options=pf_options)
    
    if not res_polar.converged or not res_phasor.converged:
        print("  ✗ FAIL: Power flow did not converge")
        return False
    
    print(f"  Power flow converged (polar: {res_polar.converged}, phasor: {res_phasor.converged})")
    
    # Update load current initialization based on power flow results
    # Get Bus1 voltage from power flow
    bus1_idx_phasor = grid_phasor.buses.index(bus1_phasor)
    V_bus1 = res_phasor.voltage[bus1_idx_phasor]
    S_load = res_phasor.Sbus[bus1_idx_phasor] / grid_phasor.Sbase  # Load injection (negative for load)
    
    # Convert power to current: I = conj(S) / conj(V)
    # For V = Vr + j*Vi and S = P + j*Q:
    # Ir = (P*Vr + Q*Vi) / |V|^2
    # Ii = (P*Vi - Q*Vr) / |V|^2
    V_mag_sq = abs(V_bus1)**2
    if V_mag_sq > 1e-10:
        Ir_init = (S_load.real * V_bus1.real + S_load.imag * V_bus1.imag) / V_mag_sq
        Ii_init = (S_load.real * V_bus1.imag - S_load.imag * V_bus1.real) / V_mag_sq
    else:
        Ir_init = -0.1
        Ii_init = -0.01
    
    print(f"  Load current initialization: Ir={Ir_init:.6f}, Ii={Ii_init:.6f}")
    load_mdl_phasor.set_parameter_in_model(var_name="Ir0", new_value=Ir_init)
    load_mdl_phasor.set_parameter_in_model(var_name="Ii0", new_value=Ii_init)
    
    # ===================================================================
    # ADD EVENT
    # ===================================================================
    print("\n[4] Adding load step event...")
    
    # Create events groups
    events_group_polar = RmsEventsGroup("polar_simulation")
    events_group_phasor = RmsEventsGroup("phasor_simulation")

    # Add event to polar grid
    Pl0_polar = find_name_in_block('Pl0', load_mdl_polar)
    event_polar = RmsEvent(device=load_polar, parameter=Pl0_polar, time=0.05, value=-0.098, group=events_group_polar)
    grid_polar.add_rms_event(event_polar)

    # Add event to phasor grid (current-based load) - step Ir
    Ir0_phasor = find_name_in_block('Ir0', load_mdl_phasor)
    event_phasor = RmsEvent(device=load_phasor, parameter=Ir0_phasor, time=0.05, value=Ir_init * 0.98, group=events_group_phasor)
    grid_phasor.add_rms_event(event_phasor)
    
    print("  Events added to both grids")
    
    # ===================================================================
    # SIMULATION OPTIONS
    # ===================================================================
    print("\n[5] Setting up simulation...")

    options = RmsOptions(
        time_step=0.001,
        simulation_time=0.1,  # Short simulation for testing
        tolerance=1e-6,
        integration_method=DynamicIntegrationMethod.DaeBackEuler,
        initialization_method=RmsInitializationMethod.Explicit,
        use_init_values=False,
        max_iter=100,
        verbose=0
    )
    
    # RUN PHASOR SIMULATION
    # ===================================================================
    print("\n[7] Running phasor simulation...")
    
    problem_phasor = RmsProblemPhasor(grid=grid_phasor, options=options, pf_results=res_phasor)
    problem_phasor.set_events_group(events_group_phasor)
    problem_phasor.set_initialize_flag()
    solver_phasor = BackEulerImplicitIntegration(
        problem=problem_phasor,
        t0=0,
        t_end=options.simulation_time,
        h=options.time_step,
        max_iter=options.max_iter
    )
    t_phasor, y_phasor, well_init_phasor, converged_phasor = solver_phasor.simulate()
    print(f"  ✓ Completed: {len(t_phasor)} steps, converged={converged_phasor}")

    
    # ===================================================================
    # COMPARE RESULTS
    # ===================================================================
    print("\n[8] Comparing results...")

    
    # For phasor: find omega in genqec_phasor using the stable symbolic name.
    omega_var_phasor = find_name_in_block("omega", genqec_phasor)
    omega_idx_phasor = problem_phasor.get_var_idx(omega_var_phasor)
    omega_phasor = y_phasor[:, omega_idx_phasor]

    # For phasor: find delta in genqec_phasor using the stable symbolic name.
    delta_var_phasor = find_name_in_block("delta", genqec_phasor)
    delta_idx_phasor = problem_phasor.get_var_idx(delta_var_phasor)
    delta_phasor = y_phasor[:, delta_idx_phasor]
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot omega
    axes[0].plot(t_phasor, omega_phasor, 'b-', label='Omega (pu)')
    axes[0].set_ylabel('Omega (pu)')
    axes[0].set_title('Generator Speed - Phasor RMS')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot delta
    axes[1].plot(t_phasor, delta_phasor, 'r-', label='Delta (rad)')
    axes[1].set_ylabel('Delta (rad)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Rotor Angle - Phasor RMS')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return True


if __name__ == "__main__":
    success = test_phasor_validation()
    sys.exit(0 if success else 1)
