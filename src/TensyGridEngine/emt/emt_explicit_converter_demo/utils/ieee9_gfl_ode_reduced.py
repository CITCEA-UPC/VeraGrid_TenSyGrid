"""IEEE 9 connectivity with resistive lines and two GFL ODE generators.

The default experiment uses resistive line surrogates. The separate
``ieee9_gfl_ode_pi_explicit_probe`` retains the IEEE-9 EMT pi-line templates
and their dynamic line states. The slack source uses sine/cosine states to
keep the model multi-affine.
"""

from __future__ import annotations

import argparse
import numpy as np

from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_gfl_ode_grid import (
    attach_gfl_ode_generator,
)
from TensyGridEngine.emt.emt_ieee9 import (
    attach_emt_models, build_emt_options, build_ieee9_grid, build_power_flow_options,
)
from VeraGridEngine.Simulations.EMT.problems.emt_problem_linear_reduced import EmtProblemLinearReduced
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Const
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.enumerations import VarPowerFlowReferenceType as Ref


def attach_resistive_line(grid, line) -> None:
    """Use a three-phase resistor with the original line's impedance magnitude."""
    vf = grid.var_factory
    input_refs = (Ref.vf_A, Ref.vf_B, Ref.vf_C, Ref.vt_A, Ref.vt_B, Ref.vt_C)
    inputs = [vf.add_var(f"{ref.value}", reference=ref) for ref in input_refs]
    from_voltage, to_voltage = inputs[:3], inputs[3:]
    from_refs = (Ref.if_A, Ref.if_B, Ref.if_C)
    to_refs = (Ref.it_A, Ref.it_B, Ref.it_C)
    from_current = [vf.add_var(f"if_{phase}", reference=ref)
                    for phase, ref in zip("ABC", from_refs)]
    to_current = [vf.add_var(f"it_{phase}", reference=ref)
                  for phase, ref in zip("ABC", to_refs)]
    conductance = 1.0 / max(float(np.hypot(line.R, line.X)), 1.0e-3)
    block = Block(
        name=f"resistive_{line.name}",
        in_vars=inputs,
        out_vars=from_current + to_current,
        algebraic_vars=from_current + to_current,
        algebraic_eqs=[current - Const(conductance) * (vf_ - vt_)
                        for current, vf_, vt_ in zip(from_current, from_voltage, to_voltage)]
                       + [to_ + from_ for from_, to_ in zip(from_current, to_current)],
    )
    block.external_mapping = {variable.ref: variable
                              for variable in inputs + from_current + to_current}
    set_emt_model(line, block, grid.var_factory)


def attach_oscillating_slack(grid, generator, voltage: complex,
                             injection_phasor: complex = 0j) -> None:
    """Replace the synchronous slack machine with a Norton oscillator."""
    vf = grid.var_factory
    voltage_refs = (Ref.v_A, Ref.v_B, Ref.v_C)
    current_refs = (Ref.i_A, Ref.i_B, Ref.i_C)
    bus_voltage = [vf.add_var(f"v_{phase}", reference=ref)
                   for phase, ref in zip("ABC", voltage_refs)]
    injection = [vf.add_var(f"i_{phase}", reference=ref)
                 for phase, ref in zip("ABC", current_refs)]
    sine = [vf.add_var(f"source_sin_{phase}") for phase in "ABC"]
    cosine = [vf.add_var(f"source_cos_{phase}") for phase in "ABC"]
    states = sine + cosine
    differentials = [vf.add_diff_var(f"d_{state.name}", base_var=state)
                     for state in states]
    omega = 2.0 * np.pi * grid.fBase
    conductance = 1000.0
    angle = float(np.angle(voltage))
    amplitude = float(np.sqrt(2.0) * abs(voltage))
    current_peak = float(np.sqrt(2.0) * abs(injection_phasor))
    current_angle = float(np.angle(injection_phasor))
    shifts = (0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0)
    block = Block(
        name="ieee9_oscillating_slack",
        in_vars=bus_voltage,
        out_vars=injection,
        state_vars=states,
        diff_vars=differentials,
        state_eqs=[Const(omega) * value for value in cosine]
                   + [Const(-omega) * value for value in sine],
        algebraic_vars=injection,
        algebraic_eqs=[current - Const(conductance) * (source - bus)
                        for current, source, bus in zip(injection, sine, bus_voltage)],
        # The bus voltage is initialized from power flow. The Norton source
        # needs an internal voltage phasor offset I/G to inject the power-flow
        # slack current. Seed both sine and cosine quadratures; seeding only
        # sine makes the oscillator's first derivative inconsistent and gives
        # a spurious 50 Hz voltage-envelope transient.
        init_eqs={**{state: Const(amplitude * np.sin(angle + shift)
                                        + current_peak * np.sin(current_angle + shift) / conductance)
                     for state, shift in zip(sine, shifts)},
                   **{state: Const(amplitude * np.cos(angle + shift)
                                        + current_peak * np.cos(current_angle + shift) / conductance)
                      for state, shift in zip(cosine, shifts)}},
    )
    block.external_mapping = {variable.ref: variable for variable in bus_voltage + injection}
    set_emt_model(generator, block, grid.var_factory)


def build_problem(*, use_pi_lines: bool = False,
                  replace_zero_line_B: float | None = None,
                  first_gfl_power_scale: float = 1.0) -> EmtProblemLinearReduced:
    """Assemble the experiment; optionally change charging or one GFL setpoint."""
    grid = build_ieee9_grid()
    if replace_zero_line_B is not None:
        if not use_pi_lines or replace_zero_line_B <= 0.0:
            raise ValueError("Positive replacement charging requires pi lines")
        for line in grid.lines:
            if abs(float(line.B)) < 1e-15:
                line.B = replace_zero_line_B
    attach_emt_models(grid)
    if not use_pi_lines:
        for line in grid.lines:
            attach_resistive_line(grid, line)

    power_flow_options = build_power_flow_options()
    balanced = PowerFlowDriver(grid=grid, options=power_flow_options)
    balanced.run()
    three_phase = PowerFlowDriver3Ph(grid=grid, options=power_flow_options)
    three_phase.run()
    if not balanced.results.converged or not three_phase.results.converged:
        raise RuntimeError("IEEE 9 power flow did not converge")

    slack = next(generator for generator in grid.generators if generator.bus.is_slack)
    slack_index = grid.buses.index(slack.bus)
    slack_voltage = complex(balanced.results.voltage[slack_index])
    slack_power = complex(balanced.results.Sbus[slack_index]) / grid.Sbase
    slack_injection = np.conj(slack_power / slack_voltage)
    attach_oscillating_slack(grid, slack, slack_voltage, slack_injection)
    first_gfl = True
    for generator_index, generator in enumerate(grid.generators):
        if not generator.bus.is_slack:
            voltage = complex(balanced.results.voltage[grid.buses.index(generator.bus)])
            reactive_power = float(balanced.results.gen_q[generator_index])
            attach_gfl_ode_generator(
                grid, generator, network_current_copies=True,
                operating_voltage=voltage,
                operating_reactive_power=reactive_power,
                power_reference_scale=first_gfl_power_scale if first_gfl else 1.0,
            )
            first_gfl = False

    return EmtProblemLinearReduced(
        grid=grid,
        options=build_emt_options(),
        pf_results=balanced.results,
        pf_results_3ph=three_phase.results,
    )


def main(*, verify_reduced_matrices: bool = False) -> None:
    full = build_problem()
    print("assembled full grid", flush=True)
    S, Phi = full.build_multilinear_matrices()
    print("validated full S/Phi", flush=True)
    reduced = full.build_reduced_model_problem()
    print("constructed reduced ODE", flush=True)
    reduced_shapes = None
    if verify_reduced_matrices:
        reduced_S, reduced_Phi = reduced.build_multilinear_matrices_fast()
        reduced_shapes = (reduced_S.shape, reduced_Phi.shape)
        print("validated reduced S/Phi", flush=True)
    end_time = 1.0e-4
    options = full.options
    solver = build_emt_solver(
        options=options, problem=full, t0=0.0, t_end=end_time,
        h=float(options.time_step), method=options.integration_method,
    )
    full_time, full_values, _, full_initialized, full_converged = solver.simulate(
        boundary_updater=full
    )
    print("simulated full grid", flush=True)
    reduced_result = reduced.simulate(t_end=end_time)
    print("simulated reduced ODE", flush=True)
    if not (full_initialized and full_converged and
            reduced_result.initialized and reduced_result.converged):
        raise RuntimeError("Full or reduced IEEE 9 simulation failed")
    if not np.allclose(full_time, reduced_result.time):
        raise RuntimeError("Full and reduced simulations used different sample times")
    state_error = float(np.max(np.abs(
        full_values[:, :len(full.get_state_vars())] - reduced_result.values
    )))
    if state_error > 1.0e-6:
        raise RuntimeError(f"Reduced ODE differs from the full DAE: {state_error:.3e}")
    print(f"full: states={len(full.get_state_vars())}, algebraics={len(full.get_algebraic_vars())}")
    print(f"reduced: states={len(reduced.get_state_vars())}, algebraics={len(reduced.get_algebraic_vars())}")
    print(f"full S={S.shape}, Phi={Phi.shape}")
    if reduced_shapes is not None:
        print(f"reduced S={reduced_shapes[0]}, Phi={reduced_shapes[1]}")
    print(f"samples={len(full_time)}, max_state_error={state_error:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-reduced-matrices", action="store_true",
                        help="Build the reduced ODE's S/Phi with coefficient maps")
    main(verify_reduced_matrices=parser.parse_args().verify_reduced_matrices)
