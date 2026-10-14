"""Simulate two GFL converters coupled through a resistive two-bus grid.

Beginner's map
--------------
1. MODEL: ``build_problem`` builds two converter ODEs and the source oscillator.
   It solves each phase's two linear KCL equations and substitutes both bus
   voltages into the converter equations. There are no algebraic variables.
2. EVENTS: ``main`` registers timed P/Q reference changes on the grid and
   connects that event group to the explicit model's runtime parameters.
3. SOLVE: ``problem.simulate()`` advances the full substituted state vector.
4. SAVE: ``simulation_data/two_gfl_kcl_ode`` contains S/Phi, named source
   voltage, setpoint and state traces in MATLAB files, and a plot.

Run with ``PYTHONPATH=src python -m TensyGridEngine.emt.emt_explicit_converter_demo.two_gfl_kcl_ode``.

The bus voltages have no dynamics: each phase's two KCL equations are solved
analytically and substituted into the converter ODEs. Thus the combined model
has states but no algebraic variables. ``EmtModelProblem`` is appropriate for
this standalone Block; it already inherits multilinear S/Phi extraction from
``EmtProblemMultilinear``. The latter's constructor expects a full grid model.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Allow direct execution by file path as well as `python -m` from the repo.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

import VeraGridEngine.api as gce
from VeraGridEngine.Devices.Events.emt_event import EmtEvent
from VeraGridEngine.Devices.Events.emt_events_group import EmtEventsGroup
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.problems.emt_model_problem import EmtModelProblem
from VeraGridEngine.Simulations.EMT.problems.emt_problem_template import EmtProblemTemplate
from VeraGridEngine.Templates.Emt.vsc_gfl_ode import build_gfl_converter_model_ode
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Const
from VeraGridEngine.enumerations import (
    DynamicIntegrationMethod, EmtInitializationMethod, EmtSolverTypes,
)
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.simulation_data import case_dir, save_case_data


PHASES = "ABC"
GRID_OMEGA = 2.0 * np.pi * 50.0


def build_problem(
    *,
    source_conductance: float = 10.0,
    line_conductance: float = 5.0,
    bus_1_shunt_conductance: float = 0.0,
    bus_2_shunt_conductance: float = 0.0,
    simulation_time: float = 2.0e-2,
) -> tuple[EmtModelProblem, dict]:
    """Eliminate six bus voltages from linear KCL and substitute into both ODEs.

    Converter 1 is connected to bus 1 and converter 2 to bus 2. The buses are
    connected by a resistive branch; bus 1 is fed by a voltage source behind a
    conductance. The same nodal matrix is solved independently for A, B and C.
    """
    if source_conductance <= 0 or line_conductance <= 0:
        raise ValueError("Source and line conductances must be positive")
    if bus_1_shunt_conductance < 0 or bus_2_shunt_conductance < 0:
        raise ValueError("Shunt conductances must be nonnegative")
    vf = VarFactory()
    time = vf.add_var("grid_time")
    blocks = []
    for index in (1, 2):
        block, _ = build_gfl_converter_model_ode(vf, multilinear=True)
        # The builder uses fixed local names; every variable needs a unique
        # compiler name in the flattened two-converter system.
        variables = [*block.state_vars, *block.diff_vars, *block.in_vars, *block.event_dict]
        for variable in variables:
            variable.name = f"{variable.name}_{index}"
        blocks.append(block)

    phase_shifts = (0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0)
    source_vars = tuple(vf.add_var(f"v_source_{phase}") for phase in PHASES)
    source_quadrature = tuple(vf.add_var(f"v_source_quadrature_{phase}") for phase in PHASES)
    # These copies stay synchronized with the physical converter currents, but
    # give KCL a distinct factor whenever its voltage result is multiplied by
    # the physical current in a converter equation. This preserves multi-affinity.
    network_currents = tuple(
        tuple(vf.add_var(f"i_network_{converter}_{phase}") for phase in PHASES)
        for converter in (1, 2)
    )
    # KCL, separately for each phase:
    # [Gsrc + Gline + Gsh1, -Gline          ] [V1] = [Gsrc*Vsrc - I1]
    # [-Gline,               Gline + Gsh2 ] [V2]   [-I2            ]
    nodal_matrix = np.array([
        [source_conductance + line_conductance + bus_1_shunt_conductance,
         -line_conductance],
        [-line_conductance, line_conductance + bus_2_shunt_conductance],
    ])
    nodal_inverse = np.linalg.inv(nodal_matrix)
    bus_voltage_expressions = ([], [])
    for phase in range(len(PHASES)):
        current_rhs = (
            Const(source_conductance) * source_vars[phase] - network_currents[0][phase],
            -network_currents[1][phase],
        )
        for bus in range(2):
            bus_voltage_expressions[bus].append(
                Const(nodal_inverse[bus, 0]) * current_rhs[0]
                + Const(nodal_inverse[bus, 1]) * current_rhs[1]
            )
    voltages = tuple(tuple(bus_voltages) for bus_voltages in bus_voltage_expressions)
    converter_eqs = []
    for bus, block in enumerate(blocks):
        replacements = {block.in_vars[phase]: voltages[bus][phase] for phase in range(len(PHASES))}
        converter_eqs.append([expr.subs(replacements) for expr in block.state_eqs])
    state_eqs = [eq for equations in converter_eqs for eq in equations]
    # Network-current copies track each converter's three physical currents.
    # Keeping distinct symbolic variables prevents a squared-current term when
    # the substituted bus voltage is multiplied by converter current.
    state_eqs.extend(equation for equations in converter_eqs for equation in equations[:3])
    # A rotating source oscillator avoids explicit time-dependent input terms.
    state_eqs.extend(Const(GRID_OMEGA) * value for value in source_quadrature)
    state_eqs.extend(Const(-GRID_OMEGA) * value for value in source_vars)

    added_states = [value for group in network_currents for value in group]
    added_states.extend(source_vars)
    added_states.extend(source_quadrature)
    added_diffs = [vf.add_diff_var(f"dt_1_{state.name}", base_var=state)
                   for state in added_states]

    event_dict = {}
    for block in blocks:
        event_dict.update(block.event_dict)
        for variable in block.in_vars[3:]:
            event_dict[variable] = Const(0.0)
    source_initial = [np.sqrt(2.0) * np.sin(shift) for shift in phase_shifts]
    quadrature_initial = [np.sqrt(2.0) * np.cos(shift) for shift in phase_shifts]
    added_init = {state: Const(0.0) for state in added_states}
    added_init.update(zip(source_vars, map(Const, source_initial)))
    added_init.update(zip(source_quadrature, map(Const, quadrature_initial)))
    combined = Block(
        name="two_gfl_kcl_ode",
        state_vars=[state for block in blocks for state in block.state_vars] + added_states,
        diff_vars=[diff for block in blocks for diff in block.diff_vars] + added_diffs,
        state_eqs=state_eqs,
        algebraic_vars=[], algebraic_eqs=[],
        in_vars=[var for block in blocks for var in block.in_vars[3:]],
        out_vars=[state for block in blocks for state in block.out_vars],
        reformulated_vars=[state for block in blocks for state in block.reformulated_vars],
        event_dict=event_dict,
        init_eqs={**{state: value for block in blocks for state, value in block.init_eqs.items()},
                  **added_init},
        diff_init_eqs={**{diff: value for block in blocks for diff, value in block.diff_init_eqs.items()},
                       **{diff: Const(0.0) for diff in added_diffs}},
    )
    options = EmtOptions(
        time_step=1.0e-5, simulation_time=simulation_time,
        explicit=True,
        # Selects the time-domain residual/Jacobian backend, not the problem
        # class. S/Phi extraction below separately validates multi-affinity.
        solver_type=EmtSolverTypes.Symbolic,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
    )
    problem = EmtModelProblem(combined, inputs={}, options=options, glob_time=time)
    return problem, {"source_vars": source_vars, "network_currents": network_currents,
                     "voltage": voltages, "blocks": blocks,
                     "source_conductance": source_conductance,
                     "line_conductance": line_conductance,
                     "bus_1_shunt_conductance": bus_1_shunt_conductance,
                     "bus_2_shunt_conductance": bus_2_shunt_conductance,
                     "nodal_matrix": nodal_matrix}


def main() -> None:
    """Run timed two-bus setpoint events and save the complete time series."""
    # MODEL: build both converters and the analytically substituted network.
    problem, network = build_problem()
    # These matrices certify that every equation is multi-affine. The solver
    # selected above still runs through the symbolic EMT residual/Jacobian.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    states = problem.get_state_vars()
    S, Phi = problem.build_multilinear_matrices()
    # This small MultiCircuit owns the devices and events. The physical network
    # equations above are still the explicitly substituted KCL equations.
    grid = gce.MultiCircuit(name="Explicit two-GFL event grid", Sbase=100.0, fbase=50.0)
    generators = []
    for index in (1, 2):
        bus = gce.Bus(name=f"bus {index}", Vnom=230.0)
        grid.add_bus(bus)
        generator = gce.Generator(name=f"GFL {index}", P=0.0, Snom=100.0)
        grid.add_generator(bus=bus, api_obj=generator)
        generators.append(generator)
    group = EmtEventsGroup(name="two converter setpoint changes")
    grid.add_emt_events_group(group)
    # Converter 1 changes active power; converter 2 changes reactive power.
    # The last event restores converter 1's original active-power reference.
    event_specs = (
        (0.005, 0, "P_ref", 0.05),
        (0.010, 1, "Q_ref", 0.03),
        (0.015, 0, "P_ref", 0.0),
    )
    for event_time, converter_index, parameter_name, value in event_specs:
        parameter = next(var for var in network["blocks"][converter_index].in_vars
                         if var.name == f"{parameter_name}_{converter_index + 1}")
        grid.add_emt_event(EmtEvent(
            name=f"GFL {converter_index + 1} {parameter_name} to {value:g}",
            device=generators[converter_index], parameter=parameter,
            time=event_time, value=value, group=group,
            force_step_alignment=True,
        ))
    problem.grid = grid
    EmtProblemTemplate.set_events_group(problem, group)
    # State order is converter 1, converter 2, current copies, then source.
    first_network_copy = 2 * len(network["blocks"][0].state_vars)
    tracked_parameters = list(problem.sys_block.in_vars)
    parameter_indices = [problem.uid2idx_event_params[var.uid] for var in tracked_parameters]
    for event_time, converter_index, parameter_name, value in event_specs:
        name = f"{parameter_name}_{converter_index + 1}"
        index = parameter_indices[[var.name for var in tracked_parameters].index(name)]
        actual = problem.def_event_params_fn(
            problem.event_params_values.copy(), event_time + problem.options.time_step
        )[index]
        if not np.isclose(actual, value):
            raise RuntimeError(f"Event did not update {name}: {actual} != {value}")
    # SOLVE: the same native method used by the other top-level demos.
    result = problem.simulate()
    if not result.initialized or not result.converged:
        raise RuntimeError(
            f"EMT simulation failed: initialized={result.initialized}, "
            f"converged={result.converged}"
        )
    times = result.time
    values = result.values.T
    converter_state_count = len(network["blocks"][0].state_vars)
    current_1 = values[0:3]
    current_2 = values[converter_state_count:converter_state_count + 3]
    network_current_1 = values[first_network_copy:first_network_copy + 3]
    network_current_2 = values[first_network_copy + 3:first_network_copy + 6]
    synchronization_error = max(
        np.max(np.abs(current_1 - network_current_1)),
        np.max(np.abs(current_2 - network_current_2)),
    )
    source_start = first_network_copy + 6
    source_trace = values[source_start:source_start + 3]
    # SAVE: include every requested setpoint at each solver output time. The
    # oscillator voltages are also included as inputs, though held as states.
    parameter_trace = np.asarray([
        problem.def_event_params_fn(problem.event_params_values.copy(), float(t))[
            parameter_indices]
        for t in times
    ])
    output_dir = case_dir("two_gfl_kcl_ode")
    save_case_data(
        output_dir, S=S, Phi=Phi, time=times, states=result.values,
        state_names=[state.name for state in states],
        inputs=np.column_stack((source_trace.T, parameter_trace)),
        input_names=[f"source_v_{phase}" for phase in PHASES]
                    + [var.name for var in tracked_parameters],
        basis_names=[state.name for state in states]
                    + [var.name for var in tracked_parameters],
        metadata={"source_conductance": network["source_conductance"],
                  "line_conductance": network["line_conductance"],
                  "event_times": np.asarray([spec[0] for spec in event_specs]),
                  "event_values": np.asarray([spec[3] for spec in event_specs])},
    )
    # Reconstruct voltages from the same KCL system used in build_problem(),
    # using the physical currents to independently check the copy equations.
    current_rhs = np.stack((
        network["source_conductance"] * source_trace - current_1,
        -current_2,
    ))
    bus_voltage = np.linalg.solve(network["nodal_matrix"], current_rhs.reshape(2, -1))
    bus_voltage = bus_voltage.reshape(current_rhs.shape)
    line_current = network["line_conductance"] * (bus_voltage[0] - bus_voltage[1])
    residual_1 = (network["source_conductance"] * (source_trace - bus_voltage[0])
                  - current_1 - line_current
                  - network["bus_1_shunt_conductance"] * bus_voltage[0])
    residual_2 = (line_current - current_2
                  - network["bus_2_shunt_conductance"] * bus_voltage[1])
    residual = np.stack((residual_1, residual_2))
    figure, axes = plt.subplots(5, 1, sharex=True, figsize=(9, 12))
    for phase, label in enumerate("ABC"):
        axes[0].plot(times * 1e3, source_trace[phase], ls="--", alpha=0.5)
        axes[0].plot(times * 1e3, bus_voltage[0, phase], label=f"bus 1 {label}")
        axes[1].plot(times * 1e3, bus_voltage[1, phase], label=f"bus 2 {label}")
        axes[2].plot(times * 1e3,
                     1e3 * (bus_voltage[1, phase] - bus_voltage[0, phase]),
                     label=f"phase {label}")
        axes[3].plot(times * 1e3, current_1[phase], label=f"converter 1 {label}")
        axes[3].plot(times * 1e3, current_2[phase], ls="--", label=f"converter 2 {label}")
    axes[0].set_ylabel("bus 1 voltage (pu)")
    axes[1].set_ylabel("bus 2 voltage (pu)")
    axes[2].set_ylabel("bus 2 − bus 1 (milli-pu)")
    axes[3].set_ylabel("converter current (pu)")
    for column, variable in enumerate(tracked_parameters):
        if variable.name.startswith(("P_ref", "Q_ref")):
            axes[4].step(times * 1e3, parameter_trace[:, column], where="post",
                         label=variable.name)
    axes[4].set_ylabel("P/Q reference (pu)")
    axes[4].set_xlabel("time (ms)")
    for axis in axes:
        for event_time, *_ in event_specs:
            axis.axvline(event_time * 1e3, color="0.6", alpha=0.45, lw=0.8)
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    output = output_dir / "comparison.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"saved={output}, initialized={result.initialized}, "
          f"converged={result.converged}, states={len(states)}, algebraics=0, "
          f"S={S.shape}, Phi={Phi.shape}, "
          f"max_kcl_residual={np.max(np.abs(residual)):.3e}, "
          f"max_copy_error={synchronization_error:.3e}")


if __name__ == "__main__":
    main()
