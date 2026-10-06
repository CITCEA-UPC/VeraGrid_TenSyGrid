"""Two explicit GFL converters on a two-bus, algebraically reduced EMT grid.

Run with ``PYTHONPATH=src python -m TensyGridEngine.emt.emt_explicit_converter_demo.two_gfl_kcl_ode``.
Each phase uses the same constant 2x2 nodal conductance matrix. The source and
line conductances make the instantaneous bus-voltage solve nonsingular.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.problems.emt_model_problem import EmtModelProblem
from VeraGridEngine.Templates.Emt.vsc_gfl_ode import build_gfl_converter_model_ode
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Const
from VeraGridEngine.enumerations import (
    DynamicIntegrationMethod, EmtInitializationMethod, EmtSolverTypes,
)


def build_problem(*, source_conductance: float = 10.0,
                  line_conductance: float = 5.0,
                  bus_1_shunt_conductance: float = 0.0,
                  bus_2_shunt_conductance: float = 0.0,
                  simulation_time: float = 1.0e-3) -> tuple[EmtModelProblem, dict]:
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
        variables = [*block.state_vars, *block.diff_vars, *block.in_vars,
                     *block.event_dict]
        for variable in variables:
            variable.name = f"{variable.name}_{index}"
        blocks.append(block)

    phase_shifts = (0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0)
    omega = 2.0 * np.pi * 50.0
    source_vars = tuple(vf.add_var(f"v_source_{phase}") for phase in "ABC")
    source_quadrature = tuple(vf.add_var(f"v_source_quadrature_{phase}") for phase in "ABC")
    # These copies stay synchronized with the physical converter currents, but
    # give KCL a distinct factor whenever its voltage result is multiplied by
    # the physical current in a converter equation. This preserves multi-affinity.
    network_currents = tuple(
        tuple(vf.add_var(f"i_network_{converter}_{phase}") for phase in "ABC")
        for converter in (1, 2)
    )
    nodal_matrix = np.array([
        [source_conductance + line_conductance + bus_1_shunt_conductance,
         -line_conductance],
        [-line_conductance, line_conductance + bus_2_shunt_conductance],
    ])
    nodal_inverse = np.linalg.inv(nodal_matrix)
    voltages = ([], [])
    for phase in range(3):
        current_rhs = (
            Const(source_conductance) * source_vars[phase] - network_currents[0][phase],
            -network_currents[1][phase],
        )
        for bus in range(2):
            voltages[bus].append(
                Const(nodal_inverse[bus, 0]) * current_rhs[0]
                + Const(nodal_inverse[bus, 1]) * current_rhs[1]
            )
    voltages = tuple(tuple(bus_voltages) for bus_voltages in voltages)
    converter_eqs = []
    for bus, block in enumerate(blocks):
        replacements = {block.in_vars[phase]: voltages[bus][phase] for phase in range(3)}
        converter_eqs.append([expr.subs(replacements) for expr in block.state_eqs])
    state_eqs = [eq for equations in converter_eqs for eq in equations]
    state_eqs.extend(converter_eqs[0][:3])
    state_eqs.extend(converter_eqs[1][:3])
    state_eqs.extend(Const(omega) * value for value in source_quadrature)
    state_eqs.extend(Const(-omega) * value for value in source_vars)

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
        algebraic_vars=[], algebraic_eqs=[], in_vars=[],
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
        solver_type=EmtSolverTypes.StructuralAD,
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


if __name__ == "__main__":
    problem, network = build_problem()
    # Build and evaluate VeraGrid's exact EmtProblemMultilinear S/Phi form.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    states = problem.get_state_vars()
    S, Phi = problem.build_multilinear_matrices()
    n_states = len(states)
    n_basis = S.shape[0]
    S_csc = S.tocsc()

    def rhs(y):
        basis = np.zeros(n_basis)
        basis[:n_states] = y
        monomials = np.ones(S.shape[1])
        for column in range(S.shape[1]):
            start, end = S_csc.indptr[column:column + 2]
            for pointer in range(start, end):
                row = S_csc.indices[pointer]
                coefficient = S_csc.data[pointer]
                monomials[column] *= coefficient * basis[row]
        return np.asarray(Phi @ monomials).reshape(-1)

    y0 = np.array([problem.init_guess[state.uid] for state in states])
    # Small phase-A current perturbation reveals the electrical coupling.
    y0[0] = 0.05
    first_network_copy = 2 * len(network["blocks"][0].state_vars)
    y0[first_network_copy] = 0.05
    step = problem.options.time_step
    times = np.arange(0.0, problem.options.simulation_time + 0.5 * step, step)
    values = np.empty((n_states, len(times)))
    values[:, 0] = y0
    for index in range(1, len(times)):
        previous = values[:, index - 1]
        k1 = rhs(previous)
        k2 = rhs(previous + 0.5 * step * k1)
        k3 = rhs(previous + 0.5 * step * k2)
        k4 = rhs(previous + step * k3)
        values[:, index] = previous + step * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    current_1 = values[:3]
    current_2 = values[len(network["blocks"][0].state_vars):][:3]
    network_current_1 = values[first_network_copy:first_network_copy + 3]
    network_current_2 = values[first_network_copy + 3:first_network_copy + 6]
    synchronization_error = max(
        np.max(np.abs(current_1 - network_current_1)),
        np.max(np.abs(current_2 - network_current_2)),
    )
    source_start = first_network_copy + 6
    source_trace = values[source_start:source_start + 3]
    current_rhs = np.stack((
        network["source_conductance"] * source_trace - current_1,
        -current_2,
    ))
    bus_voltage = np.einsum("ij,jpt->ipt", np.linalg.inv(network["nodal_matrix"]),
                            current_rhs)
    line_current = network["line_conductance"] * (bus_voltage[0] - bus_voltage[1])
    residual_1 = (network["source_conductance"] * (source_trace - bus_voltage[0])
                  - current_1 - line_current
                  - network["bus_1_shunt_conductance"] * bus_voltage[0])
    residual_2 = (line_current - current_2
                  - network["bus_2_shunt_conductance"] * bus_voltage[1])
    residual = np.stack((residual_1, residual_2))
    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 10))
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
    axes[3].set_xlabel("time (ms)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    output = Path(__file__).with_name("two_gfl_kcl_ode.png")
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"saved={output}, states={len(states)}, algebraics=0, "
          f"S={S.shape}, Phi={Phi.shape}, "
          f"max_kcl_residual={np.max(np.abs(residual)):.3e}, "
          f"max_copy_error={synchronization_error:.3e}")
