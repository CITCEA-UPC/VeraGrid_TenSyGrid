"""Run one GFL ODE generator inside the assembled IEEE 9 EMT grid.

This is a full ``MultiCircuit`` simulation. Bus KCL and device equations are
assembled by ``EmtProblemMultilinear``; they are not reduced to an ODE here.

Run with ``PYTHONPATH=src python -m TensyGridEngine.emt.emt_explicit_converter_demo.ieee9_gfl_ode_grid``.
"""

from __future__ import annotations

from TensyGridEngine.emt.emt_ieee9 import (
    attach_emt_models,
    build_emt_options,
    build_ieee9_grid,
    build_power_flow_options,
)
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_multilinear import EmtProblemMultilinear
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Templates.Emt.vsc_gfl_ode import build_gfl_converter_model_ode
from VeraGridEngine.Utils.Symbolic.symbolic import Const
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.enumerations import VarPowerFlowReferenceType


def attach_gfl_ode_generator(grid, generator=None, *, network_current_copies: bool = False):
    """Replace a non-slack machine with the state-only GFL model."""
    if generator is None:
        generator = next(g for g in grid.generators if not g.bus.is_slack)
    if generator.bus.is_slack:
        raise ValueError("Use a voltage source for the slack generator")
    block, _ = build_gfl_converter_model_ode(
        grid.var_factory, multilinear=True, dc_state="stiff_voltage"
    )
    block.name = f"gfl_ode_{generator.name}"

    # Bus voltages are genuine electrical input ports. P and Q references are
    # fixed parameters, not extra bus connections.
    voltage_refs = (
        VarPowerFlowReferenceType.v_A,
        VarPowerFlowReferenceType.v_B,
        VarPowerFlowReferenceType.v_C,
    )
    for variable, reference in zip(block.in_vars[:3], voltage_refs):
        variable.ref = reference
        block.external_mapping[reference] = variable
    for variable, value in zip(block.in_vars[3:], (generator.P / grid.Sbase, 0.0)):
        block.event_dict[variable] = Const(value)
    block.in_vars = block.in_vars[:3]

    # The converter's line current flows from the bus into the converter;
    # the generator injection used by KCL has the opposite sign.
    current_refs = (
        VarPowerFlowReferenceType.i_A,
        VarPowerFlowReferenceType.i_B,
        VarPowerFlowReferenceType.i_C,
    )
    physical_currents = block.out_vars[:3]
    if network_current_copies:
        # The network sees synchronized copies. Once its voltages are
        # substituted, converter power terms then contain distinct factors.
        network_currents = [grid.var_factory.add_var(f"i_network_{generator.name}_{phase}")
                            for phase in "ABC"]
        block.state_vars.extend(network_currents)
        block.diff_vars.extend(
            grid.var_factory.add_diff_var(f"d_{current.name}", base_var=current)
            for current in network_currents
        )
        block.state_eqs.extend(block.state_eqs[:3])
        for copy, physical in zip(network_currents, physical_currents):
            block.init_eqs[copy] = block.init_eqs[physical]
        for differential in block.diff_vars[-3:]:
            block.diff_init_eqs[differential] = Const(0.0)
    else:
        network_currents = physical_currents

    for phase, current, reference in zip("ABC", network_currents, current_refs):
        injection = grid.var_factory.add_var(f"i_gfl_inj_{phase}", reference=reference)
        block.algebraic_vars.append(injection)
        block.algebraic_eqs.append(injection + current)
        block.out_vars.append(injection)
        block.external_mapping[reference] = injection

    set_emt_model(generator, block, grid.var_factory)
    return generator


def build_problem() -> EmtProblemMultilinear:
    grid = build_ieee9_grid()
    attach_emt_models(grid)
    attach_gfl_ode_generator(grid)

    power_flow_options = build_power_flow_options()
    balanced = PowerFlowDriver(grid=grid, options=power_flow_options)
    balanced.run()
    three_phase = PowerFlowDriver3Ph(grid=grid, options=power_flow_options)
    three_phase.run()
    if not balanced.results.converged or not three_phase.results.converged:
        raise RuntimeError("IEEE 9 power flow did not converge")

    return EmtProblemMultilinear(
        grid=grid,
        options=build_emt_options(),
        pf_results=balanced.results,
        pf_results_3ph=three_phase.results,
    )


def main() -> None:
    problem = build_problem()
    options = problem.options
    solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=1.0e-4,
        h=float(options.time_step),
        method=options.integration_method,
    )
    times, _, _, initialized, converged = solver.simulate(boundary_updater=problem)
    print(
        f"states={len(problem.get_state_vars())}, "
        f"algebraics={len(problem.get_algebraic_vars())}, "
        f"samples={len(times)}, initialized={initialized}, converged={converged}"
    )


if __name__ == "__main__":
    main()
