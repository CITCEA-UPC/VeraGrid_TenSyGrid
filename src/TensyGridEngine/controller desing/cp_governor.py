# SPDX-License-Identifier: MPL-2.0
"""Factored symbolic governor exercise using VeraGrid's native RMS DAE tools."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys

# Always import the engine from this checkout, even when run from another folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import VeraGridEngine.api as vge
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Expr
from VeraGridEngine.Utils.Symbolic.bus_rms_template import initialize_bus_rms
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model
from VeraGridEngine.Templates.Rms.gencls_rms_template import (
    get_gencls_rms_template, configure_gencls_block,
    GenclsMechanicalInput, GenclsRmsParameters,
)
from VeraGridEngine.Templates.Rms.line_rms_template import get_line_rms_template
from VeraGridEngine.Templates.Rms.load_rms_template import get_load_rms_template
from VeraGridEngine.Simulations.Rms.problems.rms_problem_dae import RmsProblemDae
from VeraGridEngine.Simulations.Rms.rms_driver import RmsSimulationDriver
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import SmallSignalStabilityRmsDriver
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_options import RmsSmallSignalStabilityOptions


@dataclass(frozen=True)
class Parameters:
    Tg: float = 0.2
    Tt: float = 0.5
    K: float = 20.0
    alpha: float = 1.0
    H: float = 3.5
    D: float = 1.0
    c_ev: float = 0.0
    c_et: float = 0.0
    c_evt: float = 0.0
    c_vt: float = 0.0

    def __post_init__(self):
        if not all(np.isfinite(x) for x in vars(self).values()):
            raise ValueError("Parameters must be finite")
        if min(self.Tg, self.Tt, self.H) <= 0 or min(self.K, self.D) < 0:
            raise ValueError("Positive time constants/inertia and nonnegative gain/damping required")


@dataclass
class SymbolicCPTensor:
    """CP coefficient tensor with numeric or VeraGrid-symbolic coefficients.

    Output mode first, followed by binary [1, z_i] variable modes. Coefficients
    may depend on held parameters, but must not depend on any variable mode.
    This is an authoring representation; VeraGrid evaluates the built Block.
    """
    rank: int
    weights: np.ndarray
    output_factor: np.ndarray
    factors: list[np.ndarray]

    @classmethod
    def empty(cls, n_variables: int, n_equations: int, rank: int):
        if any(not isinstance(n, int) or n <= 0 for n in (n_variables, n_equations, rank)):
            raise ValueError("Variable count, equation count and rank must be positive integers")
        # Unused variable factors represent 1. Output coefficients start at zero.
        factors = [np.array([[1] * rank, [0] * rank], dtype=object)
                   for _ in range(n_variables)]
        return cls(rank, np.ones(rank, dtype=object),
                   np.zeros((n_equations, rank), dtype=object), factors)

    def expressions(self, variables):
        """Contract CP factors with [1, variable] without monomial expansion."""
        if (self.weights.shape != (self.rank,) or self.output_factor.ndim != 2
                or self.output_factor.shape[1] != self.rank
                or len(self.factors) != len(variables)
                or any(f.shape != (2, self.rank) for f in self.factors)):
            raise ValueError("Inconsistent CP factor dimensions")
        mode_uids = {v.uid for v in variables}
        if len(mode_uids) != len(variables):
            raise ValueError("Each CP variable mode must be distinct")
        for array in [self.weights, self.output_factor, *self.factors]:
            for coefficient in array.flat:
                if isinstance(coefficient, Expr) and any(v.uid in mode_uids for v in coefficient.get_vars()):
                    raise ValueError("CP coefficients must be independent of the variable modes")
        terms = []
        for r in range(self.rank):
            term = Const(1) * self.weights[r]
            for variable, factor in zip(variables, self.factors):
                term = term * (Const(1) * factor[0, r] + factor[1, r] * variable)
            terms.append(term)
        return [sum((coefficient * term for coefficient, term in zip(row, terms)), Const(0)).simplify()
                for row in self.output_factor]


def block_from_cp(*, name, n_states, states, inputs, outputs, tensor_variables,
                  tensor, parameters, initial_values) -> Block:
    """Build native equations: first n_states CP rows are RHS, rest are outputs."""
    if n_states != len(states) or n_states <= 0:
        raise ValueError("State count must match the declared state variables")
    if tensor.output_factor.shape[0] != n_states + len(outputs):
        raise ValueError("CP output rows must match state derivatives plus output ports")
    expressions = tensor.expressions(tensor_variables)
    block = Block(name=name)
    block.state_vars = list(states)
    block.state_eqs = expressions[:n_states]
    block.algebraic_vars = list(outputs)
    block.algebraic_eqs = [var - expr for var, expr in zip(outputs, expressions[n_states:])]
    block.in_vars = list(inputs)
    block.out_vars = list(outputs)
    block.event_dict = dict(parameters)
    block.init_eqs = dict(initial_values)
    return block


def governor(vf: VarFactory, p: Parameters) -> Block:
    """Build two states from eight CP components, including nonlinear interactions."""
    # Step 1: declare the controller dimensions, states and ports.
    n_states = 2
    cp_rank = 8
    states = [vf.add_var(name) for name in ("valve", "turbine_torque")]
    v, tau = states
    omega = vf.add_var("omega_in", shared_reference="omega_reference")
    te = vf.add_var("Te_init", shared_reference="te_reference")
    tm = vf.add_var("Tm_gov", shared_reference="tm_reference")
    Tg, Tt, K, alpha, tref = [vf.add_var(name) for name in ("Tg", "Tt", "K", "alpha", "Tref")]
    parameters = {Tg: Const(p.Tg), Tt: Const(p.Tt), K: Const(p.K),
                  alpha: Const(p.alpha), tref: te}
    interaction_gains = [vf.add_var(name) for name in ("c_ev", "c_et", "c_evt", "c_vt")]
    parameters.update({var: Const(value) for var, value in zip(
        interaction_gains, (p.c_ev, p.c_et, p.c_evt, p.c_vt))})

    # Step 2: define the coefficient tensor in CP form, with shape (3, 2, 2, 2).
    # Variable modes: [v, tau, omega]. Output rows: [dv/dt, dtau/dt, Tm].
    # Columns: Tref, v, (omega-1)*(1+alpha*(v-Tref)), tau.
    # Tref is a held parameter, not a variable mode; coefficients may contain it.
    tensor = SymbolicCPTensor.empty(n_variables=3, n_equations=n_states + 1, rank=cp_rank)
    tensor.weights[:4] = [tref, 1, 1, 1]
    tensor.factors[0][:, :4] = [[1, 0, 1 - alpha * tref, 1], [0, 1, alpha, 0]]
    tensor.factors[1][:, :4] = [[1, 1, 1, 0], [0, 0, 0, 1]]
    tensor.factors[2][:, :4] = [[1, 1, -1, 1], [0, 0, 1, 0]]
    tensor.output_factor[:, :4] = [
        [1 / Tg, -1 / Tg, -K / Tg, 0],
        [0, 1 / Tt, 0, -1 / Tt],
        [0, 0, 0, 1],
    ]
    # Extra multilinear features: e*a, e*b, e*a*b, a*b, with
    # e=(1-omega)/0.005, a=(v-Tref)/0.1, b=(tau-Tref)/0.1.
    # Each column is a product of distinct variable modes, never a square.
    for column, active_modes in enumerate(((0, 2), (1, 2), (0, 1, 2), (0, 1)), start=4):
        for mode in active_modes:
            tensor.factors[mode][:, column] = ([200, -200] if mode == 2 else [-10 * tref, 10])
        tensor.output_factor[0, column] = interaction_gains[column - 4] / Tg

    # Step 3: contract the CP definition into ordinary VeraGrid expressions.
    return block_from_cp(name="Factored governor", n_states=n_states, states=states,
                         inputs=[omega, te], outputs=[tm], tensor_variables=[v, tau, omega],
                         tensor=tensor, parameters=parameters,
                         initial_values={v: te, tau: te, tm: te})


def build_grid(p=Parameters(), step=0.05, event_time=2.0, controller_factory=governor):
    """Native two-bus circuit, GENCLS, governor, line, load and RMS event."""
    if not np.isfinite(step) or step <= -0.8 or event_time <= 0:
        raise ValueError("Finite load step above -0.8 pu and positive event time required")
    grid = vge.MultiCircuit(name="Factored governor exercise", Sbase=100., fbase=50.)
    gen_bus = vge.Bus(name="Generator bus", Vnom=20., is_slack=True)
    load_bus = vge.Bus(name="Load bus", Vnom=20.)
    for bus in (gen_bus, load_bus):
        grid.add_bus(bus)
        initialize_bus_rms(bus, vf=grid.var_factory)
    line = vge.Line(name="Feeder", bus_from=gen_bus, bus_to=load_bus,
                    r=0.01, x=0.1, b=0., rate=100.)
    grid.add_line(line)
    load = vge.Load(name="Demand", P=80., Q=10.)
    grid.add_load(bus=load_bus, api_obj=load)
    gen = vge.Generator(name="Generator", P=80., vset=1., Snom=100., freq=50.)
    grid.add_generator(bus=gen_bus, api_obj=gen)

    vf = grid.var_factory
    # On the machine-only factory, TGOV1 selects an open Tm input. It does not
    # build a TGOV1 controller; the factored governor below supplies that port.
    model = get_gencls_rms_template(vf, mechanical_input=GenclsMechanicalInput.TGOV1,
                                    name="GENCLS with factored governor").block
    machine = model.children[0]
    configure_gencls_block(machine, GenclsRmsParameters(fn=50., d=p.D, m=2 * p.H,
                                                       ra=0., xd1=0.3), vf)
    controller = controller_factory(vf, p)
    vf.add_connection(machine.in_vars[2], controller.out_vars[0])
    vf.add_connection(controller.in_vars[0], machine.out_vars[2])
    vf.add_connection(controller.in_vars[1], machine.out_vars[3])
    model.children.append(controller)
    set_rms_model(device=gen, model=model, var_factory=vf)
    set_rms_model(device=line, model=get_line_rms_template(vf).block, var_factory=vf)
    load_model = get_load_rms_template(vf).block
    set_rms_model(device=load, model=load_model, var_factory=vf)

    group = vge.RmsEventsGroup(name="Load increase")
    grid.add_rms_events_group(group)
    load_parameter = next(var for var in load_model.children[0].event_dict if var.name == "Pl0")
    # Loads use negative injection in VeraGrid RMS; the event sets an absolute value.
    grid.add_rms_event(vge.RmsEvent(device=load, parameter=load_parameter,
                                   time=event_time, value=-(load.P / grid.Sbase + step),
                                   group=group))
    return grid, machine, controller


def options(duration=20., dt=0.01):
    return vge.RmsOptions(time_step=dt, simulation_time=duration, tolerance=1e-8,
                          max_iter=50, problem_type=vge.RmsProblemTypes.PowerBalance,
                          initialization_method=vge.RmsInitializationMethod.Explicit,
                          integration_method=vge.DynamicIntegrationMethod.DaeBackEuler)


def run_study(p=Parameters(), step=0.05, duration=20., dt=0.01, event_time=2.,
              controller_factory=governor):
    if not np.isfinite(duration) or not np.isfinite(dt) or duration <= event_time or dt <= 0:
        raise ValueError("Duration must exceed event time, with a positive finite time step")
    grid, machine, controller = build_grid(p, step, event_time, controller_factory)
    pf = vge.power_flow(grid, vge.PowerFlowOptions(tolerance=1e-9))
    if not pf.converged:
        raise RuntimeError("VeraGrid power flow did not converge")
    opts = options(duration, dt)
    modal = SmallSignalStabilityRmsDriver(
        grid=grid, rms_options=opts, pf_results=pf,
        sss_options=RmsSmallSignalStabilityOptions(k=0, ss_assessment_time=0.))
    modal.run()
    rms = RmsSimulationDriver(grid=grid, options=opts, pf_results=pf)
    rms.run()
    if not isinstance(rms.problem, RmsProblemDae):
        raise RuntimeError("Expected the normal RMS DAE problem")
    if rms.results is None or not np.all(rms.results.well_initialized) or not np.all(rms.results.converged):
        raise RuntimeError(f"RMS initialization/integration failed: {rms.logger}")
    return grid, machine, controller, pf, modal, rms


def export_study(study, output, p, step, event_time):
    grid, machine, controller, pf, modal, rms = study
    output.mkdir(parents=True, exist_ok=True)
    result = rms.results
    time = (result.time_array - result.time_array[0]).total_seconds().to_numpy()
    def series(var):
        return result.values[:, result.uid2idx[var.uid], 0]
    omega = series(machine.out_vars[2])
    valve = series(controller.state_vars[0])
    torque = series(controller.out_vars[0])
    power = torque * omega
    voltage = series(grid.buses[1].rms_model.external_mapping[vge.VarPowerFlowReferenceType.Vm])
    data = np.column_stack((time, omega, valve, torque, power, voltage))
    np.savetxt(output / "response.csv", data, delimiter=",",
               header="time_s,omega_pu,valve_pu,Tm_pu,Pm_pu,load_voltage_pu", comments="")
    eigenvalues = modal.results.eigenvalues
    np.savetxt(output / "eigenvalues.csv", np.column_stack((eigenvalues.real, eigenvalues.imag)),
               delimiter=",", header="real_per_s,imag_rad_per_s", comments="")
    np.savetxt(output / "state_matrix.csv", modal.results.state_matrix, delimiter=",")
    (output / "governor_equations.txt").write_text("\n".join(
        f"d({v.name})/dt = {eq}" for v, eq in zip(controller.state_vars, controller.state_eqs)))
    initial = modal.problem.get_x0()
    dx = modal.problem.get_dx0()
    residual = max(np.linalg.norm(modal.problem.rhs_state(initial, dx), np.inf),
                   np.linalg.norm(modal.problem.rhs_algebraic(initial, dx), np.inf))
    report = (
        f"VeraGrid GENCLS + factored symbolic governor; RmsProblemDae\n"
        f"Parameters: {p}\nLoad step: {step} pu at {event_time} s\n"
        f"Power flow converged: {pf.converged}\nInitial residual: {residual:.3e}\n"
        f"RMS initialized: {result.well_initialized}; converged: {result.converged}\n"
        f"Initial eigenvalues [1/s]: {eigenvalues}\n"
        f"Sampled frequency nadir: {50 * omega.min():.6f} Hz\n"
        f"Frequency at {time[-1]:g} s: {50 * omega[-1]:.6f} Hz\n"
        "A near-zero mode is the island's arbitrary common angle.\n"
    )
    (output / "study.txt").write_text(report)
    print(report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, layout="constrained")
    axes[0].plot(time, 50 * omega)
    axes[0].set_ylabel("Frequency [Hz]")
    axes[1].plot(time, valve, label="Valve")
    axes[1].plot(time, torque, label="Mechanical torque")
    axes[1].plot(time, power, "--", label="Mechanical power")
    axes[1].set_ylabel("Per unit")
    axes[1].legend()
    axes[2].plot(time, voltage)
    axes[2].set(xlabel="Time [s]", ylabel="Load voltage [pu]")
    for ax in axes:
        ax.axvline(event_time, color="gray", ls=":")
        ax.grid(alpha=0.25)
    fig.suptitle(f"VeraGrid RMS DAE | GENCLS + governor | alpha={p.alpha:g}")
    fig.savefig(output / "response.png", dpi=160)
    plt.close(fig)
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--parameters", type=Path, help="Load a tuned Parameters JSON file")
    parser.add_argument("--step", type=float, default=0.05, help="Load increase in pu on 100 MVA")
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    p = Parameters(**json.loads(args.parameters.read_text())) if args.parameters else Parameters()
    if args.alpha is not None:
        p = replace(p, alpha=args.alpha)
    export_study(run_study(p, args.step, args.duration, args.dt), args.output, p, args.step, 2.)
