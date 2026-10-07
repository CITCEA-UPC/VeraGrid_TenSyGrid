"""Compare noise responses of the experimental algebraic-free GFL ODE."""

from __future__ import annotations

from pathlib import Path
import sys

# Allow this demo to run directly by path without installing the source tree.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import numpy as np

from TensyGridEngine.emt import ieee9_emt_ibr_simulation as ieee9
from TensyGridEngine.emt.emt_explicit_converter_demo.gfl_ode_time_inputs import (
    build_gfl_ode_problem,
    dc_voltage_trace,
    runtime_input_traces,
)
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver


OUTPUT = Path(__file__).with_name("ieee9_gfl_ode_noise_comparison.png")
NOISE_SEED = 20260914


def main() -> None:
    grid, gfl_devices, _gfl_buses, _gfm_buses, _gfm_blocks, _validation = (
        ieee9.build_ibr_grid(1, 0, multilinear_inverters=True)
    )
    power_flow = PowerFlowDriver(grid=grid, options=ieee9.build_power_flow_options())
    power_flow.run()
    if not power_flow.results.converged:
        raise RuntimeError("IEEE9 power flow did not converge")

    _generator, bus, _grid_model, (p_mw, q_mvar) = gfl_devices[0]
    bus_index = grid.buses.index(bus)
    voltage = complex(power_flow.results.voltage[bus_index])
    p0 = float(p_mw / grid.Sbase)
    q0 = float(q_mvar / grid.Sbase)

    cases = (
        ("2% independent white phase noise", "white"),
        ("2% frequency-adapted grid noise", "grid_colored"),
    )
    simulations = []
    for title, noise_kind in cases:
        problem = build_gfl_ode_problem(
            multilinear=True,
            p0=p0,
            q0=q0,
            voltage=voltage,
            vdc0=2.0,
            voltage_noise_rms=0.02,
            voltage_noise_kind=noise_kind,
            simulation_time=0.15,
        )
        np.random.seed(NOISE_SEED)
        result = problem.simulate()
        simulations.append((title, problem, result, runtime_input_traces(problem, result.time)))

    figure, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True, constrained_layout=True)
    phase_shift = {"A": 0.0, "B": -2.0 * np.pi / 3.0, "C": 2.0 * np.pi / 3.0}

    metrics = []
    for column, (title, problem, result, input_traces) in enumerate(simulations):
        for phase, color in zip(("A", "B", "C"), ("tab:blue", "tab:orange", "tab:green")):
            clean = np.sqrt(2.0) * abs(voltage) * np.sin(
                2.0 * np.pi * 50.0 * result.time + np.angle(voltage) + phase_shift[phase]
            )
            axes[0, column].plot(
                result.time,
                1e3 * (input_traces[f"vg_{phase}"] - clean),
                color=color,
                label=phase,
            )

        cosine_v = problem.trace(result, "u_cos_v")
        sine_v = problem.trace(result, "u_sin_v")
        cosine_i = problem.trace(result, "u_cos_i")
        sine_i = problem.trace(result, "u_sin_i")
        vg_a, vg_b, vg_c = (input_traces[f"vg_{phase}"] for phase in "ABC")
        ia, ib, ic = (problem.trace(result, f"i_line_{phase}") for phase in "ABC")

        def park(a, b, c, cosine, sine):
            d_axis = (
                2.0 * cosine * a
                + (-cosine + np.sqrt(3.0) * sine) * b
                + (-cosine - np.sqrt(3.0) * sine) * c
            ) / 3.0
            q_axis = (
                2.0 * sine * a
                + (-sine - np.sqrt(3.0) * cosine) * b
                + (-sine + np.sqrt(3.0) * cosine) * c
            ) / 3.0
            return d_axis, q_axis

        vg_d, vg_q = park(vg_a, vg_b, vg_c, cosine_v, sine_v)
        i_d, i_q = park(ia, ib, ic, cosine_i, sine_i)
        p_trace = 0.5 * (vg_q * i_q + vg_d * i_d)
        q_trace = 0.5 * (vg_d * i_q - vg_q * i_d)
        vdc_trace = dc_voltage_trace(problem, result)
        axes[1, column].plot(result.time, 1e3 * (p_trace - p0), color="tab:blue")
        axes[2, column].plot(result.time, 1e3 * (q_trace - q0), color="tab:orange")
        axes[3, column].plot(result.time, 1e3 * (vdc_trace - 2.0), color="tab:green")
        axes[0, column].set_title(title)
        disturbed = result.time >= 0.02
        metrics.append((
            title,
            float(np.sqrt(np.mean((p_trace[disturbed] - p0) ** 2))),
            float(np.sqrt(np.mean((q_trace[disturbed] - q0) ** 2))),
            result,
        ))

    for row, label in enumerate(("Grid disturbance (mpu)", "P - P_ref (mpu)",
                                 "Q - Q_ref (mpu)", "Vdc - Vdc_ref (mpu)")):
        axes[row, 0].set_ylabel(label)
    for column in range(2):
        axes[3, column].set_xlabel("Time (s)")
        axes[0, column].legend(ncol=3)
    for axis in axes.flat:
        axis.grid(alpha=0.3)
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)

    print(
        f"bus={bus.name}, V={abs(voltage):.9f} pu, angle={np.angle(voltage):.9f} rad, "
        f"P0={p0:.9f} pu, Q0={q0:.9f} pu"
    )
    for title, p_rms, q_rms, result in metrics:
        print(
            f"{title}: initialized={result.initialized}, converged={result.converged}, "
            f"P_error_rms={1e3 * p_rms:.6f} mpu, Q_error_rms={1e3 * q_rms:.6f} mpu"
        )
    print(f"plot={OUTPUT}")


if __name__ == "__main__":
    main()
