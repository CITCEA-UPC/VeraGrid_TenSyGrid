"""Run and compare voltage disturbances on a grid-following converter.

Beginner's map of this example
------------------------------
1. MODEL: ``main`` builds the IEEE 9-bus grid and runs a power flow. The power
   flow supplies the starting voltage and power of one grid-following (GFL)
   converter. ``build_gfl_ode_problem`` then makes the converter simulation.
   The actual differential equations live in
   ``VeraGridEngine/Templates/Emt/vsc_gfl_ode.py``; their input signals and
   initial conditions are assembled in ``gfl_ode_time_inputs.py``.
   The DC voltage is fixed at 1.03 pu, as in the IEEE9 GFL setup. There is no
   DC energy state or DC power-source input in this comparison.
2. INPUTS/NOISE: ``build_input_expressions`` defines all five converter inputs
   directly as symbolic functions of time. Edit it to change the disturbances
   or power references. ``CASES`` selects which expressions to simulate.
3. SOLVE: ``problem.simulate()`` returns sampled time, model values, and solver
   status. The runs use the same random seed for repeatability.
4. SAVE: ``save_case`` creates a folder per disturbance in
   ``simulation_data/standalone_gfl_ode_ieee9`` with four
   MATLAB data files: ``S.mat``, ``Phi.mat``, ``inputs.mat``, and
   ``states.mat``. The plot is ``OUTPUT_DIR/comparison.png``.
   ``S.mat`` includes ``basis_names``: state rows followed by the five inputs.
   ``inputs.mat`` contains ``time``, ``inputs``, and ``input_names``.
   ``states.mat`` contains ``time``, ``states``, and ``state_names``.
   Column j of ``inputs`` has name ``input_names{j}``; column j of ``states``
   has name ``state_names{j}``. Rows in both arrays follow ``time``.

Run from the repository root with:
    PYTHONPATH=src python -m TensyGridEngine.emt.emt_explicit_converter_demo.standalone_gfl_ode_ieee9
In MATLAB: x = load('random_phase_voltage_changes/states.mat');
           plot(x.time, x.states(:,1))
           x.state_names{1}  % name of the plotted state
"""

from __future__ import annotations

from pathlib import Path
import sys

# Allow this demo to run directly by path without installing the source tree.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from TensyGridEngine.emt import ieee9_emt_ibr_simulation as ieee9
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.gfl_ode_time_inputs import (
    build_gfl_ode_problem,
    runtime_input_traces,
)
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.simulation_data import case_dir
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Var, cos, heaviside, rand, sin


# Each disturbance gets a folder under this standalone simulation.
OUTPUT_DIR = case_dir("standalone_gfl_ode_ieee9")
NOISE_SEED = 20260914       # Makes random disturbances repeatable between runs.
SIMULATION_TIME = 0.15      # Seconds of simulated physical time, not wall time.
FIXED_DC_VOLTAGE_PU = 1.03  # Stiff DC voltage used by the IEEE9 GFL adapter.
DISTURBANCE_START = 0.02    # Seconds; before this, the voltage is clean.
NOISE_SCALE = 0.02          # Change this to control disturbance intensity in every case.

# Each row is (output folder, plot title, disturbance defined below).
CASES = (
    ("random_phase_voltage_changes", "random phase voltage changes", "random"),
    ("fixed_frequency_voltage_changes", "fixed-frequency voltage changes", "fixed_frequencies"),
    ("rotating_dq_voltage_changes", "voltage changes defined in rotating d/q", "rotating_dq"),
    # ("no_voltage_disturbance", "No disturbance", "none"),
)


def build_input_expressions(time: Var, voltage: complex, p0: float, q0: float,
                            disturbance: str, noise_scale: float) -> dict:
    """Return all five converter inputs as symbolic expressions in ``time``.

    The voltage starts as a balanced 50 Hz three-phase signal. The selected
    disturbance changes that signal after DISTURBANCE_START. The last two
    inputs, P_ref and Q_ref, remain constant here; edit their expressions below
    if a system-identification experiment should excite them too. A scale of
    zero gives clean voltage; 0.02 gives the example's nominal 2% intensity.
    The exact RMS voltage disturbance depends on the selected case.
    """
    if noise_scale < 0:
        raise ValueError("noise_scale must be nonnegative")
    peak = np.sqrt(2.0) * abs(voltage)  # Convert RMS bus voltage to sine peak.
    theta = Const(2.0 * np.pi * 50.0) * time + Const(np.angle(voltage))
    one_third_cycle = Const(2.0 * np.pi / 3.0)
    gate = heaviside(time - Const(DISTURBANCE_START))

    # Clean A/B/C voltages. Their phase angles are 120 degrees apart.
    clean_a = Const(peak) * sin(theta)
    clean_b = Const(peak) * sin(theta - one_third_cycle)
    clean_c = Const(peak) * sin(theta + one_third_cycle)

    if disturbance == "random":
        # Independent random variation on each phase. rand(time) is a symbolic
        # random draw evaluated during the simulation. sqrt(12) makes a
        # uniform draw in [-0.5, 0.5] have unit RMS before noise_scale.
        random_scale = Const(np.sqrt(12.0) * noise_scale)
        vg_a = clean_a + gate * random_scale * (rand(time) - Const(0.5))
        vg_b = clean_b + gate * random_scale * (rand(time) - Const(0.5))
        vg_c = clean_c + gate * random_scale * (rand(time) - Const(0.5))

    elif disturbance == "fixed_frequencies":
        # Repeatable modulation at 5, 15, and 35 Hz. The two expressions
        # below change voltage magnitude and electrical angle, respectively.
        cycle_5 = Const(2.0 * np.pi * 5.0) * time
        cycle_15 = Const(2.0 * np.pi * 15.0) * time
        cycle_35 = Const(2.0 * np.pi * 35.0) * time
        normalized_scale = np.sqrt(2.0 / 3.0)
        magnitude_change = Const(noise_scale * normalized_scale) * (
            sin(cycle_5 + Const(0.31))
            + sin(cycle_15 + Const(2.07))
            + sin(cycle_35 + Const(4.19))
        )
        angle_change = Const(0.5 * noise_scale * normalized_scale) * (
            sin(cycle_5 + Const(1.13))
            + sin(cycle_15 + Const(3.41))
            + sin(cycle_35 + Const(5.27))
        )
        disturbed_peak = Const(peak) * (Const(1.0) + gate * magnitude_change)
        shifted_theta = theta + gate * angle_change
        vg_a = disturbed_peak * sin(shifted_theta)
        vg_b = disturbed_peak * sin(shifted_theta - one_third_cycle)
        vg_c = disturbed_peak * sin(shifted_theta + one_third_cycle)

    elif disturbance == "rotating_dq":
        # Define the disturbance in a frame rotating at the nominal 50 Hz.
        # In this model's Park convention, clean voltage has d=0, q=peak.
        # The 5 Hz d-axis and 15 Hz q-axis signals below each have a peak
        # amplitude of noise_scale times the clean phase-voltage peak.
        noise_d = gate * Const(noise_scale * peak) * (
            sin(Const(2.0 * np.pi * 5.0) * time)
        )
        noise_q = gate * Const(noise_scale * peak) * (
            sin(Const(2.0 * np.pi * 15.0) * time)
        )
        vd = noise_d
        vq = Const(peak) + noise_q

        # Inverse Park transform used by this converter. No zero-sequence term
        # is included, so the three resulting voltages always sum to zero.
        cosine = cos(theta)
        sine = sin(theta)
        half = Const(0.5)
        sqrt3_over_2 = Const(np.sqrt(3.0) / 2.0)
        vg_a = vd * cosine + vq * sine
        vg_b = vd * (-half * cosine + sqrt3_over_2 * sine) + vq * (
            -half * sine - sqrt3_over_2 * cosine
        )
        vg_c = vd * (-half * cosine - sqrt3_over_2 * sine) + vq * (
            -half * sine + sqrt3_over_2 * cosine
        )

    elif disturbance == "none":
        vg_a, vg_b, vg_c = clean_a, clean_b, clean_c
    else:
        raise ValueError(f"Unknown disturbance: {disturbance}")

    # This dictionary is the full converter input definition for one run.
    # There is no P_dc_source: the model uses a fixed 1.03 pu DC voltage.
    return {
        "vg_A": vg_a,
        "vg_B": vg_b,
        "vg_C": vg_c,
        "P_ref": Const(p0),
        "Q_ref": Const(q0),
    }


def save_case(folder: Path, problem, result, input_traces):
    """Save each kind of result in its own MATLAB-readable file.

    Both time-series files include time, so either can be loaded alone.
    Names are cell arrays in the same order as their data columns.
    """
    # Explicit S/Phi use state and input rows; derivative rows are omitted.
    # scipy.io.savemat preserves these as MATLAB sparse matrices.
    folder.mkdir(parents=True, exist_ok=True)
    if problem.S is None or problem.Phi is None:
        problem.build_multilinear_matrices()
    basis_names = ([variable.name for variable in problem.get_state_vars()]
                   + [variable.name for variable in problem.sys_block.in_vars])
    savemat(folder / "S.mat", {
        "S": problem.S,
        "basis_names": np.asarray(basis_names, dtype=object).reshape(1, -1),
    }, do_compression=True)
    savemat(folder / "Phi.mat", {"Phi": problem.Phi}, do_compression=True)

    # result.values can contain more than the differential states. Select only
    # state columns, keeping state_names in exactly the same column order.
    state_vars = problem.get_state_vars()
    state_indices = [problem.get_var_idx(variable) for variable in state_vars]
    # Inputs are prescribed model signals (including the three grid voltages).
    # Their sampled traces are replayed on result.time to align every row.
    input_names = [variable.name for variable in problem.sys_block.in_vars]
    # In both files, time is N x 1. Inputs/states have N rows and one column
    # per named variable. Row k always refers to time(k). Names are MATLAB
    # cell arrays whose entries match those columns.
    savemat(folder / "inputs.mat", {
        "time": result.time.reshape(-1, 1),
        "inputs": np.column_stack([input_traces[name] for name in input_names]),
        "input_names": np.asarray(input_names, dtype=object).reshape(1, -1),
    }, do_compression=True)
    savemat(folder / "states.mat", {
        "time": result.time.reshape(-1, 1),
        "states": result.values[:, state_indices],
        "state_names": np.asarray([variable.name for variable in state_vars], dtype=object).reshape(1, -1),
        "fixed_dc_voltage_pu": FIXED_DC_VOLTAGE_PU,
    }, do_compression=True)


def plot_comparison(simulations, voltage: complex, p0: float, q0: float) -> None:
    """Plot the solved cases and print their post-disturbance RMS errors.

    This only reads simulation results; it does not change the model or files
    saved by ``save_case``.
    """
    # ------------------------------------------------------------------
    # RESULTS: Three plot rows per case, one case per column.
    # This section computes display metrics only; the raw traces are already
    # saved in each case's .mat files above.
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(3, len(simulations), figsize=(7.5 * len(simulations), 8),
                                sharex=True, constrained_layout=True, squeeze=False)
    phase_shift = {"A": 0.0, "B": -2.0 * np.pi / 3.0, "C": 2.0 * np.pi / 3.0}
    phase_color = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}

    metrics = []
    for column, (title, problem, result, input_traces) in enumerate(simulations):
        for phase in ("A", "B", "C"):
            # Compare the applied voltage input with an ideal 50 Hz sine wave.
            # A pu RMS voltage has a sqrt(2) times larger sine-wave peak.
            clean = np.sqrt(2.0) * abs(voltage) * np.sin(
                2.0 * np.pi * 50.0 * result.time + np.angle(voltage) + phase_shift[phase]
            )
            axes[0, column].plot(
                result.time,
                1e3 * (input_traces[f"vg_{phase}"] - clean),
                color=phase_color[phase],
                label=phase,
            )

        # These cos/sin states define the converter's rotating voltage and
        # current reference frames. "trace" looks up a state by its name.
        cosine_v = problem.trace(result, "u_cos_v")
        sine_v = problem.trace(result, "u_sin_v")
        cosine_i = problem.trace(result, "u_cos_i")
        sine_i = problem.trace(result, "u_sin_i")
        vg_a, vg_b, vg_c = (input_traces[f"vg_{phase}"] for phase in "ABC")
        ia, ib, ic = (problem.trace(result, f"i_line_{phase}") for phase in "ABC")

        # Park transform: convert three phase values (A/B/C) into two rotating
        # coordinates (d/q). The expressions follow this converter's convention.
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
        # Electrical active/reactive power in pu. The factor 0.5 follows the
        # peak-valued phase signals used by the converter model.
        p_trace = 0.5 * (vg_q * i_q + vg_d * i_d)
        q_trace = 0.5 * (vg_d * i_q - vg_q * i_d)
        # Plot deviations from the initial operating point, multiplied by 1000
        # to report milli-per-unit (mpu): 0.001 pu = 1 mpu.
        axes[1, column].plot(result.time, 1e3 * (p_trace - p0), color="tab:blue")
        axes[2, column].plot(result.time, 1e3 * (q_trace - q0), color="tab:orange")
        axes[0, column].set_title(title)
        # Summarize error only after the noise is switched on at 0.02 s.
        # RMS here means root-mean-square deviation over those time samples.
        disturbed = result.time >= DISTURBANCE_START
        metrics.append((
            title,
            float(np.sqrt(np.mean((p_trace[disturbed] - p0) ** 2))),
            float(np.sqrt(np.mean((q_trace[disturbed] - q0) ** 2))),
            result,
        ))

    for row, label in enumerate(("Grid disturbance (mpu)", "P - P_ref (mpu)",
                                 "Q - Q_ref (mpu)")):
        axes[row, 0].set_ylabel(label)
    for column in range(len(simulations)):
        axes[2, column].set_xlabel("Time (s)")
        axes[0, column].legend(ncol=3)
    for axis in axes.flat:
        axis.grid(alpha=0.3)
    # The .mat files are for analysis in MATLAB; this PNG gives a quick visual
    # comparison without requiring MATLAB.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "comparison.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)


    # Report the RMS errors computed for the plot above.
    for title, p_rms, q_rms, result in metrics:
        print(
            f"{title}: initialized={result.initialized}, converged={result.converged}, "
            f"P_error_rms={1e3 * p_rms:.6f} mpu, Q_error_rms={1e3 * q_rms:.6f} mpu"
        )
    print(f"results={OUTPUT_DIR}, plot={plot_path}")


def main() -> None:
    # ------------------------------------------------------------------
    # MODEL, PART 1: Find a physically consistent starting operating point.
    # ------------------------------------------------------------------
    # IEEE9 is a small nine-bus test grid. The (1, 0) arguments request one
    # grid-following inverter and zero grid-forming inverters. We use the grid
    # only to determine the converter's initial bus voltage and P/Q power.
    grid, gfl_devices, _gfl_buses, _gfm_buses, _gfm_blocks, _validation = (
        ieee9.build_ibr_grid(1, 0, multilinear_inverters=True)
    )
    # Power flow solves steady-state bus voltages before the time-domain ODE.
    power_flow = PowerFlowDriver(grid=grid, options=ieee9.build_power_flow_options())
    power_flow.run()
    if not power_flow.results.converged:
        raise RuntimeError("IEEE9 power flow did not converge")

    # Unused tuple entries have an underscore prefix. The chosen bus tells us
    # which solved voltage belongs to this converter.
    _generator, bus, _grid_model, (p_mw, q_mvar) = gfl_devices[0]
    bus_index = grid.buses.index(bus)
    voltage = complex(power_flow.results.voltage[bus_index])
    # MW and MVAr are divided by the grid's MVA base to obtain per-unit power.
    # The builder accepts voltage, active power p0, and reactive power q0 in pu.
    p0 = float(p_mw / grid.Sbase)
    q0 = float(q_mvar / grid.Sbase)

    # ------------------------------------------------------------------
    # MODEL, PART 2 + NOISE + SOLVE + SAVE: Repeat for each selected case.
    # ------------------------------------------------------------------
    simulations = []  # Keep results in memory for the comparison plot below.
    for name, description, disturbance in CASES:
        # Build symbolic expressions first, using one time variable shared by
        # the input dictionary and the simulation problem.
        time = Var("gfl_ode_time")
        inputs = build_input_expressions(
            time, voltage, p0, q0, disturbance, noise_scale=NOISE_SCALE
        )
        # This builds a standalone converter ODE; it does not simulate all nine
        # buses dynamically. The bus power-flow result supplies its starting
        # point, while prescribed voltage inputs drive it during the run.
        # The DC voltage is fixed, matching the IEEE9 GFL adapter. It is not
        # part of the time-varying input or state arrays. multilinear=True
        # enables extraction of the S and Phi matrices saved below.
        problem = build_gfl_ode_problem(
            multilinear=True,
            p0=p0,
            q0=q0,
            voltage=voltage,
            vdc0=FIXED_DC_VOLTAGE_PU,
            dc_state="stiff_voltage",
            prescribed_inputs=inputs,
            time_variable=time,
            simulation_time=SIMULATION_TIME,
        )
        # Reset the seed so repeated executions of this script are comparable.
        # Fixed-frequency cases contain no randomness, so the seed has no effect.
        np.random.seed(NOISE_SEED)
        result = problem.simulate()
        # Reconstruct the input values at the same times as the solved states.
        input_traces = runtime_input_traces(problem, result.time, seed=NOISE_SEED)
        save_case(OUTPUT_DIR / name, problem, result, input_traces)
        title = f"{100 * NOISE_SCALE:g}% {description}"
        simulations.append((title, problem, result, input_traces))

    print(
        f"bus={bus.name}, V={abs(voltage):.9f} pu, angle={np.angle(voltage):.9f} rad, "
        f"P0={p0:.9f} pu, Q0={q0:.9f} pu"
    )
    plot_comparison(simulations, voltage, p0, q0)


if __name__ == "__main__":
    main()
