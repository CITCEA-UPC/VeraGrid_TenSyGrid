# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np
from scipy.io import savemat

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def ensure_repo_import_paths() -> None:
    """
    Add VeraGrid local source folders to ``sys.path``.

    :return: ``None``.
    """
    repo_root: Path = Path(__file__).resolve().parents[3]
    path_item: Path
    path_text: str

    for path_item in (repo_root / "src", repo_root / "trunk", repo_root):
        path_text = str(path_item)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
        else:
            pass


ensure_repo_import_paths()

import VeraGridEngine.api as gce
from VeraGridEngine.Devices.Events.emt_event import EmtEvent
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowOptions
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Devices.Dynamic.static_parameter_mapping import converter_control_code
from VeraGridEngine.Templates.Emt.converter_emt_template import get_full_pseudo_emt_converter
from VeraGridEngine.Templates.Emt.dc_load_emt_template import get_dc_load_emt_template
from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
from VeraGridEngine.Templates.Emt.thevenin_equivalent_emt_generator_template import (
    get_generator_thevenin_rl_emt_template_with_ref,
)
from VeraGridEngine.Utils.Symbolic.block import Block, Var, find_name_in_block
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.Utils.procedural_logic import build_boundary_updater_from_block
from VeraGridEngine.enumerations import (
    BranchImpedanceMode,
    ConverterControlType,
    DynamicIntegrationMethod,
    EmtInitializationMethod,
    EmtSolverTypes,
    SolverType,
    VarPowerFlowReferenceType,
)


class GridCase:
    """
    Container for the physical grid objects used by the identification case.

    :param grid: VeraGrid ``MultiCircuit``.
    :param bus_slack: AC Thevenin source bus.
    :param bus_pcc: Converter AC terminal bus.
    :param bus_dc: Converter DC terminal bus.
    :param generator: Thevenin source device.
    :param line: Optional AC line between source and PCC.
    :param converter: VSC device.
    :param dc_load: DC load device.
    :return: ``None``.
    """

    __slots__ = ["grid", "bus_slack", "bus_pcc", "bus_dc", "generator", "line", "converter", "dc_load"]

    def __init__(
        self,
        grid: gce.MultiCircuit,
        bus_slack: gce.Bus,
        bus_pcc: gce.Bus,
        bus_dc: gce.Bus,
        generator: gce.Generator,
        line: gce.Line | None,
        converter: gce.VSC,
        dc_load: gce.Load,
    ) -> None:
        self.grid: gce.MultiCircuit = grid
        self.bus_slack: gce.Bus = bus_slack
        self.bus_pcc: gce.Bus = bus_pcc
        self.bus_dc: gce.Bus = bus_dc
        self.generator: gce.Generator = generator
        self.line: gce.Line | None = line
        self.converter: gce.VSC = converter
        self.dc_load: gce.Load = dc_load


def build_multicircuit_grid() -> GridCase:
    """
    Build the EMT identification grid as a real VeraGrid ``MultiCircuit``.

    The topology is AC Thevenin source -> damped line -> PCC -> VSC -> DC bus -> DC load.

    :return: Grid case with all physical devices.
    """
    grid: gce.MultiCircuit = gce.MultiCircuit(name="TensyGrid EMT identification grid", Sbase=100.0, fbase=50.0)
    bus_slack: gce.Bus = gce.Bus(name="Bus_Source", Vnom=20.0, is_slack=True)
    bus_pcc: gce.Bus = gce.Bus(name="Bus_PCC", Vnom=20.0)
    bus_dc: gce.Bus = gce.Bus(name="Bus_DC", Vnom=3.0, is_dc=True)
    generator: gce.Generator = gce.Generator(name="Thevenin_Source", vset=1.0, Snom=100.0, freq=50.0, r1=0.005, x1=0.050)
    line: gce.Line | None = gce.Line(
        name="Source_to_PCC_Line",
        bus_from=bus_slack,
        bus_to=bus_pcc,
        r=0.500,
        x=0.0005,
        b=0.0,
        rate=10.0,
    )
    converter: gce.VSC = gce.VSC(
        name="pseudo_converter_emt",
        bus_from=bus_dc,
        bus_to=bus_pcc,
        rate=1.0,
        control1=ConverterControlType.Qac,
        control2=ConverterControlType.Vm_dc,
        control1_val=0.0,
        control2_val=1.0,
    )
    dc_load: gce.Load = gce.Load(name="DC_Load", P=0.8, Q=0.0)

    grid.add_bus(bus_slack)
    if bus_pcc is bus_slack:
        pass
    else:
        grid.add_bus(bus_pcc)
    grid.add_bus(bus_dc)
    grid.add_generator(bus=bus_slack, api_obj=generator)
    if line is None:
        pass
    else:
        grid.add_line(line)
    grid.add_vsc(converter)
    grid.add_load(bus=bus_dc, api_obj=dc_load)

    converter.control1 = ConverterControlType.Pdc
    converter.control2 = ConverterControlType.Qac
    converter.control1_val = dc_load.P
    converter.control2_val = dc_load.Q

    return GridCase(
        grid=grid,
        bus_slack=bus_slack,
        bus_pcc=bus_pcc,
        bus_dc=bus_dc,
        generator=generator,
        line=line,
        converter=converter,
        dc_load=dc_load,
    )


def attach_emt_models(case: GridCase) -> Block:
    """
    Attach EMT templates to every bus and physical device in the case.

    :param case: MultiCircuit case to populate with EMT models.
    :return: Attached converter EMT block.
    """
    bus: gce.Bus
    grid: gce.MultiCircuit = case.grid
    converter_model: Block = get_full_pseudo_emt_converter(vf=grid.var_factory, name=case.converter.name).block
    generator_model: Block = get_generator_thevenin_rl_emt_template_with_ref(vf=grid.var_factory).block
    dc_load_model: Block = get_dc_load_emt_template(vf=grid.var_factory, name="dc_load_emt_identification").block
    initial_transfer: float = float(case.dc_load.P)
    initial_reactive: float = float(case.dc_load.Q)

    for bus in grid.buses:
        get_bus_emt_template(grid=grid, bus=bus)

    converter_model.set_parameter_in_model(var_name="sbase", new_value=grid.Sbase)
    converter_model.set_parameter_in_model(var_name="omega_base", new_value=2.0 * np.pi * grid.fBase)
    converter_model.set_parameter_in_model(var_name="control1", new_value=converter_control_code(ConverterControlType.Pdc))
    converter_model.set_parameter_in_model(var_name="control2", new_value=converter_control_code(ConverterControlType.Qac))
    converter_model.set_parameter_in_model(var_name="control1_val", new_value=initial_transfer)
    converter_model.set_parameter_in_model(var_name="control2_val", new_value=initial_reactive)
    converter_model.set_parameter_in_model(var_name="P0", new_value=initial_transfer)
    converter_model.set_parameter_in_model(var_name="Vdc_nom", new_value=1.0)
    converter_model.set_parameter_in_model(var_name="R_eq", new_value=0.03)
    converter_model.set_parameter_in_model(var_name="L_eq", new_value=0.12)
    converter_model.set_parameter_in_model(var_name="C_dc", new_value=0.08)
    converter_model.set_parameter_in_model(var_name="R_dc", new_value=1.0e6)
    converter_model.set_parameter_in_model(var_name="R_dc_term", new_value=1.0e-4)
    converter_model.set_parameter_in_model(var_name="pll_kp", new_value=40.0)
    converter_model.set_parameter_in_model(var_name="pll_ki", new_value=400.0)
    converter_model.set_parameter_in_model(var_name="i_kp", new_value=0.6)
    converter_model.set_parameter_in_model(var_name="i_ki", new_value=50.0)
    converter_model.set_parameter_in_model(var_name="vdc_kp", new_value=2.0)
    converter_model.set_parameter_in_model(var_name="vdc_ki", new_value=30.0)
    converter_model.set_parameter_in_model(var_name="q_kp", new_value=0.8)
    converter_model.set_parameter_in_model(var_name="q_ki", new_value=20.0)
    converter_model.set_parameter_in_model(var_name="i_max", new_value=1.3)
    converter_model.set_parameter_in_model(var_name="m_max", new_value=0.95)
    converter_model.set_parameter_in_model(var_name="P_loss_i1", new_value=0.0)
    converter_model.set_parameter_in_model(var_name="P_loss_i2", new_value=0.0)
    converter_model.set_parameter_in_model(var_name="tau_meas", new_value=0.01)
    converter_model.set_parameter_in_model(var_name="aw_gain", new_value=0.1)
    converter_model.set_parameter_in_model(var_name="vdc_floor", new_value=0.05)

    set_emt_model(
        device=case.generator,
        model=generator_model,
        var_factory=grid.var_factory,
    )
    if case.line is None:
        pass
    else:
        set_emt_model(
            device=case.line,
            model=get_pi_line_emt_template(vf=grid.var_factory, phN=False, phA=True, phB=True, phC=True).block,
            var_factory=grid.var_factory,
        )
    set_emt_model(device=case.converter, model=converter_model, var_factory=grid.var_factory)
    set_emt_model(
        device=case.dc_load,
        model=dc_load_model,
        var_factory=grid.var_factory,
    )

    add_identification_events(case=case, generator_model=generator_model)

    return converter_model


def add_identification_events(case: GridCase, generator_model: Block) -> None:
    """
    Add controlled EMT events to excite the converter through the physical grid.

    :param case: MultiCircuit case receiving EMT events.
    :param generator_model: Thevenin generator EMT model containing runtime grid perturbation variables.
    :return: ``None``.
    """
    e_scale_var: Var | None = None
    theta_deviation_var: Var | None = None
    frequency_scale_var: Var | None = None
    event_var: Var

    for event_var in generator_model.event_dict.keys():
        if event_var.name == "E_scale":
            e_scale_var = event_var
        elif event_var.name == "theta_deviation_param":
            theta_deviation_var = event_var
        elif event_var.name == "f_scale":
            frequency_scale_var = event_var
        else:
            pass

    if e_scale_var is None:
        raise KeyError("Could not find generator E_scale runtime variable for EMT events")
    else:
        pass

    if theta_deviation_var is None:
        raise KeyError("Could not find generator theta_deviation_param runtime variable for EMT events")
    else:
        pass

    if frequency_scale_var is None:
        raise KeyError("Could not find generator f_scale runtime variable for EMT events")
    else:
        pass

    case.grid.add_emt_event(EmtEvent(device=case.generator, parameter=e_scale_var, time=0.010, value=0.950))


def build_power_flow_options() -> PowerFlowOptions:
    """
    Build local power-flow options for the EMT identification case.

    :return: VeraGrid power-flow options.
    """
    return PowerFlowOptions(
        solver_type=SolverType.NR,
        retry_with_other_methods=False,
        verbose=0,
        initialize_with_existing_solution=True,
        tolerance=1.0e-6,
        max_iter=30,
        control_q=False,
        control_taps_modules=True,
        control_taps_phase=True,
        control_remote_voltage=True,
        orthogonalize_controls=True,
        apply_temperature_correction=True,
        branch_impedance_tolerance_mode=BranchImpedanceMode.Specified,
        distributed_slack=False,
        ignore_single_node_islands=False,
        trust_radius=1.0,
        backtracking_parameter=0.05,
        use_stored_guess=False,
        initialize_angles=False,
        generate_report=False,
    )


def build_options() -> EmtOptions:
    """
    Build EMT options for the MultiCircuit identification simulation.

    :return: EMT options.
    """
    options: EmtOptions = EmtOptions(
        time_step=1.0e-5,
        simulation_time=0.050,
        tolerance=1.0e-6,
        solver_type=EmtSolverTypes.StructuralCompiled,
        integration_method=DynamicIntegrationMethod.DaeBackEuler,
        initialization_method=EmtInitializationMethod.Explicit,
        verbose=0,
    )
    options.newton_max_iter = 30
    options.init_newton_max_iter = 30
    return options


def get_signal(problem: EmtProblemDae, trajectory: np.ndarray, variable_name: str) -> np.ndarray:
    """
    Extract a named variable from an EMT trajectory.

    :param problem: EMT DAE problem.
    :param trajectory: Solver trajectory matrix.
    :param variable_name: Exact symbolic variable name.
    :return: One-dimensional signal vector.
    """
    variable: Var | None = find_name_in_block(variable_name, problem.sys_block)
    if variable is None:
        raise KeyError(f"Variable '{variable_name}' was not found")
    else:
        pass
    return trajectory[:, int(problem.get_var_idx(variable))]


def get_signal_from_var(problem: EmtProblemDae, trajectory: np.ndarray, variable: Var) -> np.ndarray:
    """
    Extract a variable object from an EMT trajectory.

    :param problem: EMT DAE problem.
    :param trajectory: Solver trajectory matrix.
    :param variable: Symbolic variable object present in the DAE.
    :return: One-dimensional signal vector.
    """
    return trajectory[:, int(problem.get_var_idx(variable))]


def get_signal_from_block(problem: EmtProblemDae, trajectory: np.ndarray, block: Block, variable_name: str) -> np.ndarray:
    """
    Extract a named variable from one EMT block.

    :param problem: EMT DAE problem.
    :param trajectory: Solver trajectory matrix.
    :param block: EMT block to search.
    :param variable_name: Variable name to extract.
    :return: One-dimensional signal vector.
    """
    variable: Var | None = find_name_in_block(variable_name, block)
    if variable is None:
        raise KeyError(f"Variable '{variable_name}' was not found in block '{block.name}'")
    else:
        pass
    return get_signal_from_var(problem=problem, trajectory=trajectory, variable=variable)


def collect_state_matrix(problem: EmtProblemDae, trajectory: np.ndarray, case: GridCase) -> np.ndarray:
    """
    Collect converter state variables for identification.

    :param problem: EMT DAE problem.
    :param trajectory: Solver trajectory matrix.
    :return: State matrix with one state per column.
    """
    state_vars: list[Var] = list(case.converter.emt_model.state_vars)
    selected_vars: list[Var] = list()
    var: Var

    for var in state_vars:
        selected_vars.append(var)

    if len(selected_vars) == 0:
        raise RuntimeError("No converter state variables were found in the EMT problem")
    else:
        pass

    return np.column_stack([trajectory[:, int(problem.get_var_idx(var))] for var in selected_vars])


def collect_input_matrix(problem: EmtProblemDae, trajectory: np.ndarray, case: GridCase) -> np.ndarray:
    """
    Collect the converter boundary inputs from the solved MultiCircuit EMT grid.

    :param problem: EMT DAE problem.
    :param trajectory: Solver trajectory matrix.
    :param case: Grid case containing bus names.
    :return: Input matrix ``[v_A_pcc, v_B_pcc, v_C_pcc, v_DC]``.
    """
    v_a: Var = case.bus_pcc.emt_model.external_mapping[VarPowerFlowReferenceType.v_A]
    v_b: Var = case.bus_pcc.emt_model.external_mapping[VarPowerFlowReferenceType.v_B]
    v_c: Var = case.bus_pcc.emt_model.external_mapping[VarPowerFlowReferenceType.v_C]
    v_dc: Var = case.bus_dc.emt_model.external_mapping[VarPowerFlowReferenceType.Vdc]

    return np.column_stack(
        [
            get_signal_from_var(problem=problem, trajectory=trajectory, variable=v_a),
            get_signal_from_var(problem=problem, trajectory=trajectory, variable=v_b),
            get_signal_from_var(problem=problem, trajectory=trajectory, variable=v_c),
            get_signal_from_var(problem=problem, trajectory=trajectory, variable=v_dc),
        ]
    )


def normalize_range_1_2(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize each column like MATLAB ``normalize(values, 'range', [1 2])``.

    :param values: Raw signal matrix.
    :return: Normalized matrix, column minima, and column maxima.
    """
    values_arr: np.ndarray = np.asarray(values, dtype=np.float64)
    min_arr: np.ndarray = np.min(values_arr, axis=0, keepdims=True)
    max_arr: np.ndarray = np.max(values_arr, axis=0, keepdims=True)
    span_arr: np.ndarray = max_arr - min_arr
    safe_span_arr: np.ndarray = np.where(np.abs(span_arr) > 0.0, span_arr, 1.0)
    normalized_arr: np.ndarray = 1.0 + (values_arr - min_arr) / safe_span_arr
    return normalized_arr, min_arr, max_arr


def add_identification_input_noise(inputs: np.ndarray, seed: int, noise_fraction: float) -> tuple[np.ndarray, float]:
    """
    Add reproducible Gaussian identification noise to the input matrix.

    The noise policy mirrors the GFM reference case: one global input range is
    used to scale white Gaussian noise before normalization.

    :param inputs: Clean boundary input matrix.
    :param seed: Deterministic random seed.
    :param noise_fraction: Fraction of the global input range used as noise standard deviation.
    :return: Noisy input matrix and applied noise standard deviation.
    """
    inputs_arr: np.ndarray = np.asarray(inputs, dtype=np.float64)
    input_range: float = float(np.max(inputs_arr) - np.min(inputs_arr))
    noise_std: float = float(noise_fraction) * input_range
    random_generator: np.random.Generator = np.random.default_rng(seed)
    noisy_inputs: np.ndarray = inputs_arr + noise_std * random_generator.standard_normal(size=inputs_arr.shape)
    return noisy_inputs, noise_std


def save_identification_mat(path: Path, time_arr: np.ndarray, states: np.ndarray, inputs: np.ndarray) -> None:
    """
    Save MATLAB-compatible identification data.

    :param path: Output ``.mat`` file path.
    :param time_arr: EMT time vector.
    :param states: Raw converter state matrix.
    :param inputs: Raw converter boundary input matrix.
    :return: ``None``.
    """
    x_n: np.ndarray
    u_n: np.ndarray
    noisy_inputs: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray
    noise_seed: int = 3
    noise_fraction: float = 0.1
    input_noise_std: float

    noisy_inputs, input_noise_std = add_identification_input_noise(
        inputs=inputs,
        seed=noise_seed,
        noise_fraction=noise_fraction,
    )
    x_n, x_min, x_max = normalize_range_1_2(values=states)
    u_n, u_min, u_max = normalize_range_1_2(values=noisy_inputs)
    _unused_x_min: np.ndarray = x_min
    _unused_x_max: np.ndarray = x_max
    _unused_u_min: np.ndarray = u_min
    _unused_u_max: np.ndarray = u_max
    savemat(
        str(path),
        {
            "x_n": x_n,
            "u_n": u_n,
            "t": np.asarray(time_arr, dtype=np.float64).reshape(-1, 1),
        },
    )
    print(f"input_noise_seed={noise_seed}")
    print(f"input_noise_fraction={noise_fraction:.9e}")
    print(f"input_noise_std={input_noise_std:.9e}")


def validate_pre_event_startup(time_arr: np.ndarray, p_arr: np.ndarray, idc_arr: np.ndarray, event_time_s: float) -> bool:
    """
    Validate that the EMT trajectory is close to steady state before the first event.

    :param time_arr: EMT time vector.
    :param p_arr: Converter active-power trace.
    :param idc_arr: Converter DC-current trace.
    :param event_time_s: First event time in seconds.
    :return: ``True`` when the pre-event startup is acceptable.
    """
    pre_event_mask: np.ndarray = time_arr < float(event_time_s)
    p_pre: np.ndarray = p_arr[pre_event_mask]
    idc_pre: np.ndarray = idc_arr[pre_event_mask]
    p_drift: float = float(np.max(np.abs(p_pre - p_pre[0]))) if p_pre.size > 0 else 0.0
    idc_sign_changes: int = int(np.count_nonzero(np.signbit(idc_pre[1:]) != np.signbit(idc_pre[:-1]))) if idc_pre.size > 1 else 0

    print(f"pre_event_max_abs_P_drift={p_drift:.9e}")
    print(f"pre_event_i_dc_sign_changes={idc_sign_changes}")

    startup_valid: bool = bool(p_drift <= 5.0e-3 and idc_sign_changes == 0)
    return startup_valid


def save_startup_diagnostics(
    output_dir: Path,
    time_arr: np.ndarray,
    p_ref_arr: np.ndarray,
    p_arr: np.ndarray,
    p_f_arr: np.ndarray,
    idc_arr: np.ndarray,
    idc_conv_arr: np.ndarray,
    p_loss_arr: np.ndarray,
    vdc_arr: np.ndarray,
    load_idc_arr: np.ndarray,
    event_time_s: float,
    sbase: float,
) -> None:
    """
    Save focused diagnostics for the pre-event converter startup.

    :param output_dir: Directory where diagnostics are written.
    :param time_arr: EMT time vector.
    :param p_ref_arr: Converter active-power reference trace.
    :param p_arr: Converter active-power trace.
    :param p_f_arr: Filtered converter active-power trace.
    :param idc_arr: Converter DC terminal current trace.
    :param idc_conv_arr: Converter bridge-side DC current trace.
    :param p_loss_arr: Converter loss trace.
    :param vdc_arr: Converter DC-link voltage trace.
    :param load_idc_arr: DC-load current trace.
    :param event_time_s: First event time in seconds.
    :param sbase: System base power used to show power in system units.
    :return: ``None``.
    """
    sample_count: int = min(200, int(time_arr.size))
    diagnostic_path: Path = output_dir / "multicircuit_emt_converter_startup_diagnostics.csv"
    plot_path: Path = output_dir / "multicircuit_emt_converter_startup_diagnostics.png"
    pre_event_mask: np.ndarray = time_arr < float(event_time_s)
    p_pre: np.ndarray = p_arr[pre_event_mask]
    p_drift: float = float(np.max(np.abs(p_pre - p_pre[0]))) if p_pre.size > 0 else 0.0
    idc_sign_changes: int = int(np.count_nonzero(np.signbit(idc_arr[1:sample_count]) != np.signbit(idc_arr[:sample_count - 1]))) if sample_count > 1 else 0

    print(f"startup_diag_P_t0={p_arr[0]:.9e}")
    print(f"startup_diag_P_converter_t0_system_units={p_arr[0] * sbase:.9e}")
    print(f"startup_diag_P_ref_t0={p_ref_arr[0]:.9e}")
    print(f"startup_diag_P_loss_t0={p_loss_arr[0]:.9e}")
    print(f"startup_diag_i_dc_t0={idc_arr[0]:.9e}")
    print(f"startup_diag_i_dc_conv_t0={idc_conv_arr[0]:.9e}")
    print(f"startup_diag_load_i_dc_t0={load_idc_arr[0]:.9e}")
    print(f"startup_diag_pre_event_P_drift={p_drift:.9e}")
    print(f"startup_diag_first_{sample_count}_i_dc_sign_changes={idc_sign_changes}")

    with diagnostic_path.open("w", encoding="utf-8") as diagnostic_file:
        diagnostic_file.write("time_s,P_ref,P,P_f,i_dc,i_dc_conv,P_loss,v_dc,load_i_dc\n")
        for row_idx in range(sample_count):
            diagnostic_file.write(
                f"{time_arr[row_idx]:.12e},"
                f"{p_ref_arr[row_idx]:.12e},"
                f"{p_arr[row_idx]:.12e},"
                f"{p_f_arr[row_idx]:.12e},"
                f"{idc_arr[row_idx]:.12e},"
                f"{idc_conv_arr[row_idx]:.12e},"
                f"{p_loss_arr[row_idx]:.12e},"
                f"{vdc_arr[row_idx]:.12e},"
                f"{load_idc_arr[row_idx]:.12e}\n"
            )

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    plot_mask: np.ndarray = time_arr <= min(float(event_time_s), 0.002)
    axes[0].plot(time_arr[plot_mask], p_ref_arr[plot_mask], label="P_ref")
    axes[0].plot(time_arr[plot_mask], p_arr[plot_mask] * sbase, label="P * Sbase")
    axes[0].plot(time_arr[plot_mask], p_f_arr[plot_mask] * sbase, label="P_f * Sbase")
    axes[0].set_title("Startup active-power diagnostics [system units]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(time_arr[plot_mask], idc_arr[plot_mask], label="i_dc converter")
    axes[1].plot(time_arr[plot_mask], idc_conv_arr[plot_mask], label="i_dc_conv")
    axes[1].plot(time_arr[plot_mask], -load_idc_arr[plot_mask], label="-i_dc load", linestyle="--")
    axes[1].set_title("Startup DC-current diagnostics")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    axes[2].plot(time_arr[plot_mask], p_loss_arr[plot_mask], label="P_loss")
    axes[2].plot(time_arr[plot_mask], vdc_arr[plot_mask], label="v_dc")
    axes[2].set_title("Startup loss and DC voltage diagnostics")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    print(f"startup_diagnostics_csv={diagnostic_path}")
    print(f"startup_diagnostics_plot={plot_path}")


def get_event_schedule_text() -> list[str]:
    """
    Return the grid-side EMT event schedule used by this identification case.

    :return: Human-readable event descriptions.
    """
    return list([
        "t=0.010 s: Thevenin E_scale -> 0.950",
    ])


def plot_summary(
    path: Path,
    time_arr: np.ndarray,
    inputs: np.ndarray,
    p_arr: np.ndarray,
    q_arr: np.ndarray,
    idc_arr: np.ndarray,
    vdc_arr: np.ndarray,
    sbase: float,
) -> None:
    """
    Save a diagnostic plot for the MultiCircuit EMT dataset.

    :param path: Output PNG path.
    :param time_arr: EMT time vector.
    :param inputs: Boundary input matrix.
    :param p_arr: Converter active power trace.
    :param q_arr: Converter reactive power trace.
    :param idc_arr: Converter DC current trace.
    :param vdc_arr: Converter DC-link voltage trace.
    :param sbase: System base power used to show power in system units.
    :return: ``None``.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(time_arr, inputs[:, 0], label="v_A PCC")
    axes[0].plot(time_arr, inputs[:, 1], label="v_B PCC")
    axes[0].plot(time_arr, inputs[:, 2], label="v_C PCC")
    axes[0].set_title("PCC voltage from MultiCircuit EMT grid")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(time_arr, inputs[:, 3], label="Vdc bus")
    axes[1].plot(time_arr, vdc_arr, label="Vdc converter", linestyle="--")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    axes[2].plot(time_arr, p_arr * sbase, label="P * Sbase")
    axes[2].plot(time_arr, q_arr * sbase, label="Q * Sbase")
    axes[2].set_title("Converter active/reactive power [system units]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")
    axes[3].plot(time_arr, idc_arr, label="i_dc")
    axes[3].set_title("Converter DC current")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_dq_diagnostics(
    path: Path,
    time_arr: np.ndarray,
    v_d_arr: np.ndarray,
    v_q_arr: np.ndarray,
    i_d_arr: np.ndarray,
    i_q_arr: np.ndarray,
    i_d_ref_arr: np.ndarray,
    i_q_ref_arr: np.ndarray,
    v_cmd_d_arr: np.ndarray,
    v_cmd_q_arr: np.ndarray,
    theta_pll_arr: np.ndarray,
    omega_pll_arr: np.ndarray,
) -> None:
    """
    Save dq-frame converter diagnostics for startup and event analysis.

    :param path: Output PNG path.
    :param time_arr: EMT time vector.
    :param v_d_arr: Converter terminal d-axis voltage.
    :param v_q_arr: Converter terminal q-axis voltage.
    :param i_d_arr: Converter d-axis current.
    :param i_q_arr: Converter q-axis current.
    :param i_d_ref_arr: Converter d-axis current reference.
    :param i_q_ref_arr: Converter q-axis current reference.
    :param v_cmd_d_arr: Converter d-axis voltage command.
    :param v_cmd_q_arr: Converter q-axis voltage command.
    :param theta_pll_arr: PLL angle trace.
    :param omega_pll_arr: PLL angular-speed trace.
    :return: ``None``.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(time_arr, v_d_arr, label="v_d")
    axes[0].plot(time_arr, v_q_arr, label="v_q")
    axes[0].set_title("Converter dq terminal voltage")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(time_arr, i_d_arr, label="i_d")
    axes[1].plot(time_arr, i_q_arr, label="i_q")
    axes[1].plot(time_arr, i_d_ref_arr, label="i_d_ref", linestyle="--")
    axes[1].plot(time_arr, i_q_ref_arr, label="i_q_ref", linestyle="--")
    axes[1].set_title("Converter dq current and references")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(time_arr, v_cmd_d_arr, label="v_cmd_d")
    axes[2].plot(time_arr, v_cmd_q_arr, label="v_cmd_q")
    axes[2].set_title("Converter dq voltage commands")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].plot(time_arr, theta_pll_arr, label="theta_pll")
    axes[3].plot(time_arr, omega_pll_arr, label="omega_pll")
    axes[3].set_title("PLL angle and angular speed")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_multicircuit_case() -> tuple[GridCase, EmtProblemDae, np.ndarray, np.ndarray, bool, bool, bool]:
    """
    Build, initialize, and solve the MultiCircuit EMT identification case.

    :return: Case, problem, time vector, state trajectory, PF flag, initialization flag, convergence flag.
    """
    case: GridCase = build_multicircuit_grid()
    converter_model: Block = attach_emt_models(case=case)
    pf_driver: PowerFlowDriver3Ph = PowerFlowDriver3Ph(grid=case.grid, options=build_power_flow_options())
    pf_results: Any = gce.power_flow(grid=case.grid, options=build_power_flow_options())
    options: EmtOptions = build_options()
    pf_driver.run()
    problem: EmtProblemDae = EmtProblemDae(grid=case.grid, options=options, pf_results=pf_results, pf_results_3ph=pf_driver.results)
    seed_dc_current_balance(problem=problem, case=case)
    solver: Any = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=float(options.simulation_time),
        h=float(options.time_step),
        method=options.integration_method,
    )
    boundary_updater: Any = build_boundary_updater_from_block(problem)
    time_arr: np.ndarray
    state_traj: np.ndarray
    diff_traj: np.ndarray
    well_initialized: bool
    converged: bool
    _unused_converter_model: Block = converter_model

    time_arr, state_traj, diff_traj, well_initialized, converged = solver.simulate(boundary_updater=boundary_updater)
    _unused_diff_traj: np.ndarray = diff_traj
    return case, problem, time_arr, state_traj, bool(pf_driver.results.converged), bool(well_initialized), bool(converged)


def seed_dc_current_balance(problem: EmtProblemDae, case: GridCase) -> None:
    """
    Seed converter DC current to balance the attached DC load at ``t=0``.

    :param problem: EMT DAE problem whose initial guesses are updated.
    :param case: Grid case with converter and DC load EMT models.
    :return: ``None``.
    """
    converter_idc: Var = case.converter.emt_model.external_mapping[VarPowerFlowReferenceType.Idc]
    load_idc: Var = case.dc_load.emt_model.external_mapping[VarPowerFlowReferenceType.Idc]
    load_guess: float = float(problem.init_guess.get(load_idc.uid, -float(case.dc_load.P) / float(case.grid.Sbase)))
    problem.init_guess[converter_idc.uid] = load_guess


def main() -> None:
    """
    Generate identification data from a proper VeraGrid ``MultiCircuit`` EMT simulation.

    :return: ``None``.
    """
    output_dir: Path = Path(__file__).resolve().parent
    mat_path: Path = output_dir / "dataforid_multicircuit_emt_converter.mat"
    plot_path: Path = output_dir / "multicircuit_emt_converter_timeseries.png"
    dq_plot_path: Path = output_dir / "multicircuit_emt_converter_dq_diagnostics.png"
    case: GridCase
    problem: EmtProblemDae
    time_arr: np.ndarray
    state_traj: np.ndarray
    power_flow_converged: bool
    well_initialized: bool
    converged: bool
    states: np.ndarray
    inputs: np.ndarray
    p_arr: np.ndarray
    q_arr: np.ndarray
    idc_arr: np.ndarray
    vdc_arr: np.ndarray
    startup_valid: bool

    case, problem, time_arr, state_traj, power_flow_converged, well_initialized, converged = run_multicircuit_case()
    if not bool(well_initialized and converged):
        print(f"power_flow_converged={power_flow_converged}")
        print(f"emt_initialized={well_initialized}, emt_converged={converged}")
        raise RuntimeError("EMT run failed; refusing to save identification data or plots")
    else:
        pass

    states = collect_state_matrix(problem=problem, trajectory=state_traj, case=case)
    inputs = collect_input_matrix(problem=problem, trajectory=state_traj, case=case)
    p_arr = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="P")
    q_arr = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="Q")
    idc_arr = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="i_dc")
    vdc_arr = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="v_dc")
    p_ref_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="P_ref")
    p_f_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="P_f")
    idc_conv_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="i_dc_conv")
    p_loss_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="P_loss")
    load_idc_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.dc_load.emt_model, variable_name="i_dc")
    v_d_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="v_d")
    v_q_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="v_q")
    i_d_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="i_d")
    i_q_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="i_q")
    i_d_ref_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="i_d_ref")
    i_q_ref_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="i_q_ref")
    v_cmd_d_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="v_cmd_d")
    v_cmd_q_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="v_cmd_q")
    theta_pll_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="theta_pll")
    omega_pll_arr: np.ndarray = get_signal_from_block(problem=problem, trajectory=state_traj, block=case.converter.emt_model, variable_name="omega_pll")

    save_startup_diagnostics(
        output_dir=output_dir,
        time_arr=time_arr,
        p_ref_arr=p_ref_arr,
        p_arr=p_arr,
        p_f_arr=p_f_arr,
        idc_arr=idc_arr,
        idc_conv_arr=idc_conv_arr,
        p_loss_arr=p_loss_arr,
        vdc_arr=vdc_arr,
        load_idc_arr=load_idc_arr,
        event_time_s=0.010,
        sbase=float(case.grid.Sbase),
    )
    plot_summary(
        path=plot_path,
        time_arr=time_arr,
        inputs=inputs,
        p_arr=p_arr,
        q_arr=q_arr,
        idc_arr=idc_arr,
        vdc_arr=vdc_arr,
        sbase=float(case.grid.Sbase),
    )
    print(f"diagnostic_plot={plot_path}")
    plot_dq_diagnostics(
        path=dq_plot_path,
        time_arr=time_arr,
        v_d_arr=v_d_arr,
        v_q_arr=v_q_arr,
        i_d_arr=i_d_arr,
        i_q_arr=i_q_arr,
        i_d_ref_arr=i_d_ref_arr,
        i_q_ref_arr=i_q_ref_arr,
        v_cmd_d_arr=v_cmd_d_arr,
        v_cmd_q_arr=v_cmd_q_arr,
        theta_pll_arr=theta_pll_arr,
        omega_pll_arr=omega_pll_arr,
    )
    print(f"dq_diagnostic_plot={dq_plot_path}")

    startup_valid = validate_pre_event_startup(time_arr=time_arr, p_arr=p_arr, idc_arr=idc_arr, event_time_s=0.010)
    if startup_valid:
        pass
    else:
        print(f"power_flow_converged={power_flow_converged}")
        print(f"emt_initialized={well_initialized}, emt_converged={converged}")
        print("identification_mat_skipped=pre_event_startup_not_steady")
        return

    save_identification_mat(path=mat_path, time_arr=time_arr, states=states, inputs=inputs)

    print(f"power_flow_converged={power_flow_converged}")
    print(f"emt_initialized={well_initialized}, emt_converged={converged}")
    print(f"samples={len(time_arr)}, converter_states={states.shape[1]}, boundary_inputs={inputs.shape[1]}")
    print("event_schedule:")
    for event_text in get_event_schedule_text():
        print(f"  {event_text}")
    print(f"dataforid_mat={mat_path}")
    print(f"timeseries_plot={plot_path}")
    print(f"dq_diagnostic_plot={dq_plot_path}")


if __name__ == "__main__":
    main()
