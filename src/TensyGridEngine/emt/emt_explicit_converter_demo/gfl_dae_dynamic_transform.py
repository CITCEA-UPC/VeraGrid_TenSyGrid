"""Apply the generic symbolic_ml dynamic transformer to a connected GFL DAE."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from TensyGridEngine.emt.emt_explicit_converter_demo.single_model_time_inputs import build_problem
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Simulations.EMT.problems.emt_model_problem import EmtModelProblem
from VeraGridEngine.Utils.Symbolic import symbolic_ml
from VeraGridEngine.Utils.Symbolic.symbolic import Const

OUTPUT = Path(__file__).with_name("gfl_dae_dynamic_transform.png")
NOISE_SEED = 20260914
LIFTED_CONTROLLER_VARIABLES = {
    "i_q_ref",
    "i_d_ref",
    "y_vd_hat",
    "y_vq_hat",
    "u_PLL_pi",
    "omega",
}
LIFTED_MEASUREMENT_VARIABLES = {
    "P", "Q", "vg_d", "vg_q", "i_line_d", "i_line_q",
}
LIFTED_VOLTAGE_COMMAND_VARIABLES = {
    "v_d_c", "v_q_c", "vc_d", "vc_q",
}


def build_original_and_transformed(
        relaxation_time: float = 5.0e-4,
        phase_relaxation_time: float = 1.0e-5,
        measurement_relaxation_time: float = 1.0e-5,
        time_step: float = 1.0e-6,
) -> tuple[EmtModelProblem, EmtModelProblem, symbolic_ml.DynamicLiftResult]:
    """Return matching original and partially dynamic-lifted GFL problems."""
    original = build_problem(
        "gfl",
        multilinear=True,
        p0=0.25,
        q0=0.02,
        voltage=1.02 * np.exp(1j * 0.1),
        voltage_noise_rms=0.005,
        simulation_time=0.04,
        dc_side="resistance",
    )
    original.options.time_step = time_step
    scalar_result = symbolic_ml.dynamic_lift_affine_algebraics(
        original.sys_block,
        VarFactory(),
        time_constants=relaxation_time,
        variables=LIFTED_CONTROLLER_VARIABLES,
    )
    phase_result = symbolic_ml.dynamic_lift_balanced_dq_to_abc(
        scalar_result.block,
        VarFactory(),
        abc_variables=("vc_A", "vc_B", "vc_C"),
        d_variable="vc_d",
        q_variable="vc_q",
        cosine_variable="u_cos",
        sine_variable="u_sin",
        time_constant=phase_relaxation_time,
    )
    remaining_affine_result = symbolic_ml.dynamic_lift_affine_algebraics(
        phase_result.block,
        VarFactory(),
        time_constants={
            **{name: measurement_relaxation_time for name in LIFTED_MEASUREMENT_VARIABLES},
            **{name: relaxation_time for name in LIFTED_VOLTAGE_COMMAND_VARIABLES},
        },
        variables=LIFTED_MEASUREMENT_VARIABLES | LIFTED_VOLTAGE_COMMAND_VARIABLES,
    )
    trig_result = symbolic_ml.promote_trig_algebraics_exact(
        remaining_affine_result.block,
        cosine_variable="u_cos",
        sine_variable="u_sin",
        angle_variable="theta",
    )
    transformed_block = trig_result.block
    transformed_result = symbolic_ml.DynamicLiftResult(
        block=transformed_block,
        lifted=(scalar_result.lifted + phase_result.lifted
                + remaining_affine_result.lifted + trig_result.lifted),
        remaining_algebraic_variables=trig_result.remaining_algebraic_variables,
    )

    # Start the relaxation states exactly on the original DAE manifold. The
    # generic transformer preserves symbolic init equations, but a composed GFL
    # has cyclic initialization dependencies; the already-solved operating point
    # provides authoritative numerical values for this experiment.
    initial_values = {
        variable.name: original.init_guess[variable.uid]
        for variable in original.get_state_vars() + original.get_algebraic_vars()
    }
    for record in transformed_result.lifted:
        transformed_block.init_eqs[record.variable] = Const(
            initial_values[record.variable.name]
        )

    input_expressions = {
        variable: transformed_block.event_dict[variable]
        for variable in transformed_block.in_vars
    }
    transformed = EmtModelProblem(
        transformed_block,
        inputs=input_expressions,
        initial_values=initial_values,
        options=original.options,
        glob_time=original.glob_time,
    )
    return original, transformed, transformed_result


def main() -> None:
    original, transformed, transform = build_original_and_transformed()
    np.random.seed(NOISE_SEED)
    original_result = original.simulate()
    np.random.seed(NOISE_SEED)
    transformed_result = transformed.simulate()

    figure, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    axes[0].plot(
        original_result.time,
        original.trace(original_result, "i_line_A"),
        label="original DAE",
    )
    axes[0].plot(
        transformed_result.time,
        transformed.trace(transformed_result, "i_line_A"),
        linestyle="--",
        label="controller dynamic lifts",
    )
    axes[0].set_ylabel("Phase-A current (pu)")
    axes[0].legend()
    axes[1].plot(
        transformed_result.time,
        transformed.trace(transformed_result, "i_q_ref"),
        label="dynamic i_q_ref",
    )
    axes[1].plot(
        transformed_result.time,
        transformed.trace(transformed_result, "i_d_ref"),
        label="dynamic i_d_ref",
    )
    axes[1].set_ylabel("Current references")
    axes[1].legend()
    axes[2].plot(
        transformed_result.time,
        transformed.trace(transformed_result, "omega"),
        label="dynamic omega",
    )
    axes[2].set_ylabel("PLL frequency (pu)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    phase_sum = sum(
        transformed.trace(transformed_result, f"vc_{phase}") for phase in "ABC"
    )
    axes[3].plot(transformed_result.time, phase_sum, label="vc_A + vc_B + vc_C")
    axes[3].set_ylabel("Zero sequence (pu)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.suptitle("Fully promoted multilinear GFL DAE")
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)

    original_current = original.trace(original_result, "i_line_A")
    transformed_current = transformed.trace(transformed_result, "i_line_A")
    print(
        f"original: states={len(original.get_state_vars())}, "
        f"algebraics={len(original.get_algebraic_vars())}, "
        f"converged={original_result.converged}"
    )
    print(
        f"transformed: states={len(transformed.get_state_vars())}, "
        f"algebraics={len(transformed.get_algebraic_vars())}, "
        f"lifted={len(transform.lifted)}, converged={transformed_result.converged}"
    )
    print(
        f"phase_A_rms_difference="
        f"{np.sqrt(np.mean((transformed_current - original_current) ** 2)):.9e} pu"
    )
    print(f"max_abs_converter_voltage_sum={np.max(np.abs(phase_sum)):.9e} pu")
    print(f"plot={OUTPUT}")


if __name__ == "__main__":
    main()
