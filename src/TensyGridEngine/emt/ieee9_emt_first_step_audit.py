"""Report IEEE9 EMT discontinuities over the first two 5 microsecond steps."""

from __future__ import annotations

from VeraGridEngine.Utils.Symbolic.block import find_name_in_block

import ieee9_emt_damping_ablation as ablation


TIME_STEP = 5.0e-6


def report_variable(problem, values, label: str, variable) -> None:
    idx = int(problem.get_var_idx(variable))
    value0 = float(values[0, idx])
    value1 = float(values[1, idx])
    value2 = float(values[2, idx])
    print(
        f"{label:42s} x0={value0:+.12e} x1={value1:+.12e} "
        f"x2={value2:+.12e} delta_10us={value2 - value0:+.12e} "
        f"slope_0_5us={(value1 - value0) / TIME_STEP:+.12e}"
    )


def main() -> None:
    ablation.TIME_STEP = TIME_STEP
    ablation.SIMULATION_TIME = 2.0 * TIME_STEP
    problem, _time, values = ablation.run_case(0.0)

    print("BUS PHASE-A VOLTAGES")
    for bus in problem.grid.buses:
        report_variable(problem, values, bus.name, bus.emt_model.out_vars[0])

    print("\nGENERATOR CURRENTS")
    for generator in problem.grid.generators:
        for phase in ("A", "B", "C"):
            variable = find_name_in_block(f"i_{phase}", generator.emt_model)
            if variable is not None:
                report_variable(problem, values, f"{generator.name} i{phase}", variable)

    print("\nTRANSFORMER PORT AND WINDING CURRENTS")
    for transformer in problem.grid.transformers2w:
        for prefix in ("if", "it", "i_f", "i_t"):
            for phase in ("A", "B", "C"):
                variable = find_name_in_block(f"{prefix}_{phase}", transformer.emt_model)
                if variable is not None:
                    report_variable(
                        problem,
                        values,
                        f"{transformer.name} {prefix}{phase}",
                        variable,
                    )

    print("\nSYNCHRONOUS GENERATOR POWER/TORQUE")
    machine = next(gen for gen in problem.grid.generators if not gen.bus.is_slack)
    for name in (
        "p_e", "Te", "Tm", "omega_", "i_d_", "i_q_", "psi_d_", "psi_q_",
        "e_qp_", "e_dp_", "psi_pp_d_", "psi_pp_q_",
    ):
        variable = find_name_in_block(name, machine.emt_model)
        if variable is not None:
            report_variable(problem, values, f"{machine.name} {name}", variable)


if __name__ == "__main__":
    main()
