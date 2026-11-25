"""End-to-end acceptance checks for serialized basic challenge cases."""
from pathlib import Path

import numpy as np

from VeraGridEngine.api import open_file
from VeraGridEngine.Simulations.Rms.problems.rms_problem_template import RmsProblemTemplate
from hackathon_files.basic.cases import basic_evaluator, initialize_grid, reference_tuning
from hackathon_files.src.rms_diagnostics import analyze_grid
from hackathon_files.src.rms_tuning import RmsTunerEvaluator, RmsTuning, RmsTuningMetrics, TunableGrid


def equilibrium_error(problem: RmsProblemTemplate) -> float:
    """Measure both stationary differential and algebraic equations.

    :param problem: Fully initialized RMS problem at its startup snapshot.
    :return: Infinity norm with zero state derivatives.
    """
    x: np.ndarray = problem.get_x0()
    dx: np.ndarray = np.zeros_like(x)
    return float(max(
        np.max(np.abs(problem.rhs_state(x, dx)), initial=0.0),
        np.max(np.abs(problem.rhs_algebraic(x, dx)), initial=0.0),
    ))


def validate_case(path: Path) -> dict:
    """Reject cases without a growing mode and a feasible stable witness.

    :param path: Saved unstable VeraGrid file, reopened for this verification.
    :return: JSON-ready measurement record, including the exact witness values.
    """
    tunable: TunableGrid = initialize_grid(open_file(str(path)))
    baseline_error: float = equilibrium_error(tunable.problem)
    if not np.isfinite(baseline_error) or baseline_error > 1e-6:
        raise ValueError(f'{path.name}: baseline is not an equilibrium ({baseline_error})')
    else:
        pass
    # Inspect every finite mode, rather than an arbitrary subset of six or 20.
    analysis: dict = analyze_grid(tunable)
    raw_eigenvalues: np.ndarray = analysis['eigenvalues']
    if np.any(np.isnan(raw_eigenvalues)):
        raise ValueError(f'{path.name}: NaN eigenvalues')
    else:
        pass
    algebraic_modes: int = int(np.count_nonzero(~np.isfinite(raw_eigenvalues)))
    if algebraic_modes and tunable.problem.get_diff_var_number() == 0:
        raise ValueError(f'{path.name}: nonfinite ordinary state eigenvalues')
    else:
        pass
    # Descriptor converter models have infinite algebraic constraint modes.
    # Keep every finite physical mode in the stability assessment.
    eigenvalues: np.ndarray = raw_eigenvalues[np.isfinite(raw_eigenvalues)]
    converter_ids: set[str] = set(generator.idtag for generator in tunable.problem.grid.generators
                                  if 'Basic dynamic' in generator.rms_model.name)
    converter_rows: np.ndarray = np.array([label['device_idtag'] in converter_ids
                                          for label in analysis['variables']], dtype=bool)
    growing_columns: np.ndarray = np.flatnonzero(np.isfinite(raw_eigenvalues) & (raw_eigenvalues.real > 1e-4))
    converter_participation: float = 0.0
    column: int
    for column in growing_columns:
        factors: np.ndarray = analysis['participation_factors'][:, column]
        converter_participation = max(converter_participation,
                                      float(np.sum(factors[converter_rows]) / max(np.sum(factors), 1e-30)))
    if converter_ids and converter_participation < 0.1:
        raise ValueError(f'{path.name}: growing modes do not involve converter controls')
    else:
        pass
    growing: np.ndarray = eigenvalues.real > 1e-4
    if not np.any(growing):
        raise ValueError(f'{path.name}: no genuinely unstable physical mode')
    else:
        pass
    evaluator: RmsTunerEvaluator = basic_evaluator()
    controller_only = tunable.problem.grid.name.endswith('_02')
    if controller_only and any((tunable.openable_lines, tunable.closable_lines, tunable.line_candidates,
                                tunable.shunt_options, tunable.dispatch_options)):
        raise ValueError(f'{path.name}: controller-only case exposes physical actions')
    if not controller_only and (not tunable.closable_lines or not tunable.line_candidates
                                or not tunable.shunt_options or len(tunable.dispatch_options) < 2):
        raise ValueError(f'{path.name}: incomplete general-action catalog')
    else:
        pass
    witness: RmsTuning = reference_tuning(tunable)
    metrics: RmsTuningMetrics = evaluator.evaluate_proposal(tunable, witness)
    if not metrics.stable or metrics.voltage_violations or metrics.thermal_violations:
        raise ValueError(f'{path.name}: witness failed acceptance: {metrics}')
    else:
        pass
    # Reinitialize after applying the very same permitted proposal and check
    # equilibrium independently of the modal/transient success flags.
    stable_input: TunableGrid = initialize_grid(open_file(str(path)))
    stable_problem: RmsProblemTemplate = evaluator.apply_tuning(stable_input, reference_tuning(stable_input))
    stable_error: float = equilibrium_error(stable_problem)
    if not np.isfinite(stable_error) or stable_error > 1e-6:
        raise ValueError(f'{path.name}: witness is not an equilibrium ({stable_error})')
    else:
        pass
    return dict(
        file=path.name,
        max_test_tunings=("unlimited" if np.isinf(tunable.max_test_tunings)
                          else tunable.max_test_tunings),
        buses=len(tunable.problem.grid.buses),
        generators=len(tunable.problem.grid.generators),
        lines=len(tunable.problem.grid.lines),
        transformers=len(tunable.problem.grid.transformers2w),
        states=tunable.problem.get_states_number(),
        baseline_equilibrium_error=baseline_error,
        baseline_max_real=float(np.max(eigenvalues.real)),
        baseline_growing_modes=int(np.count_nonzero(growing)),
        baseline_finite_modes=int(eigenvalues.size),
        algebraic_constraint_modes=algebraic_modes,
        baseline_growing_converter_participation=converter_participation,
        converters=[dict(device=generator.name, kind=generator.rms_model.name, rating_mva=generator.Snom)
                    for generator in tunable.problem.grid.generators if generator.idtag in converter_ids],
        tunable_parameters=[dict(device=parameter.device_name, name=parameter.name,
                                 initial=parameter.initial_value, minimum=parameter.minimum, maximum=parameter.maximum)
                            for parameter in tunable.tunable_parameters],
        witness_parameters={parameter.runtime_key: value for parameter, value in witness.parameters.items()},
        initial_open_lines=[dict(idtag=line.idtag, name=line.name) for line in tunable.problem.grid.lines if not line.active],
        action_catalog=dict(
            open_lines=[dict(idtag=option.idtag, name=option.name) for option in tunable.openable_lines],
            close_lines=[dict(idtag=option.idtag, name=option.name) for option in tunable.closable_lines],
            dispatch=[dict(idtag=option.device_idtag, initial=option.initial_mw,
                           minimum=option.minimum_mw, maximum=option.maximum_mw) for option in tunable.dispatch_options],
            new_lines=[dict(name=option.name, bus_from=option.bus_from_idtag, bus_to=option.bus_to_idtag,
                            r=option.resistance, x=option.reactance, b=option.susceptance, rate_mva=option.rate_mva)
                       for option in tunable.line_candidates],
            new_shunts=[dict(bus=option.bus_idtag, conductance_mw=option.conductance_mw,
                             susceptance_mvar=option.susceptance_mvar) for option in tunable.shunt_options],
        ),
        witness_close_lines=[option.idtag for option in witness.close_lines],
        witness_equilibrium_error=stable_error,
        witness_max_real=metrics.max_eigenvalue_real,
        witness_min_damping=metrics.min_oscillatory_damping,
        required_minimum_damping_ratio=evaluator.minimum_damping_ratio,
        witness_neutral_modes=metrics.neutral_mode_count,
        witness_finite_modes=metrics.eigenvalue_count,
        witness_voltage_violations=metrics.voltage_violations,
        witness_thermal_violations=metrics.thermal_violations,
        witness_transient_converged=metrics.transient_converged,
        witness_peak_frequency_hz=metrics.peak_frequency_deviation_hz,
        witness_peak_voltage_pu=metrics.peak_voltage_deviation_pu,
        witness_losses_mw=metrics.active_losses_mw,
    )
