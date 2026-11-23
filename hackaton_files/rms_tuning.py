"""Small submission interface for the hackathon RMS tuning exercise.

Participants select parameter handles from the tunable grid's catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from abc import ABC, abstractmethod
from typing import Mapping, Callable
from threading import Lock
from copy import deepcopy
import math
import warnings
import numpy as np

from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Devices.Injections.shunt import Shunt
from VeraGridEngine.Devices.Branches.line import Line
from VeraGridEngine.Devices.Events.rms_event import RmsEvent
from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup
from VeraGridEngine.Devices.Parents.dynamic_parent import DynamicDevice
from VeraGridEngine.Devices.Parents.dynamic_bus_parent import DynamicBusDevice
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.Rms.problems.rms_problem_dae import RmsProblemDae
from VeraGridEngine.Simulations.Rms.problems.rms_problem_template import RmsProblemTemplate
from VeraGridEngine.Simulations.Rms.rms_driver import RmsSimulationDriver
from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import SmallSignalStabilityRmsDriver
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_options import RmsSmallSignalStabilityOptions
from VeraGridEngine.Templates.Rms.line_rms_template import get_line_rms_template
from VeraGridEngine.Templates.Rms.shunt_template import get_shunt_template
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model
from VeraGridEngine.Utils.Symbolic.symbolic import Const
from .rms_diagnostics import analyze_grid, get_modes_from_unstable_eigenvalues
from hackaton_files.rms_controllers import (ControllerBlock, ControllerMode, controller_blocks,
                                           controller_parameters, controller_modes)
from .rms_sensitivity import (build_parameterized_state_matrix, evaluate_state_matrix,
                              state_matrix_derivative, eigenvalue_sensitivities, equilibrium_residual)


@dataclass(frozen=True)
class TunableParameter:
    """One allowed controller parameter and its inclusive tuning range."""

    device_name: str
    device_idtag: str
    name: str
    initial_value: float
    minimum: float
    maximum: float

    @property
    def key(self) -> str:
        return f"{self.device_name}.{self.name}"

    @property
    def runtime_key(self) -> str:
        return f"{self.device_idtag}:{self.name}"


# Each grid can provide its own control names and search bounds. These are the
# chosen limits for the bundled Kundur case, not stability guarantees.
KUNDUR_CONTROL_RANGES: dict[str, tuple[float, float]] = {
    "Kp": (0.0, 150.0),
    "Ki": (0.0, 10.0),
}


@dataclass(frozen=True)
class DispatchOption:
    """One generator's active-power setpoint and permitted MW range."""

    device_idtag: str
    device_name: str
    initial_mw: float
    minimum_mw: float
    maximum_mw: float


@dataclass(frozen=True)
class LineOption:
    """An existing line offered for a specific opening or closing action."""

    idtag: str
    name: str


@dataclass(frozen=True)
class LineCandidate:
    """A predefined corridor that may be built once."""

    bus_from_idtag: str
    bus_to_idtag: str
    name: str
    resistance: float
    reactance: float
    susceptance: float
    rate_mva: float


@dataclass(frozen=True)
class ShuntOption:
    """A fixed shunt candidate supplied by the challenge organizer."""

    bus_idtag: str
    bus_name: str
    conductance_mw: float
    susceptance_mvar: float


def _sanitize_action_handles[T: (LineOption, LineCandidate, ShuntOption)](
        values: object, catalog: tuple[T, ...], limit: int, label: str) -> tuple[T, ...]:
    """Validate an action list and return the actual organizer handles.

    :param values: Submitted action container, potentially malformed.
    :param catalog: Permitted handles of one explicit action type.
    :param limit: Maximum number of distinct actions.
    :param label: Human-readable warning label.
    :return: Valid actions in submission order, with every omission reported.
    """
    result: tuple[T, ...]
    if not isinstance(values, (tuple, list)):
        warnings.warn(f"Skipped {label}: expected a tuple or list of action handles", UserWarning, stacklevel=3)
        result = tuple()
    else:
        kept: list[T] = list()
        action: object
        for action in values:
            selected: T | None = next((allowed for allowed in catalog
                                       if type(action) is type(allowed) and action == allowed), None)
            if selected is None:
                warnings.warn(f"Skipped {label} {action!r}: unknown or unavailable handle", UserWarning, stacklevel=3)
            elif selected in kept:
                warnings.warn(f"Skipped {label} {action!r}: duplicate action", UserWarning, stacklevel=3)
            elif len(kept) >= limit:
                warnings.warn(f"Skipped {label} {action!r}: action budget exceeded", UserWarning, stacklevel=3)
            else:
                kept.append(selected)
        result = tuple(kept)
    return result


KUNDUR_SHUNT_BUSES = ("Bus7", "Bus9")
KUNDUR_LINE_CORRIDORS = (("Bus7", "Bus9"),)


def build_rms_tunables(
    grid: MultiCircuit,
    control_ranges: Mapping[str, tuple[float, float]],
) -> tuple[TunableParameter, ...]:
    """Discover selected scalar event parameters in a grid's RMS models."""
    tunables: dict[str, TunableParameter] = {}
    devices = [
        device for device in grid.get_all_elements_iter()
        if isinstance(device, (DynamicDevice, DynamicBusDevice))
    ]
    name_counts: dict[str, int] = {}
    for device in devices:
        name_counts[device.name] = name_counts.get(device.name, 0) + 1
    for device in devices:
        device_label = (
            device.name if name_counts[device.name] == 1
            else f"{device.name}[{device.idtag}]"
        )
        pending = [device.rms_model]
        while pending:
            block = pending.pop()
            pending.extend(block.children)
            for variable, expression in block.event_dict.items():
                if variable.name not in control_ranges or not isinstance(expression, Const):
                    continue
                if not isinstance(expression.value, (int, float)):
                    continue
                minimum, maximum = control_ranges[variable.name]
                if not minimum <= expression.value <= maximum:
                    raise ValueError(f"{device_label}.{variable.name} starts outside its tuning range")
                spec = TunableParameter(
                    device_name=device_label,
                    device_idtag=device.idtag,
                    name=variable.name,
                    initial_value=float(expression.value),
                    minimum=minimum,
                    maximum=maximum,
                )
                if spec.key in tunables:
                    raise ValueError(f"Ambiguous RMS tuning key: {spec.key}")
                tunables[spec.key] = spec
    return tuple(tunables.values())


def _parameter_indices(problem: RmsProblemTemplate) -> dict[str, int]:
    variables = problem._variable_parameters
    device_ids = problem._event_parameter_device_idtags
    indices: dict[str, int] = {}
    counts: dict[str, int] = {}
    for variable in variables:
        counts[variable.name] = counts.get(variable.name, 0) + 1
    for index, variable in enumerate(variables):
        device_id = device_ids.get(variable.uid)
        if device_id is not None:
            indices[f"{device_id}:{variable.name}"] = index
        if counts[variable.name] == 1:
            indices[variable.name] = index
    return indices


@dataclass(frozen=True)
class TunableGrid:
    """One initialized RMS problem and the controls participants may change."""

    problem: RmsProblemTemplate
    tunable_parameters: tuple[TunableParameter, ...]
    dispatch_options: tuple[DispatchOption, ...]
    openable_lines: tuple[LineOption, ...]
    line_candidates: tuple[LineCandidate, ...]
    shunt_options: tuple[ShuntOption, ...]
    max_open_lines: int = 2
    max_new_lines: int = 1
    max_new_shunts: int = 1
    _trial: Callable | None = field(default=None, repr=False, compare=False)
    _budget_status: Callable | None = field(default=None, repr=False, compare=False)
    closable_lines: tuple[LineOption, ...] = ()
    max_close_lines: int = 1

    @property
    def max_test_tunings(self) -> int:
        """Read-only trial allowance for this scenario."""
        return self._budget_status()[0] if self._budget_status else 0

    @property
    def tests_used(self) -> int:
        """Read-only count; the evaluator session owns the counter."""
        return self._budget_status()[1] if self._budget_status else 0

    @property
    def control_tensor(self) -> np.ndarray:
        """Read-only [value, minimum, maximum] rows aligned with parameters."""
        tensor = np.array(
            [(p.initial_value, p.minimum, p.maximum)
             for p in self.tunable_parameters],
            dtype=np.float64,
        ).reshape((-1, 3))
        tensor.flags.writeable = False
        return tensor

    def test_tuning(self, tuning: RmsTuning) -> RmsTuningMetrics:
        """Evaluate one proposal without changing this grid or RMS problem."""
        if self._trial is None:
            raise RuntimeError("This TunableGrid has no trial evaluation session")
        return self._trial(tuning)

    @classmethod
    def from_problem(
        cls,
        problem: RmsProblemTemplate,
        control_ranges: Mapping[str, tuple[float, float]],
        shunt_bus_names: tuple[str, ...] = KUNDUR_SHUNT_BUSES,
        line_corridors: tuple[tuple[str, str], ...] = KUNDUR_LINE_CORRIDORS,
        max_open_lines: int = 2,
        max_new_lines: int = 1,
        max_new_shunts: int = 1,
        max_test_tunings: int = 5,
        evaluator: RmsTunerEvaluator | None = None,
        max_close_lines: int = 1,
    ) -> "TunableGrid":
        grid = problem.grid
        buses_by_name = {bus.name: bus for bus in grid.buses}
        snapshot = cls(
            problem=problem,
            tunable_parameters=build_rms_tunables(grid, control_ranges),
            dispatch_options=tuple(
                DispatchOption(generator.idtag, generator.name, generator.P,
                               max(generator.Pmin, generator.P * 0.9),
                               min(generator.Pmax, generator.P * 1.1))
                for generator in grid.generators if generator.active
            ),
            openable_lines=tuple(LineOption(line.idtag, line.name) for line in grid.lines if line.active),
            closable_lines=tuple(LineOption(line.idtag, line.name) for line in grid.lines if not line.active),
            max_close_lines=max_close_lines,
            line_candidates=tuple(
                LineCandidate(
                    buses_by_name[from_name].idtag,
                    buses_by_name[to_name].idtag,
                    f"Candidate {from_name}-{to_name}",
                    0.01, 0.1, 0.1, 400.0,
                )
                for from_name, to_name in line_corridors
            ),
            shunt_options=tuple(
                ShuntOption(bus.idtag, bus.name, 1.0, 10.0)
                for bus in grid.buses if bus.name in shunt_bus_names
            ),
            max_open_lines=max_open_lines,
            max_new_lines=max_new_lines,
            max_new_shunts=max_new_shunts,
        )
        if evaluator is not None:
            return evaluator.start_session(snapshot, max_test_tunings)
        return snapshot


@dataclass(frozen=True)
class RmsTuning:
    """A candidate set of finite RMS parameter values."""

    parameters: Mapping[TunableParameter, float] = field(default_factory=dict)
    dispatch_mw: Mapping[DispatchOption, float] = field(default_factory=dict)
    open_lines: tuple[LineOption, ...] = ()
    add_lines: tuple[LineCandidate, ...] = ()
    add_shunts: tuple[ShuntOption, ...] = ()
    close_lines: tuple[LineOption, ...] = ()

    def is_accepted(
        self,
        tunable_grid: TunableGrid,
    ) -> bool:
        """Accept an unchanged model or in-range updates to allowed controls."""
        tunables = set(tunable_grid.tunable_parameters)
        indices = _parameter_indices(tunable_grid.problem)
        controls_valid = all(
            isinstance(parameter, TunableParameter)
            and parameter in tunables
            and parameter.runtime_key in indices
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and parameter.minimum <= value <= parameter.maximum
            for parameter, value in self.parameters.items()
        )
        dispatch_valid = all(
            isinstance(option, DispatchOption)
            and option in tunable_grid.dispatch_options
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and option.minimum_mw <= value <= option.maximum_mw
            for option, value in self.dispatch_mw.items()
        )
        dispatch_balanced = math.isclose(
            sum(self.dispatch_mw.get(option, option.initial_mw)
                for option in tunable_grid.dispatch_options),
            sum(option.initial_mw for option in tunable_grid.dispatch_options),
            rel_tol=0.0, abs_tol=1e-6,
        )
        return (
            controls_valid
            and dispatch_valid
            and dispatch_balanced
            and len(self.close_lines) <= tunable_grid.max_close_lines
            and len(self.open_lines) <= tunable_grid.max_open_lines
            and len(self.add_lines) <= tunable_grid.max_new_lines
            and len(self.add_shunts) <= tunable_grid.max_new_shunts
            and len(set(self.close_lines)) == len(self.close_lines)
            and set(self.close_lines).issubset(tunable_grid.closable_lines)
            and len(set(self.open_lines)) == len(self.open_lines)
            and len(set(self.add_lines)) == len(self.add_lines)
            and len(set(self.add_shunts)) == len(self.add_shunts)
            and set(self.open_lines).issubset(tunable_grid.openable_lines)
            and set(option.idtag for option in self.open_lines).isdisjoint(option.idtag for option in self.close_lines)
            and set(self.add_lines).issubset(tunable_grid.line_candidates)
            and set(self.add_shunts).issubset(tunable_grid.shunt_options)
        )


class RmsTuner(ABC):
    """Base class for participant tuners, with a one-grid evaluation helper."""

    def __init__(self, max_tests: int = 10) -> None:
        self.max_tests = max_tests
        self.tests_used = 0

    @abstractmethod
    def tune(self, tunable_grid: TunableGrid) -> RmsTuning:
        """Return the participant's proposed control values."""

    def test_tuning(
        self,
        grid: MultiCircuit,
        evaluator: RmsTunerEvaluator | None = None,
    ) -> RmsTuningMetrics:
        """Run power flow and RMS SSA for this tuner on one grid snapshot."""
        return test_tuning(self, grid, evaluator)


    def get_modal_analysis(
        self, grid: MultiCircuit | TunableGrid | dict,
    ) -> dict:
        """Inspect the supplied operating point without tuning or using test budget.

        Accepts the TunableGrid passed to tune(), a raw grid, or a reusable
        analysis snapshot. This does not automatically apply self.tune().
        """
        return analyze_grid(grid)

    def get_controller_blocks(self, device: DynamicDevice | DynamicBusDevice) -> tuple[ControllerBlock, ...]:
        """Discover native controller views without running PF/SSA or spending trials.

        :param device: RMS device whose actual symbolic model is inspected.
        :return: Classified template views and explicitly unclassified blocks.
        """
        return controller_blocks(device)

    def get_controller_parameters(self, grid: TunableGrid, controller: ControllerBlock) -> tuple[TunableParameter, ...]:
        """Select permitted handles, never merely all parameters in an equation.

        :param grid: Participant snapshot providing permissions and bounds.
        :param controller: Device-specific discovered controller view.
        :return: Existing tunable handles for this controller.
        """
        return controller_parameters(grid, controller)

    def get_controller_modes(self, grid: TunableGrid | dict, controller: ControllerBlock,
                             minimum_participation: float = 0.01,
                             include_conjugates: bool = False) -> tuple[ControllerMode, ...]:
        """Associate system modes with a controller without using tuning trials.

        :param grid: Participant snapshot or cached modal-analysis result.
        :param controller: Discovered controller view.
        :param minimum_participation: Minimum fraction of modal participation.
        :param include_conjugates: Return both conjugate members when true.
        :return: Physical modes with their controller participation and scope.
        """
        return controller_modes(grid, controller, minimum_participation, include_conjugates)

    def get_parameterized_state_matrix(self, grid: TunableGrid) -> dict:
        """Keep editable controls symbolic for fixed-point matrix/mode sensitivities."""
        return build_parameterized_state_matrix(grid)

    def evaluate_state_matrix(self, grid: TunableGrid, values=None, **options) -> np.ndarray:
        """Evaluate A at proposed controls, constructing symbolic blocks internally."""
        return evaluate_state_matrix(build_parameterized_state_matrix(grid), values, **options)

    def get_state_matrix_derivative(self, grid: TunableGrid, parameter, values=None, **options) -> np.ndarray:
        """Return analytic dA/dparameter at the grid's initialized operating point."""
        return state_matrix_derivative(build_parameterized_state_matrix(grid), parameter, values, **options)

    def get_eigenvalue_sensitivities(self, grid: TunableGrid, values=None, **options) -> dict:
        """Return mode sensitivities using the supplied grid and editable controls."""
        return eigenvalue_sensitivities(build_parameterized_state_matrix(grid), values, **options)

    def get_equilibrium_residual(self, grid: TunableGrid, values=None) -> float:
        """Check the grid's initialized operating point under proposed controls."""
        return equilibrium_residual(build_parameterized_state_matrix(grid), values)

    def get_state_matrix(
        self, grid: MultiCircuit | TunableGrid | dict, values=None, **options,
    ) -> np.ndarray:
        """Return numerical A directly from a grid.

        TunableGrid supports parameter overrides and equilibrium-check options.
        Raw grids and modal snapshots use native SSA without parameter overrides;
        descriptor snapshots also provide mass_matrix for E delta_z_dot = A delta_z.
        """
        if isinstance(grid, TunableGrid):
            return self.evaluate_state_matrix(grid, values, **options)
        if values is not None or options:
            raise TypeError('Parameter overrides and equilibrium options require a TunableGrid')
        return self.get_modal_analysis(grid)['state_matrix']

    def get_modes_from_unstable_eigenvalues(
        self, grid: MultiCircuit | TunableGrid | dict, *,
        real_tolerance: float = 1e-8, top_n: int | None = 5,
        include_conjugates: bool = False,
    ) -> tuple[dict, ...]:
        """Rank growing modes and their participating variables, worst first."""
        return get_modes_from_unstable_eigenvalues(self.get_modal_analysis(grid),
            real_tolerance=real_tolerance, top_n=top_n,
            include_conjugates=include_conjugates,
        )


class BasicRmsTuner(RmsTuner):
    """Baseline participant entry point that leaves the RMS model unchanged."""

    def tune(self, tunable_grid: TunableGrid) -> RmsTuning:
        return RmsTuning()


class RmsTunerEvaluator:
    """Validate, apply, and evaluate one RMS tuning proposal."""

    def __init__(self,
                 control_ranges: Mapping[str, tuple[float, float]] = KUNDUR_CONTROL_RANGES,
                 shunt_bus_names: tuple[str, ...] = KUNDUR_SHUNT_BUSES,
                 line_corridors: tuple[tuple[str, str], ...] = KUNDUR_LINE_CORRIDORS,
                 max_open_lines: int = 2,
                 max_new_lines: int = 1,
                 max_new_shunts: int = 1,
                 max_test_tunings: int = 5,
                 load_step_fraction: float = 0.02,
                 load_step_time_s: float = 0.2,
                 transient_duration_s: float = 3.0,
                 transient_time_step_s: float = 0.02,
                 minimum_damping_ratio: float = 0.05,
                 max_close_lines: int = 1) -> None:
        """Configure tuning permissions and acceptance thresholds.

        :param minimum_damping_ratio: Minimum oscillatory damping ratio;
            existing exercises retain the default five percent.
        :return: None.
        """
        if not math.isfinite(minimum_damping_ratio) or not 0.0 <= minimum_damping_ratio <= 1.0:
            raise ValueError('minimum_damping_ratio must be finite and between zero and one')
        else:
            pass
        self.minimum_damping_ratio: float = minimum_damping_ratio
        self.control_ranges = control_ranges
        self.shunt_bus_names = shunt_bus_names
        self.line_corridors = line_corridors
        self.max_open_lines = max_open_lines
        self.max_close_lines: int = max_close_lines
        self.max_new_lines = max_new_lines
        self.max_new_shunts = max_new_shunts
        self.max_test_tunings = max_test_tunings
        self.load_step_fraction = load_step_fraction
        self.load_step_time_s = load_step_time_s
        self.transient_duration_s = transient_duration_s
        self.transient_time_step_s = transient_time_step_s

    def start_session(self, baseline: TunableGrid, limit: int | None = None) -> TunableGrid:
        """Create a fresh per-scenario budget and an isolated participant problem.

        The closure retains the authoritative baseline and counter. Participant
        edits to their diagnostic problem cannot change the scoring baseline.
        This is cooperative API enforcement, not a Python security sandbox.
        """
        limit = self.max_test_tunings if limit is None else limit
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("Trial budget must be a nonnegative integer")
        used = 0
        lock = Lock()

        def trial(proposal):
            nonlocal used
            with lock:
                if used >= limit:
                    raise RuntimeError("The tuning test budget is exhausted")
                used += 1  # Invalid proposals and failed solves also cost a trial.
            return self.evaluate_proposal(baseline, proposal)

        def status():
            with lock:
                return limit, used

        problem = baseline.problem
        participant_problem = SmallSignalStabilityRmsDriver(
            grid=problem.grid.copy(),
            pf_results=deepcopy(problem.power_flow_results),
            sss_options=RmsSmallSignalStabilityOptions(k=0, ss_assessment_time=0.0),
        ).problem
        return replace(baseline, problem=participant_problem,
                       _trial=trial, _budget_status=status)

    def evaluate_load_step(self, grid: MultiCircuit, pf_results) -> tuple[float, float, bool]:
        """Run a fixed load increase and measure peak speed and voltage excursions."""
        disturbance_grid = grid.copy()
        if not disturbance_grid.loads:
            raise ValueError("The RMS disturbance test requires a load")
        load = disturbance_grid.loads[0]
        pending = [load.rms_model]
        load_parameter = None
        while pending:
            block = pending.pop()
            pending.extend(block.children)
            for variable in block.event_dict:
                if variable.name == "Pl0":
                    load_parameter = variable
                    break
            if load_parameter is not None:
                break
        if load_parameter is None:
            raise ValueError("The selected load has no Pl0 RMS event parameter")
        for group in disturbance_grid.rms_events_groups:
            group.active = False
        event_group = RmsEventsGroup(name="hackathon_load_step")
        disturbance_grid.add_rms_events_group(event_group)
        disturbance_grid.add_rms_event(RmsEvent(
            device=load,
            parameter=load_parameter,
            time=self.load_step_time_s,
            value=-float(load.P) / disturbance_grid.Sbase * (1.0 + self.load_step_fraction),
            group=event_group,
        ))
        driver = RmsSimulationDriver(
            grid=disturbance_grid,
            options=RmsOptions(
                time_step=self.transient_time_step_s,
                simulation_time=self.transient_duration_s,
            ),
            pf_results=pf_results,
        )
        driver.run()
        results = driver.results
        group_index = len(disturbance_grid.rms_events_groups) - 1
        converged = bool(results.converged[group_index])
        if not converged or not np.all(np.isfinite(results.values[:, :, group_index])):
            return math.inf, math.inf, False
        speed_indices = [
            index for index, name in enumerate(results.variable_array)
            if str(name).startswith("omega")
        ]
        voltage_indices = [
            index for index, name in enumerate(results.variable_array)
            if str(name).startswith("Vm")
        ]
        if not speed_indices or not voltage_indices:
            raise RuntimeError("The RMS results lack generator speeds or bus voltages")
        speeds = results.values[:, speed_indices, group_index]
        voltages = results.values[:, voltage_indices, group_index]
        peak_frequency_hz = float(np.max(np.abs(speeds - speeds[0])) * grid.fBase)
        peak_voltage_pu = float(np.max(np.abs(voltages - voltages[0])))
        return peak_frequency_hz, peak_voltage_pu, converged

    def sanitize_tuning(self, tunable_grid: TunableGrid, tuning: RmsTuning) -> RmsTuning:
        """Keep valid actions in submission order and warn about every omission.

        No values are clamped. Redispatch is atomic because dropping only one
        generator update can break the scheduled MW balance. A wrong return
        type remains an error; malformed action containers are skipped.
        """
        if not isinstance(tuning, RmsTuning):
            raise ValueError("The tuner must return an RmsTuning proposal")

        parameters = dict()
        if not isinstance(tuning.parameters, Mapping):
            warnings.warn("Skipped parameters: expected a mapping of handles to values", UserWarning, stacklevel=2)
        else:
            for parameter, value in tuning.parameters.items():
                if RmsTuning(parameters={parameter: value}).is_accepted(tunable_grid):
                    parameters[parameter] = value
                else:
                    warnings.warn(f"Skipped parameter {parameter!r}: unknown/unavailable handle "
                                  "or non-finite/out-of-range value", UserWarning, stacklevel=2)

        dispatch = dict()
        if not isinstance(tuning.dispatch_mw, Mapping):
            warnings.warn("Skipped entire redispatch: expected a mapping", UserWarning, stacklevel=2)
        elif tuning.dispatch_mw:
            valid_values = all(
                isinstance(option, DispatchOption)
                and option in tunable_grid.dispatch_options
                and isinstance(value, (int, float)) and math.isfinite(value)
                and option.minimum_mw <= value <= option.maximum_mw
                for option, value in tuning.dispatch_mw.items()
            )
            if valid_values and RmsTuning(dispatch_mw=tuning.dispatch_mw).is_accepted(tunable_grid):
                dispatch = dict(tuning.dispatch_mw)
            else:
                warnings.warn("Skipped entire redispatch: invalid handle/value or unbalanced total MW", UserWarning, stacklevel=2)

        opened: tuple[LineOption, ...] = _sanitize_action_handles(
            tuning.open_lines, tunable_grid.openable_lines, tunable_grid.max_open_lines, "open line")
        requested_closed: tuple[LineOption, ...] = _sanitize_action_handles(
            tuning.close_lines, tunable_grid.closable_lines, tunable_grid.max_close_lines, "close line")
        opened_ids: set[str] = set(option.idtag for option in opened)
        closed: tuple[LineOption, ...] = tuple(option for option in requested_closed if option.idtag not in opened_ids)
        if len(closed) != len(requested_closed):
            warnings.warn("Skipped conflicting close line: the same line is also opened", UserWarning, stacklevel=2)
        else:
            pass
        return RmsTuning(
            parameters=parameters, dispatch_mw=dispatch,
            open_lines=opened, close_lines=closed,
            add_lines=_sanitize_action_handles(tuning.add_lines, tunable_grid.line_candidates,
                              tunable_grid.max_new_lines, "new line"),
            add_shunts=_sanitize_action_handles(tuning.add_shunts, tunable_grid.shunt_options,
                               tunable_grid.max_new_shunts, "new shunt"),
        )

    def apply_tuning(self, tunable_grid: TunableGrid, tuning: RmsTuning) -> RmsProblemTemplate:
        """Apply grid and control choices, then build the candidate RMS problem."""
        problem = tunable_grid.problem
        tuning = self.sanitize_tuning(tunable_grid, tuning)
        grid = problem.grid
        generators = {generator.idtag: generator for generator in grid.generators}
        for option, value in tuning.dispatch_mw.items():
            generators[option.device_idtag].P = float(value)
        lines = {line.idtag: line for line in grid.lines}
        for line_option in tuning.open_lines:
            lines[line_option.idtag].active = False
            lines[line_option.idtag].rms_model.set_parameter_in_model('u', 0.0)
        for line_option in tuning.close_lines:
            lines[line_option.idtag].active = True
            lines[line_option.idtag].rms_model.set_parameter_in_model('u', 1.0)
        buses = {bus.idtag: bus for bus in grid.buses}
        for candidate in tuning.add_lines:
            new_line: Line = Line(
                name=candidate.name,
                bus_from=buses[candidate.bus_from_idtag],
                bus_to=buses[candidate.bus_to_idtag],
                r=candidate.resistance,
                x=candidate.reactance,
                b=candidate.susceptance,
                rate=candidate.rate_mva,
            )
            grid.add_line(new_line)
            set_rms_model(new_line, get_line_rms_template(grid.var_factory).block, grid.var_factory)
        for shunt_option in tuning.add_shunts:
            new_shunt: Shunt = Shunt(
                name=f"Tuning shunt {shunt_option.bus_name}",
                G=shunt_option.conductance_mw, B=shunt_option.susceptance_mvar,
            )
            grid.add_shunt(bus=buses[shunt_option.bus_idtag], api_obj=new_shunt)
            set_rms_model(new_shunt, get_shunt_template(grid.var_factory, phasor=False).block, grid.var_factory)
        devices = {device.idtag: device for device in problem.grid.get_all_elements_iter()
                   if isinstance(device, (DynamicDevice, DynamicBusDevice))}
        for parameter, value in tuning.parameters.items():
            devices[parameter.device_idtag].rms_model.set_parameter_in_model(
                parameter.name, float(value)
            )
        pf = problem.power_flow_results
        if tuning.dispatch_mw or tuning.open_lines or tuning.close_lines or tuning.add_lines or tuning.add_shunts:
            power_flow = PowerFlowDriver(grid)
            power_flow.run()
            pf = power_flow.results
            if not bool(pf.converged):
                raise RuntimeError("Tuned grid power flow did not converge")
        tuned_problem = RmsProblemDae(
            grid=grid,
            options=problem.options,
            pf_results=pf,
        )
        if problem._active_events_group is not None:
            tuned_problem.set_events_group(problem._active_events_group)
        return tuned_problem

    def evaluate_tuning(self, grid: MultiCircuit, tuner: RmsTuner) -> "RmsTuningMetrics":
        """Give the tuner a baseline model and evaluate its final proposal."""
        baseline_grid = grid.copy()
        power_flow = PowerFlowDriver(baseline_grid)
        power_flow.run()
        pf = power_flow.results
        if not bool(pf.converged):
            raise RuntimeError("Power flow did not converge")

        ssa = SmallSignalStabilityRmsDriver(
            grid=baseline_grid,
            pf_results=pf,
            sss_options=RmsSmallSignalStabilityOptions(ss_assessment_time=0.0),
        )
        tunable_grid = TunableGrid.from_problem(
            ssa.problem,
            self.control_ranges,
            self.shunt_bus_names,
            self.line_corridors,
            self.max_open_lines,
            self.max_new_lines,
            self.max_new_shunts,
            max_close_lines=self.max_close_lines,
        )
        participant_grid = self.start_session(tunable_grid)
        tuning = tuner.tune(participant_grid)
        return self.evaluate_proposal(tunable_grid, tuning)

    def evaluate_proposal(self, tunable_grid: TunableGrid, tuning: RmsTuning) -> "RmsTuningMetrics":
        """Score a proposal on a fresh copy of the same baseline grid."""
        tuning = self.sanitize_tuning(tunable_grid, tuning)
        tuned_grid = tunable_grid.problem.grid.copy()
        power_flow = PowerFlowDriver(tuned_grid)
        power_flow.run()
        pf = power_flow.results
        if not bool(pf.converged):
            raise RuntimeError("Power flow did not converge")
        ssa = SmallSignalStabilityRmsDriver(
            grid=tuned_grid,
            pf_results=pf,
            sss_options=RmsSmallSignalStabilityOptions(ss_assessment_time=0.0),
        )
        candidate_grid: TunableGrid = TunableGrid(
            problem=ssa.problem,
            tunable_parameters=tunable_grid.tunable_parameters,
            dispatch_options=tunable_grid.dispatch_options,
            openable_lines=tunable_grid.openable_lines,
            closable_lines=tunable_grid.closable_lines,
            line_candidates=tunable_grid.line_candidates,
            shunt_options=tunable_grid.shunt_options,
            max_open_lines=tunable_grid.max_open_lines,
            max_close_lines=tunable_grid.max_close_lines,
            max_new_lines=tunable_grid.max_new_lines,
            max_new_shunts=tunable_grid.max_new_shunts,
        )
        ssa.problem = self.apply_tuning(candidate_grid, tuning)
        pf = ssa.problem.power_flow_results
        ssa.run()

        eigenvalues = np.asarray(ssa.results.eigenvalues, dtype=complex)
        # Descriptor systems include infinite eigenvalues for algebraic
        # constraints. Score their finite physical spectrum, while still
        # rejecting NaNs and non-finite results from an ordinary state matrix.
        if np.any(np.isnan(eigenvalues)):
            raise RuntimeError("Small-signal analysis returned NaN eigenvalues")
        if ssa.problem.get_diff_var_number():
            eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
        if eigenvalues.size == 0 or not np.all(np.isfinite(eigenvalues)):
            raise RuntimeError("Small-signal analysis returned no finite eigenvalues")
        neutral_modes = np.abs(eigenvalues) <= 1e-8
        physical_modes = eigenvalues[~neutral_modes]
        if physical_modes.size == 0:
            raise RuntimeError("Small-signal analysis returned no physical modes")
        frequencies_hz = np.abs(eigenvalues.imag) / (2.0 * np.pi)
        oscillatory = frequencies_hz > 0.05
        damping = np.zeros(eigenvalues.size, dtype=float)
        damping[~neutral_modes] = -eigenvalues[~neutral_modes].real / np.abs(physical_modes)
        min_damping = float(np.min(damping[oscillatory])) if np.any(oscillatory) else None
        max_real = float(np.max(physical_modes.real))
        voltage = np.abs(np.asarray(pf.voltage))
        loading = np.abs(np.asarray(pf.loading))
        voltage_violations = int(np.count_nonzero((voltage < 0.95) | (voltage > 1.05)))
        thermal_violations = int(np.count_nonzero(loading > 1.0))
        voltage_by_bus_id = {
            bus.idtag: abs(pf.voltage[index])
            for index, bus in enumerate(tuned_grid.buses)
        }
        shunt_consumption = sum(
            shunt.G * voltage_by_bus_id[shunt.bus.idtag] ** 2
            for shunt in tuned_grid.shunts if shunt.active
        )
        peak_frequency_hz, peak_voltage_pu, transient_converged = self.evaluate_load_step(
            tuned_grid, pf
        )
        return RmsTuningMetrics(
            active_losses_mw=float(np.sum(np.asarray(pf.losses).real) + shunt_consumption),
            max_eigenvalue_real=max_real,
            min_oscillatory_damping=min_damping,
            voltage_violations=voltage_violations,
            thermal_violations=thermal_violations,
            stable=(transient_converged
                    and int(np.count_nonzero(neutral_modes)) <= 1
                    and max_real < -1e-5
                    and (min_damping is None or min_damping >= self.minimum_damping_ratio)),
            eigenvalue_count=int(eigenvalues.size),
            neutral_mode_count=int(np.count_nonzero(neutral_modes)),
            peak_frequency_deviation_hz=peak_frequency_hz,
            peak_voltage_deviation_pu=peak_voltage_pu,
            transient_converged=transient_converged,
        )


@dataclass(frozen=True)
class RmsTuningMetrics:
    """Power-flow, small-signal, and load-step RMS evaluation results.

    active_losses_mw includes branch and shunt real-power losses. Generator
    internal losses are absent when the RMS generator model omits resistance.
    """

    active_losses_mw: float
    max_eigenvalue_real: float
    min_oscillatory_damping: float | None
    voltage_violations: int
    thermal_violations: int
    stable: bool
    eigenvalue_count: int
    neutral_mode_count: int
    peak_frequency_deviation_hz: float
    peak_voltage_deviation_pu: float
    transient_converged: bool


def test_tuning(
    tuner: RmsTuner,
    grid: MultiCircuit,
    evaluator: RmsTunerEvaluator | None = None,
) -> RmsTuningMetrics:
    """Evaluate one participant tuner against a grid snapshot."""
    if tuner.tests_used >= tuner.max_tests:
        raise RuntimeError("The tuner has used all of its test evaluations")
    tuner.tests_used += 1
    if evaluator is None:
        evaluator = RmsTunerEvaluator()
    return evaluator.evaluate_tuning(grid, tuner)
