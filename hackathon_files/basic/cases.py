"""Reproducible grid-action and controller RMS stabilization challenges."""
from pathlib import Path
import math

import numpy as np

from VeraGridEngine.api import open_file
from VeraGridEngine.enumerations import TapModuleControl, TapPhaseControl
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Devices.Injections.generator import Generator
from VeraGridEngine.Templates.Rms.shunt_template import get_shunt_template
from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import SmallSignalStabilityRmsDriver
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_options import RmsSmallSignalStabilityOptions
from VeraGridEngine.Templates.Rms.genrow1_rms_template import get_genrow1_rms_template
from VeraGridEngine.Templates.Rms.line_rms_template import get_line_rms_template
from VeraGridEngine.Templates.Rms.load_rms_template import get_load_rms_template
from VeraGridEngine.Templates.Rms.transformer_rms_template import get_transformer2w_rms
from VeraGridEngine.Utils.Symbolic.bus_rms_template import initialize_bus_rms
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model
from hackathon_files.basic.controllers import add_variant_controllers
from hackathon_files.basic.actions import BasicActionCatalog, prepare_action_scenario
from hackathon_files.src.rms_tuning import RmsTunerEvaluator, RmsTuning, TunableGrid, build_rms_tunables


class NetworkSpec:
    """Identify a source network without retaining mutable grid objects."""

    __slots__ = ('name', 'source', 'load_scale', 'voltage', 'tap_blend')

    def __init__(self, name: str, source: str, load_scale: float = 1.0,
                 voltage: float = 1.02, tap_blend: float | None = None) -> None:
        """Store the source relative to the repository root.

        :param name: Short network identifier.
        :param source: Relative source file path.
        :param load_scale: Loading multiplier for a feasible benchmark snapshot.
        :param voltage: Generator voltage setpoint in per unit.
        :param tap_blend: Fraction of original tap deviation to retain, or None.
        """
        self.name: str = name
        self.source: str = source
        self.load_scale: float = load_scale
        self.voltage: float = voltage
        self.tap_blend: float | None = tap_blend


def network_specs() -> tuple[NetworkSpec, ...]:
    """List ten electrically different small and medium networks.

    :return: Source specifications in deterministic generation order.
    """
    return (
        NetworkSpec('lynn5', 'Grids_and_profiles/grids/Lynn 5 Bus pv.gridcal', load_scale=0.8),
        NetworkSpec('six_bus', 'Grids_and_profiles/grids/6-node-ACOPF.gridcal', load_scale=0.45),
        NetworkSpec('ieee9', 'hackathon_files/grids/ieee9_rms_transformers.veragrid'),
        NetworkSpec('kundur11', 'hackathon_files/grids/kundur_rms_65.veragrid'),
        NetworkSpec('ieee14', 'src/tests/data/grids/case14.gridcal'),
        NetworkSpec('ieee24', 'src/tests/data/grids/case24_ieee_rts.matpower'),
        NetworkSpec('ieee30', 'hackathon_files/grids/ieee30_rms.veragrid'),
        NetworkSpec('ieee39', 'hackathon_files/grids/ieee39_rms.veragrid'),
        NetworkSpec('ieee57', 'src/tests/data/grids/IEEE57.gridcal', load_scale=0.38, voltage=1.0, tap_blend=0.3),
        NetworkSpec('ieee118', 'Grids_and_profiles/grids/IEEE 118 Bus - ntc_areas.gridcal', load_scale=0.4, voltage=1.0, tap_blend=0.0),
    )


def case_trial_budget(name: str) -> int | float:
    """Split the 60 public cases evenly between unlimited and 20 trials."""
    for network_index, spec in enumerate(network_specs()):
        if name == f'basic_switching_{spec.name}_01':
            return math.inf if network_index % 2 else 20
        for variant in range(1, 6):
            if name == f"basic_{spec.name}_{variant:02d}":
                return math.inf if (network_index + variant) % 2 else 20
    return 20


def is_switching_case(name: str) -> bool:
    """Identify serialized exercises with fixed controls and network-action permissions."""
    return name.startswith(('basic_switching_', 'switching_'))


def basic_evaluator() -> RmsTunerEvaluator:
    """Expose controller and physical-action search with case-local permissions.

    :return: Existing evaluator with parameter-only action permissions.
    """
    # Genrow uses +Kp*(omega-omega_ref); negative Kp is negative feedback.
    return RmsTunerEvaluator(
        control_ranges=dict(Kp=(-1000.0, 300.0), Kp_icl=(0.001, 5.0), Ki_icl=(1.0, 500.0),
                            Kp_pol=(0.01, 1.0), Ki_pol=(0.01, 5.0),
                            Kp_vac=(0.01, 1.0), Ki_vac=(0.01, 5.0),
                            Kp_pll=(0.001, 0.5), Ki_pll=(0.01, 1.0),
                            Kdp=(0.001, 1.0), Kdq=(0.001, 1.0),
                            tau_P=(0.001, 1.0), tau_Q=(0.001, 1.0)),
        shunt_bus_names=(), line_corridors=(),
        max_open_lines=1, max_new_lines=1, max_new_shunts=1,
        max_close_lines=1, minimum_damping_ratio=0.025,
    )


def build_network(spec: NetworkSpec) -> MultiCircuit:
    """Attach educational RMS models to an existing AC benchmark.

    :param spec: Static source network or previously prepared RMS network.
    :return: Independent grid with consistent Genrow and constant-power models.
    """
    root: Path = Path(__file__).resolve().parents[2]
    grid: MultiCircuit = open_file(str(root / spec.source))
    # No source file is edited. Dynamic models are intentionally synthesized,
    # rather than represented as the original benchmark's machine data.
    for group in grid.rms_events_groups:
        group.active = False
    # Omit out-of-service equipment from this static topology snapshot; its
    # default RMS equations would otherwise energize a disconnected device.
    for line in tuple(grid.lines):
        if not line.active:
            grid.delete_line(line)
        else:
            pass
    for transformer in tuple(grid.transformers2w):
        if not transformer.active:
            grid.delete_transformer2w(transformer)
        else:
            pass
    for generator in tuple(grid.generators):
        if not generator.active:
            grid.delete_generator(generator)
        else:
            pass
    for bus in grid.buses:
        # IEEE57 leaves nominal kV unspecified. A common positive base leaves
        # its per-unit AC data intact and permits RMS SI-base conversion.
        if bus.Vnom <= 0.0:
            bus.Vnom = 110.0
        else:
            pass
        initialize_bus_rms(bus, grid.var_factory)
    for bus in grid.buses:
        if bus.is_slack and not any(generator.bus is bus and generator.active for generator in grid.generators):
            grid.add_generator(bus=bus, api_obj=Generator(name=f'Source at {bus.name}', vset=1.02))
        else:
            pass
    for generator in grid.generators:
        # Static load-flow files do not supply machine impedances. Use the
        # template's educational system-base impedance for every new machine.
        generator.R1 = 0.0
        generator.X1 = 0.03333333333333333
        generator.Vset = spec.voltage
        generator.P *= spec.load_scale
        set_rms_model(generator, get_genrow1_rms_template(grid.var_factory).block, grid.var_factory)
    for load in grid.loads:
        load.P *= spec.load_scale
        load.Q *= spec.load_scale
        set_rms_model(load, get_load_rms_template(grid.var_factory).block, grid.var_factory)
    for line in grid.lines:
        set_rms_model(line, get_line_rms_template(grid.var_factory).block, grid.var_factory)
    for shunt in grid.shunts:
        set_rms_model(shunt, get_shunt_template(grid.var_factory, phasor=False).block, grid.var_factory)
    for transformer in grid.transformers2w:
        # The original 57/118 snapshots contain tap settings that cause
        # persistent limit violations. Fix a feasible tap schedule before the
        # challenge; the participant may subsequently change controller gains only.
        if spec.tap_blend is not None:
            transformer.tap_module = 1.0 + spec.tap_blend * (transformer.tap_module - 1.0)
            transformer.tap_phase = 0.0
            transformer.tap_module_control_mode = TapModuleControl.fixed
            transformer.tap_phase_control_mode = TapPhaseControl.fixed
        else:
            pass
        set_rms_model(transformer, get_transformer2w_rms(grid.var_factory).block, grid.var_factory)
    return grid


def make_variant(base: MultiCircuit, name: str, variant: int) -> MultiCircuit:
    """Change operating point, inertia, damping and initial feedback.

    :param base: Network that remains untouched.
    :param name: Family identifier used in the saved name.
    :param variant: Zero-based variant index, from zero to four.
    :return: A reproducible candidate; validation is performed before saving.
    """
    grid: MultiCircuit = base.copy()
    rng: np.random.Generator = np.random.default_rng(20261119 + variant)
    scale: float = 0.84 + 0.025 * variant
    grid.name = f'basic_{name}_{variant + 1:02d}'
    for load in grid.loads:
        load.P *= scale
        load.Q *= scale * (0.96 + 0.02 * variant)
    for generator in grid.generators:
        generator.P *= scale
        generator.rms_model.set_parameter_in_model('M', float(rng.uniform(95.0, 145.0)))
        generator.rms_model.set_parameter_in_model('D', float(rng.uniform(80.0, 95.0)))
        generator.rms_model.set_parameter_in_model('Kp', float(rng.uniform(120.0 + 8 * variant, 145.0 + 8 * variant)))
        # Ki would introduce absolute-angle feedback; this collection studies
        # speed feedback while retaining the usual neutral angle symmetry.
        generator.rms_model.set_parameter_in_model('Ki', 0.0)
    add_variant_controllers(grid, variant)
    prepare_action_scenario(grid, variant)
    return grid


def initialize_grid(grid: MultiCircuit) -> TunableGrid:
    """Solve PF and initialize the exact problem presented to a tuner.

    :param grid: Independent case grid, which may be initialized in place.
    :return: Existing TunableGrid API with a case-specific evaluation budget.
    """
    power_flow: PowerFlowDriver = PowerFlowDriver(grid, options=PowerFlowOptions(tolerance=1e-10, max_iter=100))
    power_flow.run()
    if not bool(power_flow.results.converged):
        raise RuntimeError(f'{grid.name}: power flow did not converge')
    else:
        pass
    driver: SmallSignalStabilityRmsDriver = SmallSignalStabilityRmsDriver(
        grid=grid, pf_results=power_flow.results,
        sss_options=RmsSmallSignalStabilityOptions(k=0, ss_assessment_time=0.0),
    )
    evaluator: RmsTunerEvaluator = basic_evaluator()
    ranges: dict[str, tuple[float, float]] = dict(evaluator.control_ranges)
    network_only = is_switching_case(grid.name)
    if any('Basic dynamic' in generator.rms_model.name for generator in grid.generators):
        ranges.pop('Kp')
    else:
        pass
    controller_only = not network_only and grid.name.startswith("basic_") and grid.name.endswith("_02")
    actions: BasicActionCatalog = BasicActionCatalog(grid)
    snapshot: TunableGrid = TunableGrid(
        problem=driver.problem,
        tunable_parameters=() if network_only else build_rms_tunables(grid, ranges),
        dispatch_options=() if controller_only or network_only else actions.dispatch,
        openable_lines=() if controller_only else actions.openable,
        closable_lines=() if controller_only else actions.closable,
        line_candidates=() if controller_only else actions.lines,
        shunt_options=() if controller_only else actions.shunts,
        max_open_lines=0 if controller_only else 1,
        max_close_lines=0 if controller_only else 1,
        max_new_lines=0 if controller_only else 1,
        max_new_shunts=0 if controller_only else 1,
    )
    return evaluator.start_session(snapshot, limit=case_trial_budget(grid.name))


def reference_tuning(grid: TunableGrid, gain: float = -400.0) -> RmsTuning:
    """Construct a stabilizing candidate using only permitted handles.

    :param grid: Current initialized case, never another case's handles.
    :param gain: Signed speed-feedback gain within the published bounds.
    :return: Controller gains and any required outage-restoration closing.
    """
    values: dict[str, float] = dict(Kp=gain, Kp_icl=0.5, Ki_icl=5.0,
                                   Kp_pol=0.2, Ki_pol=0.2, Kp_pll=0.05, Ki_pll=0.1, Kp_vac=0.2, Ki_vac=0.2,
                                   Kdp=0.05, Kdq=0.05, tau_P=0.01, tau_Q=0.01)
    if grid.problem.grid.name.startswith('basic_six_bus_'):
        values['Kp_icl'] = 3.0
    else:
        pass
    restoration: tuple = tuple(option for option in grid.closable_lines
                                if option.name.startswith(('Restore outage:', 'Switching relief:')))
    return RmsTuning(parameters={parameter: values[parameter.name] for parameter in grid.tunable_parameters},
                     close_lines=restoration)


LIMITED_CASE_STEMS = (
    'basic_lynn5_01', 'basic_lynn5_02', 'basic_lynn5_03',
    'basic_lynn5_04', 'basic_lynn5_05', 'basic_switching_lynn5_01',
    'basic_six_bus_01', 'basic_six_bus_02', 'basic_six_bus_03', 'basic_six_bus_04',
)


def case_files(selection: str = 'all') -> tuple[Path, ...]:
    """Find the bundled challenge grids in reproducible order.

    :param selection: All 60 basic cases, or the ten smallest in limited.
    :return: Serialized controller and network-action grids, one per case.
    """
    directory: Path = Path(__file__).with_name('testgrids')
    if selection == 'limited':
        return tuple(directory / f'{name}.veragrid' for name in LIMITED_CASE_STEMS)
    if selection != 'all':
        raise ValueError("Case selection must be 'all' or 'limited'")
    return (tuple(directory / f'basic_{spec.name}_{variant + 1:02d}.veragrid'
                  for spec in network_specs() for variant in range(5))
            + tuple(directory / f'basic_switching_{spec.name}_01.veragrid' for spec in network_specs()))
