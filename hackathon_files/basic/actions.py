"""Case-local switching, reinforcement, shunt and dispatch permissions."""
import numpy as np

from VeraGridEngine.Devices.Branches.line import Line
from VeraGridEngine.Devices.Substation.bus import Bus
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
from VeraGridEngine.Templates.Rms.line_rms_template import get_line_rms_template
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model
from hackathon_files.src.rms_tuning import DispatchOption, LineCandidate, LineOption, ShuntOption


def connected_without(grid: MultiCircuit, excluded_id: str) -> bool:
    """Check active AC connectivity before offering a single-line opening.

    :param grid: Connected basic network with lines and two-winding transformers.
    :param excluded_id: Line omitted from the candidate topology.
    :return: Whether every bus remains reachable from the first bus.
    """
    indices: dict[str, int] = dict((bus.idtag, index) for index, bus in enumerate(grid.buses))
    reached: np.ndarray = np.zeros(len(grid.buses), dtype=bool)
    reached[0] = True
    changed: bool = True
    while changed:
        changed = False
        for collection in (grid.lines, grid.transformers2w):
            for branch in collection:
                if branch.active and branch.idtag != excluded_id:
                    left: int = indices[branch.bus_from.idtag]
                    right: int = indices[branch.bus_to.idtag]
                    if reached[left] != reached[right]:
                        reached[left] = True
                        reached[right] = True
                        changed = True
                    else:
                        pass
                else:
                    pass
    return bool(np.all(reached))


def prepare_action_scenario(grid: MultiCircuit, variant: int) -> None:
    """Introduce real outages or normally-open standby circuits.

    :param grid: Independently built controller scenario.
    :param variant: Zero selects a screened outage; others add a standby circuit.
    """
    selected: Line | None = None
    if variant == 0:
        line: Line
        for line in grid.lines:
            if selected is None and line.active and connected_without(grid, line.idtag):
                line.active = False
                trial: PowerFlowDriver = PowerFlowDriver(grid, options=PowerFlowOptions(tolerance=1e-10, max_iter=100))
                trial.run()
                voltage: np.ndarray = np.abs(trial.results.voltage)
                loading: np.ndarray = np.abs(trial.results.loading)
                if (bool(trial.results.converged) and np.all(np.isfinite(voltage))
                        and np.all((voltage >= 0.95) & (voltage <= 1.05))
                        and np.all(np.isfinite(loading)) and np.all(loading <= 1.0)):
                    selected = line
                    line.name = f'Restore outage: {line.name}'
                    line.rms_model.set_parameter_in_model('u', 0.0)
                else:
                    line.active = True
            else:
                pass
    else:
        pass
    if selected is None:
        original: Line = grid.lines[0]
        standby: Line = Line(
            name=f'Normally open standby: {original.name}',
            bus_from=original.bus_from, bus_to=original.bus_to,
            r=original.R, x=original.X, b=original.B, g=original.G,
            rate=original.rate, active=False,
        )
        grid.add_line(standby)
        set_rms_model(standby, get_line_rms_template(grid.var_factory).block, grid.var_factory)
        standby.rms_model.set_parameter_in_model('u', 0.0)
    else:
        pass


class BasicActionCatalog:
    """Bound physical actions to the current snapshot's device identities."""

    __slots__ = ('dispatch', 'openable', 'closable', 'lines', 'shunts')

    def __init__(self, grid: MultiCircuit) -> None:
        """Construct feasible ranges and non-islanding single-line options.

        :param grid: Initialized basic case with its original device identities.
        """
        # Redispatch remains atomic and MW-balanced in the shared evaluator.
        self.dispatch: tuple[DispatchOption, ...] = tuple(
            DispatchOption(generator.idtag, generator.name, generator.P,
                           max(generator.Pmin, generator.P - max(abs(generator.P) * 0.05, grid.Sbase * 0.005)),
                           min(generator.Pmax, generator.P + max(abs(generator.P) * 0.05, grid.Sbase * 0.005)))
            for generator in grid.generators
            if generator.active and generator.Pmin <= generator.P <= generator.Pmax
            and max(generator.Pmin, generator.P - max(abs(generator.P) * 0.05, grid.Sbase * 0.005))
            < min(generator.Pmax, generator.P + max(abs(generator.P) * 0.05, grid.Sbase * 0.005)))
        self.openable: tuple[LineOption, ...] = tuple(
            LineOption(line.idtag, line.name) for line in grid.lines
            if line.active and connected_without(grid, line.idtag))
        self.closable: tuple[LineOption, ...] = tuple(
            LineOption(line.idtag, line.name) for line in grid.lines if not line.active)
        # Reinforcement is a new circuit, distinct from closing the existing
        # standby circuit. Use a physical source corridor and its actual rating.
        corridor: Line = grid.lines[min(1, len(grid.lines) - 1)]
        self.lines: tuple[LineCandidate, ...] = tuple((LineCandidate(
            corridor.bus_from.idtag, corridor.bus_to.idtag,
            f'Reinforce corridor: {corridor.name}', corridor.R, corridor.X,
            corridor.B, corridor.rate,
        ),))
        # Both capacitor and reactor choices are available at an actual load
        # bus; no artificial conductance or hidden active-power injection.
        bus: Bus = grid.loads[0].bus
        self.shunts: tuple[ShuntOption, ...] = tuple(
            ShuntOption(bus.idtag, bus.name, 0.0, sign * 0.02 * grid.Sbase)
            for sign in (-1.0, 1.0))
