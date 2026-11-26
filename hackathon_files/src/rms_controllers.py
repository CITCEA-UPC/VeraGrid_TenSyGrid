"""Controller views for native flattened templates and arbitrary block trees."""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from VeraGridEngine.Devices.Parents.dynamic_parent import DynamicDevice
from VeraGridEngine.Devices.Parents.dynamic_bus_parent import DynamicBusDevice
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Expr, Var
from hackathon_files.src.rms_diagnostics import analyze_grid

if TYPE_CHECKING:
    from hackathon_files.src.rms_tuning import TunableGrid, TunableParameter
else:
    pass


class ControllerRole(Enum):
    """Explicit roles supported by controller discovery adapters."""

    PLL = 'pll'
    GOVERNOR = 'governor'
    OUTER_POWER = 'outer_power'
    OUTER_VOLTAGE = 'outer_voltage'
    INNER_CURRENT = 'inner_current'
    DROOP = 'droop'
    UNCLASSIFIED = 'unclassified'


class ParticipationScope(Enum):
    """Distinguish controller-owned coordinates from affected machine states."""

    OWNED = 'owned_variables'
    AFFECTED = 'affected_states'


class ControllerBlock:
    """Diagnostic view; views may share equations and parameter handles."""

    __slots__ = ('device_idtag', 'device_name', 'role', 'path', 'variables',
                 'state_equations', 'algebraic_equations', 'parameter_names',
                 'affected_states', 'template_view')

    def __init__(self, device: DynamicDevice | DynamicBusDevice, block: Block,
                 role: ControllerRole, path: str, variable_names: tuple[str, ...],
                 parameter_names: tuple[str, ...], template_view: bool = True) -> None:
        """Select a known controller's symbols from a physical model block.

        :param device: Owner of the selected symbolic equations.
        :param block: Actual block; no equations are synthesized or changed.
        :param role: Explicit role supplied by a template adapter or the caller.
        :param path: Diagnostic path, including a view name for flattened models.
        :param variable_names: Exact template-owned coordinate names.
        :param parameter_names: Exact controller parameter names, not permissions.
        :param template_view: Whether a flattened-template adapter made this view.
        """
        self.device_idtag: str = device.idtag
        self.device_name: str = device.name
        self.role: ControllerRole = role
        self.path: str = path
        self.template_view: bool = template_view
        self.variables: tuple[Var, ...] = tuple(
            variable for collection in (block.state_vars, block.algebraic_vars)
            for variable in collection if variable.name in variable_names)
        # Differential symbols must be included when selecting PI/filter
        # equations; participation itself is attached to their base coordinates.
        owned: set[int] = set(variable.uid for variable in self.variables)
        selection: set[int] = owned.copy()
        variable: Var
        for variable in self.variables:
            if variable.diff_var is not None:
                selection.add(variable.diff_var.uid)
            else:
                pass
        self.state_equations: tuple[Expr, ...] = tuple(
            equation for variable, equation in zip(block.state_vars, block.state_eqs)
            if variable.uid in owned)
        # Include equations directly containing these coordinates. These are
        # connected equation views, not a claim that the plant decomposes into
        # independent closed-loop subsystems.
        self.algebraic_equations: tuple[Expr, ...] = tuple(
            equation for equation in block.algebraic_eqs
            if any(variable.uid in selection for variable in equation.get_vars()))
        used: set[int] = set(variable.uid
                             for collection in (self.state_equations, self.algebraic_equations)
                             for equation in collection for variable in equation.get_vars())
        self.parameter_names: tuple[str, ...] = tuple(
            variable.name for variable in block.event_dict
            if variable.name in parameter_names and variable.uid in used)
        self.affected_states: tuple[Var, ...] = tuple(
            variable for variable, equation in zip(block.state_vars, block.state_eqs)
            if variable.uid not in owned
            and any(signal.uid in owned for signal in equation.get_vars()))


class ControllerMode:
    """One system mode associated with a controller's selected coordinates."""

    __slots__ = ('index', 'eigenvalue', 'frequency_hz', 'damping_ratio',
                 'participation', 'scope', 'variable_uids')

    def __init__(self, index: int, eigenvalue: complex, participation: float,
                 scope: ParticipationScope, variable_uids: tuple[int, ...]) -> None:
        """Record a finite physical system mode without recomputing SSA.

        :param index: Column in the supplied modal snapshot.
        :param eigenvalue: System eigenvalue, not an isolated controller pole.
        :param participation: Fraction of the mode's participation in this view.
        :param scope: Whether coordinates are owned or affected downstream states.
        :param variable_uids: Coordinates used to compute this fraction.
        """
        self.index: int = index
        self.eigenvalue: complex = eigenvalue
        self.frequency_hz: float = abs(eigenvalue.imag) / (2.0 * np.pi)
        self.damping_ratio: float = -eigenvalue.real / abs(eigenvalue)
        self.participation: float = participation
        self.scope: ParticipationScope = scope
        self.variable_uids: tuple[int, ...] = variable_uids


def controller_blocks(device: DynamicDevice | DynamicBusDevice) -> tuple[ControllerBlock, ...]:
    """Discover native template views and retain unknown blocks unclassified.

    :param device: Device whose RMS model is inspected without mutation.
    :return: Explicitly classified native loops or unclassified physical blocks.
    """
    result: list[ControllerBlock] = list()
    root: Block = device.rms_model
    pending: list[tuple[Block, str]] = list(((root, root.name or 'root'),))
    block: Block
    path: str
    while pending:
        block, path = pending.pop()
        names: set[str] = set(variable.name for variable in block.event_dict)
        coordinates: set[str] = set(variable.name for collection in (block.state_vars, block.algebraic_vars)
                                    for variable in collection)
        # These are explicit adapters for the native template contracts. The
        # full parameter/output signature is required; arbitrary names such as
        # 'PLL' in a custom model never trigger automatic classification.
        profiles: tuple[tuple[ControllerRole, tuple[str, ...], tuple[str, ...]], ...]
        gfl_parameters = {'Kp_pll', 'Ki_pll', 'Kp_pol', 'Ki_pol', 'Kp_icl', 'Ki_icl'}
        gfl_coordinates = {'theta', 'y_PLL_integrator', 'i_d_ref', 'i_q_ref', 'y_vd_hat', 'y_vq_hat'}
        gfl_active_modes = {'u_Pac_ctrl', 'u_Pdc_ctrl', 'u_Vdc_ctrl'}
        gfl_reactive_modes = {'u_Qac_ctrl', 'u_Vac_ctrl'}
        if (gfl_parameters <= names and gfl_coordinates <= coordinates
                and gfl_active_modes & coordinates and gfl_reactive_modes & coordinates):
            profiles = (
                (ControllerRole.PLL, ('theta', 'vd', 'vq', 'y_PLL_integrator', 'u_PLL_integrator'), ('Kp_pll', 'Ki_pll')),
                (ControllerRole.OUTER_POWER, ('i_d_ref', 'i_q_ref', 'u_Pac_ctrl', 'u_Qac_ctrl', 'u_Pdc_ctrl', 'u_Vdc_ctrl', 'u_Vac_ctrl'), ('Kp_pol', 'Ki_pol', 'Kp_vdc', 'Ki_vdc', 'Kp_vac', 'Ki_vac')),
                (ControllerRole.INNER_CURRENT, ('y_vd_hat', 'u_vd_hat', 'y_vq_hat', 'u_vq_hat'), ('Kp_icl', 'Ki_icl')),
            )
        elif {'Kdp', 'Kdq', 'tau_P', 'tau_Q', 'Kp_icl', 'Ki_icl'}.issubset(names) and {'y_P_loop', 'y_Q_loop', 'y_vd_ctrl', 'y_vq_ctrl', 'y_id_ctrl', 'y_iq_ctrl'}.issubset(coordinates):
            profiles = (
                (ControllerRole.DROOP, ('omega', 'V', 'theta', 'y_P_loop', 'y_Q_loop'), ('Kdp', 'Kdq', 'tau_P', 'tau_Q')),
                (ControllerRole.OUTER_VOLTAGE, ('y_vd_ctrl', 'u_vd_ctrl', 'y_vq_ctrl', 'u_vq_ctrl', 'y_aw_d', 'y_aw_q', 'i_d_ref_aw', 'i_q_ref_aw', 'aw_d_in', 'aw_q_in'), ('Kp_icl', 'Ki_icl')),
                (ControllerRole.INNER_CURRENT, ('y_id_ctrl', 'u_id_ctrl', 'y_iq_ctrl', 'u_iq_ctrl'), ('Kp_icl', 'Ki_icl')),
            )
        elif {'M', 'D', 'Kp', 'Ki', 'tm0'}.issubset(names) and {'delta', 'omega', 'tm', 'et'}.issubset(coordinates):
            profiles = ((ControllerRole.GOVERNOR, ('tm', 'et'), ('Kp', 'Ki')),)
        else:
            profiles = tuple()
        role: ControllerRole
        variables: tuple[str, ...]
        parameters: tuple[str, ...]
        if profiles:
            for role, variables, parameters in profiles:
                result.append(ControllerBlock(device, block, role, f'{path}/{role.value}', variables, parameters))
        elif block.state_vars or block.algebraic_vars:
            result.append(ControllerBlock(device, block, ControllerRole.UNCLASSIFIED, path,
                                          tuple(coordinates), tuple(names), template_view=False))
        else:
            pass
        index: int
        child: Block
        for index, child in reversed(tuple(enumerate(block.children))):
            pending.append((child, f'{path}/{index}:{child.name or "block"}'))
    return tuple(result)


def controller_parameters(grid: TunableGrid, controller: ControllerBlock) -> tuple[TunableParameter, ...]:
    """Return only the current snapshot's permitted controller parameter handles.

    :param grid: Initialized participant snapshot owning the permission catalog.
    :param controller: Device-specific view returned by discovery.
    :return: Existing handles, including initial values and inclusive bounds.
    """
    # Parameters present in equations are not necessarily editable. Intersect
    # with the catalog rather than inventing new handles or widening bounds.
    return tuple(parameter for parameter in grid.tunable_parameters
                 if parameter.device_idtag == controller.device_idtag
                 and parameter.name in controller.parameter_names)


def controller_modes(source: TunableGrid | dict, controller: ControllerBlock,
                     minimum_participation: float = 0.01,
                     include_conjugates: bool = False) -> tuple[ControllerMode, ...]:
    """Associate full-system modes with controller coordinates by symbolic UID.

    :param source: Participant snapshot or cached get_modal_analysis() result.
    :param controller: Selected loop; no isolated controller eigensolve is made.
    :param minimum_participation: Inclusive fraction of total modal participation.
    :param include_conjugates: Retain both members of complex conjugate pairs.
    :return: Finite non-neutral modes, ordered by decreasing participation.
    """
    if not np.isfinite(minimum_participation) or not 0.0 <= minimum_participation <= 1.0:
        raise ValueError('minimum_participation must be between zero and one')
    else:
        pass
    analysis: dict = analyze_grid(source)
    owned: set[int] = set(variable.uid for variable in controller.variables)
    labels: tuple = analysis['variables']
    selected: np.ndarray = np.array(list(label['device_idtag'] == controller.device_idtag
                                         and label['uid'] in owned for label in labels), dtype=bool)
    scope: ParticipationScope = ParticipationScope.OWNED
    mass: np.ndarray | None = analysis['mass_matrix']
    if mass is not None:
        # A descriptor snapshot includes algebraic coordinates as well. A
        # zero mass column does not provide a controller-owned dynamic state.
        selected &= np.any(mass != 0.0, axis=0)
    else:
        pass
    if not np.any(selected):
        # Static governor outputs disappear during ordinary state reduction.
        # Report their downstream machine-state association explicitly, rather
        # than pretending an algebraic governor owns the rotor state.
        owned = set(variable.uid for variable in controller.affected_states)
        selected = np.array(list(label['device_idtag'] == controller.device_idtag
                                 and label['uid'] in owned for label in labels), dtype=bool)
        scope = ParticipationScope.AFFECTED
    else:
        pass
    factors: np.ndarray = analysis['participation_factors']
    totals: np.ndarray = np.sum(factors, axis=0)
    fractions: np.ndarray = np.divide(np.sum(factors[selected, :], axis=0), totals,
                                     out=np.zeros_like(totals), where=totals > 0.0)
    values: np.ndarray = analysis['eigenvalues']
    indices: np.ndarray = np.argsort(-fractions, kind='stable')
    matches: tuple[int, ...] = tuple(label['uid'] for label, chosen in zip(labels, selected) if chosen)
    result: list[ControllerMode] = list()
    index: int
    for index in indices:
        value: complex = complex(values[index])
        duplicate: bool = (not include_conjugates and value.imag != 0.0
                           and any(np.isclose(value, mode.eigenvalue.conjugate(), rtol=1e-7, atol=1e-10)
                                   for mode in result))
        if (np.isfinite(value) and abs(value) > 1e-8 and fractions[index] >= minimum_participation
                and fractions[index] > 0.0 and not duplicate):
            result.append(ControllerMode(int(index), value, float(fractions[index]), scope, matches))
        else:
            pass
    return tuple(result)
