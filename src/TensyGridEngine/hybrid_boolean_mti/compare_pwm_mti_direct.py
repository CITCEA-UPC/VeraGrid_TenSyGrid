# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp


def ensure_repo_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    for path in (str(src_root), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


ensure_repo_import_paths()

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_template import EmtProblemTemplate
from VeraGridEngine.Simulations.Rms.numerical.back_euler_mti import BackEulerImplicitIntegrationMTI
from VeraGridEngine.Templates.Emt.bridge_2level_3ph_emt_template import get_bridge_2level_3ph_emt_template
from VeraGridEngine.Utils.procedural_logic import ThreePhaseCarrierPwmLogic, build_boundary_updater_from_block
from VeraGridEngine.Utils.Symbolic.block import Block, find_name_in_block
from VeraGridEngine.Utils.Symbolic.symbolic_ml import (
    mti_three_phase_carrier_pwm_internal_carrier_params,
    mti_three_phase_carrier_pwm_scheduled_params,
)
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Expr, Var
import VeraGridEngine.Utils.Symbolic.symbolic as sym
from VeraGridEngine.enumerations import DynamicIntegrationMethod, EmtSolverTypes


class GenericEmtProblem(EmtProblemTemplate):
    """Minimal EMT problem used to bind the procedural PWM logic."""

    __slots__ = []


@dataclass(frozen=True)
class PwmComparisonTrace:
    t: np.ndarray
    ref_a: np.ndarray
    ref_b: np.ndarray
    ref_c: np.ndarray
    carrier: np.ndarray
    proc_gate_a: np.ndarray
    proc_gate_b: np.ndarray
    proc_gate_c: np.ndarray
    mti_gate_a: np.ndarray
    mti_gate_b: np.ndarray
    mti_gate_c: np.ndarray


class RecordingBoundaryUpdater:
    """Record gate modes after delegating boundary updates."""

    __slots__ = ["base_updater", "mode_indices", "trace_t", "trace_gates"]

    def __init__(self, base_updater, mode_indices: tuple[int, int, int]) -> None:
        self.base_updater = base_updater
        self.mode_indices = tuple(int(i) for i in mode_indices)
        self.trace_t: list[float] = []
        self.trace_gates: list[tuple[float, float, float]] = []

    def update(self, t: float, x: np.ndarray, params: np.ndarray) -> None:
        self.base_updater.update(t, x, params)
        self.trace_t.append(float(t))
        self.trace_gates.append(tuple(float(params[i]) for i in self.mode_indices))

    def get_next_forced_event_time(self, t_prev: float, t_target: float):
        return self.base_updater.get_next_forced_event_time(t_prev, t_target)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.trace_t, dtype=float), np.asarray(self.trace_gates, dtype=float)


class DrivenProceduralRecordingBoundaryUpdater(RecordingBoundaryUpdater):
    """Drive modulation variables as external inputs before procedural PWM runs."""

    __slots__ = ["mod_indices", "f_ref", "modulation_index"]

    def __init__(
        self,
        base_updater,
        mode_indices: tuple[int, int, int],
        mod_indices: tuple[int, int, int],
        f_ref: float,
        modulation_index: float,
    ) -> None:
        super().__init__(base_updater=base_updater, mode_indices=mode_indices)
        self.mod_indices = tuple(int(i) for i in mod_indices)
        self.f_ref = float(f_ref)
        self.modulation_index = float(modulation_index)

    def update(self, t: float, x: np.ndarray, params: np.ndarray) -> None:
        theta = 2.0 * np.pi * self.f_ref * float(t)
        refs = (
            float(self.modulation_index * np.sin(theta)),
            float(self.modulation_index * np.sin(theta - 2.0 * np.pi / 3.0)),
            float(self.modulation_index * np.sin(theta + 2.0 * np.pi / 3.0)),
        )
        for idx, value in zip(self.mod_indices, refs):
            x[idx] = value
        super().update(t=t, x=x, params=params)


class DirectComparatorBoundaryUpdater:
    """
    Root-detected direct comparator PWM updater.

    This is the EMT executable counterpart of the direct MTI comparator model:
    gates switch when ``ref_phase(t) - carrier(t)`` crosses zero.
    """

    __slots__ = [
        "mode_indices",
        "f_ref",
        "f_sw",
        "modulation_index",
        "carrier_phase",
        "trace_t",
        "trace_gates",
    ]

    def __init__(
        self,
        mode_indices: tuple[int, int, int],
        f_ref: float,
        f_sw: float,
        modulation_index: float,
        carrier_phase: float = 0.0,
    ) -> None:
        self.mode_indices = tuple(int(i) for i in mode_indices)
        self.f_ref = float(f_ref)
        self.f_sw = float(f_sw)
        self.modulation_index = float(modulation_index)
        self.carrier_phase = float(carrier_phase)
        self.trace_t: list[float] = []
        self.trace_gates: list[tuple[float, float, float]] = []

    def _refs(self, t: float) -> tuple[float, float, float]:
        theta = 2.0 * np.pi * self.f_ref * float(t)
        return (
            float(self.modulation_index * np.sin(theta)),
            float(self.modulation_index * np.sin(theta - 2.0 * np.pi / 3.0)),
            float(self.modulation_index * np.sin(theta + 2.0 * np.pi / 3.0)),
        )

    def _switching_functions(self, t: float) -> tuple[float, float, float]:
        carrier = triangular_carrier(
            float(t),
            omega_sw=2.0 * np.pi * self.f_sw,
            carrier_phase=self.carrier_phase,
        )
        refs = self._refs(float(t))
        return tuple(float(ref - carrier) for ref in refs)

    def update(self, t: float, x: np.ndarray, params: np.ndarray) -> None:
        _unused_x = x
        s_values = self._switching_functions(float(t))
        for idx, s_val in zip(self.mode_indices, s_values):
            params[idx] = 1.0 if s_val >= 0.0 else 0.0
        self.trace_t.append(float(t))
        self.trace_gates.append(tuple(float(params[i]) for i in self.mode_indices))

    def get_next_forced_event_time(self, t_prev: float, t_target: float):
        roots: list[float] = []
        for phase_idx in range(3):
            root = self._find_phase_root(float(t_prev), float(t_target), phase_idx)
            if root is not None:
                roots.append(float(root))
        if len(roots) == 0:
            return None
        return min(roots)

    def _find_phase_root(self, t_prev: float, t_target: float, phase_idx: int):
        if t_target <= t_prev + 1e-15:
            return None

        s0 = self._switching_functions(t_prev)[phase_idx]
        s1 = self._switching_functions(t_target)[phase_idx]
        if abs(s1) <= 1e-12:
            return t_target
        if abs(s0) <= 1e-12:
            probe = t_prev + min(1e-12, 0.25 * (t_target - t_prev))
            s0 = self._switching_functions(probe)[phase_idx]
        if s0 * s1 > 0.0:
            return None

        lo = t_prev
        hi = t_target
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            sm = self._switching_functions(mid)[phase_idx]
            if abs(sm) <= 1e-13:
                return mid
            if s0 * sm <= 0.0:
                hi = mid
                s1 = sm
            else:
                lo = mid
                s0 = sm
        return hi

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.trace_t, dtype=float), np.asarray(self.trace_gates, dtype=float)


class MinimalBlockMtiPwmProblem:
    """Small block-backed problem adapter for BackEulerImplicitIntegrationMTI."""

    __slots__ = [
        "block",
        "f_ref",
        "f_sw",
        "modulation_index",
        "sampled_inputs",
        "_x0",
        "algebraic_vars",
        "state_vars",
        "diff_vars",
        "_algebraic_eqs",
        "_state_eqs",
        "_variable_parameters_values",
        "_variable_parameters",
        "_uid2idx_event_params",
        "_uid2idx_params",
        "_uid2idx_diff",
        "uid2idx_vars",
        "_constant_params",
        "direction_eps",
        "omega_sw",
        "carrier_idx",
        "gate_slice",
        "carrier_bool_pos",
        "_mti_bool_param_indices",
        "_mti_col_meta",
        "_mti_row_meta",
        "_mti_solving_order",
        "logger",
        "_last_sample_half_index",
    ]

    def __init__(
        self,
        f_ref: float = 50.0,
        f_sw: float = 1000.0,
        modulation_index: float = 0.8,
        sampled_inputs: bool = True,
        direction_eps: float = 1.0e-8,
    ) -> None:
        vf = VarFactory()
        block, gate_a, gate_b, gate_c, ref_a, ref_b, ref_c, omega_sw, carrier, b_carrier_rise = mti_three_phase_carrier_pwm_internal_carrier_params(
            vf=vf,
            name="be_mti_compare",
            omega_sw0=2.0 * np.pi * float(f_sw),
            direction_eps=float(direction_eps),
            ref_a0=0.0,
            ref_b0=float(modulation_index * np.sin(-2.0 * np.pi / 3.0)),
            ref_c0=float(modulation_index * np.sin(2.0 * np.pi / 3.0)),
            carrier0=triangular_carrier(0.0, 2.0 * np.pi * f_sw),
            carrier_rising0=1.0,
        )
        self.block = block
        self.f_ref = float(f_ref)
        self.f_sw = float(f_sw)
        self.modulation_index = float(modulation_index)
        self.sampled_inputs = bool(sampled_inputs)
        self.direction_eps = float(direction_eps)
        self.omega_sw = 2.0 * np.pi * float(f_sw)
        self.state_vars = [carrier]
        self.algebraic_vars = [gate_a, gate_b, gate_c]
        self.diff_vars = list(block.diff_vars)
        self._algebraic_eqs = list(block.algebraic_eqs)
        self._state_eqs = list(block.state_eqs)
        self.carrier_idx = 0
        self.gate_slice = slice(1, 4)
        self._x0 = np.zeros(4, dtype=float)

        bool_vars = list(block.boolean_guards.keys())
        self.carrier_bool_pos = bool_vars.index(b_carrier_rise)
        input_vars = [ref_a, ref_b, ref_c, omega_sw]
        self._variable_parameters = input_vars + bool_vars
        self._uid2idx_event_params = {var.uid: idx for idx, var in enumerate(self._variable_parameters)}
        self._mti_bool_param_indices = [self._uid2idx_event_params[var.uid] for var in bool_vars]
        self._variable_parameters_values = np.zeros(len(self._variable_parameters), dtype=float)
        self._uid2idx_params = {}
        self._uid2idx_diff = {}
        self.uid2idx_vars = {carrier.uid: 0, gate_a.uid: 1, gate_b.uid: 2, gate_c.uid: 3}
        self._uid2idx_diff = {self.diff_vars[0].uid: 0} if len(self.diff_vars) > 0 else {}
        self._constant_params = np.zeros(0, dtype=float)
        self.logger = SimpleNamespace(add_error=lambda **_kwargs: None)
        self._last_sample_half_index = None

        self._variable_parameters_values[3] = self.omega_sw
        self.update_variable_params(0.0)
        self._x0[self.carrier_idx] = triangular_carrier(0.0, self.omega_sw)
        self._initialize_mti_booleans_at_t0()
        self._x0[self.gate_slice] = self._variable_parameters_values[self._mti_bool_param_indices[:3]]
        self._mti_col_meta = []
        self._mti_row_meta = []
        self._mti_solving_order = []

    @property
    def get_mti_boolean_parameter_indices(self) -> list[int]:
        return list(self._mti_bool_param_indices)

    def get_all_vars_number(self) -> int:
        return 4

    def get_x0(self) -> np.ndarray:
        return self._x0.copy()

    def get_diff_var_number(self) -> int:
        return 1

    def get_states_number(self) -> int:
        return 1

    def get_algebraic_var_number(self) -> int:
        return 3

    def get_dx(self, x_new: np.ndarray, x_prev: np.ndarray, dx_last: np.ndarray, h_eff: float) -> np.ndarray:
        _unused = dx_last
        if h_eff <= 0.0:
            return np.zeros(1, dtype=float)
        return np.asarray([(float(x_new[self.carrier_idx]) - float(x_prev[self.carrier_idx])) / float(h_eff)], dtype=float)

    def report_progress2(self, step_idx: int, steps: int) -> None:
        _unused = (step_idx, steps)

    def _refs(self, t: float) -> tuple[float, float, float]:
        theta = 2.0 * np.pi * self.f_ref * float(t)
        return (
            float(self.modulation_index * np.sin(theta)),
            float(self.modulation_index * np.sin(theta - 2.0 * np.pi / 3.0)),
            float(self.modulation_index * np.sin(theta + 2.0 * np.pi / 3.0)),
        )

    def _half_index(self, t: float) -> int:
        omega_sw = 2.0 * np.pi * self.f_sw
        return int(np.floor((omega_sw * float(t) + 0.5 * np.pi) / np.pi))

    def update_variable_params(self, t: float, x_snapshot: np.ndarray | None = None, scheduled_t: float | None = None) -> None:
        _unused = (x_snapshot, scheduled_t)
        self._update_reference_inputs(float(t))
        self._variable_parameters_values[3] = self.omega_sw

    def _update_reference_inputs(self, t: float) -> None:
        if self.sampled_inputs:
            half_index = self._half_index(float(t))
            if self._last_sample_half_index is None or half_index != self._last_sample_half_index:
                refs = self._refs(float(t))
                self._variable_parameters_values[0:3] = refs
                self._last_sample_half_index = half_index
        else:
            self._variable_parameters_values[0:3] = self._refs(float(t))

    def update(self, t: float, x: np.ndarray, params: np.ndarray) -> None:
        _unused = (x, params)
        self._update_reference_inputs(float(t))
        self._variable_parameters_values[3] = self.omega_sw

    def get_next_forced_event_time(self, t_prev: float, t_target: float):
        half_period = 0.5 / self.f_sw
        next_boundary = (np.floor(float(t_prev) / half_period) + 1.0) * half_period
        if float(t_prev) < next_boundary <= float(t_target):
            return float(next_boundary)
        return None

    def _initialize_mti_booleans_at_t0(self) -> None:
        z = self.update_mti_boolean_state(self._x0, np.zeros(0), self._x0, 0.0)
        self.set_mti_boolean_state(z)

    def set_mti_boolean_state(self, z: np.ndarray) -> None:
        for k, idx in enumerate(self._mti_bool_param_indices):
            self._variable_parameters_values[idx] = float(z[k])

    def update_mti_boolean_state(self, x: np.ndarray, dx: np.ndarray, xn: np.ndarray, h: float) -> np.ndarray:
        _unused = (x, dx, xn, h)
        s = self._switching_residuals(x)
        carrier_value = float(np.asarray(x, dtype=float)[self.carrier_idx]) if len(np.asarray(x)) > 0 else triangular_carrier(0.0, self.omega_sw)
        z_gate = [1.0 if val >= 0.0 else 0.0 for val in s]
        z_carrier = 0.0 if carrier_value >= 1.0 - 1.0e-12 else 1.0
        if carrier_value <= -1.0 + 1.0e-12:
            z_carrier = 1.0
        return np.asarray(z_gate + [z_carrier], dtype=float)

    def rhs_algebraic(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        _unused = dx
        z = self._variable_parameters_values[self._mti_bool_param_indices]
        return np.asarray(x, dtype=float)[self.gate_slice] - z[:3]

    def compute_mti_equalities(self, x: np.ndarray, dx: np.ndarray, xn: np.ndarray, h: float) -> np.ndarray:
        _unused = dx
        z = self._variable_parameters_values[self._mti_bool_param_indices]
        x_arr = np.asarray(x, dtype=float)
        xn_arr = np.asarray(xn, dtype=float)
        carrier_res = x_arr[self.carrier_idx] - xn_arr[self.carrier_idx] - float(h) * self._carrier_slope_from_z(z)
        gate_res = x_arr[self.gate_slice] - z[:3]
        return np.r_[carrier_res, gate_res]

    def compute_mti_inequalities(self, x: np.ndarray, dx: np.ndarray, xn: np.ndarray, h: float) -> np.ndarray:
        _unused = (dx, xn, h)
        z = self._variable_parameters_values[self._mti_bool_param_indices]
        x_arr = np.asarray(x, dtype=float)
        s = self._switching_residuals(x_arr)
        carrier = float(x_arr[self.carrier_idx])
        g_gate = -(2.0 * z[:3] - 1.0) * s
        g_carrier = z[3] * (carrier - 1.0) + (1.0 - z[3]) * (-carrier - 1.0)
        return np.r_[g_gate, g_carrier]

    def total_derivative_inequalities(self, x: np.ndarray, dx: np.ndarray, xpp: np.ndarray | None = None) -> np.ndarray:
        _unused = (x, dx, xpp)
        return np.zeros(4, dtype=float)

    @staticmethod
    def inequalities_satisfied(g: np.ndarray, tol: float = 1e-9) -> bool:
        return bool(np.all(np.asarray(g, dtype=float) <= tol))

    def get_j22(self, x: np.ndarray, dx: np.ndarray, h: float):
        _unused = (x, dx, h)
        return sp.eye(3, format="csc")

    def get_j11(self, x: np.ndarray, dx: np.ndarray, h: float):
        _unused = (x, dx, h)
        return sp.csc_matrix((1, 1), dtype=float)

    def get_j12(self, x: np.ndarray, dx: np.ndarray, h: float):
        _unused = (x, dx, h)
        return sp.csc_matrix((1, 3), dtype=float)

    def get_j21(self, x: np.ndarray, dx: np.ndarray, h: float):
        _unused = (x, dx, h)
        return sp.csc_matrix((3, 1), dtype=float)

    def rhs_state(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        _unused = (x, dx)
        z = self._variable_parameters_values[self._mti_bool_param_indices]
        return np.asarray([self._carrier_slope_from_z(z)], dtype=float)

    def build_mti_incidence_and_order(self, x: np.ndarray, dx: np.ndarray, h: float) -> None:
        _unused = (x, dx, h)
        self._mti_row_meta = [("eq", 0), ("eq", 1), ("eq", 2), ("eq", 3), ("ineq", 0), ("ineq", 1), ("ineq", 2), ("ineq", 3)]
        self._mti_col_meta = [("xp", 0, self.diff_vars[0]), ("y", 0, self.algebraic_vars[0]), ("y", 1, self.algebraic_vars[1]), ("y", 2, self.algebraic_vars[2])]
        self._mti_solving_order = [
            SimpleNamespace(eq_idx=i + 1, var_idx=min(i, 3) + 1, subset=i, subproblem=i, explicit=1)
            for i in range(8)
        ]

    def get_event_local_boolean_candidates(self, ineq_idx: int, z_prev: np.ndarray) -> list[np.ndarray]:
        out = []
        for bit in (0.0, 1.0):
            z = np.asarray(z_prev, dtype=float).copy()
            if 0 <= int(ineq_idx) < 3:
                z[int(ineq_idx)] = bit
            elif int(ineq_idx) == 3:
                z[3] = bit
            out.append(z)
        return out

    def get_event_subset_ids(self, ineq_idx: int) -> set[int]:
        return {int(ineq_idx)}

    def get_event_solving_stages(self, ineq_idx: int):
        _unused = ineq_idx
        return [], [(np.asarray([1, 2, 3, 4], dtype=int), np.asarray([1, 2, 3, 4], dtype=int))], []

    def get_inequality_row_indices(self, row_idx: np.ndarray) -> np.ndarray:
        rows = np.asarray(row_idx, dtype=int)
        return rows[rows >= 4] - 4

    def get_equality_row_indices(self, row_idx: np.ndarray) -> np.ndarray:
        rows = np.asarray(row_idx, dtype=int)
        return rows[rows < 4]

    def get_continuous_equality_row_indices(self, row_idx: np.ndarray) -> np.ndarray:
        return self.get_equality_row_indices(row_idx)

    def get_fixed_boolean_equality_row_indices(self, row_idx: np.ndarray) -> np.ndarray:
        _unused = row_idx
        return np.zeros(0, dtype=int)

    def split_explicit_subproblem_pairs(self, eq_idx: np.ndarray, var_idx: np.ndarray):
        pairs = [(int(eq), int(eq)) for eq in np.asarray(eq_idx, dtype=int) if int(eq) < 4]
        return pairs, np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    def get_subproblem_boolean_positions(self, eq_idx: np.ndarray, var_idx: np.ndarray) -> list[int]:
        _unused = (eq_idx, var_idx)
        return [0, 1, 2, 3]

    def get_group_subset_ids(self, eq_idx: np.ndarray, var_idx: np.ndarray) -> set[int]:
        _unused = (eq_idx, var_idx)
        return {0, 1, 2, 3}

    def evaluate_boolean_guard(self, bool_position: int, x: np.ndarray, dx: np.ndarray):
        _unused = (x, dx)
        if int(bool_position) < 3:
            s = self._switching_residuals(x)
            return 1.0 if s[int(bool_position)] >= 0.0 else 0.0
        carrier_value = float(np.asarray(x, dtype=float)[self.carrier_idx])
        if carrier_value >= 1.0 - 1.0e-12:
            return 0.0
        if carrier_value <= -1.0 + 1.0e-12:
            return 1.0
        z = self._variable_parameters_values[self._mti_bool_param_indices]
        return float(z[3])

    def has_boolean_guard(self, bool_position: int) -> bool:
        _unused = bool_position
        return True

    def _switching_residuals(self, x: np.ndarray) -> np.ndarray:
        z = self._variable_parameters_values[self._mti_bool_param_indices]
        direction_for_switching_surface = 1.0 - 2.0 * z[3]
        return (
            self._variable_parameters_values[0:3]
            - float(np.asarray(x, dtype=float)[self.carrier_idx])
            + self.direction_eps * direction_for_switching_surface
        )

    def _carrier_slope_from_z(self, z: np.ndarray) -> float:
        return float((2.0 * z[3] - 1.0) * (2.0 / np.pi) * self.omega_sw)


class ScheduledBlockMtiPwmProblem:
    """BackEulerMTI adapter for the scheduled-transition PWM block."""

    __slots__ = [
        "block",
        "f_ref",
        "f_sw",
        "modulation_index",
        "transition_eps",
        "sampled_inputs",
        "_x0",
        "algebraic_vars",
        "_algebraic_eqs",
        "_state_eqs",
        "_variable_parameters",
        "_variable_parameters_values",
        "_uid2idx_event_params",
        "_uid2idx_params",
        "_uid2idx_diff",
        "uid2idx_vars",
        "_constant_params",
        "_mti_bool_param_indices",
        "_mti_col_meta",
        "_mti_row_meta",
        "_mti_solving_order",
        "logger",
        "_last_sample_half_index",
        "_local_start_time",
    ]

    def __init__(
        self,
        f_ref: float = 50.0,
        f_sw: float = 1000.0,
        modulation_index: float = 0.8,
        sampled_inputs: bool = True,
        transition_eps: float = 1.0e-9,
    ) -> None:
        vf = VarFactory()
        time_var = vf.add_var("t_sched_pwm_compare")
        half_period = 0.5 / float(f_sw)
        interval_start, carrier_rising = self._interval_start_and_rise_static(0.0, f_sw)
        refs0 = self._refs_static(0.0, f_ref, modulation_index)
        (
            block,
            gate_a,
            gate_b,
            gate_c,
            t_cross_a,
            t_cross_b,
            t_cross_c,
            ref_a,
            ref_b,
            ref_c,
            interval_start_var,
            half_period_var,
            q_a,
            q_b,
            q_c,
            b_carrier_rise,
        ) = mti_three_phase_carrier_pwm_scheduled_params(
            vf=vf,
            time=time_var,
            name="be_mti_sched_compare",
            transition_eps=float(transition_eps),
            ref_a0=refs0[0],
            ref_b0=refs0[1],
            ref_c0=refs0[2],
            interval_start0=interval_start,
            half_period0=half_period,
            carrier_rising0=carrier_rising,
        )
        self.block = block
        self.f_ref = float(f_ref)
        self.f_sw = float(f_sw)
        self.modulation_index = float(modulation_index)
        self.transition_eps = float(transition_eps)
        self.sampled_inputs = bool(sampled_inputs)
        self.algebraic_vars = [t_cross_a, t_cross_b, t_cross_c, gate_a, gate_b, gate_c]
        self._algebraic_eqs = list(block.algebraic_eqs)
        self._state_eqs = []
        self._x0 = np.zeros(6, dtype=float)

        bool_vars = [q_a, q_b, q_c, b_carrier_rise]
        input_vars = [ref_a, ref_b, ref_c, interval_start_var, half_period_var]
        self._variable_parameters = input_vars + bool_vars
        self._uid2idx_event_params = {var.uid: idx for idx, var in enumerate(self._variable_parameters)}
        self._mti_bool_param_indices = [self._uid2idx_event_params[var.uid] for var in bool_vars]
        self._variable_parameters_values = np.zeros(len(self._variable_parameters), dtype=float)
        self._uid2idx_params = {}
        self._uid2idx_diff = {}
        self.uid2idx_vars = {var.uid: idx for idx, var in enumerate(self.algebraic_vars)}
        self._constant_params = np.zeros(0, dtype=float)
        self._mti_col_meta = []
        self._mti_row_meta = []
        self._mti_solving_order = []
        self.logger = SimpleNamespace(add_error=lambda **_kwargs: None)
        self._last_sample_half_index = None
        self._local_start_time = 0.0

        self.update_variable_params(0.0)
        self._initialize_mti_booleans_at_t0()
        self._x0[:] = self._scheduled_equalities_target(0.0)

    @staticmethod
    def _refs_static(t: float, f_ref: float, modulation_index: float) -> tuple[float, float, float]:
        theta = 2.0 * np.pi * float(f_ref) * float(t)
        return (
            float(modulation_index * np.sin(theta)),
            float(modulation_index * np.sin(theta - 2.0 * np.pi / 3.0)),
            float(modulation_index * np.sin(theta + 2.0 * np.pi / 3.0)),
        )

    @staticmethod
    def _interval_start_and_rise_static(t: float, f_sw: float) -> tuple[float, float]:
        omega_sw = 2.0 * np.pi * float(f_sw)
        shifted_phase = omega_sw * float(t) + 0.5 * np.pi
        interval_index = int(np.floor(shifted_phase / np.pi))
        interval_start = (float(interval_index) * np.pi - 0.5 * np.pi) / omega_sw
        return float(interval_start), 1.0 if interval_index % 2 == 0 else 0.0

    @property
    def get_mti_boolean_parameter_indices(self) -> list[int]:
        return list(self._mti_bool_param_indices)

    def get_all_vars_number(self) -> int:
        return 6

    def get_x0(self) -> np.ndarray:
        return self._x0.copy()

    def get_diff_var_number(self) -> int:
        return 0

    def get_states_number(self) -> int:
        return 0

    def get_algebraic_var_number(self) -> int:
        return 6

    def get_dx(self, x_new: np.ndarray, x_prev: np.ndarray, dx_last: np.ndarray, h_eff: float) -> np.ndarray:
        _unused = (x_new, x_prev, dx_last, h_eff)
        return np.zeros(0, dtype=float)

    def report_progress2(self, step_idx: int, steps: int) -> None:
        _unused = (step_idx, steps)

    def _half_index(self, t: float) -> int:
        omega_sw = 2.0 * np.pi * self.f_sw
        return int(np.floor((omega_sw * float(t) + 0.5 * np.pi) / np.pi))

    def update_variable_params(self, t: float, x_snapshot: np.ndarray | None = None, scheduled_t: float | None = None) -> None:
        _unused = (x_snapshot, scheduled_t)
        self._local_start_time = float(t)
        self._update_schedule_inputs(float(t))

    def update(self, t: float, x: np.ndarray, params: np.ndarray) -> None:
        _unused = (x, params)
        self._update_schedule_inputs(float(t))

    def _update_schedule_inputs(self, t: float) -> None:
        half_index = self._half_index(float(t))
        if self.sampled_inputs:
            if self._last_sample_half_index is None or half_index != self._last_sample_half_index:
                self._variable_parameters_values[0:3] = self._refs_static(t, self.f_ref, self.modulation_index)
                self._last_sample_half_index = half_index
        else:
            self._variable_parameters_values[0:3] = self._refs_static(t, self.f_ref, self.modulation_index)
        interval_start, carrier_rising = self._interval_start_and_rise_static(t, self.f_sw)
        self._variable_parameters_values[3] = interval_start
        self._variable_parameters_values[4] = 0.5 / self.f_sw
        self._variable_parameters_values[self._mti_bool_param_indices[3]] = carrier_rising

    def get_next_forced_event_time(self, t_prev: float, t_target: float):
        candidates: list[float] = []
        half_period = 0.5 / self.f_sw
        next_boundary = (np.floor(float(t_prev) / half_period) + 1.0) * half_period
        if float(t_prev) < next_boundary <= float(t_target):
            candidates.append(float(next_boundary))

        t_cross = self._scheduled_equalities_target(float(t_prev))[:3]
        for t_evt in t_cross:
            if float(t_prev) < float(t_evt) <= float(t_target):
                candidates.append(float(t_evt))

        if len(candidates) == 0:
            return None
        return min(candidates)

    def _initialize_mti_booleans_at_t0(self) -> None:
        z = self.update_mti_boolean_state(self._x0, np.zeros(0), self._x0, 0.0)
        self.set_mti_boolean_state(z)

    def set_mti_boolean_state(self, z: np.ndarray) -> None:
        z_arr = np.asarray(z, dtype=float).copy()
        # Carrier direction is owned by the schedule boundary, not by phase-event candidates.
        _interval_start, carrier_rising = self._interval_start_and_rise_static(self._local_start_time, self.f_sw)
        if z_arr.size >= 4:
            z_arr[3] = carrier_rising
        for k, idx in enumerate(self._mti_bool_param_indices):
            self._variable_parameters_values[idx] = float(z_arr[k])

    def update_mti_boolean_state(self, x: np.ndarray, dx: np.ndarray, xn: np.ndarray, h: float) -> np.ndarray:
        _unused = (x, dx, xn)
        t_eval = self._local_start_time + float(h)
        tau = self._tau_values(np.asarray(x, dtype=float), t_eval)
        q = [1.0 if val >= 0.0 else 0.0 for val in tau]
        _interval_start, carrier_rising = self._interval_start_and_rise_static(t_eval, self.f_sw)
        return np.asarray(q + [carrier_rising], dtype=float)

    def rhs_algebraic(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return self.compute_mti_equalities(x, dx, x, 0.0)

    def compute_mti_equalities(self, x: np.ndarray, dx: np.ndarray, xn: np.ndarray, h: float) -> np.ndarray:
        _unused = (dx, xn, h)
        return np.asarray(x, dtype=float) - self._scheduled_equalities_target(self._local_start_time + float(h))

    def compute_mti_inequalities(self, x: np.ndarray, dx: np.ndarray, xn: np.ndarray, h: float) -> np.ndarray:
        _unused = (dx, xn)
        z = self._variable_parameters_values[self._mti_bool_param_indices]
        tau = self._tau_values(np.asarray(x, dtype=float), self._local_start_time + float(h))
        return -(2.0 * z[:3] - 1.0) * tau

    def _scheduled_equalities_target(self, t_eval: float) -> np.ndarray:
        refs = self._variable_parameters_values[0:3]
        interval_start = float(self._variable_parameters_values[3])
        half_period = float(self._variable_parameters_values[4])
        z = self._variable_parameters_values[self._mti_bool_param_indices]
        carrier_rising = float(z[3])
        t_cross = np.zeros(3, dtype=float)
        for idx, ref in enumerate(refs):
            rising_cross = interval_start + 0.5 * (float(ref) + 1.0) * half_period
            falling_cross = interval_start + 0.5 * (1.0 - float(ref)) * half_period
            t_cross[idx] = carrier_rising * rising_cross + (1.0 - carrier_rising) * falling_cross
        q = z[:3]
        gates = (1.0 - q) * carrier_rising + q * (1.0 - carrier_rising)
        _unused = t_eval
        return np.r_[t_cross, gates]

    def _tau_values(self, x: np.ndarray, t_eval: float) -> np.ndarray:
        return float(t_eval) - np.asarray(x[:3], dtype=float) + self.transition_eps

    def total_derivative_inequalities(self, x: np.ndarray, dx: np.ndarray, xpp: np.ndarray | None = None) -> np.ndarray:
        _unused = (x, dx, xpp)
        return np.zeros(3, dtype=float)

    @staticmethod
    def inequalities_satisfied(g: np.ndarray, tol: float = 1e-9) -> bool:
        return bool(np.all(np.asarray(g, dtype=float) <= tol))

    def get_j22(self, x: np.ndarray, dx: np.ndarray, h: float):
        _unused = (x, dx, h)
        return sp.eye(6, format="csc")

    def build_mti_incidence_and_order(self, x: np.ndarray, dx: np.ndarray, h: float) -> None:
        _unused = (x, dx, h)
        self._mti_row_meta = [("eq", i) for i in range(6)] + [("ineq", i) for i in range(3)]
        self._mti_col_meta = [("y", i, self.algebraic_vars[i]) for i in range(6)]
        self._mti_solving_order = [
            SimpleNamespace(eq_idx=i + 1, var_idx=min(i, 5) + 1, subset=i, subproblem=i, explicit=1)
            for i in range(9)
        ]

    def get_event_local_boolean_candidates(self, ineq_idx: int, z_prev: np.ndarray) -> list[np.ndarray]:
        out = []
        for bit in (0.0, 1.0):
            z = np.asarray(z_prev, dtype=float).copy()
            if 0 <= int(ineq_idx) < 3:
                z[int(ineq_idx)] = bit
            if z.size >= 4:
                _interval_start, carrier_rising = self._interval_start_and_rise_static(self._local_start_time, self.f_sw)
                z[3] = carrier_rising
            out.append(z)
        return out

    def get_event_subset_ids(self, ineq_idx: int) -> set[int]:
        return {int(ineq_idx)}

    def get_event_solving_stages(self, ineq_idx: int):
        _unused = ineq_idx
        return [], [(np.arange(1, 7, dtype=int), np.arange(1, 7, dtype=int))], []

    def get_inequality_row_indices(self, row_idx: np.ndarray) -> np.ndarray:
        rows = np.asarray(row_idx, dtype=int)
        return rows[rows >= 6] - 6

    def get_equality_row_indices(self, row_idx: np.ndarray) -> np.ndarray:
        rows = np.asarray(row_idx, dtype=int)
        return rows[rows < 6]

    def get_continuous_equality_row_indices(self, row_idx: np.ndarray) -> np.ndarray:
        return self.get_equality_row_indices(row_idx)

    def get_fixed_boolean_equality_row_indices(self, row_idx: np.ndarray) -> np.ndarray:
        _unused = row_idx
        return np.zeros(0, dtype=int)

    def split_explicit_subproblem_pairs(self, eq_idx: np.ndarray, var_idx: np.ndarray):
        pairs = [(int(eq), int(eq)) for eq in np.asarray(eq_idx, dtype=int) if int(eq) < 6]
        return pairs, np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    def get_subproblem_boolean_positions(self, eq_idx: np.ndarray, var_idx: np.ndarray) -> list[int]:
        _unused = (eq_idx, var_idx)
        return [0, 1, 2]

    def get_group_subset_ids(self, eq_idx: np.ndarray, var_idx: np.ndarray) -> set[int]:
        _unused = (eq_idx, var_idx)
        return {0, 1, 2}

    def evaluate_boolean_guard(self, bool_position: int, x: np.ndarray, dx: np.ndarray):
        _unused = dx
        if int(bool_position) < 3:
            tau = self._tau_values(np.asarray(x, dtype=float), self._local_start_time)
            return 1.0 if tau[int(bool_position)] >= 0.0 else 0.0
        _interval_start, carrier_rising = self._interval_start_and_rise_static(self._local_start_time, self.f_sw)
        return carrier_rising

    def has_boolean_guard(self, bool_position: int) -> bool:
        _unused = bool_position
        return True


def build_constant_algebraic_var(vf: VarFactory, name: str, value: float) -> tuple[Var, Expr]:
    variable = vf.add_var(name=name)
    return variable, variable - Const(float(value))


def build_bridge_standalone_problem() -> tuple[GenericEmtProblem, dict[str, Var]]:
    vf = VarFactory()
    bridge_name = "bridge_2level_pwm_compare"
    omega_sw = vf.add_var(name="omega_sw_pwm_compare")
    carrier_phase = vf.add_var(name="carrier_phase_pwm_compare")
    bridge_block = get_bridge_2level_3ph_emt_template(vf=vf, name=bridge_name).block

    theta_pll, theta_pll_eq = build_constant_algebraic_var(vf, "theta_pll_pwm_compare", 0.0)
    v_cmd_d, v_cmd_d_eq = build_constant_algebraic_var(vf, "v_cmd_d_pwm_compare", 0.3)
    v_cmd_q, v_cmd_q_eq = build_constant_algebraic_var(vf, "v_cmd_q_pwm_compare", 0.0)
    v_cmd_0, v_cmd_0_eq = build_constant_algebraic_var(vf, "v_cmd_0_pwm_compare", 0.0)
    v_dc, v_dc_eq = build_constant_algebraic_var(vf, "v_dc_pwm_compare", 1.0)
    k_v_conv, k_v_conv_eq = build_constant_algebraic_var(vf, "k_v_conv_pwm_compare", 0.5)
    m_max, m_max_eq = build_constant_algebraic_var(vf, "m_max_pwm_compare", 0.95)
    vdc_floor, vdc_floor_eq = build_constant_algebraic_var(vf, "vdc_floor_pwm_compare", 0.05)

    bridge_block.connect(
        bridge_block.in_vars,
        [
            theta_pll,
            v_cmd_d,
            v_cmd_q,
            v_cmd_0,
            v_dc,
            k_v_conv,
            m_max,
            vdc_floor,
            omega_sw,
            carrier_phase,
        ],
    )

    root_block = Block(
        name="PwmProceduralVsMtiDirectCase",
        children=[bridge_block],
        algebraic_vars=[theta_pll, v_cmd_d, v_cmd_q, v_cmd_0, v_dc, k_v_conv, m_max, vdc_floor],
        algebraic_eqs=[
            theta_pll_eq,
            v_cmd_d_eq,
            v_cmd_q_eq,
            v_cmd_0_eq,
            v_dc_eq,
            k_v_conv_eq,
            m_max_eq,
            vdc_floor_eq,
        ],
        event_dict={
            omega_sw: Const(2.0 * np.pi * 1000.0),
            carrier_phase: Const(0.0),
        },
        init_eqs={
            theta_pll: Const(0.0),
            v_cmd_d: Const(0.3),
            v_cmd_q: Const(0.0),
            v_cmd_0: Const(0.0),
            v_dc: Const(1.0),
            k_v_conv: Const(0.5),
            m_max: Const(0.95),
            vdc_floor: Const(0.05),
        },
    )
    root_block.unify_blocks()

    problem = GenericEmtProblem(
        sys_block=root_block,
        glob_time=vf.add_var("t_pwm_compare"),
        static_parameter_values_mapping={},
    )
    tracked = {
        "m_a": find_name_in_block(f"m_a_{bridge_name}", problem.sys_block),
        "m_b": find_name_in_block(f"m_b_{bridge_name}", problem.sys_block),
        "m_c": find_name_in_block(f"m_c_{bridge_name}", problem.sys_block),
    }

    all_problem_vars = list(problem.get_state_vars()) + list(problem.get_algebraic_vars())
    init_values = {
        "theta_pll_pwm_compare": 0.0,
        "v_cmd_d_pwm_compare": 0.3,
        "v_cmd_q_pwm_compare": 0.0,
        "v_cmd_0_pwm_compare": 0.0,
        "v_dc_pwm_compare": 1.0,
        "k_v_conv_pwm_compare": 0.5,
        "m_max_pwm_compare": 0.95,
        "vdc_floor_pwm_compare": 0.05,
        f"m_a_{bridge_name}": 0.0,
        f"m_b_{bridge_name}": -0.6,
        f"m_c_{bridge_name}": 0.6,
    }
    for init_name, init_value in init_values.items():
        init_var = next((var for var in all_problem_vars if var.name == init_name), None)
        if init_var is not None:
            problem.init_guess[init_var.uid] = float(init_value)

    return problem, tracked


def build_pwm_signal_problem(
    use_procedural_pwm: bool,
    f_ref: float,
    f_sw: float,
    modulation_index: float,
) -> tuple[GenericEmtProblem, tuple[int, int, int], tuple[int, int, int]]:
    """
    Build a small solvable EMT DAE with time-driven modulation references.

    The DAE only solves the reference algebraic equations. Gate switching is
    handled by the provided boundary updater, matching how EMT runtime modes are
    updated in the full converter models.
    """
    vf = VarFactory()
    t_var = vf.add_var("t_pwm_solver_compare")
    omega_sw = vf.add_var("omega_sw_pwm_solver_compare")
    carrier_phase = vf.add_var("carrier_phase_pwm_solver_compare")
    gate_a_mode = vf.add_var("gate_a_mode_pwm_solver_compare")
    gate_b_mode = vf.add_var("gate_b_mode_pwm_solver_compare")
    gate_c_mode = vf.add_var("gate_c_mode_pwm_solver_compare")
    m_a = vf.add_var("m_a_pwm_solver_compare")
    m_b = vf.add_var("m_b_pwm_solver_compare")
    m_c = vf.add_var("m_c_pwm_solver_compare")

    _unused_f_ref = f_ref
    eqs = [
        m_a,
        m_b,
        m_c,
    ]

    procedural_logic = []
    if use_procedural_pwm:
        procedural_logic.append(
            ThreePhaseCarrierPwmLogic(
                mod_a_var_name=m_a.name,
                mod_b_var_name=m_b.name,
                mod_c_var_name=m_c.name,
                gate_a_mode_var_name=gate_a_mode.name,
                gate_b_mode_var_name=gate_b_mode.name,
                gate_c_mode_var_name=gate_c_mode.name,
                omega_sw_var_name=omega_sw.name,
                carrier_phase_var_name=carrier_phase.name,
                name="solver_backed_proc_pwm",
            )
        )

    block = Block(
        name="PwmSolverBackedSignalCase",
        algebraic_vars=[m_a, m_b, m_c],
        algebraic_eqs=eqs,
        event_dict={
            omega_sw: Const(2.0 * np.pi * float(f_sw)),
            carrier_phase: Const(0.0),
            gate_a_mode: Const(0.0),
            gate_b_mode: Const(0.0),
            gate_c_mode: Const(0.0),
        },
        init_eqs={
            m_a: Const(0.0),
            m_b: Const(float(modulation_index * np.sin(-2.0 * np.pi / 3.0))),
            m_c: Const(float(modulation_index * np.sin(2.0 * np.pi / 3.0))),
        },
        procedural_logic=procedural_logic,
    )
    block.unify_blocks()

    problem = GenericEmtProblem(
        sys_block=block,
        glob_time=t_var,
        static_parameter_values_mapping={},
    )
    for var in problem.get_algebraic_vars():
        if var.name == m_a.name:
            problem.init_guess[var.uid] = 0.0
        elif var.name == m_b.name:
            problem.init_guess[var.uid] = float(modulation_index * np.sin(-2.0 * np.pi / 3.0))
        elif var.name == m_c.name:
            problem.init_guess[var.uid] = float(modulation_index * np.sin(2.0 * np.pi / 3.0))

    mode_indices = (
        problem.uid2idx_event_params[gate_a_mode.uid],
        problem.uid2idx_event_params[gate_b_mode.uid],
        problem.uid2idx_event_params[gate_c_mode.uid],
    )
    mod_indices = (
        problem.get_var_idx(m_a),
        problem.get_var_idx(m_b),
        problem.get_var_idx(m_c),
    )
    return problem, mode_indices, mod_indices


def run_solver_backed_case(
    use_procedural_pwm: bool,
    f_ref: float = 50.0,
    f_sw: float = 1000.0,
    modulation_index: float = 0.8,
    t_end: float = 4.0e-3,
    h: float = 50.0e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool, str]:
    problem, mode_indices, mod_indices = build_pwm_signal_problem(
        use_procedural_pwm=use_procedural_pwm,
        f_ref=f_ref,
        f_sw=f_sw,
        modulation_index=modulation_index,
    )
    options = EmtOptions()
    options.solver_type = EmtSolverTypes.StructuralCompiled
    options.init_newton_max_iter = 12

    solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=float(t_end),
        h=float(h),
        method=DynamicIntegrationMethod.DaeBackEuler,
    )

    if use_procedural_pwm:
        updater = DrivenProceduralRecordingBoundaryUpdater(
            base_updater=build_boundary_updater_from_block(problem),
            mode_indices=mode_indices,
            mod_indices=mod_indices,
            f_ref=f_ref,
            modulation_index=modulation_index,
        )
        solver_label = "StructuralCompiledSolver + DaeBackEuler + ThreePhaseCarrierPwmLogic scheduled updater"
    else:
        updater = DirectComparatorBoundaryUpdater(
            mode_indices=mode_indices,
            f_ref=f_ref,
            f_sw=f_sw,
            modulation_index=modulation_index,
        )
        solver_label = "StructuralCompiledSolver + DaeBackEuler + root-detected direct comparator updater"

    t, y, _dy, well_initialized, converged = solver.simulate(boundary_updater=updater)
    gate_t, gate_values = updater.arrays()
    return t, y, gate_t, gate_values, bool(well_initialized), bool(converged), solver_label


def plot_solver_backed_comparison(
    proc_t: np.ndarray,
    proc_gates: np.ndarray,
    direct_t: np.ndarray,
    direct_gates: np.ndarray,
) -> Path:
    out_path = Path(__file__).resolve().with_name("pwm_solver_backed_compare.png")
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    labels = ["A", "B", "C"]
    for idx, ax in enumerate(axes):
        ax.step(proc_t, proc_gates[:, idx], where="post", label=f"procedural {labels[idx]}", linewidth=1.5)
        ax.step(direct_t, direct_gates[:, idx], where="post", label=f"direct comparator {labels[idx]}", linestyle="--", linewidth=1.2)
        ax.set_title(f"Solver-backed gate {labels[idx]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("EMT solver-backed procedural PWM vs direct comparator switching")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def print_solver_backed_summary(proc_t: np.ndarray, proc_gates: np.ndarray, direct_t: np.ndarray, direct_gates: np.ndarray) -> None:
    labels = ["A", "B", "C"]
    sample_t = np.union1d(proc_t, direct_t)
    for idx, label in enumerate(labels):
        proc_interp = previous_value_sample(proc_t, proc_gates[:, idx], sample_t)
        direct_interp = previous_value_sample(direct_t, direct_gates[:, idx], sample_t)
        mismatch = np.abs(proc_interp - direct_interp) > 0.5
        print(
            f"solver_backed_phase={label} mismatch_samples={int(np.count_nonzero(mismatch))}/{sample_t.size} "
            f"({100.0 * np.count_nonzero(mismatch) / max(1, sample_t.size):.3f}%) "
            f"proc_updates={proc_t.size} direct_updates={direct_t.size} "
            f"proc_transitions={count_transitions(proc_gates[:, idx])} "
            f"direct_transitions={count_transitions(direct_gates[:, idx])}"
        )


def previous_value_sample(t_src: np.ndarray, v_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    if t_src.size == 0:
        return np.zeros_like(t_query, dtype=float)
    indices = np.searchsorted(t_src, t_query, side="right") - 1
    indices = np.clip(indices, 0, t_src.size - 1)
    return np.asarray(v_src[indices], dtype=float)


def run_back_euler_mti_block_case(
    f_ref: float = 50.0,
    f_sw: float = 1000.0,
    modulation_index: float = 0.8,
    t_end: float = 4.0e-3,
    h: float = 50.0e-6,
    sampled_inputs: bool = True,
    direction_eps: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool, str]:
    problem = MinimalBlockMtiPwmProblem(
        f_ref=f_ref,
        f_sw=f_sw,
        modulation_index=modulation_index,
        sampled_inputs=sampled_inputs,
        direction_eps=direction_eps,
    )
    solver = BackEulerImplicitIntegrationMTI(
        problem=problem,
        t0=0.0,
        t_end=float(t_end),
        h=float(h),
        max_iter=12,
        tolerance=1.0e-8,
        inequality_tolerance=1.0e-9,
    )
    t, y, well_initialized, converged = solver.simulate()
    z = np.asarray(solver.z[: len(t), :], dtype=float)
    label = "BackEulerImplicitIntegrationMTI + internal-carrier block PWM + runtime-parameter references"
    return t, y, z, bool(well_initialized), bool(converged), label


def run_back_euler_mti_scheduled_block_case(
    f_ref: float = 50.0,
    f_sw: float = 1000.0,
    modulation_index: float = 0.8,
    t_end: float = 4.0e-3,
    h: float = 50.0e-6,
    sampled_inputs: bool = True,
    transition_eps: float = 1.0e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool, str]:
    problem = ScheduledBlockMtiPwmProblem(
        f_ref=f_ref,
        f_sw=f_sw,
        modulation_index=modulation_index,
        sampled_inputs=sampled_inputs,
        transition_eps=transition_eps,
    )
    solver = BackEulerImplicitIntegrationMTI(
        problem=problem,
        t0=0.0,
        t_end=float(t_end),
        h=float(h),
        max_iter=12,
        tolerance=1.0e-8,
        inequality_tolerance=1.0e-9,
    )
    t, y, well_initialized, converged = solver.simulate()
    z = np.asarray(solver.z[: len(t), :3], dtype=float)
    # Convert q_after booleans to gates using the stored algebraic gate columns.
    gates = np.asarray(y[:, 3:6], dtype=float)
    label = "BackEulerImplicitIntegrationMTI + scheduled-transition PWM block"
    return t, gates, z, bool(well_initialized), bool(converged), label


def reference_and_carrier_arrays(
    t: np.ndarray,
    f_ref: float = 50.0,
    f_sw: float = 1000.0,
    modulation_index: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = 2.0 * np.pi * float(f_ref) * np.asarray(t, dtype=float)
    ref_a = float(modulation_index) * np.sin(theta)
    ref_b = float(modulation_index) * np.sin(theta - 2.0 * np.pi / 3.0)
    ref_c = float(modulation_index) * np.sin(theta + 2.0 * np.pi / 3.0)
    carrier = np.asarray([triangular_carrier(float(ti), 2.0 * np.pi * float(f_sw)) for ti in t], dtype=float)
    return ref_a, ref_b, ref_c, carrier


def plot_emt_vs_back_euler_mti(proc_t: np.ndarray, proc_gates: np.ndarray, mti_t: np.ndarray, mti_z: np.ndarray) -> Path:
    out_path = Path(__file__).resolve().with_name("pwm_emt_vs_back_euler_mti.png")
    sample_t = np.union1d(proc_t, mti_t)
    ref_a, ref_b, ref_c, carrier = reference_and_carrier_arrays(sample_t)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(sample_t, carrier, label="carrier", linewidth=1.5)
    axes[0].plot(sample_t, ref_a, label="ref_a", linewidth=1.2)
    axes[0].plot(sample_t, ref_b, label="ref_b", linewidth=1.2)
    axes[0].plot(sample_t, ref_c, label="ref_c", linewidth=1.2)
    axes[0].set_title("References and triangular carrier")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=4)

    labels = ["A", "B", "C"]
    for idx, ax in enumerate(axes[1:]):
        ax.step(proc_t, proc_gates[:, idx], where="post", label=f"EMT procedural {labels[idx]}", linewidth=1.5)
        ax.step(mti_t, mti_z[:, idx], where="post", label=f"BackEulerMTI {labels[idx]}", linestyle="--", linewidth=1.2)
        ax.set_title(f"Gate {labels[idx]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("EMT procedural PWM vs BackEulerMTI block PWM")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def plot_emt_vs_scheduled_mti(proc_t: np.ndarray, proc_gates: np.ndarray, mti_t: np.ndarray, mti_gates: np.ndarray) -> Path:
    out_path = Path(__file__).resolve().with_name("pwm_emt_vs_back_euler_mti_scheduled.png")
    sample_t = np.union1d(proc_t, mti_t)
    ref_a, ref_b, ref_c, carrier = reference_and_carrier_arrays(sample_t)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(sample_t, carrier, label="carrier", linewidth=1.5)
    axes[0].plot(sample_t, ref_a, label="ref_a", linewidth=1.2)
    axes[0].plot(sample_t, ref_b, label="ref_b", linewidth=1.2)
    axes[0].plot(sample_t, ref_c, label="ref_c", linewidth=1.2)
    axes[0].set_title("References and triangular carrier")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=4)

    for idx, label in enumerate(["A", "B", "C"]):
        ax = axes[idx + 1]
        ax.step(proc_t, proc_gates[:, idx], where="post", label=f"EMT procedural {label}", linewidth=1.5)
        ax.step(mti_t, mti_gates[:, idx], where="post", label=f"Scheduled BackEulerMTI {label}", linestyle="--", linewidth=1.2)
        ax.set_title(f"Gate {label}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("EMT procedural PWM vs scheduled-transition BackEulerMTI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def plot_emt_vs_back_euler_mti_mismatch_zoom(proc_t: np.ndarray, proc_gates: np.ndarray, mti_t: np.ndarray, mti_z: np.ndarray) -> Path:
    out_path = Path(__file__).resolve().with_name("pwm_emt_vs_back_euler_mti_mismatch_zoom.png")
    sample_t = np.union1d(proc_t, mti_t)
    labels = ["A", "B", "C"]
    mismatch_times: list[float] = []
    for idx in range(3):
        proc_sample = previous_value_sample(proc_t, proc_gates[:, idx], sample_t)
        mti_sample = previous_value_sample(mti_t, mti_z[:, idx], sample_t)
        mismatch_times.extend(sample_t[np.abs(proc_sample - mti_sample) > 0.5].tolist())

    if len(mismatch_times) == 0:
        mismatch_times = [float(sample_t[len(sample_t) // 2])] if sample_t.size > 0 else [0.0]

    selected_times = []
    for t_mismatch in sorted(set(float(ti) for ti in mismatch_times)):
        if len(selected_times) >= 6:
            break
        if len(selected_times) == 0 or abs(t_mismatch - selected_times[-1]) > 1.0e-4:
            selected_times.append(t_mismatch)

    fig, axes = plt.subplots(len(selected_times), 1, figsize=(12, max(3, 2.2 * len(selected_times))), sharex=False)
    if len(selected_times) == 1:
        axes = np.asarray([axes])

    for ax, center in zip(axes, selected_times):
        window = 1.5e-4
        t0 = max(0.0, float(center) - window)
        t1 = float(center) + window
        dense_t = np.linspace(t0, t1, 500)
        ref_a, ref_b, ref_c, carrier = reference_and_carrier_arrays(dense_t)
        ax.plot(dense_t, carrier, color="black", label="carrier", linewidth=1.2)
        ax.plot(dense_t, ref_a, label="ref_a", linewidth=1.0)
        ax.plot(dense_t, ref_b, label="ref_b", linewidth=1.0)
        ax.plot(dense_t, ref_c, label="ref_c", linewidth=1.0)
        for idx, label in enumerate(labels):
            proc_sample = previous_value_sample(proc_t, proc_gates[:, idx], dense_t)
            mti_sample = previous_value_sample(mti_t, mti_z[:, idx], dense_t)
            ax.step(dense_t, 1.2 + 0.16 * idx + 0.08 * proc_sample, where="post", linewidth=1.0, label=f"EMT {label}")
            ax.step(dense_t, 1.2 + 0.16 * idx + 0.08 * mti_sample, where="post", linestyle="--", linewidth=0.9, label=f"MTI {label}")
        ax.axvline(center, color="red", linestyle=":", linewidth=1.0, label="mismatch")
        ax.set_title(f"Mismatch zoom around t={center:.6e} s")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.15, 1.85)

    axes[0].legend(loc="upper right", ncol=5, fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("EMT procedural vs BackEulerMTI: mismatch zooms")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def run_slow_tie_break_improvement_demo() -> Path:
    f_ref = 1.0
    f_sw = 20.0
    t_end = 10.0e-2
    h = 2.5e-3
    modulation_index = 0.8

    _proc_t_grid, _proc_y, proc_t, proc_gates, proc_wi, proc_conv, _proc_label = run_solver_backed_case(
        use_procedural_pwm=True,
        f_ref=f_ref,
        f_sw=f_sw,
        modulation_index=modulation_index,
        t_end=t_end,
        h=h,
    )
    mti_t_no, _mti_y_no, mti_z_no, no_wi, no_conv, _no_label = run_back_euler_mti_block_case(
        f_ref=f_ref,
        f_sw=f_sw,
        modulation_index=modulation_index,
        t_end=t_end,
        h=h,
        direction_eps=0.0,
    )
    mti_t_bias, _mti_y_bias, mti_z_bias, bias_wi, bias_conv, _bias_label = run_back_euler_mti_block_case(
        f_ref=f_ref,
        f_sw=f_sw,
        modulation_index=modulation_index,
        t_end=t_end,
        h=h,
        direction_eps=1.0e-8,
    )

    print("\nSlow tie-break improvement demo:")
    print(f"slow_demo_procedural_well_initialized={proc_wi} procedural_converged={proc_conv}")
    print(f"slow_demo_no_bias_well_initialized={no_wi} no_bias_converged={no_conv}")
    print(f"slow_demo_direction_bias_well_initialized={bias_wi} direction_bias_converged={bias_conv}")
    for idx, label in enumerate(["A", "B", "C"]):
        count_no, n_no, pct_no = mismatch_count_on_union(proc_t, proc_gates, mti_t_no, mti_z_no, idx)
        count_bias, n_bias, pct_bias = mismatch_count_on_union(proc_t, proc_gates, mti_t_bias, mti_z_bias, idx)
        print(
            f"slow_demo_phase={label} no_bias={count_no}/{n_no} ({pct_no:.3f}%) "
            f"direction_bias={count_bias}/{n_bias} ({pct_bias:.3f}%)"
        )

    out_path = Path(__file__).resolve().with_name("pwm_slow_tie_break_improvement.png")
    sample_t = np.union1d(np.union1d(proc_t, mti_t_no), mti_t_bias)
    ref_a, _ref_b, _ref_c, carrier = reference_and_carrier_arrays(
        sample_t,
        f_ref=f_ref,
        f_sw=f_sw,
        modulation_index=modulation_index,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(sample_t, carrier, label="carrier", linewidth=1.5)
    axes[0].plot(sample_t, ref_a, label="ref_a", linewidth=1.3)
    axes[0].set_title("Slow tie-break demo inputs")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].step(proc_t, proc_gates[:, 0], where="post", label="EMT procedural A", linewidth=1.7)
    axes[1].step(mti_t_no, mti_z_no[:, 0], where="post", label="MTI no direction bias A", linestyle="--", linewidth=1.3)
    axes[1].step(mti_t_bias, mti_z_bias[:, 0], where="post", label="MTI internal direction bias A", linestyle=":", linewidth=1.8)
    axes[1].set_title("Phase-A boundary ownership")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.suptitle("Slow PWM tie-break improvement: f_ref=1 Hz, f_sw=20 Hz")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def print_emt_vs_mti_summary(proc_t: np.ndarray, proc_gates: np.ndarray, mti_t: np.ndarray, mti_z: np.ndarray) -> None:
    sample_t = np.union1d(proc_t, mti_t)
    for idx, label in enumerate(["A", "B", "C"]):
        proc_sample = previous_value_sample(proc_t, proc_gates[:, idx], sample_t)
        mti_sample = previous_value_sample(mti_t, mti_z[:, idx], sample_t)
        mismatch = np.abs(proc_sample - mti_sample) > 0.5
        print(
            f"emt_vs_back_euler_mti_phase={label} mismatch_samples={int(np.count_nonzero(mismatch))}/{sample_t.size} "
            f"({100.0 * np.count_nonzero(mismatch) / max(1, sample_t.size):.3f}%) "
            f"emt_transitions={count_transitions(proc_gates[:, idx])} "
            f"mti_transitions={count_transitions(mti_z[:, idx])}"
        )


def mismatch_count_on_union(
    ref_t: np.ndarray,
    ref_values: np.ndarray,
    test_t: np.ndarray,
    test_values: np.ndarray,
    phase_idx: int,
) -> tuple[int, int, float]:
    sample_t = np.union1d(ref_t, test_t)
    ref_sample = previous_value_sample(ref_t, ref_values[:, phase_idx], sample_t)
    test_sample = previous_value_sample(test_t, test_values[:, phase_idx], sample_t)
    count = int(np.count_nonzero(np.abs(ref_sample - test_sample) > 0.5))
    pct = 100.0 * count / max(1, sample_t.size)
    return count, int(sample_t.size), pct


def triangular_carrier(t: float, omega_sw: float, carrier_phase: float = 0.0) -> float:
    if abs(omega_sw) <= 1.0e-12:
        return -1.0

    shifted_phase = omega_sw * t + carrier_phase + 0.5 * np.pi
    interval_index = int(np.floor(shifted_phase / np.pi))
    half_period = np.pi / abs(omega_sw)
    interval_start = (float(interval_index) * np.pi - carrier_phase - 0.5 * np.pi) / omega_sw
    alpha = np.clip((t - interval_start) / half_period, 0.0, 1.0)

    if interval_index % 2 == 0:
        return float(-1.0 + 2.0 * alpha)
    return float(1.0 - 2.0 * alpha)


def carrier_direction_for_time(t: float, f_sw: float, carrier_phase: float = 0.0) -> float:
    """Return sign of d(ref-carrier)/dt for sampled references."""
    omega_sw = 2.0 * np.pi * float(f_sw)
    if abs(omega_sw) <= 1.0e-12:
        return -1.0
    shifted_phase = omega_sw * float(t) + float(carrier_phase) + 0.5 * np.pi
    interval_index = int(np.floor(shifted_phase / np.pi))
    # Rising carrier means d(ref-carrier)/dt < 0; falling means > 0.
    return -1.0 if interval_index % 2 == 0 else 1.0


def simulate_comparison(
    t_end: float = 4.0e-3,
    n_samples: int = 4001,
    f_ref: float = 50.0,
    f_sw: float = 1000.0,
    modulation_index: float = 0.8,
) -> PwmComparisonTrace:
    problem, tracked = build_bridge_standalone_problem()
    boundary_updater = build_boundary_updater_from_block(problem)
    x = problem.get_x0().copy()
    params = problem.event_params_values.copy()
    omega_sw = 2.0 * np.pi * float(f_sw)
    carrier_phase = 0.0

    mode_a_idx = problem.uid2idx_event_params[
        next(var.uid for var in problem.get_runtime_mode_parameters() if var.name == "gate_a_mode_bridge_2level_pwm_compare")
    ]
    mode_b_idx = problem.uid2idx_event_params[
        next(var.uid for var in problem.get_runtime_mode_parameters() if var.name == "gate_b_mode_bridge_2level_pwm_compare")
    ]
    mode_c_idx = problem.uid2idx_event_params[
        next(var.uid for var in problem.get_runtime_mode_parameters() if var.name == "gate_c_mode_bridge_2level_pwm_compare")
    ]

    idx_m_a = problem.get_var_idx(tracked["m_a"])
    idx_m_b = problem.get_var_idx(tracked["m_b"])
    idx_m_c = problem.get_var_idx(tracked["m_c"])

    t_arr = np.linspace(0.0, float(t_end), int(n_samples), dtype=float)
    ref_a = np.zeros_like(t_arr)
    ref_b = np.zeros_like(t_arr)
    ref_c = np.zeros_like(t_arr)
    carrier = np.zeros_like(t_arr)
    proc_gate_a = np.zeros_like(t_arr)
    proc_gate_b = np.zeros_like(t_arr)
    proc_gate_c = np.zeros_like(t_arr)
    mti_gate_a = np.zeros_like(t_arr)
    mti_gate_b = np.zeros_like(t_arr)
    mti_gate_c = np.zeros_like(t_arr)

    for k, t in enumerate(t_arr):
        theta = 2.0 * np.pi * float(f_ref) * float(t)
        ref_a[k] = modulation_index * np.sin(theta)
        ref_b[k] = modulation_index * np.sin(theta - 2.0 * np.pi / 3.0)
        ref_c[k] = modulation_index * np.sin(theta + 2.0 * np.pi / 3.0)
        carrier[k] = triangular_carrier(float(t), omega_sw=omega_sw, carrier_phase=carrier_phase)

        x[idx_m_a] = ref_a[k]
        x[idx_m_b] = ref_b[k]
        x[idx_m_c] = ref_c[k]
        boundary_updater.update(float(t), x, params)

        proc_gate_a[k] = params[mode_a_idx]
        proc_gate_b[k] = params[mode_b_idx]
        proc_gate_c[k] = params[mode_c_idx]

        mti_gate_a[k] = 1.0 if ref_a[k] - carrier[k] >= 0.0 else 0.0
        mti_gate_b[k] = 1.0 if ref_b[k] - carrier[k] >= 0.0 else 0.0
        mti_gate_c[k] = 1.0 if ref_c[k] - carrier[k] >= 0.0 else 0.0

    return PwmComparisonTrace(
        t=t_arr,
        ref_a=ref_a,
        ref_b=ref_b,
        ref_c=ref_c,
        carrier=carrier,
        proc_gate_a=proc_gate_a,
        proc_gate_b=proc_gate_b,
        proc_gate_c=proc_gate_c,
        mti_gate_a=mti_gate_a,
        mti_gate_b=mti_gate_b,
        mti_gate_c=mti_gate_c,
    )


def print_summary(trace: PwmComparisonTrace) -> None:
    for phase, proc, mti in [
        ("A", trace.proc_gate_a, trace.mti_gate_a),
        ("B", trace.proc_gate_b, trace.mti_gate_b),
        ("C", trace.proc_gate_c, trace.mti_gate_c),
    ]:
        mismatch = np.abs(proc - mti) > 0.5
        mismatch_count = int(np.count_nonzero(mismatch))
        mismatch_pct = 100.0 * mismatch_count / max(1, trace.t.size)
        print(
            f"phase={phase} mismatch_samples={mismatch_count}/{trace.t.size} "
            f"({mismatch_pct:.3f}%) proc_transitions={count_transitions(proc)} "
            f"mti_transitions={count_transitions(mti)}"
        )


def count_transitions(values: np.ndarray) -> int:
    return int(np.count_nonzero(np.abs(np.diff(values)) > 0.5))


def plot_trace(trace: PwmComparisonTrace) -> Path:
    out_path = Path(__file__).resolve().with_name("pwm_mti_direct_compare.png")
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(trace.t, trace.carrier, label="carrier", linewidth=1.5)
    axes[0].plot(trace.t, trace.ref_a, label="ref_a", linewidth=1.2)
    axes[0].plot(trace.t, trace.ref_b, label="ref_b", linewidth=1.2)
    axes[0].plot(trace.t, trace.ref_c, label="ref_c", linewidth=1.2)
    axes[0].set_title("References and triangular carrier")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=4)

    axes[1].step(trace.t, trace.proc_gate_a, where="post", label="procedural A", linewidth=1.5)
    axes[1].step(trace.t, trace.mti_gate_a, where="post", label="direct MTI A", linewidth=1.1, linestyle="--")
    axes[1].set_title("Gate A")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].step(trace.t, trace.proc_gate_b, where="post", label="procedural B", linewidth=1.5)
    axes[2].step(trace.t, trace.mti_gate_b, where="post", label="direct MTI B", linewidth=1.1, linestyle="--")
    axes[2].set_title("Gate B")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    axes[3].step(trace.t, trace.proc_gate_c, where="post", label="procedural C", linewidth=1.5)
    axes[3].step(trace.t, trace.mti_gate_c, where="post", label="direct MTI C", linewidth=1.1, linestyle="--")
    axes[3].set_title("Gate C")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")

    fig.suptitle("Procedural sampled PWM vs direct MTI comparator PWM")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def main() -> None:
    trace = simulate_comparison()
    print("Direct logic comparison without DAE solve:")
    print_summary(trace)
    out_path = plot_trace(trace)
    print(f"Plot saved to: {out_path}")

    print("\nSolver-backed EMT comparison:")
    proc_t, _proc_y, proc_gate_t, proc_gates, proc_wi, proc_conv, proc_solver = run_solver_backed_case(
        use_procedural_pwm=True,
    )
    direct_t, _direct_y, direct_gate_t, direct_gates, direct_wi, direct_conv, direct_solver = run_solver_backed_case(
        use_procedural_pwm=False,
    )
    _unused_proc_t = proc_t
    _unused_direct_t = direct_t
    print(f"procedural_solver={proc_solver}")
    print(f"procedural_well_initialized={proc_wi} procedural_converged={proc_conv}")
    print(f"direct_solver={direct_solver}")
    print(f"direct_well_initialized={direct_wi} direct_converged={direct_conv}")
    print_solver_backed_summary(proc_gate_t, proc_gates, direct_gate_t, direct_gates)
    solver_out_path = plot_solver_backed_comparison(proc_gate_t, proc_gates, direct_gate_t, direct_gates)
    print(f"Solver-backed plot saved to: {solver_out_path}")

    print("\nEMT procedural solver vs BackEulerMTI block comparison:")
    mti_t, _mti_y, mti_z, mti_wi, mti_conv, mti_solver = run_back_euler_mti_block_case()
    print(f"mti_solver={mti_solver}")
    print(f"mti_well_initialized={mti_wi} mti_converged={mti_conv}")
    print_emt_vs_mti_summary(proc_gate_t, proc_gates, mti_t, mti_z)
    mti_out_path = plot_emt_vs_back_euler_mti(proc_gate_t, proc_gates, mti_t, mti_z)
    print(f"EMT-vs-MTI plot saved to: {mti_out_path}")
    mti_zoom_path = plot_emt_vs_back_euler_mti_mismatch_zoom(proc_gate_t, proc_gates, mti_t, mti_z)
    print(f"EMT-vs-MTI mismatch zoom plot saved to: {mti_zoom_path}")

    print("\nEMT procedural solver vs scheduled-transition BackEulerMTI block comparison:")
    sched_t, sched_gates, _sched_q, sched_wi, sched_conv, sched_solver = run_back_euler_mti_scheduled_block_case()
    print(f"scheduled_mti_solver={sched_solver}")
    print(f"scheduled_mti_well_initialized={sched_wi} scheduled_mti_converged={sched_conv}")
    print_emt_vs_mti_summary(proc_gate_t, proc_gates, sched_t, sched_gates)
    sched_path = plot_emt_vs_scheduled_mti(proc_gate_t, proc_gates, sched_t, sched_gates)
    print(f"Scheduled EMT-vs-MTI plot saved to: {sched_path}")

    slow_demo_path = run_slow_tie_break_improvement_demo()
    print(f"Slow tie-break improvement plot saved to: {slow_demo_path}")


if __name__ == "__main__":
    main()
