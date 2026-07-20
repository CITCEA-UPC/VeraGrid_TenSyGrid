import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Simulations.Rms.numerical.back_euler_ts import BackEulerImplicitTensygrid
from VeraGridEngine.Simulations.Rms.numerical.trapezoidal import TrapezoidalImplicitIntegration
from VeraGridEngine.Simulations.Rms.numerical.midpoint import MidpointImplicitIntegration
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.compiled_functions import SymbolicDerivative, SymbolicJacobian, SymbolicVector
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Expr, Var
from VeraGridEngine.Utils.Symbolic import symbolic as sym
from VeraGridEngine.Utils.Symbolic.symbolic_ml import trig_transform

USE_ARTIFICIAL_REGULARIZATION = True


@dataclass
class RunResult:
    solver: str
    reformulation: str
    t: np.ndarray
    y: np.ndarray
    elapsed_s: float
    well_initialized: bool
    converged: bool
    rms_cos: float
    rms_sin: float
    max_cos: float
    max_sin: float
    theta: np.ndarray
    u_cos: np.ndarray
    u_sin: np.ndarray
    int_sin2: np.ndarray | None = None
    int_sin2_ref: np.ndarray | None = None


class SingleBlockDaeProblem:
    VARS_NAME = "vrs"
    VARIABLE_PARAMS_NAME = "vprms"
    CONSTANT_PARAMS_NAME = "cprms"
    DIFF_NAME = "diff"

    def __init__(self, block: Block):
        self.block = block
        self._promote_differential_eqs_to_algebraic_all_blocks()
        self.block.unify_blocks()

        self._state_vars = list(self.block.state_vars)
        self._algebraic_vars = list(self.block.algebraic_vars)
        self._all_vars = self._state_vars + self._algebraic_vars
        self._diff_vars = list(self.block.diff_vars)
        self._state_eqs = list(self.block.state_eqs)
        self._algebraic_eqs = list(self.block.algebraic_eqs)
        # differential_eqs are promoted before flattening

        self.uid2idx_vars = {v.uid: i for i, v in enumerate(self._all_vars)}
        self._uid2idx_diff = {v.uid: i for i, v in enumerate(self._diff_vars)}

        self._compiler_names_dict = {}
        self._alias_names_dict = {}
        for uid, i in self.uid2idx_vars.items():
            self._compiler_names_dict[uid] = f"{self.VARS_NAME}[{i}]"
            self._alias_names_dict[uid] = f"{self.VARS_NAME}_{i}"
        for uid, i in self._uid2idx_diff.items():
            self._compiler_names_dict[uid] = f"{self.DIFF_NAME}[{i}]"
            self._alias_names_dict[uid] = f"{self.DIFF_NAME}_{i}"

        self._variable_parameters_values = np.zeros(0, dtype=float)
        self._constant_params = np.zeros(0, dtype=float)

        all_eqs = self._state_eqs + self._algebraic_eqs
        self._rhs_fn = SymbolicVector(
            all_eqs,
            self._compiler_names_dict,
            self._alias_names_dict,
            self.VARS_NAME,
            self.DIFF_NAME,
            self.VARIABLE_PARAMS_NAME,
            self.CONSTANT_PARAMS_NAME,
            use_jit=True,
        ) if len(all_eqs) else None

        self._jacobian_fn = SymbolicJacobian(
            eqs=all_eqs,
            variables=self._all_vars,
            compiler_names_dict=self._compiler_names_dict,
            alias_names_dict=self._alias_names_dict,
            VARS_NAME=self.VARS_NAME,
            DIFF_NAME=self.DIFF_NAME,
            EVENT_PARAMS_NAME=self.VARIABLE_PARAMS_NAME,
            PARAMS_NAME=self.CONSTANT_PARAMS_NAME,
            use_jit=True,
        ) if len(all_eqs) else None

        self._derivative_fn = SymbolicDerivative(
            vars=self._all_vars,
            uid2idx_vars=self.uid2idx_vars,
            diff_vars=self._diff_vars,
            compiler_names_dict=self._compiler_names_dict,
            use_jit=True,
        )

        self.init_guess: dict[int, float] = {}
        self._populate_init_guess()
        self._x0 = self._build_x0_from_init_eqs()

    def _populate_init_guess(self) -> None:
        uid_bindings: dict[int, float] = {}

        pending = list(self.block.init_eqs.items())
        unresolved_last: list[tuple[Var, Expr | Const | float]] = []
        for _ in range(30):
            changed = False
            next_pending = []
            for var, expr in pending:
                if not isinstance(var, Var) or var.uid not in self.uid2idx_vars:
                    continue
                try:
                    if isinstance(expr, Var) and expr.uid == var.uid:
                        raise ValueError("self-referential init expression")
                    if isinstance(expr, Const):
                        val = 0.0 if expr.value is None else float(expr.value)
                    elif isinstance(expr, Expr):
                        val = float(expr.eval_uid(uid_bindings))
                    else:
                        val = float(expr)
                    self.init_guess[var.uid] = val
                    uid_bindings[var.uid] = val
                    changed = True
                except Exception:
                    next_pending.append((var, expr))
            pending = next_pending
            unresolved_last = next_pending
            if not changed:
                break

        if unresolved_last:
            unresolved_names = [v.name for v, _ in unresolved_last if isinstance(v, Var)]
            print(f"init resolution unresolved vars: {unresolved_names}")

        # Fill unresolved vars with robust defaults based on trig semantics.
        theta0 = 0.0
        for var in self._all_vars:
            if var.name == "theta" and var.uid in self.init_guess:
                theta0 = self.init_guess[var.uid]
                break

        for var in self._all_vars:
            if var.uid in self.init_guess:
                continue
            if var.name in {"u_cos", "u_cos_n", "u_cos_aux"}:
                self.init_guess[var.uid] = float(np.cos(theta0))
            elif var.name in {"u_sin", "u_sin_n", "u_sin_aux"}:
                self.init_guess[var.uid] = float(np.sin(theta0))
            elif var.name == "int_xsinx":
                self.init_guess[var.uid] = float(np.sin(theta0) - theta0 * np.cos(theta0))
            elif var.name == "int_xcosx":
                self.init_guess[var.uid] = float(theta0 * np.sin(theta0) + np.cos(theta0))
            elif var.name in {"norm", "norm_aux"}:
                self.init_guess[var.uid] = 1.0
            else:
                self.init_guess[var.uid] = 0.0

        critical_names = {"theta", "u_cos", "u_sin", "int_xsinx", "int_xcosx"}
        unresolved_critical = [var.name for var in self._all_vars if var.name in critical_names and var.uid not in self.init_guess]
        if unresolved_critical:
            raise ValueError(f"Critical init vars unresolved after fallback: {sorted(set(unresolved_critical))}")

    def _build_x0_from_init_eqs(self) -> np.ndarray:
        x0 = np.zeros(len(self._all_vars), dtype=float)
        for var in self._all_vars:
            idx = self.uid2idx_vars[var.uid]
            x0[idx] = float(self.init_guess.get(var.uid, 0.0))
        return x0

    def _promote_differential_eqs_to_algebraic_all_blocks(self) -> None:
        for blk in self.block.get_all_blocks():
            if len(blk.differential_eqs):
                blk.algebraic_eqs.extend(blk.differential_eqs)

    def get_all_vars_number(self) -> int:
        return len(self._all_vars)

    def get_states_number(self) -> int:
        return len(self._state_vars)

    def get_algebraic_var_number(self) -> int:
        return len(self._algebraic_vars)

    def get_diff_var_number(self) -> int:
        return len(self._diff_vars)

    @property
    def algebraic_vars(self):
        return self._algebraic_vars

    @property
    def _algebraic_vars_(self):
        return self._algebraic_vars

    def get_x0(self) -> np.ndarray:
        return self._x0.copy()

    def update_variable_params_ts(self, x_snapshot: np.ndarray, t: float) -> None:
        _ = (x_snapshot, t)

    def rhs_algebraic(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        if self._rhs_fn is None:
            return np.array([])
        full = self._rhs_fn(x, dx, self._variable_parameters_values, self._constant_params)
        ns = self.get_states_number()
        return full[ns:]

    def rhs_state(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        if self._rhs_fn is None or self.get_states_number() == 0:
            return np.array([])
        full = self._rhs_fn(x, dx, self._variable_parameters_values, self._constant_params)
        ns = self.get_states_number()
        return full[:ns]

    def get_dx(self, x: np.ndarray, xn: np.ndarray, dx: np.ndarray, h: float) -> np.ndarray:
        return self._derivative_fn(x, xn, dx, h)

    def _jac_full(self, x: np.ndarray, dx: np.ndarray, h: float) -> sp.csc_matrix:
        if self._jacobian_fn is None:
            return sp.csc_matrix((0, 0))
        return self._jacobian_fn(x, dx, self._variable_parameters_values, self._constant_params, h).tocsc()

    def get_j11(self, x: np.ndarray, dx: np.ndarray, h: float) -> sp.csc_matrix:
        ns = self.get_states_number()
        if ns == 0:
            return sp.csc_matrix((0, 0))
        J = self._jac_full(x, dx, h)
        return J[:ns, :ns]

    def get_j12(self, x: np.ndarray, dx: np.ndarray, h: float) -> sp.csc_matrix:
        ns = self.get_states_number()
        na = self.get_algebraic_var_number()
        if ns == 0:
            return sp.csc_matrix((0, na))
        J = self._jac_full(x, dx, h)
        return J[:ns, ns:ns + na]

    def get_j21(self, x: np.ndarray, dx: np.ndarray, h: float) -> sp.csc_matrix:
        ns = self.get_states_number()
        na = self.get_algebraic_var_number()
        if na == 0:
            return sp.csc_matrix((0, ns))
        J = self._jac_full(x, dx, h)
        return J[ns:ns + na, :ns]

    def get_j22(self, x: np.ndarray, dx: np.ndarray, h: float) -> sp.csc_matrix:
        ns = self.get_states_number()
        na = self.get_algebraic_var_number()
        if na == 0:
            return sp.csc_matrix((0, 0))
        J = self._jac_full(x, dx, h)
        return J[ns:ns + na, ns:ns + na]

    def get_var_idx(self, v: Var) -> int:
        return self.uid2idx_vars[v.uid]

    def report_progress2(self, step_idx: int, steps: int) -> None:
        _ = (step_idx, steps)

    def update_variable_params(self, t: float, x_snapshot: np.ndarray | None = None) -> None:
        _ = (t, x_snapshot)

    def update(self, t: float, x_snapshot: np.ndarray, event_values: np.ndarray) -> None:
        _ = (t, x_snapshot, event_values)

    def get_next_forced_event_time(self, t_prev: float, t_target: float):
        _ = (t_prev, t_target)
        return None

    def initialize_fmu_cs_devices(self, x0: np.ndarray, t0: float) -> None:
        _ = (x0, t0)

    def initialize_fmu_me_devices(self, x0: np.ndarray, t0: float) -> None:
        _ = (x0, t0)

    def advance_fmu_cs_devices(self, t: float, x_snapshot: np.ndarray, h: float) -> None:
        _ = (t, x_snapshot, h)

    def advance_fmu_me_devices(self, t: float, x_snapshot: np.ndarray, h: float) -> None:
        _ = (t, x_snapshot, h)

    def close_fmu_cs_devices(self) -> None:
        return None

    def close_fmu_me_devices(self) -> None:
        return None


class BackEulerImplicitTensygridWithDx0(BackEulerImplicitTensygrid):
    def __init__(self, *args, dx0_init: np.ndarray | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx0_init = dx0_init

    def simulate(self):
        converged = False
        well_initialized = True
        x0 = self.problem.get_x0()
        dx0 = np.zeros(self.problem.get_diff_var_number(), dtype=float) if self.dx0_init is None else np.array(self.dx0_init, dtype=float, copy=True)
        self.t[0] = self.t0
        self.y[0, :] = x0.copy()
        dx_last = dx0.copy()
        x_new = x0.copy()

        for step_idx in range(self.steps):
            converged = False
            n_iter = 0
            xn = self.y[step_idx, :]
            while not converged and n_iter < self.max_iter_0:
                dx = self.problem.get_dx(x_new, xn, dx_last, self.h)
                rhs = self._rhs_implicit(x_new, dx, xn, self.h)
                converged = np.linalg.norm(rhs, np.inf) < 1e-7
                if not converged:
                    Jf = self._jacobian_implicit(x_new, dx, self.h)
                    delta = sp.linalg.spsolve(Jf, -rhs)
                    if not np.all(np.isfinite(delta)):
                        delta, *_ = sp.linalg.lsqr(Jf, -rhs)
                    x_new += delta
                    n_iter += 1

            if converged:
                self.y[step_idx + 1, :] = x_new
                self.t[step_idx + 1] = self.t[step_idx] + self.h
                dx_last = dx.copy()
            else:
                well_initialized = False
                break

        return self.t, self.y, well_initialized, converged


def find_name_in_block(name: str, block: Block) -> Var:
    for var in block.algebraic_vars + block.state_vars + block.diff_vars:
        if var.name == name:
            return var
    raise KeyError(f"Variable '{name}' not found")


def try_find_name_in_block(name: str, block: Block) -> Var | None:
    try:
        return find_name_in_block(name, block)
    except KeyError:
        return None


def build_trig_test_problem(reformulation: str, omega: float = 40.0, theta0: float = 0.0) -> tuple[SingleBlockDaeProblem, Var, Var, Var]:
    vf = VarFactory()
    theta = vf.add_var("theta")

    trig_block, u_cos, u_sin = trig_transform(vf=vf, u=theta, type=reformulation)

    drive_block = Block(
        state_vars=[theta],
        state_eqs=[Const(omega)],
        init_eqs={theta: Const(theta0)},
    )

    model = Block(children=[drive_block, trig_block])
    problem = SingleBlockDaeProblem(model)
    return problem, theta, u_cos, u_sin


def run_case(reformulation: str, solver_name: str, t_end: float, h: float, omega: float, theta0: float) -> RunResult:
    problem, theta, u_cos, u_sin = build_trig_test_problem(reformulation=reformulation, omega=omega, theta0=theta0)

    # Enforce consistent trig initial conditions explicitly to avoid
    # homogeneous-zero trajectories due to accidental zero seeds.
    x0 = problem.get_x0()
    itheta = problem.get_var_idx(theta)
    icos = problem.get_var_idx(u_cos)
    isin = problem.get_var_idx(u_sin)
    theta0 = float(x0[itheta])
    x0[icos] = float(np.cos(theta0))
    x0[isin] = float(np.sin(theta0))
    problem._x0 = x0

    def build_dx0() -> np.ndarray:
        dx0 = np.zeros(problem.get_diff_var_number(), dtype=float)
        for i, dvar in enumerate(problem._diff_vars):
            base_name = dvar.base_var.name if dvar.base_var is not None else dvar.name
            if base_name == "theta":
                dx0[i] = omega
            elif base_name == "u_cos":
                dx0[i] = -omega * np.sin(theta0)
            elif base_name == "u_sin":
                dx0[i] = omega * np.cos(theta0)
        return dx0

    dx0_seed = build_dx0()

    if solver_name == "back_euler":
        solver = BackEulerImplicitTensygridWithDx0(
            problem=problem,
            t0=0.0,
            t_end=t_end,
            h=h,
            max_iter=60,
            dx0_init=dx0_seed,
        )
    elif solver_name == "trapezoidal":
        solver = TrapezoidalImplicitIntegration(
            problem=problem,
            t0=0.0,
            t_end=t_end,
            h=h,
            max_iter=200,
            dx0_init=dx0_seed,
            use_fd_jacobian=True,
            use_chain_rule_jacobian=True,
        )
    elif solver_name == "midpoint":
        solver = MidpointImplicitIntegration(
            problem=problem,
            t0=0.0,
            t_end=t_end,
            h=h,
            max_iter=200,
            dx0_init=dx0_seed,
        )
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    t0 = time.perf_counter()
    t, y, well_initialized, converged = solver.simulate()
    elapsed = time.perf_counter() - t0

    theta_ref = y[:, itheta]
    cos_ref = np.cos(theta_ref)
    sin_ref = np.sin(theta_ref)

    # Diagnostic for integral formulation consistency
    int_sin2_series = None
    int_sin2_ref = None
    if reformulation == "integral":
        int_sin2_var = try_find_name_in_block("int_sin2", problem.block)
        if int_sin2_var is not None and int_sin2_var.uid in problem.uid2idx_vars:
            i_int_sin2 = problem.uid2idx_vars[int_sin2_var.uid]
            int_sin2_series = y[:, i_int_sin2]
            int_sin2_ref = 0.5 * (theta_ref - sin_ref * cos_ref)
            residual = int_sin2_series - int_sin2_ref
            print(
                f"check[{solver_name}/integral] int_sin2 residual: "
                f"max_abs={np.nanmax(np.abs(residual)):.3e}, "
                f"rms={np.sqrt(np.nanmean(residual * residual)):.3e}"
            )

    print(
        f"init[{reformulation}] t=0 -> theta={y[0, itheta]:+.6f}, "
        f"u_cos={y[0, icos]:+.6f}, u_sin={y[0, isin]:+.6f}, "
        f"ref_cos={np.cos(y[0, itheta]):+.6f}, ref_sin={np.sin(y[0, itheta]):+.6f}"
    )

    cos_err = y[:, icos] - cos_ref
    sin_err = y[:, isin] - sin_ref

    return RunResult(
        solver=solver_name,
        reformulation=reformulation,
        t=t,
        y=y,
        elapsed_s=elapsed,
        well_initialized=well_initialized,
        converged=converged,
        rms_cos=float(np.sqrt(np.mean(cos_err * cos_err))),
        rms_sin=float(np.sqrt(np.mean(sin_err * sin_err))),
        max_cos=float(np.max(np.abs(cos_err))),
        max_sin=float(np.max(np.abs(sin_err))),
        theta=theta_ref.copy(),
        u_cos=y[:, icos].copy(),
        u_sin=y[:, isin].copy(),
        int_sin2=int_sin2_series,
        int_sin2_ref=int_sin2_ref,
    )


def main() -> None:
    solvers = ["back_euler", "trapezoidal", "midpoint"]
    solvers = ["trapezoidal"]
    reformulations = ["usual", "norm", 'singular', 'integral']
    t_end = 1.0
    h = 1e-3
    omega = 65.0
    theta0 = 0.0

    results = []
    for solver_name in solvers:
        for reform in reformulations:
            results.append(run_case(reform, solver_name=solver_name, t_end=t_end, h=h, omega=omega, theta0=theta0))

    print("\n=== Trig Reformulation Comparison ===")
    print(f"simulation_time={t_end}s, h={h}, omega={omega} rad/s")
    print("solver      | reformulation | elapsed_s | init_ok | converged | rms_cos | rms_sin | max_cos | max_sin")
    for r in results:
        print(
            f"{r.solver:11s} | {r.reformulation:12s} | {r.elapsed_s:8.4f} | {str(r.well_initialized):7s} | {str(r.converged):9s} | "
            f"{r.rms_cos:8.3e} | {r.rms_sin:8.3e} | {r.max_cos:8.3e} | {r.max_sin:8.3e}"
        )

    for solver_name in solvers:
        fig, axs = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
        plotted_any = False

        for r in results:
            if r.solver != solver_name:
                continue

            t = r.t
            cos_ref = np.cos(r.theta)
            sin_ref = np.sin(r.theta)
            err_cos = r.u_cos - cos_ref
            err_sin = r.u_sin - sin_ref

            finite_series = (
                np.isfinite(t)
                & np.isfinite(r.u_cos)
                & np.isfinite(r.u_sin)
                & np.isfinite(err_cos)
                & np.isfinite(err_sin)
                & (np.abs(r.u_cos) < 1e6)
                & (np.abs(r.u_sin) < 1e6)
                & (np.abs(err_cos) < 1e6)
                & (np.abs(err_sin) < 1e6)
            )
            if not np.any(finite_series):
                continue

            plotted_any = True
            axs[0, 0].plot(t[finite_series], r.u_cos[finite_series], label=f"{r.reformulation}")
            axs[0, 1].plot(t[finite_series], r.u_sin[finite_series], label=f"{r.reformulation}")
            axs[1, 0].plot(t[finite_series], err_cos[finite_series], label=f"{r.reformulation}")
            axs[1, 1].plot(t[finite_series], err_sin[finite_series], label=f"{r.reformulation}")
            axs[2, 0].plot(t[finite_series], r.theta[finite_series], label=f"{r.reformulation}")

            # Mark initial sample explicitly
            axs[0, 0].plot(t[0], r.u_cos[0], marker="o", linestyle="None", markersize=4)
            axs[0, 1].plot(t[0], r.u_sin[0], marker="o", linestyle="None", markersize=4)
            axs[2, 0].plot(t[0], r.theta[0], marker="o", linestyle="None", markersize=4)

        # Add reference on u_cos subplot
        ref_added = False
        for r in results:
            if r.solver != solver_name:
                continue
            t = r.t
            cos_ref = np.cos(r.theta)
            finite_ref = np.isfinite(t) & np.isfinite(cos_ref) & (np.abs(cos_ref) < 1e6)
            if np.any(finite_ref):
                axs[0, 0].plot(t[finite_ref], cos_ref[finite_ref], "k--", linewidth=1.4, label="reference cos(theta)")
                ref_added = True
                break

        axs[0, 0].set_title(f"{solver_name}: u_cos")
        axs[0, 1].set_title(f"{solver_name}: u_sin")
        axs[1, 0].set_title(f"{solver_name}: u_cos - cos(theta)")
        axs[1, 1].set_title(f"{solver_name}: u_sin - sin(theta)")
        axs[2, 0].set_title(f"{solver_name}: theta")
        axs[2, 1].set_title(f"{solver_name}: int_sin2 check")

        for r in results:
            if r.solver != solver_name:
                continue
            if r.int_sin2 is None or r.int_sin2_ref is None:
                continue
            t = r.t
            finite_i = np.isfinite(t) & np.isfinite(r.int_sin2) & np.isfinite(r.int_sin2_ref)
            if not np.any(finite_i):
                continue
            axs[2, 1].plot(t[finite_i], r.int_sin2[finite_i], label="int_sin2")
            axs[2, 1].plot(t[finite_i], r.int_sin2_ref[finite_i], "k--", label="0.5*(x-sin*cos)")

        for ax in axs.flat:
            ax.grid(True)
            if plotted_any:
                handles, labels = ax.get_legend_handles_labels()
                if len(labels) > 0:
                    ax.legend()
        axs[2, 0].set_xlabel("Time (s)")

        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), f"trig_reformulations_{solver_name}.png")
        #plt.savefig(out_path, dpi=140)
        #print(f"Saved plot: {out_path}")
        if os.environ.get("DISPLAY"):
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
