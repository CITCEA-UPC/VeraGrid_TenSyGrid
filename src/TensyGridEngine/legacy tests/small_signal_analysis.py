# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Small Signal Analysis for Phasor-based RMS Grid.

This script creates the phasor problem manually and injects it into the SmallSignalStabilityRmsDriver.
"""

import numpy as np
import sys, os
import concurrent.futures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from matplotlib import pyplot as plt
import scipy.linalg as la
import scipy.sparse as sp

import VeraGridEngine.api as vge
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import (
    run_dense_small_signal_stability,
    run_sparse_small_signal_stability,
)
from VeraGridEngine.Simulations.Rms.numerical.back_euler_fx import BackEulerImplicitIntegration
import PolynomialMatrixBuilder as pmb_module
from PolynomialMatrixBuilder import PolynomialMatrixBuilder


def generate_plots(eigenvalues: np.ndarray, participation_factors: np.ndarray) -> None:
    """Generate eigenvalue map and participation factor plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Eigenvalue map
    ax1 = axes[0]
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    colors = ['red' if r > 0 else 'blue' for r in real_parts]
    ax1.scatter(real_parts, imag_parts, c=colors, alpha=0.6, s=50)
    ax1.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax1.set_xlabel('Real Part')
    ax1.set_ylabel('Imaginary Part')
    ax1.set_title('Eigenvalue Map (Phasor RMS with Multilinear Generator)')
    ax1.grid(True, alpha=0.3)

    # Participation factors
    ax2 = axes[1]
    if participation_factors is not None and len(participation_factors) > 0:
        n_modes = min(3, participation_factors.shape[1])
        n_states = min(10, participation_factors.shape[0])
        x = np.arange(n_states)
        width = 0.25

        for mode_idx in range(n_modes):
            pf = participation_factors[:n_states, mode_idx]
            pf = pf / np.sum(pf) if np.sum(pf) > 0 else pf
            ax2.bar(x + mode_idx * width, pf, width,
                    label=f'Mode {mode_idx + 1} ({np.real(eigenvalues[mode_idx]):.3f})')

        ax2.set_xlabel('State Variable')
        ax2.set_ylabel('Participation Factor')
        ax2.set_title('Participation Factors (Dominant Modes)')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels([f'State {i}' for i in range(n_states)])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def generate_dynamic_plots(dynamic_runs: list[dict]) -> None:
    """Plot dynamic simulation traces for all modes."""
    if not dynamic_runs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    ax_vm0, ax_va0 = axes[0]
    ax_vm1, ax_va1 = axes[1]

    plotted = False
    for run in dynamic_runs:
        t = run.get("t")
        if t is None or len(t) == 0:
            continue

        label = run.get("hard_sat_type", "mode")
        vm0 = run.get("vm0")
        va0 = run.get("va0")
        vm1 = run.get("vm1")
        va1 = run.get("va1")

        if vm0 is not None:
            ax_vm0.plot(t, vm0, label=label, linewidth=1.8)
            plotted = True
        if va0 is not None:
            ax_va0.plot(t, np.degrees(va0), label=label, linewidth=1.8)
            plotted = True
        if vm1 is not None:
            ax_vm1.plot(t, vm1, label=label, linewidth=1.8)
            plotted = True
        if va1 is not None:
            ax_va1.plot(t, np.degrees(va1), label=label, linewidth=1.8)
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax_vm0.set_title("Bus0 Voltage Magnitude")
    ax_va0.set_title("Bus0 Voltage Angle")
    ax_vm1.set_title("Bus1 Voltage Magnitude")
    ax_va1.set_title("Bus1 Voltage Angle")

    ax_vm0.set_ylabel("Vm [pu]")
    ax_vm1.set_ylabel("Vm [pu]")
    ax_vm1.set_xlabel("Time [s]")
    ax_va1.set_xlabel("Time [s]")
    ax_va0.set_ylabel("Va [deg]")
    ax_va1.set_ylabel("Va [deg]")

    for ax in [ax_vm0, ax_va0, ax_vm1, ax_va1]:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plt.suptitle("Dynamic RMS Comparison by Generator Saturation Model")
    plt.tight_layout()
    plt.show()


def _seed_problem_init_guess_from_vector(problem, x_vec: np.ndarray) -> None:
    """Write a full state/algebraic vector back into problem.init_guess."""
    for uid, idx in problem.uid2idx_vars.items():
        if idx < len(x_vec):
            problem.init_guess[uid] = float(x_vec[idx])


def _build_eigen_matrices(problem, x: np.ndarray, dx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build (A, B) for eigenproblem A v = lambda B v used by RMS SSS dense path."""
    nx = problem.get_states_number()
    ny = problem.get_algebraic_var_number()

    if problem.get_diff_var_number() == 0:
        h = problem.get_dt_value()
        fx = problem.get_j11(x, dx, h)
        fy = problem.get_j12(x, dx, h)
        gx = problem.get_j21(x, dx, h)
        gy = problem.get_j22(x, dx, h)
        A = fx.toarray() - fy.toarray() @ np.linalg.pinv(gy.toarray()) @ gx.toarray()
        B = np.eye(A.shape[0], dtype=np.float64)
        return A, B

    # Generalized eigenproblem branch (same structure as run_dense_small_signal_stability)
    E_matrix = problem.get_E_matrix(x, dx)
    if nx == 0:
        fx = sp.csr_matrix((0, 0))
        fy = sp.csr_matrix((0, ny))
        gx = sp.csr_matrix((ny, 0))
    else:
        fx = problem.get_j11(x, dx, 1e10)
        fy = problem.get_j12(x, dx, 1e10)
        gx = problem.get_j21(x, dx, 1e10)
    gy = problem.get_j22(x, dx, 1e15)

    j_top = sp.hstack([fx, fy])
    j_bot = sp.hstack([gx, gy])
    a_matrix = sp.vstack([j_top, j_bot]) + sp.eye(nx + ny) * 1e-10
    a_dense = a_matrix.toarray()

    # Dense generalized eigenproblem: A v = lambda (-E) v
    return a_dense, -E_matrix


def validate_eigenvalues(problem,
                         x: np.ndarray,
                         dx: np.ndarray,
                         eigenvalues: np.ndarray,
                         singular_tol: float = 1e-6,
                         relative_tol: float = 1e-8) -> dict:
    """Validate eigenpairs by singularity of pencil: sigma_min(A - lambda B)."""
    A, B = _build_eigen_matrices(problem=problem, x=x, dx=dx)

    mask_rep = np.isfinite(eigenvalues) & (np.abs(eigenvalues) < 1e6)
    ev_rep = eigenvalues[mask_rep]
    if len(ev_rep) == 0:
        return {"ok": False, "reason": "no finite eigenvalues"}

    abs_sigmins = []
    rel_sigmins = []
    abs_pass = 0
    rel_pass = 0
    for lam in ev_rep:
        M = A - lam * B
        svals = la.svdvals(M)
        sigma_min = float(svals[-1]) if svals.size > 0 else float("inf")
        sigma_max = float(svals[0]) if svals.size > 0 else float("inf")
        rel_sigma = sigma_min / sigma_max if sigma_max > 0 else float("inf")

        abs_sigmins.append(sigma_min)
        rel_sigmins.append(rel_sigma)
        if sigma_min <= singular_tol:
            abs_pass += 1
        if rel_sigma <= relative_tol:
            rel_pass += 1

    n = len(ev_rep)
    max_abs_sigma_min = float(np.max(abs_sigmins)) if abs_sigmins else float("inf")
    mean_abs_sigma_min = float(np.mean(abs_sigmins)) if abs_sigmins else float("inf")
    max_rel_sigma_min = float(np.max(rel_sigmins)) if rel_sigmins else float("inf")
    mean_rel_sigma_min = float(np.mean(rel_sigmins)) if rel_sigmins else float("inf")

    # Success if all reported lambdas make pencil numerically singular
    ok = (abs_pass == n) or (rel_pass == n)
    return {
        "ok": ok,
        "model_sizes": {
            "states": int(problem.get_states_number()),
            "diff_vars": int(problem.get_diff_var_number()),
            "alg_vars": int(problem.get_algebraic_var_number()),
        },
        "reported_modes": int(n),
        "singular_abs_pass": int(abs_pass),
        "singular_rel_pass": int(rel_pass),
        "max_sigma_min": max_abs_sigma_min,
        "mean_sigma_min": mean_abs_sigma_min,
        "max_rel_sigma_min": max_rel_sigma_min,
        "mean_rel_sigma_min": mean_rel_sigma_min,
        "tol_sigma_min": singular_tol,
        "tol_rel_sigma": relative_tol,
    }


def run_polynomial_builder_analysis(problem) -> dict:
    """Run the TenSyGrid PolynomialMatrixBuilder small-signal flow."""
    original_executor_selector = pmb_module._parallel_executor_cls
    pmb_module._parallel_executor_cls = lambda: concurrent.futures.ThreadPoolExecutor
    builder = PolynomialMatrixBuilder(problem=problem, verbose=False)
    v_dict = PolynomialMatrixBuilder.op_extraction(problem=problem)
    builder.linearize(v_dict=v_dict)
    res = builder.compute_stability()
    pmb_module._parallel_executor_cls = original_executor_selector

    eigenvalues = res[0]
    finite_eigs = eigenvalues[np.isfinite(eigenvalues) & (np.abs(eigenvalues) < 1e6)]
    max_real = float(np.max(np.real(finite_eigs))) if len(finite_eigs) > 0 else float("nan")
    stable = bool(res[4])
    margin = float(res[5]) if res[5] is not None else float("nan")

    return {
        "eigenvalues": finite_eigs,
        "n_eigs": int(len(finite_eigs)),
        "max_real": max_real,
        "stable": stable,
        "margin": margin,
    }


def compare_eigen_sets(ref_eigs: np.ndarray, poly_eigs: np.ndarray, tol: float = 1e-2) -> dict:
    """Compare two eigenvalue sets with global one-to-one matching."""
    if len(ref_eigs) == 0 or len(poly_eigs) == 0:
        return {"matches": 0, "ref_only": int(len(ref_eigs)), "poly_only": int(len(poly_eigs)), "mean_err": float("nan")}

    # Build all admissible pairings (distance <= tol), then select globally
    # by increasing distance with one-to-one constraints.
    candidates = []
    for i, ev_ref in enumerate(ref_eigs):
        d = np.abs(poly_eigs - ev_ref)
        valid = np.where(d <= tol)[0]
        for j in valid:
            candidates.append((float(d[j]), i, int(j)))

    candidates.sort(key=lambda x: x[0])

    used_ref = set()
    used_poly = set()
    errs = []
    for err, i, j in candidates:
        if i in used_ref or j in used_poly:
            continue
        used_ref.add(i)
        used_poly.add(j)
        errs.append(err)

    matches = len(errs)

    return {
        "matches": int(matches),
        "ref_only": int(len(ref_eigs) - matches),
        "poly_only": int(len(poly_eigs) - matches),
        "mean_err": float(np.mean(errs)) if errs else float("nan"),
    }


def small_signal_analysis(show_plt: bool = False,
                          test_mti=False,
                          hard_sat_type: str = "ml",
                          use_multilinear_problem: bool = True):
    """Perform small signal analysis on the phasor grid."""
    print("=" * 60)
    print("SMALL SIGNAL ANALYSIS - PHASOR RMS")
    print("=" * 60)

    # ===================================================================
    # CREATE GRID (Same as phasor validation)
    # ===================================================================
    print("\n[1] Creating phasor grid...")

    Sbase = 100.0
    grid = vge.MultiCircuit(Sbase=Sbase, fbase=50.0)

    bus0 = vge.Bus(name="Bus0", Vnom=10, is_slack=True)
    bus1 = vge.Bus(name="Bus1", Vnom=10)
    grid.add_bus(bus0)
    grid.add_bus(bus1)

    for bus in grid.buses:
        vge.initialize_bus_phasor_rms(bus, vf=grid.var_factory)

    line = vge.Line(name="Line", bus_from=bus0, bus_to=bus1,
                    r=0.029585798816568046, x=0.07100591715976332, b=0.03, rate=900.0)
    grid.add_line(line)

    load = vge.Load(P=9.999999, Q=0.999999)
    grid.add_load(bus=bus1, api_obj=load)

    gen = vge.Generator(name="Gen0", P=10, vset=1.0, Snom=900)
    grid.add_generator(bus=bus0, api_obj=gen)

    # Build RMS models
    genqec = vge.get_complete_generator_template_phasor(
        grid.var_factory,
        name="Gen0",
        hard_sat_type=hard_sat_type,
    ).block
    genqec = vge.to_implicit(genqec, grid.var_factory)

    line_mdl = vge.get_line_phasor_rms_template(grid.var_factory).block
    grid.var_factory.add_connections([line_mdl.in_vars[0]], [bus0.rms_model.out_vars[0]])
    grid.var_factory.add_connections([line_mdl.in_vars[1]], [bus0.rms_model.out_vars[1]])
    grid.var_factory.add_connections([line_mdl.in_vars[2]], [bus1.rms_model.out_vars[0]])
    grid.var_factory.add_connections([line_mdl.in_vars[3]], [bus1.rms_model.out_vars[1]])

    # Attach line model to the line device
    set_rms_model(device=line, model=line_mdl, var_factory=grid.var_factory)

    grid.var_factory.add_connections([genqec.in_vars[0]], [bus0.rms_model.out_vars[0]])
    grid.var_factory.add_connections([genqec.in_vars[1]], [bus0.rms_model.out_vars[1]])

    set_rms_model(device=gen, model=genqec, var_factory=grid.var_factory)

    load_mdl = vge.get_load_phasor_current_rms_template(grid.var_factory).block
    grid.var_factory.add_connections([load_mdl.in_vars[0]], [bus1.rms_model.out_vars[0]])
    grid.var_factory.add_connections([load_mdl.in_vars[1]], [bus1.rms_model.out_vars[1]])
    set_rms_model(device=load, model=load_mdl, var_factory=grid.var_factory)

    print(f"  ✓ Grid created with generator hard_sat_type='{hard_sat_type}'")

    # ===================================================================
    # POWER FLOW
    # ===================================================================
    print("\n[2] Running power flow...")
    pf_options = vge.PowerFlowOptions()
    pf_results = vge.power_flow(grid, pf_options)

    if pf_results.converged:
        print(f"  ✓ Power flow converged")
        for i, bus in enumerate(grid.buses):
            v = pf_results.voltage[i]
            print(f"    {bus.name}: Vm={abs(v):.4f} pu")
    else:
        print("  ✗ Power flow failed")
        return False

    # ===================================================================
    # CREATE PHASOR PROBLEM
    # ===================================================================
    print("\n[3] Creating phasor problem...")

    rms_options = vge.RmsOptions(
        time_step=0.01,
        simulation_time=1.0,
        tolerance=1e-6,
        max_iter=20,
        problem_type=vge.RmsProblemTypes.CurrentBalance,
    )

    if use_multilinear_problem:
        phasor_problem = vge.RmsProblemMultilinear(grid=grid, options=rms_options, pf_results=pf_results)
        print("  ✓ Using RmsProblemMultilinear")
    else:
        phasor_problem = vge.RmsProblemPhasor(grid=grid, options=rms_options, pf_results=pf_results)
        print("  ✓ Using RmsProblemPhasor")
    print(f"  ✓ Phasor problem created with {phasor_problem.get_states_number()} states")
    #if test_mti: tensor_problem = MTImatrix(phasor_problem)

    # ===================================================================
    # SMALL SIGNAL CONFIGURATION (PHASOR ONLY)
    # ===================================================================
    print("\n[4] Setting up small signal analysis (phasor only)...")

    ss_options = vge.RmsSmallSignalStabilityOptions(ss_assessment_time=0)
    k = ss_options.k if ss_options.k is not None else phasor_problem.get_states_number()

    print("  ✓ Configured to run directly on RmsProblemPhasor")

    # ===================================================================
    # INSPECT JACOBIAN
    # ===================================================================
    print("\n[4.5] Inspecting Jacobian...")
    import scipy.sparse as sp
    import scipy.linalg as la

    def inspect_gy(gy: sp.spmatrix, problem, tol: float = 1e-12) -> None:
        """Inspect Jacobian matrix to identify singularities."""
        gy_csc = gy.tocsc()
        gy_csr = gy.tocsr()
        m, n = gy_csc.shape
        print(f"gy shape: {m} x {n}, nnz={gy_csc.nnz}")
        if m != n:
            print("gy is not square")
            return
        zero_rows = np.flatnonzero(np.diff(gy_csr.indptr) == 0)
        zero_cols = np.flatnonzero(np.diff(gy_csc.indptr) == 0)
        print(f"zero rows: {zero_rows.tolist()}")
        print(f"zero cols: {zero_cols.tolist()}")

        # Print details of zero rows
        if len(zero_rows) > 0:
            print("\n=== Zero Row Details ===")
            for idx in zero_rows:
                if idx < len(problem._algebraic_vars):
                    print(f"\nRow {idx}:")
                    print(f"  Variable: {problem._algebraic_vars[idx]}")
                    if idx < len(problem._algebraic_eqs):
                        eq = problem._algebraic_eqs[idx]
                        print(f"  Equation: {eq}")
                        vars_in_eq = eq.get_vars()
                        print(f"  Variables in eq ({len(vars_in_eq)}): {[str(v) for v in vars_in_eq]}")

                        # Check if variables are actually in algebraic_vars by UID
                        print(f"  Variable UID check:")
                        for v in vars_in_eq:
                            in_alg = any(av.uid == v.uid for av in problem._algebraic_vars)
                            print(f"    {v.name} (uid={v.uid}) in algebraic_vars: {in_alg}")

        # Print details of zero columns
        if len(zero_cols) > 0:
            print("\n=== Zero Column Details ===")
            for idx in zero_cols:
                if idx < len(problem._algebraic_vars):
                    print(f"\nCol {idx}: {problem._algebraic_vars[idx]}")

        dense = gy_csc.toarray()
        s = la.svdvals(dense)
        smin = s[-1]
        smax = s[0]
        cond = np.inf if smin < tol else smax / smin
        print(f"\nsigma_min = {smin:.3e}")
        print(f"sigma_max = {smax:.3e}")
        print(f"cond      = {cond:.3e}")
        if smin < tol:
            ns = la.null_space(dense)
            print(f"nullity ≈ {ns.shape[1]}")

            # Analyze null space to find problematic variables
            print("\n=== Null Space Analysis ===")
            for i in range(ns.shape[1]):
                vec = ns[:, i]
                # Find largest components
                abs_vec = np.abs(vec)
                top_indices = np.argsort(abs_vec)[-5:][::-1]
                print(f"\nNull vector {i} (top 5 components):")
                for idx in top_indices:
                    if abs_vec[idx] > 1e-6:
                        if idx < len(problem._algebraic_vars):
                            var_name = str(problem._algebraic_vars[idx])
                            print(f"  [{idx}] {var_name}: {vec[idx]:.4f}")
                        else:
                            print(f"  [{idx}] (no var): {vec[idx]:.4f}")

    try:
        problem = phasor_problem
        x = problem.get_x0()
        dx = np.zeros(problem.get_diff_var_number())
        h = 0.001
        gy = problem.get_j22(x, dx, h)
        inspect_gy(gy, problem)
    except Exception as e:
        print(f"Error inspecting Jacobian: {e}")

    # ===================================================================
    # RUN ANALYSIS
    # ===================================================================
    print("\n[5] Running small signal analysis...")
    x = phasor_problem.get_x0()
    dx = np.zeros_like(x)
    n = phasor_problem.get_states_number()
    if k >= n - 1 or k == 0:
        (eigenvalues,
         participation_factors,
         damping_ratios,
         frequencies,
         state_matrix,
         _) = run_dense_small_signal_stability(problem=phasor_problem, x=x, dx=dx, verbose=ss_options.verbose)
    else:
        (eigenvalues,
         participation_factors,
         damping_ratios,
         frequencies,
         state_matrix,
         _) = run_sparse_small_signal_stability(problem=phasor_problem, x=x, dx=dx, k=k, verbose=ss_options.verbose)

    if use_multilinear_problem:
        eig_ml, stable_ml, margin_ml = phasor_problem.compute_stability_multilinear(x=x)
        status_ml = "STABLE" if stable_ml else "UNSTABLE/MARGINAL"
        print(f"  [Multilinear S/Phi eig] modes={len(eig_ml)} status={status_ml} margin={margin_ml:.6e}")

        finite_driver = eigenvalues[np.isfinite(eigenvalues) & (np.abs(eigenvalues) < 1e6)]
        finite_ml = eig_ml[np.isfinite(eig_ml) & (np.abs(eig_ml) < 1e6)]
        drv_ml_cmp = compare_eigen_sets(ref_eigs=finite_driver, poly_eigs=finite_ml, tol=1e-2)
        print(
            "  [Driver vs Multilinear] "
            f"matches={drv_ml_cmp['matches']} "
            f"driver_only={drv_ml_cmp['ref_only']} "
            f"ml_only={drv_ml_cmp['poly_only']} "
            f"mean_err={drv_ml_cmp['mean_err']:.3e}"
        )

    # ===================================================================
    # DISPLAY RESULTS
    # ===================================================================
    print("\n[6] Analysis complete!")

    # Filter out infinite eigenvalues (algebraic constraints)
    finite_mask = np.isfinite(eigenvalues) & (np.abs(eigenvalues) < 1e6)
    eigenvalues = eigenvalues[finite_mask]
    if participation_factors is not None:
        participation_factors = participation_factors[:, finite_mask]
    if damping_ratios is not None and len(damping_ratios) == len(finite_mask):
        damping_ratios = damping_ratios[finite_mask]
    if frequencies is not None and len(frequencies) == len(finite_mask):
        frequencies = frequencies[finite_mask]

    print(f"  Found {len(eigenvalues)} finite eigenvalues")

    state_vars = phasor_problem.state_vars

    # Display dominant modes
    print(f"\n  Dominant eigenvalues (top 10):")
    print(f"  {'Mode':<6} {'Real':<15} {'Imag':<15} {'Freq (Hz)':<12} {'Damping':<12} {'Status':<10} {'Dominant State'}")
    print(f"  {'-' * 130}")

    for i in range(min(50, len(eigenvalues))):
        lam = eigenvalues[i]
        real_part = np.real(lam)
        imag_part = np.imag(lam)

        # Calculate frequency and damping from eigenvalue directly
        freq = abs(imag_part) / (2 * np.pi)
        # Damping ratio: -Real / |lambda| for oscillatory modes, sign(Real) for real modes
        magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)
        if magnitude > 1e-12:
            damping = -real_part / magnitude
        else:
            damping = 0.0

        status = "Stable" if real_part < -1e-6 else "UNSTABLE" if real_part > 1e-6 else "Marginal"

        dominant_state = "N/A"
        if participation_factors is not None and i < participation_factors.shape[1] and participation_factors.shape[0] > 0:
            pf = np.abs(participation_factors[:, i])
            max_pf_idx = np.argmax(pf)
            dominant_state = str(state_vars[max_pf_idx]) if max_pf_idx < len(state_vars) else f"State {max_pf_idx}"

        print(
            f"  {i + 1:<6} {real_part:<15.6f} {imag_part:<15.6f} {freq:<12.4f} "
            f"{damping:<12.4f} {status:<10} {dominant_state}"
        )

    # Check for unstable eigenvalues
    unstable = [(i, ev) for i, ev in enumerate(eigenvalues) if np.real(ev) > 1e-6]
    if unstable:
        print(f"\n  ⚠️  Found {len(unstable)} unstable eigenvalues:")
        print(f"  {'Mode':<6} {'Real':<15} {'Imag':<15} {'Dominant State'}")
        print(f"  {'-' * 80}")

        for i, ev in unstable:
            # Find the state with the highest participation for this mode
            if participation_factors is not None and i < participation_factors.shape[1]:
                pf = np.abs(participation_factors[:, i])
                max_pf_idx = np.argmax(pf)
                dominant_state = str(state_vars[max_pf_idx]) if max_pf_idx < len(state_vars) else f"State {max_pf_idx}"
                pf_val = pf[max_pf_idx]
            else:
                dominant_state = "N/A"
                pf_val = 0.0

            print(f"  {i + 1:<6} {np.real(ev):<15.6f} {np.imag(ev):<15.6f} {dominant_state:<30} (PF={pf_val:.3f})")

    # Stability
    max_real = np.max(np.real(eigenvalues))
    print(f"\n  Stability: {'STABLE' if max_real < -1e-6 else 'UNSTABLE' if max_real > 1e-6 else 'MARGINAL'}")

    # ===================================================================
    # EIGENVALUE VALIDATION
    # ===================================================================
    x_ref = phasor_problem.get_x0()
    dx_ref = np.zeros_like(x_ref)
    validation = validate_eigenvalues(
        problem=phasor_problem,
        x=x_ref,
        dx=dx_ref,
        eigenvalues=eigenvalues,
        singular_tol=1e-6,
        relative_tol=1e-8,
    )
    print("\n  Eigenvalue validation:")
    if validation.get("ok", False):
        print(
            f"    ✓ PASS | singular_abs={validation['singular_abs_pass']}/{validation['reported_modes']} "
            f"singular_rel={validation['singular_rel_pass']}/{validation['reported_modes']} "
            f"max sigma_min={validation['max_sigma_min']:.3e}, max rel sigma={validation['max_rel_sigma_min']:.3e}"
        )
    else:
        print(
            f"    ✗ FAIL | singular_abs={validation['singular_abs_pass']}/{validation['reported_modes']} "
            f"singular_rel={validation['singular_rel_pass']}/{validation['reported_modes']} "
            f"max sigma_min={validation['max_sigma_min']:.3e}, max rel sigma={validation['max_rel_sigma_min']:.3e}"
        )
    print(
        f"    info | sizes={validation.get('model_sizes')} "
        f"reported={validation.get('reported_modes')}"
    )

    # ===================================================================
    # DYNAMIC RMS RUN
    # ===================================================================
    print("\n[7] Running dynamic RMS simulation on phasor problem...")
    dyn_h = 0.01
    dyn_t_end = 1.0
    dyn_max_iter = 100 if hard_sat_type == "normal" else 20

    # Optional warm-start for the normal saturation model to improve initialization.
    if hard_sat_type == "normal":
        print("  Running warm-start micro-step for initialization...")
        phasor_problem.reset_boundary_update_state(0.0)
        warm_solver = BackEulerImplicitIntegration(
            problem=phasor_problem,
            t0=0.0,
            t_end=1e-4,
            h=1e-4,
            max_iter=300,
        )
        t_warm, y_warm, warm_well_initialized, warm_converged = warm_solver.simulate()
        if warm_converged and y_warm is not None and y_warm.shape[0] > 0:
            _seed_problem_init_guess_from_vector(phasor_problem, y_warm[-1, :])
            print(f"  Warm-start status: converged={bool(warm_converged)}, well_initialized={bool(warm_well_initialized)}")
        else:
            print("  Warm-start status: failed, continuing with original initialization")

    dyn_solver = BackEulerImplicitIntegration(
        problem=phasor_problem,
        t0=0.0,
        t_end=dyn_t_end,
        h=dyn_h,
        max_iter=dyn_max_iter,
    )
    phasor_problem.reset_boundary_update_state(0.0)
    print(f"  Dynamic Newton max_iter: {dyn_max_iter} (mode={hard_sat_type})")
    t_sol, y_sol, well_initialized, dyn_converged = dyn_solver.simulate()

    vm0 = va0 = vm1 = va1 = None
    t_dyn = np.array(t_sol, dtype=float)
    if y_sol is not None and y_sol.size > 0:
        uid2idx = phasor_problem.uid2idx_vars

        def extract_by_uid(var_uid):
            idx = uid2idx.get(var_uid)
            if idx is None:
                return None
            return y_sol[:, idx]

        vm0 = extract_by_uid(bus0.rms_model.out_vars[0].uid)
        va0 = extract_by_uid(bus0.rms_model.out_vars[1].uid)
        vm1 = extract_by_uid(bus1.rms_model.out_vars[0].uid)
        va1 = extract_by_uid(bus1.rms_model.out_vars[1].uid)

    print(f"  Dynamic simulation converged: {bool(dyn_converged)}")
    print(f"  Dynamic simulation well initialized: {bool(well_initialized)}")

    # ===================================================================
    # POLYNOMIAL MATRIX BUILDER ANALYSIS
    # ===================================================================
    print("\n[7.5] Running PolynomialMatrixBuilder analysis...")
    poly_res = run_polynomial_builder_analysis(phasor_problem)
    poly_status = "STABLE" if poly_res["stable"] else "UNSTABLE"
    print(
        f"  PolynomialBuilder: n_eigs={poly_res['n_eigs']} "
        f"max_real={poly_res['max_real']:.6e} status={poly_status} margin={poly_res['margin']:.6e}"
    )
    eig_cmp = compare_eigen_sets(eigenvalues, poly_res["eigenvalues"], tol=1e-2)
    print(
        f"  Eig comparison (tol=1e-2): matches={eig_cmp['matches']} "
        f"ref_only={eig_cmp['ref_only']} poly_only={eig_cmp['poly_only']} "
        f"mean_err={eig_cmp['mean_err']:.3e}"
    )

    # ===================================================================
    # PLOT
    # ===================================================================
    if show_plt:
        print("\n[8] Generating small-signal plots...")
        generate_plots(eigenvalues, participation_factors)

    # vge.save_file(grid, "test.gridcal")

    return {
        "ok": True,
        "hard_sat_type": hard_sat_type,
        "n_eigs": int(len(eigenvalues)),
        "max_real": float(np.max(np.real(eigenvalues))) if len(eigenvalues) > 0 else float("nan"),
        "validation_ok": bool(validation.get("ok", False)),
        "validation_max_err": float(validation.get("max_sigma_min", float("nan"))),
        "dyn": {
            "t": t_dyn,
            "vm0": vm0,
            "va0": va0,
            "vm1": vm1,
            "va1": va1,
            "converged": dyn_converged,
        },
        "poly": poly_res,
        "eig_cmp": eig_cmp,
    }


if __name__ == "__main__":

    modes = ["ml"]
    summaries = []

    for mode in modes:
        print("\n" + "#" * 80)
        print(f"Running mode: {mode}")
        print("#" * 80)
        summaries.append(
            small_signal_analysis(
                show_plt=True,
                test_mti=True,
                hard_sat_type=mode,
                use_multilinear_problem=True,
            )
        )

    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    for res in summaries:
        status = "STABLE" if res["max_real"] < -1e-6 else "UNSTABLE" if res["max_real"] > 1e-6 else "MARGINAL"
        vstatus = "PASS" if res.get("validation_ok", False) else "FAIL"
        verr = res.get("validation_max_err", float("nan"))
        dconv = res.get("dyn", {}).get("converged", False)
        poly = res.get("poly", {})
        cmp = res.get("eig_cmp", {})
        poly_max_real = poly.get("max_real", float("nan"))
        poly_status = "STABLE" if poly.get("stable", False) else "UNSTABLE"
        print(
            f"  mode={res['hard_sat_type']:<7} n_eigs={res['n_eigs']:<4} "
            f"max_real={res['max_real']:.6e} status={status} eig_check={vstatus} "
            f"max_sigma_min={verr:.3e} dyn_conv={dconv} "
            f"poly_max_real={poly_max_real:.6e} poly_status={poly_status} "
            f"eig_matches={cmp.get('matches', 0)}"
        )

    dynamic_runs = []
    for res in summaries:
        dyn = res.get("dyn", {})
        dynamic_runs.append({
            "hard_sat_type": res.get("hard_sat_type", "mode"),
            "t": dyn.get("t"),
            "vm0": dyn.get("vm0"),
            "va0": dyn.get("va0"),
            "vm1": dyn.get("vm1"),
            "va1": dyn.get("va1"),
        })

    print("\nGenerating combined dynamic plots...")
    generate_dynamic_plots(dynamic_runs)

    sys.exit(0)
