# Analysis: Is Floquet necessary for EMT stability with multilinear models?

Date: 2026-10-28

## Context
- Project documentation in `deliverable/docs/` (TenSyGrid proposal, paper "Small-Signal Stability Analysis of Power Systems by Implicit Multilinear Models", etc.)
- Code developed in `src/TensyGridEngine/` and `src/VeraGridEngine/`
- Current point: It is assumed that Floquet is needed to compute stability in EMT models, but this is not certain. Moreover, it is very slow.

## Answers to questions
1. **Switching**: These are models with switching (not averaged). For systems with periodic switching (PWM), the behavior is truly periodic and the correct analysis is Floquet over the limit cycle.
2. **Multilinear coverage**: In principle, it should cover the entire system (grid, loads, converters, machines). Currently, the implementation appears partial/integration is in progress.
3. **Objective**: We want to compute stability. According to TenSyGrid, the final path is to move to the iMTI + generalized eigenvalue approach, but it must be validated against Floquet.
4. **Speed**: Total time is the problem. As the number of buses grows, computation time skyrockets (poor scalability).
5. **Validation vs RMS**: There are differences between RMS and EMT. This justifies using EMT to obtain correct results.
6. **Use case**: We need this for large networks.

## Initial code observations

### Current approach (Floquet in time domain)
- `src/TensyGridEngine/small_signal_analysis_main_emt.py`: loads EMT network, forces multilinear models (`FORCE_MULTILINEAR_MACHINE=True`), performs power flow, simulates one period with `EmtOptions`, and runs `SmallSignalStabilityEmtDriver` with `build_type=Arnoldi`/`HybridArnoldi`. Computes Floquet multipliers $\mu_i$ of the monodromy matrix $\Phi = \partial x(T)/\partial x(0)$, evaluates stability with $|\mu_{max}|$, and saves results.
- `src/VeraGridEngine/Simulations/SmallSignalStabilityEmt/small_signal_stability_emt_driver.py`: implements `run_arnoldi()` (Arnoldi via SciPy over `EmtFloquetOperator`/`BlockEmtFloquetOperator` with LU factorizations cached per step) and `run_arnoldi_hybrid()` (BIRAM with block-Arnoldi, deflation, reorthogonalization, adaptive restarts). Also supports the `AkStackBlockEmtFloquetOperator` operator and consistent tangent projection for DAEs (`ProjectedEmtFloquetOperator`).
- `src/VeraGridEngine/Simulations/SmallSignalStabilityEmt/emt_floquet_operator.py`: matrix-free operator that propagates perturbations over the entire period by integrating the EMT DAE system with Jacobians/JIT; block version for BLAS3; explicit "Ak stack" version.
- Experiments with trigonometric reduction: `emt/emt_ieee9_projected_trig_small_signal.py` projects onto the tangent manifold of the relations $\cos\theta,u_sin$-$\sin\theta,u_cos$; `emt/emt_ieee9_trig_manifold_reduced_small_signal.py` and `manifold_reduced_emt_floquet_operator.py` eliminate lifted trigonometric states per step (reducing dimension). Comparisons with the "original machine" case are observed.

### Alternative approach / related to multilinear
- `src/VeraGridEngine/Simulations/EMT/problems/emt_problem_multilinear.py`: adds utilities to extract multilinear matrices $(S,\Phi)$ (monomial structure) by reusing logic from `RmsProblemMultilinear`.
- `src/VeraGridEngine/Simulations/Rms/problems/rms_problem_multilinear.py`: much more extensive infrastructure (CP/CPN, approximators, efficient linearization with tensor structure).
- Paper/proposal (TenSyGrid): proposes "small-signal stability analysis based on linearization of implicit multilinear models" computing generalized eigenvalues from a linearized descriptor model of the iMTI. It also shows significant gains in linearization time with tensor structure.
- `src/TensyGridEngine/fast_small_signal.py`: example of shift-and-invert on precomputed matrices $A,E$ (pencil method / for linear/descriptor systems) — conceptually much faster if $A,E$ are already available.

## Reasoning

### When is Floquet needed vs generalized eigenvalues?
- **LTI systems/linearized around an equilibrium point** (classical phasor RMS, or constant equilibrium): local stability is given by eigenvalues of $A$ (or generalized eigenvalues of $(A,B)$ / pencil). Floquet is not needed.
- **Systems with periodic regime (limit cycle)**: if the system is periodic in steady state (PWM switching, etc.), linearization in the classical sense around a point is not valid; it is necessary to linearize around the periodic orbit and obtain Floquet multipliers of the monodromy matrix. Stability holds if all $|\mu_i| < 1$.
- **Multilinear models / trig-lift / transformations**: the key point of TenSyGrid's work is that by transforming with implicit multilinear models (iMTI) and appropriate lifts (e.g., representing $\sin/\cos$ or products), one can obtain a formulation where linearization yields a time-invariant linear descriptor system in the transformed coordinates, allowing generalized eigenvalues to be computed directly and avoiding temporal integration over periods to build the monodromy. If this is done correctly, **Floquet would not be necessary** and would be much faster.

### Why is it slow now?
- It is necessary to simulate until periodic steady state and then integrate **an entire period** for **each matvec** in Arnoldi; with $k_{krylov}$ iterations and possibly restarts, many periods are integrated.
- Each step requires solving sparse DAE systems (LU per step, cached).
- Large EMT systems (many states, networks with converters) make each solve expensive.
- Experiments with trig reduction (eliminating lifted states) show dimension reduction helps, but fidelity must be ensured.

### Strengths and weaknesses of each approach

#### Approach A: Floquet (currently implemented)
**Strengths:**
- Theoretically correct for periodic regime/true EMT with switching or periodic dynamics.
- General, works with existing models without rewriting everything as pure iMTI.
- Mature infrastructure (operators, block-Arnoldi, Ak-stack, reductions, consistent tangent for trapezoidal DAE).
- Allows direct analysis of behavior on the limit cycle as modeled.

**Weaknesses:**
- Very computationally expensive (multiple period integrations + LU solves).
- Depends on limit cycle convergence (ss_assessment_time), time step accuracy (steps_per_period), etc.
- Difficult to scale to large networks within reasonable time for iterative use.
- Less direct than obtaining $A,E$ and solving the pencil eigenvalue problem.

#### Approach B: Direct linearization of multilinear models (iMTI) with generalized eigenvalues (proposed in TenSyGrid/paper)
**Strengths:**
- Potentially much faster if $(A,E)$ are obtained (avoids repeated period integrations). Consistent with the paper: "generalized eigenvalues are computed from linear descriptor models of the linearized implicit multilinear model".
- Compatible with tensor approach (CPN1) which provides large gains in linearization (orders of magnitude according to literature).
- More similar to classical workflow (operating point -> linearize -> eigenvalues), familiar to power system engineers.
- Scalable if tensor structure is properly exploited.

**Weaknesses:**
- Requires the model to be correctly formulated in multilinear/iMTI form and trigonometric/product lifts to be done well. If there is hard-switching or dynamics not captured by pure multilinear, validity must be checked.
- Must ensure linearization around the equilibrium of the transformed formulation truly corresponds to correct local dynamics (and that lifts don't introduce artifacts).
- Infrastructure is less developed compared to the Floquet path already integrated in VeraGridEngine.
- Must be carefully validated against Floquet/time-domain (trig-manifold experiments point in this direction).
