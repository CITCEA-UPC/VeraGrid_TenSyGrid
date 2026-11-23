# EMT SSA performance changes

The performance changes keep the physical equations and numerical tolerances
unchanged. Monomial derivatives are evaluated by a cached Numba CSC kernel;
Floquet Jacobians use the shared BDF assembler instead of dense sparse-slice
assignment; standard Arnoldi searches compute the requested candidates and
honor the configured Krylov dimension. Frequency-targeted searches retain
oversampling.

Numba preserves the previous 1e-12 zero-factor rule and disables fast-math.
The assembler retains its pattern checks and exact-zero cancellation fallback.
A smaller Krylov space may require more iterations for strongly clustered
spectra; callers can increase max_krylov_dim.

TenSyGrid additionally uses sparse matched elimination for its existing
consistent-tangent analysis, with the original dense SVD retained as fallback.
The capture path rounds almost-integer period step counts and rejects truly
noninteger periods.

Recorded working-tree benchmarks (six modes, 600 steps/period) before and after
Numba/assembler reuse: IEEE39 26.97 -> 22.36 s; IEEE118 104.72 -> 92.07 s on the
first run and 81.25 s on a repeat. Trajectories were identical; IEEE118 mode
families agreed within 2.02e-12 /s. Individual wall-clock runs are not a controlled
multi-repeat performance study. These measurements used the separately prepared
benchmark models; model corrections and experiment archives are outside this
performance commit.

Validation: 57 tests passed (23 November 2026):

```sh
PYTHONPATH=src /home/pablo/myenv/bin/python -m pytest -q \
  src/tests/RMS/test_multilinear_jacobian_numba.py \
  src/tests/RMS/test_rms_implicit_jacobian_assembler.py \
  src/tests/EMT/test_multilinear_floquet_jacobian.py \
  src/tests/EMT/test_emt_floquet_operator_state_space.py \
  src/tests/EMT/test_sparse_consistent_tangent.py
```

The earlier initialization optimizations are already present in this repository's
base revision. No benchmark binary captures or unrelated template edits are part
of this commit.
