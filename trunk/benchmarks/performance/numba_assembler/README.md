# Numba and JacobianAssembler reuse

The existing JacobianAssembler was useful. Its newer implementation in the
sibling VeraGrid checkout provides a compiled BDF data-fill routine and cached
CSC scatter maps. That routine and `assemble_jacobian_bdf` were ported into
TenSyGrid's existing `ImplicitJacobianAssembler`; the EMT multilinear Floquet
adapter now calls it. The original RMS Backward Euler API remains supported.
No code was changed in the sibling checkout.

The BDF coefficient is the current integration coefficient (2/h for trapezoidal),
so this produces the same differential rows `alpha*I - J` and unchanged algebraic
rows. Existing pattern checks and the exact-zero cancellation fallback are reused.
Returned matrices own their storage. The new tests check changing coefficients,
pattern reuse, identical CSC storage and cancellation fallback.

A separate cached Numba kernel differentiates the multilinear monomials directly
into CSC buffers. This replaces Python loops and list construction shared by RMS
and multilinear EMT. It preserves the existing 1e-12 zero-factor rule and uses
no fast-math reassociation. SciPy still constructs the matrix and performs sparse
products, and SuperLU still solves the tangent systems.

## Observed end-to-end results

Six modes; 600 steps per period; one period; StructuralAD; consistent DAE tangent;
same integration, eigensolver and closure tolerances as the 105-second reference.

| Case | Before this change | With Numba + assembler | Time reduction |
|---|---:|---:|---:|
| IEEE118, first run | 104.72 s | 92.07 s | 12.1% |
| IEEE118, repeat with warm cache | 104.72 s | **81.25 s** | **22.4%** |
| IEEE39 | 26.97 s | **22.36 s** | **17.1%** |

These are individual wall-clock observations, not a controlled multi-repeat
performance study. The first IEEE118 run overlapped a short regression suite;
the repeat also overlapped a short final regression check. Timing variation
cannot be attributed entirely to JIT compilation or cache warming. Both IEEE118
measurements are retained. The changes to assembly and monomial evaluation were
measured together; their separate full-run contributions have not been isolated.

Both trajectories are identical to their references (maximum absolute difference
zero), and the same periodicity checks pass at 1e-4. The six IEEE118 mode families
match within 2.02e-12 /s; IEEE39's match exactly. Compare against the previous sparse-
Jacobian optimization, not against the superseded faulty physical models.

The synthetic monomial benchmark uses 12,972 variables, 20,000 monomials and 59,993
stored coefficients. Python median: 112.10 ms; compiled median: 0.369 ms, about
304x faster for that kernel alone, with zero difference from the retained Python
reference. Its first reported call loaded an existing cache; this is not a cold
compilation measurement. The whole SSA calculation gains much less because other
work remains in integration, sparse assembly/products, initialization and LU solves.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/pablo/myenv/bin/python \
  trunk/benchmarks/multilinear_ssa.py --grid ieee118 --formulation emt \
  --steps-per-period 600 --assessment-periods 1 --modes 6 \
  --output trunk/benchmarks/performance/ieee118_numba_assembler_warm
/home/pablo/myenv/bin/python \
  trunk/benchmarks/performance/numba_assembler/time_kernel.py
```

Validation: **59 tests passed** in 12.58 s. These cover independent product
 differentiation with zero factors and 32/64-bit CSC indices; the existing
near-zero rule; the reused assembler and its fallback; independent discrete
residual finite differences for BE/trapezoidal/BDF2; known-spectrum Arnoldi;
consistent tangent reduction; and periodicity diagnostics. See
`regression_tests.log`, `summary.json` and `kernel_timing.json`.
