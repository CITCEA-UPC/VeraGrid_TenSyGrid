# Faster corrected EMT SSA benchmarks

> **Numba update (23 November 2026):** Reusing the JacobianAssembler BDF fill and compiling monomial derivatives reduces IEEE118 from 104.72 s to 92.07 s on the first run and **81.25 s** on a repeat; IEEE39 takes **22.36 s**. Trajectories are identical and the mode families agree. See [Numba and assembler results](numba_assembler/README.md) for timing conditions and tests.


Observed full-analysis times with six modes, 600 steps per period, one captured
period, StructuralAD integration, and consistent DAE tangent reduction:

| Grid | Previous total | Optimized total | Speedup | Previous SSA | Optimized SSA |
|---|---:|---:|---:|---:|---:|
| IEEE39 | 81.09 s | **26.97 s** | **3.01×** | 73.79 s | 20.46 s |
| IEEE118 | 634.26 s | **104.72 s** | **6.06×** | 609.31 s | 77.82 s |

These are observed wall-clock runs, not repeated isolated timing statistics.
The previous IEEE118 baseline ran alongside a capture refinement and a short
test suite; see the earlier report for its conditions.

## Changes

The multilinear Floquet adapter previously converted each Jacobian to LIL and
negated its differential rows through slice assignment. For IEEE118 that slice
covers 2811 × 10161 entries. SciPy's assignment path materialized the large
dense slice, including its zeros, at every one of the 600 time steps.
A partial profile recorded about 64 s in LIL assignment and 4.3 s in dense
conversion after only 98 Jacobian evaluations. No period-propagation matvec
had yet completed. Thus the apparent Arnoldi bottleneck was principally
Jacobian preparation.

The adapter now applies the same row signs through a sparse diagonal matrix
and adds a prebuilt sparse state-mass diagonal. It never materializes that
large dense slice. The residual formula, integration method, time step and
physical model are unchanged.

For ordinary largest-magnitude Arnoldi searches, the driver now requests the
number of modes the caller actually needs rather than twice that count.
Frequency-targeted searches retain oversampling. The benchmark's default
Krylov dimension is now 24 rather than 180; the driver still enforces its
minimum required dimension. Increase `--krylov-dim` for cases with difficult
spectral clustering. The timing comparison includes both changes and does
not isolate their individual contributions.

## Numerical agreement

Both optimized trajectories are identical to their saved reference trajectories
(maximum absolute difference zero). Both retain the same one-period closure:
IEEE39 5.7431e-5 and IEEE118 8.8152e-5, within the declared 1e-4 tolerance.
No integration or eigensolver tolerance was relaxed.

All six IEEE118 exponents match within **1.46e-8 /s**. IEEE39 mode families
match within **1.01e-7 /s**. Its sixth returned mode is the opposite conjugate
representative at the six-mode cutoff; the associated growth rate and
oscillation frequency agree. This is documented rather than treating the
sign of that representative's imaginary part as a physical discrepancy.

The instability conclusions for these benchmark models are unchanged.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/pablo/myenv/bin/python \
  trunk/benchmarks/multilinear_ssa.py --grid ieee118 --formulation emt \
  --steps-per-period 600 --assessment-periods 1 --modes 6 --krylov-dim 24 \
  --output trunk/benchmarks/performance/ieee118_sparse_jacobian
```

Substitute `ieee39` for the smaller case. JSON, modes and trajectories are in
`ieee39_sparse_jacobian/` and `ieee118_sparse_jacobian/`; comparison metrics are
in `summary.json`.

Validation: **31 focused tests passed**, including an independent finite-
difference check of the multilinear adapter for Backward Euler, trapezoidal
and BDF2 integration, at zero and nonzero states; a known-spectrum Arnoldi
test; consistent DAE reduction; and periodicity diagnostics. See
`regression_tests.log`.

The `ieee118_krylov40/` and `ieee118_search6_krylov24/` trials were interrupted
before a completed result and are excluded from comparisons. The saved
`partial_jacobian_profile.prof` is an intentionally interrupted diagnostic
using the existing verified capture, not a completed full benchmark.
