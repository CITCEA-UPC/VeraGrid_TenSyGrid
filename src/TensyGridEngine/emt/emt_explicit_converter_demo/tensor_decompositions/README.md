# IEEE-9 EMT tensor decomposition experiments

These experiments use the reduced IEEE-9 model built by
`../utils/ieee9_gfl_ode_reduced.py`. Run from the repository root:

```bash
VERAGRID_EMT_INIT_CACHE_DIR=/tmp/veragrid_emt_tensor PYTHONPATH=src \
  python -m \
  TensyGridEngine.emt.emt_explicit_converter_demo.tensor_decompositions.ieee9_gfl_ode_full_converter_cp
```

The cache setting keeps EMT initialization files in a writable directory.

## Experiments

- `ieee9_gfl_ode_nodal_tensor.py`: eliminate 75 local device equations and
  form the 27-by-27 nodal matrix. Its inverse has bus/phase separation rank
  two in this balanced resistive example.
- `ieee9_gfl_ode_nodal_cp.py`: encode the 27-input linear nodal solve as a
  binary-monomial coefficient tensor. Its exact CP rank is 27. It also checks
  coefficient reconstruction from the two bus/phase factors and benchmarks
  the dense, explicit-CP, and structured evaluators.
- `ieee9_gfl_ode_cp_rank_probe.py`: inspect the full 49-output, 49-binary-mode
  coefficient tensor. Its 1,628 monomial supports provide a CP upper bound;
  the strongest tested unfolding gives a lower bound of 125.
- `ieee9_gfl_ode_degree5_tt.py`: isolate the degree-five part of each
  converter's six current equations. The grouped tensor has shape
  (6, 18, 3, 2, 2, 2), and exact TT-SVD ranks (2, 4, 4, 2, 2). It uses
  232 values instead of 2,592 dense coefficients per converter.
- `ieee9_gfl_ode_degree5_cp.py`: fit eight grouped CP terms to that same
  degree-five block. It uses 272 values and reconstructs coefficients to
  about 1e-13 relative error. The complete numerical factors and state
  groups are in `ieee9_gfl_ode_degree5_cp_expression.md`.
- `ieee9_gfl_ode_full_converter_cp.py`: verify a genuine binary-mode CP
  representation including every degree. The exact monomial construction
  takes 822 terms for each 17-equation converter and 1,628 for all 49
  equations. It reproduces every coefficient, but is larger than sparse
  `S/Phi`; no elementary one-mode affine merges were found.
- `ieee9_gfl_ode_full_grouped_tensor.py`: represent all degrees of a converter
  in one grouped tensor of shape (17, 22, 4, 3, 3, 3, 6), using a constant
  coordinate for absent state families. Exact TT-SVD ranks are
  (10, 20, 23, 15, 9, 6), with 8,048 stored values versus 4,390 or 4,418
  nonzero sparse coefficients. Optional `--cp-ranks` runs grouped CP-ALS.
  Converter 1 fits had coefficient errors of 8.72% at 24 terms, 0.777% at
  32, 0.401% at 48, 0.288% at 64, and 0.184% at 80. The last two used
  250 iterations; these fits are not exact replacements.
- `ieee9_gfl_ode_tt_benchmark.py`: compare complete 17-output converter RHS
  evaluations from sparse `S/Phi` and the exact grouped TT. With 400 random
  49-state inputs and one BLAS thread, sparse evaluation took 25.17 and
  26.08 microseconds per call for converters 1 and 2; TT took 50.68 and
  50.33 microseconds. The TT was about 2.0 and 1.9 times slower, with
  relative evaluation errors below 1.4e-13. Model/S/Phi extraction took
  2.96 seconds, and TT factorization took another 0.02–0.03 seconds per
  converter. These are evaluator timings, not full EMT solver timings.
- `ieee9_gfl_ode_grouped_cp_benchmark.py`: fit 80 grouped CP terms for every
  degree of both converters, then compare accuracy and per-call time against
  sparse `S/Phi`. Coefficient errors were 0.184% and 0.187%. Near the
  initialized state, median derivative-vector errors were 0.098% and 0.220%;
  the largest among 400 nearby samples were 0.124% and 0.248%. NumPy CP was
  slower than sparse evaluation (32–33 versus 25–27 microseconds), but with
  `--compiled` both paths use Numba loops: CP took 3.12–3.20 microseconds
  versus 7.81–8.08 for sparse, about 2.5 times faster per converter call.
  Fitting each CP took roughly one minute. With `--trajectory`, a mixed
  49-state RHS uses CP for both converters and exact sparse evaluation for
  the other 15 states. A short Radau run over 1e-4 seconds had maximum state
  error 5.11e-5 and final relative state error 3.89e-5. The approximate
  solve required 82 RHS calls versus 62 exact calls, but best-of-three wall
  time was 5.9 versus 6.5 milliseconds (about 9% faster). This short
  microbenchmark excludes the roughly two-minute CP fitting cost and does
  not establish accuracy or speed on longer trajectories or the production
  EMT solver.

## What the ranks mean

For the nodal inverse, the useful factorization is
`G^-1 = B_common ⊗ P_common + B_differential ⊗ P_differential`, where
`P_common = 11ᵀ/3` and `P_differential = I - P_common`. The two bus matrices
store 162 values instead of the dense inverse's 729. On the IEEE-9 nodal
matrix, coefficient reconstruction error was 4.31e-16 and maximum voltage
error over 20 random currents was 1.14e-15. Single-call NumPy timings were
1.44 microseconds for dense multiplication, 7.52 for the structured
evaluator, and 27.57 for the explicit Python CP loop. The phase symmetry
causes this factorization; it need not hold for unbalanced models.

The grouped converter tensors have one mode per physical state family.
Their CP and TT ranks are **not** CP ranks of the full tensor with one binary
mode for each of the 49 states. Only the monomial construction in
`ieee9_gfl_ode_full_converter_cp.py` directly represents that full tensor.
No exact compact CP decomposition for the complete 49-equation model has been
established. The measured Python/NumPy TT evaluator was slower than sparse
`S/Phi` for a complete converter. The approximate 80-term grouped CP has a
measurable compiled converter-evaluation speedup and a small speedup in one
short solver comparison; longer runs and the production EMT solver remain
unverified.
