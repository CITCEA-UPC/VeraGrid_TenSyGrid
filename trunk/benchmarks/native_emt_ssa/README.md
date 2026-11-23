# Native DAE versus multilinear EMT SSA

This comparison separates the equation-evaluation backend from the choice of
device templates. Both completed native cases use **EmtProblemDae (CurrentBalance)**
and its **native StructuralAD discrete Jacobian**, with no S/Phi Floquet adapter.
The generator, ZIP load, PI line and coupled-core transformer equations are the
same auxiliary-model equations used by the existing multilinear reference.
They have the same state and algebraic variables. Thus these are native DAE
backend timings, not timings for a fully non-lifted replacement of every device.

## Completed, same-equation comparison

Six modes, Krylov dimension 24, 600 steps per fundamental period, one captured
period, trapezoidal integration, 1e-6 Newton tolerance, consistent tangent
reduction and a 1e-4 scaled periodicity tolerance. BLAS/OpenMP threads: one.
The standard IEEE39 input has fBase=50 Hz; IEEE118 has 60 Hz.

| Grid | Native DAE total | Native SSA including capture | Native construction | Optimized S/Phi total | S/Phi SSA including capture |
|---|---:|---:|---:|---:|---:|
| IEEE39 | **24.58 s** | 16.88 s | 1.01 s | 22.36 s | 15.83 s |
| IEEE118 | **87.99 s** | 60.94 s | 15.00 s | 81.25 s | 59.09 s |

These are individual observed wall-clock runs, not statistical evidence that one
backend is consistently faster. The first optimized S/Phi IEEE118 run took 92.07 s;
its repeat took 81.25 s, so the native 87.99 s result lies within that observed
range. These implementations are in the same performance range on these grids.

The full captured trajectories are **identical** (maximum absolute difference
zero). The matched mode families differ by at most 1.27e-11 /s for IEEE39 and
2.32e-12 /s for IEEE118. Both pass the same periodicity tolerance, with scaled
closure 5.7431e-5 and 8.8152e-5 respectively. Native dimensions are 620 states /
1502 algebraic variables for IEEE39 and 2811 / 7350 for IEEE118, matching the
multilinear references. The benchmark-model instability conclusions are unchanged.

## Fully standard, non-lifted templates: diagnostic only

A second check replaces the auxiliary generator and ZIP templates with their
standard nonlinear counterparts and uses the standard transformer template.
This changes more than the equation-evaluation backend: transformer core
structure/defaults and terminal capacitance differ, and auxiliary variables
are removed. It must pass its own operating-point checks before a valid SSA
timing can be compared.

* IEEE39: 600 states / 1187 algebraic variables. Capture reached a scaled closure
  of **0.0017415**, above 1e-4, so the benchmark stopped before mode extraction.
  The 11.69 s diagnostic total is **not an SSA timing**.
* IEEE118: 2703 states / 5475 algebraic variables. Its explicit initial DAE
  residual was **0.00103544**, above the common 1e-6 tolerance. The optional
  capture diagnostic was interrupted; no validated period or completed SSA
  timing is available. The saved partial report is explicitly marked interrupted.

The completed 25/88-second results must therefore not be described as timings
for the fully non-lifted standard-template models. Those models need separate
initialization/periodicity work before that second comparison is meaningful.
No model physics or validation tolerances were changed to force them through.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/pablo/myenv/bin/python \
  trunk/benchmarks/multilinear_ssa.py --grid ieee118 --formulation emt \
  --emt-problem CurrentBalance --emt-models auxiliary \
  --steps-per-period 600 --assessment-periods 1 --modes 6 \
  --output trunk/benchmarks/native_emt_ssa/ieee118_same_models
```

Substitute `ieee39` for the other grid. To reproduce the standard-template
precondition check, use `--emt-models canonical --capture-only` and a separate
output directory. The flag `--emt-problem Multilinear` selects the original
S/Phi backend. Transformer model selection and earlier benchmark options remain
available.

Artifacts: each run's JSON, mode CSV and captured NPZ are in its directory;
`summary.json` contains the comparisons. The existing two benchmark periodicity
regression checks pass (`benchmark_tests.log`).
