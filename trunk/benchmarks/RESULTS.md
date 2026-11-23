# IEEE39 and IEEE118 multilinear SSA: results and timing

Date: 2026-11-23. Checkout: VeraGrid_TenSyGrid, including the requested upstream
connectivity-validator disablement. Python 3.12.3; NumPy 2.3.0.

## Recorded setup runs

Six requested modes, 120 EMT steps per electrical period, one assessment period,
template-default machines, fixed initialized torque/field voltage. RMS uses
sparse descriptor SSA near zero; EMT uses Arnoldi Floquet analysis with the
multilinear Jacobian adapter. See [benchmark configuration](README.md) for grid
paths and the generic sequence-impedance assumptions.

| Grid | Formulation | Outcome | Build (s) | SSA (s) | Total (s) |
| --- | --- | --- | ---: | ---: | ---: |
| IEEE39 | RMS | completed | 0.817 | 0.333 | 9.162 |
| IEEE39 | EMT | completed | 38.062 | 19.436 | 65.652 |
| IEEE118 | RMS | completed | 4.677 | 2.527 | 14.318 |
| IEEE118 | EMT | timeout | — | — | — |

IEEE118 EMT was stopped at **300 seconds** and returned no modes. Its old log
only establishes balanced power-flow convergence; it does not resolve whether
the timeout occurred in three-phase PF or problem construction. The timeout
must not be reported as a 300-second SSA time.

These are individual exploratory runs, with cache and concurrent-load effects,
not controlled repeated performance measurements. The old EMT `build_seconds`
includes three-phase PF as well as DAE construction. RMS matrix assembly and
imports/model attachment were not all timed separately, so build plus SSA need
not equal total. The updated runner separates those phases and saves progress
before three-phase PF, DAE construction, and SSA.

Raw evidence: [outcome summary](observed/summary.json),
[IEEE39 RMS](observed/ieee39_rms.json), [IEEE39 EMT](observed/ieee39_emt.json),
[IEEE118 RMS](observed/ieee118_rms.json), [IEEE118 EMT timeout](observed/ieee118_emt.json).
The `observed` directory also contains mode CSVs, logs, and the IEEE39 trajectory.

## Numerical validity

Both RMS runs returned six finite modes, but their explicit operating-point
residuals are approximately **8.86** (IEEE39) and **5.94** (IEEE118). Equilibrium
is not validated. IEEE39 EMT returned six finite modes with 620 states and 1,502
algebraic variables; explicit initialization residual is approximately **23.59**
and scaled last-period closure error approximately **1.94**. Periodic steady state
is not verified. These timings describe executable analysis paths; the modes
must not be interpreted as validated physical stability results.

## Fresh IEEE39 timing profile

A subsequent run used the same six modes, 120 steps per period, and one assessment
period, with the new timing instrumentation. It completed with the same maximum
multiplier magnitude (approximately 7.136071) and the
same unverified periodicity as the earlier run.

| Phase | Time (s) |
| --- | ---: |
| Grid load | 1.158 |
| Attach models | 0.210 |
| Balanced PF | 2.922 |
| Three-phase PF | 1.446 |
| DAE construction, excluding PF | 37.551 |
| Trajectory capture + Floquet Jacobian setup | 1.581 |
| Floquet work after capture | 16.968 |
| Complete SSA, including output/closure handling | 18.591 |
| End-to-end | 65.741 |

SSA contains the capture and post-capture phases: do not add those rows twice.
Totals also include imports and miscellaneous overhead. In the engine's nested
build report, `events_group_finalize_s` is
36.653 seconds. This is a broad
finalization region, not proof that event processing alone takes that time;
it includes subsequent runtime/initialization bookkeeping and residual checks.
`numeric_refresh_s` overlaps this region and must not be added to it.

Capture plus Jacobian setup is only
2.4% of total runtime.
Even eliminating that entire measured phase would therefore save only about
1.58 seconds on this one-period case.
The stronger optimization targets are DAE finalization and the remaining Floquet
work. Longer assessment windows may change that balance and need a new profile.

Evidence: [profile JSON](timing_profile/ieee39_emt.json) and
[profile log](timing_profile/ieee39_emt.log). Single run, existing caches; no
statistically established solver speedup is claimed.

## Solver selection: original behavior and comparison change

**The original SSA path already used StructuralAD for trajectory capture.**
It constructed `StructuralVectorizedSolver` directly, bypassing the solver
factory. Merely changing `EmtOptions.solver_type` could not change capture.
The original timing tables above describe that behavior.

For the subsequent solver comparison, SSA now calls `build_emt_solver` and
honors `EmtOptions.solver_type`. Backend-specific Jacobian interfaces are
normalized after simulation, including dense and sparse Symbolic variants.
The multilinear problem still supplies the S/Phi Floquet Jacobian adapter;
only the trajectory backend changes in these comparisons.

The benchmark exposes `--emt-solver StructuralAD|StructuralCompiled|Automatic|Symbolic`.
`compare_emt_solvers.py` runs the four choices in isolated, sequential processes,
with identical settings and a configurable wall-clock limit per solver. Its
summary compares trajectories and matches the returned eigenvalue sets to the
StructuralAD reference. Seventeen regression tests pass for SSA capture and
backend Jacobian selection.

Relevant source:

- `src/VeraGridEngine/Simulations/SmallSignalStabilityEmt/small_signal_stability_emt_driver.py`
- `src/VeraGridEngine/Simulations/EMT/emt_solver_factory.py`
- `src/VeraGridEngine/Simulations/EMT/problems/emt_problem_multilinear.py`

## Practical acceleration options

1. **Identify the IEEE118 timeout phase first.** Three-phase PF and DAE build
   occur before trajectory simulation; a faster integration solver cannot speed
   up those phases. The runner now retains phase timings even on a batch timeout.
2. **Profile construction and Floquet work separately.** The runner now records
   the engine's build breakdown, trajectory capture plus Jacobian setup, and the
   remaining Floquet/Arnoldi work. This indicates whether to optimize symbolic
   construction, multilinear S/Phi generation, or propagation/eigensolution.
3. **Compare the actual trajectory backends.** The driver now uses the public
   solver factory; measured outcomes are recorded below. Compile/setup costs
   are included in capture time. Longer simulations may amortize those costs
   differently, so short-run results must not be extrapolated blindly.
4. **Use sparse RMS SSA for these larger cases.** It avoids computing thousands
   of algebraic/infinite descriptor modes. For EMT, coarser steps, fewer modes,
   and shorter assessment reduce work but also change resolution/coverage;
   validate time-step convergence and periodicity before using them for conclusions.

## Commands

```bash
python trunk/benchmarks/multilinear_ssa.py --all --timeout 300 \
  --steps-per-period 120 --assessment-periods 1 --modes 6

# Longer default-resolution attempt (1,800-second limit per case)
python trunk/benchmarks/multilinear_ssa.py --all --timeout 1800
```

## Measured solver comparison

Same IEEE39 model and SSA settings for all four trials: six requested modes,
120 steps per electrical period, one assessment period, tolerance 1e-6,
seed 0, and the multilinear S/Phi Floquet Jacobian. Each backend ran in a fresh
process, sequentially, with a **240-second limit for the entire child process**.
Existing compilation caches were retained. The source change makes solver
selection effective; these trials are actual different capture implementations.

| Capture backend | Outcome | Parent wall (s) | DAE build (s) | Capture + Jacobian setup (s) | Complete SSA (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| StructuralAD | completed | 69.370 | 41.593 | 1.790 | 20.047 |
| StructuralCompiled | timeout | 240.077 | 63.224 | — | — |
| Automatic | timeout | 240.081 | 106.717 | — | — |
| Symbolic | timeout | 240.100 | 95.764 | — | — |

StructuralAD returned six finite modes. Its worker-measured end-to-end time was
**68.005 s**; parent wall time additionally includes process startup and exit.
All other trials reached SSA but returned no modes before the limit. The
240-second values are **whole-process timeout observations, not SSA durations**.
A timeout does not distinguish compilation, integration, or later SSA work
within the unfinished call. No cross-backend trajectory or modal agreement
could be established; the zero comparison errors in the summary are the
StructuralAD reference compared with itself.

An initial Automatic attempt was interrupted before completion and is preserved
in `solver_comparison/Automatic_interrupted`; the table uses the subsequent
bounded retry. There were no additional completed trials of the other backends.
Differences in DAE build time, although the topology and formulation are held
fixed, also show why these single trials should not be treated as statistically
controlled performance rankings.

**Recommendation for this benchmark: retain StructuralAD.** No faster complete
alternative was obtained under the measured budget. This does not establish
that the other backends are intrinsically slower, or that they cannot benefit
from warm compilation caches and longer trajectories. Construction/finalization
and Floquet work remain the more promising measured optimization targets.
The large initialization residual and unverified periodic orbit still apply;
these are implementation/performance checks, not validated stability studies.

Evidence: [comparison summary](solver_comparison/summary.json),
[StructuralAD](solver_comparison/StructuralAD/ieee39_emt.json),
[StructuralCompiled](solver_comparison/StructuralCompiled/ieee39_emt.json),
[Automatic](solver_comparison/Automatic/ieee39_emt.json),
[Symbolic](solver_comparison/Symbolic/ieee39_emt.json).
Logs are alongside each JSON. [Regression test output](solver_comparison/regression_tests.log)
records **17 passed** for the SSA/Floquet test module, including backend selection
and dense/sparse Symbolic Jacobian interfaces.

## Reproduce the solver sweep

```bash
python trunk/benchmarks/compare_emt_solvers.py --timeout 240

python trunk/benchmarks/multilinear_ssa.py --grid ieee39 --formulation emt \
  --emt-solver StructuralCompiled --steps-per-period 120 --assessment-periods 1 --modes 6
```

The sweep keeps existing caches and defaults BLAS/OpenMP to one thread unless
the environment explicitly sets otherwise. Single trials include cold/warm
cache differences and process startup; they are exploratory measurements.

## Initialization acceleration and pinned upstream trial

### Concrete optimization

Profiling found repeated compilation of the state-RHS function used only to
complete initial derivatives (`_compute_missing_dx0`). The profiled old run
made 374 million calls and took 176.55 seconds; 127.68 seconds were inside
that initialization helper, dominated by Numba compilation. Profiling adds
overhead: **176.55 seconds is not the ordinary baseline runtime**.

The implementation now uses the existing generated Python evaluator for this
initialization-only function, instead of eagerly JIT-compiling large RHS
vectors. The persistent cache key was versioned so old JIT-enabled payloads
cannot silently restore that policy. Integration solver kernels remain compiled.
Initial-state and derivative vectors are also snapshotted once when building
final UID bindings, replacing per-UID full-vector rebuilds. The latter accounted
for about 5.33 seconds combined in the profiled run, much less than compilation.

| Downstream IEEE39, six modes / 120 steps / one assessment period | Before (s) | Optimized (s) |
| --- | ---: | ---: |
| DAE build | 37.551 | 0.862 |
| Complete SSA | 18.591 | 20.666 |
| End-to-end | 65.741 | 30.047 |

These single unprofiled observations show about
2.19x lower end-to-end runtime and
43.6x lower build time. They are not
controlled repeated trials: the optimized trial loaded saved model blocks,
whereas the earlier baseline attached them from templates. The decisive
numerical check is that the captured **121 x 2122 trajectory values and all six
matched modes are identical** in the saved before/after output comparison
(maximum recorded differences both zero). Initialization residual and periodicity
caveats are unchanged. Twenty-four tests passed, including real EMT residual,
Jacobian, and short-trajectory checks across all four backends.

IEEE118 now builds in **14.137 seconds**, compared with
previous whole-case timeouts. The run reaches SSA but fails trajectory capture
(failed, nonfinite, or incomplete history) and exits after
**28.561 seconds**. It returns no modes; this is an exposed
numerical initialization/capture issue, not a successful stability analysis.
Its explicit initialization residual is approximately
14.607.

Evidence: [optimization summary](upstream_comparison/summary.json),
[optimized IEEE39](upstream_comparison/downstream_optimized/ieee39_emt.json),
[before/after agreement](upstream_comparison/downstream_optimized/agreement.json),
[optimized IEEE118](upstream_comparison/downstream_optimized_ieee118/ieee118_emt.json),
[profile summary](upstream_comparison/profile_summary.txt),
[24-test output](upstream_comparison/regression_tests.log).

### What happened on upstream?

Pinned upstream: [SanPen/VeraGrid `37e7d867`, version 6.8.5](https://github.com/SanPen/VeraGrid/commit/37e7d867ba8241b03c5e0208cbec606ca3b90534).
A source snapshot was downloaded into `/tmp/veragrid-upstream-37e7d867`;
neither the upstream snapshot nor the sibling user checkout was patched.
The upstream and optimized downstream trials load the same saved IEEE39 grid
and downstream-defined multilinear device blocks, rather than different model
defaults. Input SHA-256:
`91297e9c35bfdb954e55dbf664f3ed58f7aef3be3f8d5679d65f1505e1cc569c`.

Upstream did **not** complete within the **180-second whole-process limit**
(measured parent wall time 180.069 s). Its last saved
phase was **DAE construction**, so there is no measured completed upstream SSA
time or mode set. This is a lower-bound/timeout observation, not proof of an
intrinsic downstream-versus-upstream speed ratio. No upstream IEEE118 timing
was measured in this experiment.

A strict public-API comparison is also unavailable at this revision:

- Upstream SSA options lack `consistent_dae_tangent`. The native trial explicitly
  disables that downstream feature.
- Upstream SSA accepts one `pf_results` argument, not the downstream separate
  balanced and three-phase inputs. The native driver received the balanced
  result after both power flows were checked by the harness.
- Upstream's public SSA constructor directly creates `EmtProblemDae`, ignoring
  the requested multilinear problem selection. Its native Floquet path therefore
  does not match the downstream `EmtProblemMultilinear` S/Phi and consistent-tangent
  path. The saved devices still have multilinear equations.

Two preliminary harness attempts stopped on those unsupported API arguments;
they are saved separately in `upstream` and `upstream_native`, and are **excluded
from timing comparisons**. The actual compatible native attempt is
[recorded here](upstream_comparison/upstream_native_compatible/ieee39_emt.json),
with its [log](upstream_comparison/upstream_native_compatible/ieee39_emt.log).
Upstream contains the same eager initialization JIT and repeated-vector code,
but no upstream speedup was inferred from source inspection alone.

To repeat using a checkout of that pinned upstream revision:

```bash
python trunk/benchmarks/multilinear_ssa.py --grid ieee39 --formulation emt \
  --engine-root /path/to/upstream --reuse-models --no-consistent-tangent \
  --grid-file trunk/benchmarks/upstream_comparison/ieee39_multilinear.veragrid \
  --steps-per-period 120 --assessment-periods 1 --modes 6 --output /tmp/upstream-result
```

The single-case command itself has no wall-clock cap; the recorded experiment
used a parent subprocess timeout of 180 seconds. `--all --timeout` enforces
per-case limits for the ordinary batch runner.
