# RMS and EMT small-signal benchmarks

The current RMS benchmark is documented in [rms_ssa/README.md](rms_ssa/README.md).
It checks original symbolic equations against S/Phi, validates the PF equilibrium,
and solves all finite descriptor modes with algebraic elimination and full-pencil
eigenpair checks. Reproduce with `python trunk/benchmarks/verify_rms_benchmarks.py`.

The current end-to-end verification is documented in [verified/README.md](verified/README.md).
It runs **both EMT backends on both grids**, sequentially, checks their source
hashes and numerical agreement, and rejects failed power flows, initialization,
cycle closure or mode extraction:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python trunk/benchmarks/verify_emt_benchmarks.py
```

Outputs default to `results/end_to_end/`. Large NPZ captures and profiles are
regenerated and ignored by Git; compact JSON, CSV and Markdown records are kept.
The reference machine data are generic assumptions, so this validates the
implementation and those benchmark models, not standard IEEE dynamic data.

The runner covers IEEE 39-bus (New England) and IEEE 118-bus, with separate
RMS eigenvalue and EMT Floquet analyses. Run from any working directory using
an environment with the repository's usual Python dependencies:

```bash
python trunk/benchmarks/multilinear_ssa.py --all
python trunk/benchmarks/multilinear_ssa.py --grid ieee39 --formulation rms
python trunk/benchmarks/multilinear_ssa.py --grid ieee118 --formulation emt
```

`--all` runs the four combinations sequentially in isolated Python processes,
with a 1,800-second timeout per case. It returns a nonzero exit status if any
case fails or times out. Each case writes a JSON record with the failing stage
and traceback, or timing, problem size, and modal summary; successful analyses
also write modes CSV files. Batch runs write `summary.json` and save complete
stdout/stderr logs.
`--output` chooses the destination (default: `results` beside the script).

## Cases and models

- IEEE39: `Grids_and_profiles/grids/IEEE39.gridcal`.
- IEEE118: `src/tests/data/grids/RAW/IEEE 118 Bus.raw`. This is the repository's
  case, not the NREL case from the topology bug report. Use `--grid-file PATH`
  with an individual run to benchmark another revision of the network.
- RMS: rectangular phasor buses, the multilinear GENQEC phasor generator,
  constant-power load current constraints, phasor PI lines, transformers, and
  shunts. Mechanical torque and field voltage are held at their initialized
  operating-point values; no exciter, governor, or stabilizer is attached.
  `RmsProblemMultilinear` supplies the multilinear matrix linearization.
  The default `--rms-spectrum reduced` computes every finite mode by algebraic
  elimination and an unchanged reduced pencil. Algebraic infinite modes are not
  returned. `--rms-spectrum sparse` targets only the requested modes near zero,
  which can mostly be neutral lifted-state/gauge modes. The legacy `full`
  solver is more expensive and adds a 1e-7 diagonal regularization; its output
  is not the unregularized reference used by the RMS report.
- EMT: multilinear simple synchronous generators, multilinear ZIP loads in
  constant-power configuration, multilinear transformers, linear PI lines,
  and multilinear ZIP shunts in constant-impedance configuration. Torque and
  field voltage are held at their
  initialized values. `EmtProblemTypes.Multilinear` selects the public problem
  factory and its multilinear Floquet Jacobian adapter. Both balanced and
  three-phase power flow must converge. Missing machine sequence impedances
  are assigned generic values: R1/X1 = 0.003/1.8, R0/X0 = 0.003/0.15,
  R2/X2 = 0.003/0.2 pu (only when that sequence impedance is zero).
  SSA uses Arnoldi with consistent DAE
  perturbations.

The models use template-default machine parameters; these are scaling and
implementation benchmarks, not validated IEEE dynamic-data studies. RMS and
EMT use different machine realizations, so their spectra should be reported
separately. No outages, redispatch, or artificial terminal elements are added.
Generic machine
sequence assumptions are the only additions to the static case data.
The upstream connectivity-validator disablement is present in this checkout.

## Resolution and interpretation

EMT defaults to 600 steps per electrical period, one assessment period,
six requested modes and a 24-vector Krylov budget. Standard Arnoldi honors
the Krylov dimension; the hybrid-only restart option does not control ARPACK. Adjust with `--steps-per-period`,
`--assessment-periods`, and `--modes`. A quick setup check is:

```bash
python trunk/benchmarks/multilinear_ssa.py --all --timeout 300 \
  --steps-per-period 120 --assessment-periods 1 --modes 6
```

That coarse run is not a convergence study. Repeat with finer time steps and
longer assessment before interpreting Floquet results as physical stability.
The earlier RMS rejection was traced to S/Phi dropping voltage-division terms
introduced by generator power-to-current conversion, not to an invalid PF
operating point. The multilinear backend now uses the generator's existing
current ports. Independent checks compare the original equations with S/Phi at
the initial state and two perturbations. RMS still stops if representation or
equilibrium validation fails; `--allow-nonequilibrium` is diagnostic only.

RMS also records the steady-state residual and `equilibrium_validated`;
`--rms-init consistent` requests the engine's Newton initialization instead of
the default explicit seed. EMT records the initialization report and scaled
last-period closure error. `periodicity_validated` reports whether the measured cycle closure meets the
declared tolerance (`--periodicity-tolerance`, default 1e-4). The benchmark
stops before SSA on a failed check unless `--allow-nonperiodic` is explicitly
requested for diagnostics. This tolerance check does not certify an exact orbit.

Initialization errors, singular matrices, and missing SSA results are recorded
as failures. A completed numerical analysis alone does not prove a valid
steady state or periodic orbit. EMT captures its last-period trajectory for
further periodicity checks. RMS CSVs can include infinite descriptor modes;
the JSON summary separately counts finite modes.

BLAS/OpenMP default to one thread per process for reproducible timings; explicit
thread-count environment variables take precedence. JSON records include the
input SHA-256, Python/NumPy versions, random seed, and run settings.

## Timing results

See [RESULTS.md](RESULTS.md) for measured timings, numerical caveats, and the
StructuralAD solver-path investigation. New EMT runs record PF and DAE build
times separately, the engine build breakdown, trajectory/Jacobian setup time,
and remaining Floquet work. Phase checkpoints survive batch timeouts.

## Compare EMT capture solvers

SSA now honors `EmtOptions.solver_type` through the public solver factory.
`--emt-solver` selects StructuralAD (the benchmark default), StructuralCompiled,
Automatic, or Symbolic. With `--emt-problem Multilinear`, the Floquet Jacobian uses the S/Phi
adapter. `--emt-problem CurrentBalance` instead uses the native StructuralAD
Jacobian on identical device equations. This is different from selecting
`--emt-models canonical`, which changes the device representations and must
pass its own initialization and periodicity checks. Run `python trunk/benchmarks/compare_emt_solvers.py --timeout 240`
for isolated sequential trials with identical IEEE39 settings (600 steps per
period by default; coarse overrides may fail the closure check). Earlier
alternate-solver timing records used older models and are historical, not
measurements of the current corrected setup. The comparison
saves per-solver JSON/log/trajectory/mode files and `summary.json`, including
trajectory differences and matched mode differences versus StructuralAD.

## Transformer comparison

See [transformer_comparison/README.md](transformer_comparison/README.md) for
the static terminal-equivalent audit, isolated passivity/termination checks,
and matched IEEE118 captures using `--transformer-model simple_linear` or
`coupled_core`. `--capture-only` saves trajectories without mode extraction.
