# Observed setup runs — 2026-11-23

Python 3.12.3 / NumPy 2.3.0, using `/home/pablo/myenv/bin/python` and this
working checkout. These are exploratory setup checks, with template-default
machine parameters, six requested modes, 120 EMT steps per period, and one
assessment period. Some cases were rerun individually as diagnostics were added;
wall-clock timings should not be compared as controlled performance measurements.

| Grid | RMS | EMT |
| --- | --- | --- |
| IEEE39 | Six finite modes returned | Six finite Floquet modes returned |
| IEEE118 | Six finite modes returned | Timed out after 300 seconds before returning SSA results |

The RMS explicit seeds have steady-state infinity-norm residuals of approximately
8.86 (IEEE39) and 5.94 (IEEE118), so `equilibrium_validated` is false. IEEE39 EMT
reports an explicit initialization residual of approximately 23.59 and a scaled
last-period closure error of approximately 1.94; periodic steady state is not
verified. **These numerical modes are not validated physical stability results.**

`summary.json` records the final outcomes. The per-case JSONs contain settings,
modal counts, residuals, and metadata. CSV files contain modes; the IEEE39 EMT
NPZ contains the captured trajectory. The IEEE118 EMT timeout report deliberately
contains no modal result. Its log confirms balanced PF convergence, but the
coarse log does not distinguish how much remaining time was spent in three-phase
PF versus DAE construction.

Reproduce the setup sweep with:

```bash
python trunk/benchmarks/multilinear_ssa.py --all --timeout 300 \
  --steps-per-period 120 --assessment-periods 1 --modes 6
```

For a longer IEEE118 EMT attempt, the normal batch timeout is 1,800 seconds:

```bash
python trunk/benchmarks/multilinear_ssa.py --all --timeout 1800
```

`--rms-init consistent` exposes the engine's Newton initialization as an optional
follow-up. A diagnostic attempt was interrupted; its convergence has not been
established. No excitation tuning, redispatch, or validator bypass beyond the
previously requested upstream change was introduced to obtain these outputs.
