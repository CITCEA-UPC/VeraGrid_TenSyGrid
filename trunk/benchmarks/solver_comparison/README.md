# EMT solver comparison

See [RESULTS.md](../RESULTS.md#measured-solver-comparison) for timings and limitations.
`summary.json` contains the final bounded trials. Per-solver subdirectories
contain JSON and logs; StructuralAD also has modes and a trajectory. The
interrupted Automatic attempt is separate from its final timed-out trial.
Only StructuralAD returned modes, so cross-backend numerical agreement is
unestablished. Tests: 17 passed (`regression_tests.log`).

To retry an individual backend with a longer budget:

```bash
python trunk/benchmarks/compare_emt_solvers.py --solvers StructuralCompiled --timeout 600
```

That command replaces the selected backend's files and summary row, retaining
other backend outcomes in the output directory. Use `--output PATH` for a
separate experiment.
