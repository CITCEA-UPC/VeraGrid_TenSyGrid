# Supporting experiments

These scripts investigate alternatives to the four top-level native
`.simulate()` demos. They include direct DAE probes, a SciPy time series,
charge-reduction and initialization checks, and converter comparisons.

Run a script as a module from the repository root, for example:

```bash
PYTHONPATH=src python -m TensyGridEngine.emt.emt_explicit_converter_demo.experiments.ieee9_pi_charge_init_check
```

The older numerical findings are retained in `README_history.md`. Generated
plots and arrays live in `../simulation_data/experiments/`.
