# Native VeraGrid IEEE-9 EMT demos

IEEE-9 tests that use VeraGrid's native synchronous-machine and converter EMT
models. They do not depend on the external STAMP repository.

Run from the repository root with `PYTHONPATH=src`, for example:

```bash
PYTHONPATH=src python -m TensyGridEngine.emt.emt_ieee_veragrid_demo.ieee9_time_domain_compare
```

`ieee9_common.py` contains the shared case builder used by the simulation and
small-signal comparisons.
