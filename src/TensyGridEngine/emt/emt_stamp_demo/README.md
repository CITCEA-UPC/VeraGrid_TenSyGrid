# STAMP EMT demos

IEEE-9/WSCC experiments based on the public STAMP SG, GFOR, and GFOL models.
The scripts discover a sibling `STAMP_Public` checkout automatically. For any
other location, set:

```bash
export STAMP_PUBLIC_ROOT=/absolute/path/to/STAMP_Public
```

Run from the VeraGrid repository root with `PYTHONPATH=src`, for example:

```bash
PYTHONPATH=src python -m TensyGridEngine.emt.emt_stamp_demo.ieee9_stamp_converter_stability --no-event
```

`stamp_multilinear_converters.py` and `stamp_multilinear_generator.py` contain
the exact multi-affine lifts used by the SSA scripts.
