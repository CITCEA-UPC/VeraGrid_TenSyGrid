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

For a comparison using the full nonlinear GFOR/GFOL controls copied into
`TensyGridEngine.emt.stamp_converters`, run:

```bash
MPLCONFIGDIR=/tmp/mpl-stamp-demo VERAGRID_EMT_INIT_CACHE_DIR=/tmp/veragrid-emt-init-stamp NUMBA_CACHE_DIR=/tmp/numba-stamp PYTHONPATH=src python3 -m TensyGridEngine.emt.emt_stamp_demo.ieee9_full_nonlinear_comparison
```

This runs both full nonlinear and linear-deviation converter models on the
same STAMP IEEE-9 network and generator, then writes two `.npz` spectra and
`ieee9_stamp_full_nonlinear_vs_linear_deviation.png` in this directory.
The network and generator still come from `STAMP_Public`; the full nonlinear
converter blocks come from this repository.
