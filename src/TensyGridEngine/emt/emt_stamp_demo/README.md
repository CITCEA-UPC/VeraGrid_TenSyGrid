# STAMP EMT demos

Run the examples from the repository root with `myenv` activated.

The IEEE9 saved-grid workflow, including the separate 60 Hz generator examples,
is described in [Saved IEEE9 EMT examples](../README_saved_grid_examples.md).

IEEE-9/WSCC experiments based on the public STAMP SG, GFOR, and GFOL models.
The scripts discover a sibling `STAMP_Public` checkout automatically. For any
other location, set:

```bash
export STAMP_PUBLIC_ROOT=/absolute/path/to/STAMP_Public
```

To save the STAMP GFOR/GFOL case and its multilinear version as model-bearing
`.veragrid` files, then verify their Floquet spectra against the in-memory
models, run:

```bash
python src/TensyGridEngine/emt/emt_stamp_demo/ieee9_stamp_converter_roundtrip.py
```

The files are written to `saved_grids/`. Both cases retain STAMP's 50 Hz
network, SG, GFOR, and GFOL parameters. The multilinear case uses the lifts
already defined in this directory.

Run from the VeraGrid repository root with `PYTHONPATH=src`, for example:

```bash
PYTHONPATH=src python -m TensyGridEngine.emt.emt_stamp_demo.ieee9_stamp_converter_stability --no-event
```

`stamp_multilinear_converters.py` and `stamp_multilinear_generator.py` contain
the exact multi-affine lifts used by the SSA scripts.

`stamp_explicit_ode_demo.py` runs the algebraic-free nonlinear GFOR and GFOL
converter ODEs from `stamp_converters/explicit_converters.py` under prescribed
three-phase voltage. It saves named input, state, and output time series in
`stamp_explicit_ode_results/gfor/time_series.mat` and
`stamp_explicit_ode_results/gfol/time_series.mat` for MATLAB. These standalone
runs do not include the IEEE-9 network or the synchronous generator.

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
