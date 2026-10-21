# Saved IEEE9 EMT examples

The 60 Hz IEEE9 examples now include two model-bearing grids:
`Grids_and_profiles/grids/IEEE 9 Bus EMT.veragrid` has the original machine,
and `IEEE 9 Bus EMT Multilinear.veragrid` has the lifted machine. Both contain
the generator, network, load, and controller EMT models, including governor
`K = 40`; the runner uses those saved models directly. From the repository
root, run the original model directly. Set `DEFAULT_MULTILINEAR_MACHINE = True`
near the top of `emt_ieee9_small_signal.py` to use the saved lifted model with
the same command:

```bash
PYTHONPATH=src /home/pablo/myenv/bin/python -m TensyGridEngine.emt.emt_ieee9_small_signal
```

The non-lifted result has a unit Floquet multiplier, and the lifted result is
near-neutral at 1200 steps per period under the example's `1e-4` tolerance.

Previously, `IEEE 9 Bus.gridcal` was a data-only grid: the script had to
attach EMT models after loading and set `K = 40` itself. That path still works,
and its Floquet spectra match the corresponding saved-model grids. The STAMP
WSCC example uses a different 50 Hz operating point with one SG, one GFOR,
and one GFOL. Its original and multilinear model-bearing grids are under
`src/TensyGridEngine/emt/emt_stamp_demo/saved_grids/`. With the sibling
`STAMP_Public` checkout available, run the following command to rebuild both
grids and check each saved spectrum against its in-memory counterpart:

```bash
PYTHONPATH=src /home/pablo/myenv/bin/python -m TensyGridEngine.emt.emt_stamp_demo.ieee9_stamp_converter_roundtrip
```

The computed dominant multipliers are neutral in both STAMP cases; the
round-trip comparison is exact for the selected modes.
