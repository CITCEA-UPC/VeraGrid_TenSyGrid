# Saved IEEE9 EMT examples

The 60 Hz IEEE9 examples now include two model-bearing grids:
`Grids_and_profiles/grids/IEEE 9 Bus EMT.veragrid` has the original machine,
and `IEEE 9 Bus EMT Multilinear.veragrid` has the multilinear machine. Both contain
the generator, network, load, and controller EMT models, including governor
`K = 40`; the runner uses those saved models directly. From the repository
root with the project dependencies installed, run the original model directly.
Set `DEFAULT_MULTILINEAR_MACHINE = True` near the top of
`emt_ieee9_small_signal.py` to use the saved multilinear model with
the same command:

```bash
python src/TensyGridEngine/emt/emt_ieee9_small_signal.py
```

The original result has a unit Floquet multiplier, and the multilinear result is
near-neutral at 1200 steps per period under the example's `1e-4` tolerance.

Previously, `IEEE 9 Bus.gridcal` was a data-only grid: the script had to
attach EMT models after loading and set `K = 40` itself. That path still works,
and its Floquet spectra match the corresponding saved-model grids. The STAMP
WSCC example uses a different 50 Hz operating point with one SG, one GFOR,
and one GFOL. Its original and multilinear model-bearing grids are under
`src/TensyGridEngine/emt/emt_stamp_demo/saved_grids/`. The editable STAMP
network and model sources are in `src/veragrid_stamp/`. Run the following
command to rebuild both grids from those sources and compare each spectrum
before and after saving and reloading. No separate checkout is needed:

```bash
python src/TensyGridEngine/emt/emt_stamp_demo/ieee9_stamp_converter_roundtrip.py
```

The computed dominant multipliers are neutral in both STAMP cases; the
round-trip comparison is exact for the selected modes.
