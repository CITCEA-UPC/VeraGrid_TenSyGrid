# Explicit converter EMT demos

Standalone converter tests and experiments for models whose electrical inputs
are prescribed explicitly rather than supplied by a complete network.

Run from the repository root with `PYTHONPATH=src`, for example:

```bash
PYTHONPATH=src python -m TensyGridEngine.emt.emt_explicit_converter_demo.gfl_ode_dynamic_limiter
```

- `single_model_time_inputs.py`: standalone GFM and connected GFL DAE inputs.
- `gfl_ode_time_inputs.py`: reusable algebraic-free GFL ODE builder.
- `gfl_dae_dynamic_transform.py`: DAE-to-dynamic multilinear lift comparison.
- `gfl_ode_dynamic_limiter.py`: explicit ODE current-limiter test.
- `ieee9_gfl_ode_noise_comparison.py`: standalone GFL initialized from an IEEE-9 operating point.
- `ieee9_gfl_ode_grid.py`: attaches the GFL ODE as an IEEE-9 generator and
  runs the assembled full-grid EMT DAE. This example does not eliminate the
  grid's algebraic variables.
- `ieee9_gfl_ode_reduced.py`: uses the IEEE-9 bus/branch connectivity with
  resistive line surrogates, an oscillator slack source and two GFL ODE
  generators. It compares the assembled DAE with the state-only problem made
  by `EmtProblemLinearReduced`. The reduction freezes initialized parameters;
  this experiment has no runtime parameter changes or topology events. Pass
  `--verify-reduced-matrices` to build the reduced ODE's S/Phi with the fast
  coefficient-map extractor.
- `two_gfl_kcl_ode.py`: two GFL ODEs on separate three-phase buses connected
  by a resistive branch. A constant 2x2 nodal conductance solve eliminates both
  buses' linear KCL equations phase by phase. Synchronized current copies keep
  the substituted model multi-affine. The time-domain run uses the ordinary
  VeraGrid `EmtModelProblem.simulate()` symbolic/trapezoidal path; construction
  of the exact `EmtProblemMultilinear` S/Phi representation is an additional
  validation and is not yet used by that time integrator.
- `tensor_decompositions/`: IEEE-9 nodal inverse, binary-mode CP, grouped CP,
  and tensor-train experiments. See its [README](tensor_decompositions/README.md)
  for entry points, results, and the distinction between their tensorizations.
