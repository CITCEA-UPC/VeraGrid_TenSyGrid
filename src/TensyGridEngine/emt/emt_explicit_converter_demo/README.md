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
- `two_gfl_kcl_ode.py`: two GFL ODEs on separate three-phase buses connected
  by a resistive branch. A constant 2x2 nodal conductance solve eliminates both
  buses' linear KCL equations phase by phase. Synchronized current copies keep
  the substituted model multi-affine. The time-domain run uses the ordinary
  VeraGrid `EmtModelProblem.simulate()` symbolic/trapezoidal path; construction
  of the exact `EmtProblemMultilinear` S/Phi representation is an additional
  validation and is not yet used by that time integrator.
