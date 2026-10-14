# Experiment notes (historical file locations)

The script names below describe earlier experiments. The four current entry
points and their locations are listed in the parent `README.md`; supporting
files now live in `utils/`, `experiments/`, and `simulation_data/`.

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
- `ieee9_gfl_ode_pi_explicit_probe.py`: retains the dynamic EMT π-line
  templates. Its 130-state, 156-algebraic-variable DAE simulates, but the
  direct algebraic matrix has rank 120. Differentiating hidden state
  constraints raises the algebraic rank to 150 and then 156. A local numerical
  explicit vector field on 88 independent state coordinates is compared with
  the DAE over a 100 μs interval. With a 2.5 μs DAE step, the maximum state
  difference was `1.19e-5` (about `8.07e-6` of the largest state magnitude);
  at a 25 μs step it was `7.48e-5`. The DAE's first reported point is not on
  the algebraic constraint manifold, so comparison starts after its first
  accepted step. This is a higher-index, state-dependent reduction; it is
  not yet a symbolic multilinear ODE or a drop-in simulation replacement.
- `ieee9_gfl_ode_pi_charge_ode.py`: controlled test of substituting each
  terminal charge `q=C*v` and using bus voltages as ODE states. It replaces
  the three zero line-charging `B` values by `0.1`, so this is not the
  original IEEE-9 parameter set. The original aggregate 27-by-27 bus
  capacitance has rank 18; the modified one has rank 27. The 54
  terminal-charge states become 27 bus-voltage states,
  giving 103 explicit states; the constant remaining current/voltage-derivative
  solve has rank 156 of 156. Over 100 μs, its max trajectory difference from
  the DAE was `8.31e-5` at a 2.5 μs DAE step, decreasing from `1.43e-3` at
  25 μs. This route substitutes charge dynamics directly and does not
  differentiate KCL. The full vector field is checked for multi-affinity by
  `ieee9_pi_charge_polynomial.py`.
- `ieee9_pi_charge_polynomial.py`: expands the complete 103-state explicit
  ODE for that **modified-capacitance** grid. All 1,946 retained coefficient
  entries have distinct state indices within each monomial: the vector field
  is multi-affine. Its 420 monomial supports, of degree at most five, give an
  exact binary-mode CP representation with at most 420 terms. Evaluation at
  20 sampled states agrees with the direct charge-reduced vector field to
  `4.26e-16` relative error. This is an upper bound on CP rank, not a
  minimal-rank decomposition; the original zero-charging grid is not covered.
- `ieee9_pi_charge_event_timeseries.py`: 0.2 s time series on the same
  modified-capacitance grid. The first GFL generator's active-power reference
  steps up 10% at 0.05 s. Converter currents, control integrators, and PLL
  phase are initialized from the power-flow operating point; leaving currents
  at the template's zero default caused an immediate voltage transient. The
  converter current is connected as positive generator injection, consistent
  with its filter law `L*di/dt = v_converter - v_bus - R*i`. The previous
  reversed connection produced two fast growing zero-sequence modes. The
  corrected 0.2 s run integrates all 103 ODE coordinates without projection;
  the maximum observed zero-sequence mean was `8.55e-11`. The PNG plots bus-voltage RMS,
  converter-current change, measured and commanded power, and phase-A
  voltage; the NPZ stores the full state series. Generator 1 measured power
  rose from 1.629 to 1.650 pu by 0.2 s, below its new 1.793 pu reference.
- `ieee9_pi_charge_native.py`: wraps the same 103-state polynomial in a
  symbolic `Block` and `EmtModelProblem`, following `two_gfl_kcl_ode.py`.
  Its differential initialization uses the polynomial RHS evaluated at the
  initial state. Seeding all derivatives as zero caused a first-step jump in
  the native trapezoidal solver, even after the physical state was corrected.
  Native `.simulate()` runs a 100 μs check with zero algebraic variables and
  validates `S=(206, 420)`, `Phi=(103, 420)`. Against a tight direct ODE
  solve, maximum state error falls from `8.72e-4` at a 2.5 μs native step to
  `3.49e-4` at a 1 μs step.
- `ieee9_pi_charge_native_event.py`: adds an `EmtEvent` to the IEEE-9
  `MultiCircuit` and maps it to the reduced model's runtime power-reference
  parameter. The raw power-flow seed left the slack Norton's internal
  oscillator voltage equal to the bus voltage, implying zero injected current
  despite the nonzero power-flow slack injection. The source initialization
  now includes the current divided by Norton conductance in both oscillator
  quadratures. A direct-start 10 ms high-accuracy ODE check stays within
  `4.1e-7` pu of the initial bus RMS values. The event script starts directly
  from this initialization without a DAE seed step. A native `.simulate()`
  run at a 50 μs step completed 0.2 s with 4,001 samples and the 10% GFL
  reference event at 0.05 s. Before the event the monitored bus RMS values
  varied by at most `4.62e-7` pu; generator 1 measured power rose from
  1.630 pu to 1.651 pu by 0.2 s. The PNG and NPZ contain this corrected run.
- `ieee9_pi_charge_init_check.py`: plots the 10 ms startup at native 100 μs
  and 25 μs steps against a high-accuracy BDF reference. With both oscillator
  quadratures and the native initial derivatives seeded correctly, all three
  traces remain within about `5e-7` pu of their initial bus RMS values.
- `ieee9_zero_charging_rl_rank.py`: checks the exact R-L simplification of
  the original three zero-charging pi lines. Eliminating their 18 zero-charge
  states and 18 zero capacitor-current algebraics leaves a 138-by-138 direct
  solve of rank 132. The six remaining constraints therefore do not come
  from those redundant zero-capacitance variables. Series-current dynamics
  are retained; no resistive surrogate is introduced.
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
