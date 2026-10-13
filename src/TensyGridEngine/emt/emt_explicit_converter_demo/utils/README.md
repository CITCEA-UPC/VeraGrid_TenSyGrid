# Building blocks

- `gfl_ode_time_inputs.py`: construct a standalone converter with prescribed inputs.
- `ieee9_gfl_ode_grid.py`: attach a GFL ODE generator to an assembled IEEE-9 grid.
- `ieee9_gfl_ode_reduced.py`: assemble the two-GFL IEEE-9 variant and its EMT templates.
- `ieee9_pi_charge_polynomial.py`: derive the complete 103-state polynomial after charge substitution.
- `ieee9_pi_charge_native.py`: wrap that polynomial in an `EmtModelProblem` and initialize its derivatives for native `.simulate()`.
- `simulation_data.py`: write named MATLAB traces and S/Phi matrices in a consistent layout.

These modules are imported by the top-level demos and the experiments.
