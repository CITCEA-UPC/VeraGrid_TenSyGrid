# Explicit GFL EMT demos

The four top-level Python scripts are the runnable examples. From the
repository root, install the dependencies with
`python -m pip install -r requirements.txt`, then run a module with
`PYTHONPATH=src python -m` plus its
module path. You can also run any of the four scripts directly by file path.
Each uses VeraGrid's native `.simulate()` method.

| Script | Simulation |
| --- | --- |
| `standalone_gfl_ode.py` | One prescribed-input GFL ODE with a current-limiter event. |
| `standalone_gfl_ode_ieee9.py` | The existing commented IEEE-9 initialized GFL ODE example, comparing prescribed voltage disturbances. The grid supplies the operating point. |
| `two_gfl_kcl_ode.py` | Two GFL converters on a two-bus resistive network; KCL is substituted into the ODE. |
| `ieee9_full_explicit.py` | The complete 103-state charge-reduced IEEE-9 ODE with dynamic π lines, a grid `EmtEvent` at 0.05 s, and a native 0.2 s simulation. |

For example:

```bash
PYTHONPATH=src python -m TensyGridEngine.emt.emt_explicit_converter_demo.ieee9_full_explicit
```

`ieee9_full_explicit.py` modifies the three zero-charging IEEE-9 lines to
`B=0.1` so the bus capacitance matrix is invertible. This is a controlled
variant of IEEE-9, not the original parameter set. Its default step is 50 µs;
the 0.2 s run produced 4,001 samples and had a pre-event bus RMS variation
below `4.62e-7` pu. The reduced ODE is multi-affine in all 103 states.

- `utils/` holds converter builders, IEEE-9 assembly, polynomial extraction,
  and the native reduced-model adapter.
- `experiments/` holds alternative reductions, rank checks, noise comparisons,
  initialization diagnostics, and runs using other solvers.
- `tensor_decompositions/` holds CP and tensor-train studies.
- `simulation_data/<demo>/` holds each demo's plot and MATLAB-readable
  `S.mat`, `Phi.mat`, `inputs.mat`, and `states.mat`. The latter two contain
  aligned `time` rows and name arrays. The IEEE-9 event demo also saves
  `S_after.mat`, `Phi_after.mat`, and `timeseries.npz`.

The standalone IEEE-9 demo stores each voltage-disturbance variant in its
own folder under `simulation_data/standalone_gfl_ode_ieee9/`.

The native EMT integrator evaluates symbolic equations. Extracting the
multilinear S/Phi matrices does not switch it to a CP tensor integrator.
