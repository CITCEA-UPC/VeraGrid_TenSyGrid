# Exercise: a factored governor in VeraGrid

Declare the governor's number of states and a CP tensor of a chosen rank, build
a native VeraGrid `Block` from those factors, connect it to GENCLS, and study the circuit using
`RmsProblemDae`. The runnable solution is `cp_governor.py`.

The circuit, power flow, symbolic variables, block connections, initialization,
load event, DAE integration, and small-signal analysis all use VeraGrid. NumPy
holds the factor arrays and exports/checks results; Matplotlib draws plots.
`SymbolicCPTensor` is an authoring representation whose coefficients can contain
VeraGrid symbolic parameters. Its factors generate ordinary symbolic equations;
the running model uses `RmsProblemDae`, with no custom CP numerical runtime or
call to `RmsProblemMultilinear`.

## Define dimensions, tensor, then block

The three numbered steps in `governor()` are the exercise's construction sequence:

1. Set `n_states = 2` and `cp_rank = 8`; declare the states and input/output ports.
2. Create `SymbolicCPTensor.empty(n_variables=3, n_equations=n_states+1, rank=cp_rank)`
   and populate its weights and factor matrices.
3. Pass these definitions to `block_from_cp(...)`, which creates the state and
   algebraic equations and registers native ports, parameters and initialization.

The tensor has shape `(3, 2, 2, 2)`. Its first mode selects the three equations
`[dv/dt, dtau/dt, Tm]`; the remaining modes use `[1, v]`, `[1, tau]`, and
`[1, omega]`. Every variable factor is `2 x rank` and the output factor is
`3 x rank`. Contraction generates

\[
h_j(z)=\sum_{r=1}^R C_{jr}\lambda_r
        \prod_i\left(F_i[0,r]+F_i[1,r]z_i\right).
\]

The first four weighted components are `Tref`, `v`,
`(omega-1)*(1+alpha*(v-Tref))`, and `tau`. Here is the actual tensor definition:

```python
tensor.weights[:4] = [tref, 1, 1, 1]
tensor.factors[0][:, :4] = [[1, 0, 1-alpha*tref, 1], [0, 1, alpha, 0]] # v
tensor.factors[1][:, :4] = [[1, 1, 1, 0], [0, 0, 0, 1]]              # tau
tensor.factors[2][:, :4] = [[1, 1, -1, 1], [0, 0, 1, 0]]             # omega
tensor.output_factor[:, :4] = [
    [1/Tg, -1/Tg, -K/Tg, 0],
    [0, 1/Tt, 0, -1/Tt],
    [0, 0, 0, 1],
]
```

`Tref`, `Tg`, `Tt`, `K`, and `alpha` are runtime parameters, so the tensor is
parameterized by them. They are not additional variable modes. `Te` is only an
initialization input and therefore also has no tensor mode. The builder rejects
factor coefficients that depend on a variable mode, which would invalidate the
claimed multilinear structure.

The first `n_states` output rows become `block.state_eqs`. Each remaining row
becomes `output_variable - expression = 0` in `block.algebraic_eqs`.
The builder also supports other state counts and ranks; the tests include a
one-state, rank-two controller. Changing the governor's dimensions requires
updating the states, factors and output rows consistently. Eight is the chosen number of stored CP components, not a claim of minimum
tensor rank. The first four reproduce the baseline; the remaining four add
the interactions described below. Zero interaction gains recover the baseline.

## Controller and circuit

```text
                     Native VeraGrid generator device
        +------------------------------------------------------+
        |   factored governor             native GENCLS         |
        |   states: valve, turbine        states: delta, omega  |
        |               Tm -------------------------------->   |
        |                  <--------------------------- omega   |
        |   initial Tref   <------------------------------ Te   |
        +------------------------------------------------------+
                                         |
                                  generator bus
                                         |
                                   native RMS line
                                         |
                                     load bus
                                         |
                                   native RMS load
```

`governor()` declares these two differential equations and one output equation:

\[
T_g\dot v=T_{ref}-v-K(\omega-1)[1+\alpha(v-T_{ref})]+I(e,a,b),\qquad
T_t\dot\tau=v-\tau,\qquad T_m=\tau.
\]

With the extra interaction gains set to zero, the CP construction produces equations equivalent to:

```python
speed_error = omega - 1
scheduled_gain = 1 + alpha * (v - tref)
block.state_eqs = [
    (tref - v - K * speed_error * scheduled_gain) / Tg,
    (v - tau) / Tt,
]
block.algebraic_eqs = [tm - tau]
```

The affine product is left intact. `alpha=0` gives ordinary droop with two lags;
`alpha=1` adds an illustrative valve-dependent gain. The latter is a teaching
control law, not a named standard governor. At fixed parameters each state/input
occurs at most linearly in each product, which is suitable for a future CP
conversion.

GENCLS expects **mechanical torque**, so `tau` and `Tm` are torque in pu, and
`v` is a valve command scaled to torque. Actual shaft power is `Tm * omega`.
Absolute shaft speed is in pu; `50 * omega` is frequency in Hz. Both device and
system bases are 100 MVA. The native machine includes rotor angle, speed, stator
algebraic equations, and a fixed internal excitation initialized from power flow.

The governor's electrical-torque input `Te` is used only to initialize the held
runtime parameter `Tref` and its states. It is not a continuously tracking
reference. Thus the initial torque includes actual feeder losses; it is not
hardcoded to the load's 0.8 pu. The three connections use `VarFactory.add_connection`:

```python
vf.add_connection(machine.in_vars[2], controller.out_vars[0])  # Tm
vf.add_connection(controller.in_vars[0], machine.out_vars[2]) # omega
vf.add_connection(controller.in_vars[1], machine.out_vars[3]) # initial Te
```

The machine-only `get_gencls_rms_template(..., mechanical_input=TGOV1)` factory
exposes its torque port. Despite the enum name, that factory does not attach a
TGOV1 controller; this exercise supplies the custom governor at that port.

The initial load is 80 MW + 10 Mvar. A VeraGrid `RmsEvent` raises active demand by
5 MW at 2 s. RMS load parameters use negative injection, so `Pl0` changes from
`-0.8` to `-0.85` pu. The event receives the actual parameter from the load
block's `event_dict`.

## Run

From the repository root, using a Python environment with this checkout's
VeraGrid dependencies installed:

```bash
python 'src/TensyGridEngine/controller desing/cp_governor.py'
python 'src/TensyGridEngine/controller desing/cp_governor.py' --alpha 0 --output /tmp/governor_linear
python -m unittest discover -s 'src/TensyGridEngine/controller desing' -p 'test_*.py'
```

The script inserts this checkout's `src` directory into the import path. It has
been run here with `/home/pablo/Desktop/eroots/VeraGrid/.venv/bin/python`; the
system Python lacks some VeraGrid dependencies.

Options include `--step` (pu on 100 MVA), `--duration` (seconds), `--dt` (seconds),
and `--alpha`. Change `Tg`, `Tt`, `K`, `H`, and `D` in `Parameters` for further
experiments.

`RmsSimulationDriver` selects `RmsProblemDae` through
`RmsProblemTypes.PowerBalance`, and uses VeraGrid's implicit backward Euler
solver. `SmallSignalStabilityRmsDriver` calculates all modes at the initialized
operating point (`ss_assessment_time=0`). No external integration or eigenvalue
calculation replaces those drivers.

The `results` directory contains:

- `response.png` and `response.csv`: frequency, valve, torque, power and voltage.
- `study.txt`: convergence, initial residual, eigenvalues and response measures.
- `eigenvalues.csv` and `state_matrix.csv`: native small-signal results. The
  matrix is exported as supplied by VeraGrid; consult its state ordering and
  scaling before interpreting entries.
- `governor_equations.txt`: the connected native symbolic state equations.

A near-zero eigenvalue represents the island's arbitrary common angle. Assess
frequency-control damping using the other modes. The power-flow slack bus is an
initial voltage-angle reference, not an infinite bus fixing dynamic frequency.
After the load step, primary droop leaves a speed offset and the absolute rotor
angle drifts. The reported final frequency is a time-simulation sample, not a
separately solved post-event equilibrium.

## Participant tasks

1. Choose the number of states and CP rank. Declare variables with `VarFactory`,
   fill the factor matrices, and use `block_from_cp` to generate the governor.
   Recover the written control equations from the CP columns.
2. Connect the three native ports. Check from the power-flow initialization that
   `v = tau = Tref = Te` and `omega = 1`, with zero DAE residual.
3. Run the ordinary droop case (`alpha=0`). Explain the initial frequency decrease,
   the turbine delay, and the remaining steady frequency offset.
4. Run `alpha=1`. Identify the single rank-one affine product that a future tensor
   compiler could retain, and compare the transient with the linear-governor case.
5. Inspect the native eigenvalues. Vary droop gain and time constants, then relate
   the mode damping to the time response. Both alpha cases share the initial
   Jacobian because the scheduling factor is centered at `Tref`.
6. Reduce the integration time step and compare the response to assess backward
   Euler's numerical damping. Increase the load step to explore the limitations
   of a controller without valve saturation, deadband, or rate limits.

The tests check CP conversion at different dimensions and ranks, reject invalid
factors, and check the symbolic equations and speed derivative, the native port
connections, initial DAE residual, actual load change, frequency response, mode
structure, and preservation of equilibrium without a disturbance. This example
uses a classical machine with fixed excitation, so voltage control and detailed
machine dynamics are outside its scope.


## Tensor interaction controller and peak-response tuning

The two states and physical actuator/turbine lag constants stay unchanged.
Define normalized coordinates

\[
e=(1-\omega)/0.005,\qquad a=(v-T_{ref})/0.1,\qquad
b=(\tau-T_{ref})/0.1.
\]

Four additional CP columns contribute the torque-command correction

\[
I(e,a,b)=c_{ev}ea+c_{et}eb+c_{evt}eab+c_{vt}ab.
\]

Each feature is a product of one affine factor per participating variable mode.
The three-way product `e*a*b` directly couples speed, valve and turbine state.
This is nonlinear state feedback, with no integrator or additional states.
All added features and their first derivatives vanish at the initial operating
point. Thus the equilibrium and initial small-signal modes match the baseline;
the differences are in the finite-disturbance response. The `e*a` column also
adjusts the interaction already present in the baseline scheduling factor.
Stored CP component count need not equal minimal rank or independent term count.

`tune_tensor_governor.py` searches the four coefficients using SciPy's bounded,
seeded differential evolution. Every objective evaluation uses the actual native
VeraGrid power flow, GENCLS circuit, RMS DAE simulation and small-signal driver.
The optimization target is the sampled peak absolute frequency deviation from
50 Hz after the +5 MW load step, over a 30-second run. It is a bounded numerical
search, not proof of a global minimum.

The search fixes `K`, `Tg`, `Tt`, `H`, `D` and `alpha`. Each interaction coefficient
has a bounded search interval: `c_ev,c_et` in [-0.12,0.12], `c_evt` in
[-0.06,0.06], and `c_vt` in [-0.015,0.015] pu of torque command. Candidate trajectories must keep
valve and torque in [0, 1.1] pu, command in [0, 1.2] pu, absolute valve rate below
0.5 pu/s, final absolute frequency error below `0.15 * abs(step_pu)/0.05` Hz, and non-neutral initial
modes to the left of -0.05/s. The peak-to-peak frequency excursion in the last
five seconds must be below 0.01 Hz, excluding persistent oscillatory responses.
Search runs use a tighter 0.4 pu/s valve-rate margin. A -5 MW load step must
also keep its peak frequency deviation below 0.25 Hz. These are illustrative exercise limits; they are
trajectory checks, not saturation or anti-windup dynamics. Failed simulations
are rejected. Equal baseline limits are reported in the comparison.

```bash
python 'src/TensyGridEngine/controller desing/tune_tensor_governor.py'
python 'src/TensyGridEngine/controller desing/cp_governor.py' \
  --parameters 'src/TensyGridEngine/controller desing/tensor_tuning_final/parameters.json' \
  --output /tmp/tensor_governor
```

The search uses a 0.01 s time step, a seeded initial population with several small
interaction candidates, and adds -5 MW and +7.5 MW disturbances as
feasibility constraints. It then verifies the baseline and selected candidate
at 0.005 s for 30 s on all three scenarios and an additional +2.5 MW holdout. The saved comparison records whether the limits hold in each case;
a good result for +5 MW alone does not establish robustness for arbitrary events.

Artifacts in `tensor_tuning_final`: selected `parameters.json`, full
`search_history.json`, reproducible settings in `search_config.json`, scenario metrics in `comparison.json`, baseline/tuned
trajectories, `comparison.png`, and the acceptance summary `verification.json`. To repeat verification without searching:

```bash
python 'src/TensyGridEngine/controller desing/tune_tensor_governor.py' \
  --parameters 'src/TensyGridEngine/controller desing/tensor_tuning_final/parameters.json'
```

## Saved tuning result

Selected gains: `c_ev=0.08`, `c_et=-0.08`, `c_evt=-0.01`, `c_vt=-0.001`.

Fine-step results (0.005 s, 30 s):

| Load change | Baseline peak [Hz] | Tensor peak [Hz] |
| --- | ---: | ---: |
| +5 MW | 0.215986 | 0.198508 |
| +2.5 MW | 0.108874 | 0.104178 |
| -5 MW | 0.223392 | 0.246310 |
| +7.5 MW | 0.321418 | 0.284832 |

The target peak is 8.09% smaller. All four tuned trajectories pass the stated limits. The load-decrease peak is larger than the baseline, and the tuned controller uses more valve movement; this is a target-specific improvement, not uniform improvement for every disturbance.

The saved final search evaluated 40 candidates (20 initial points plus one differential-evolution generation). Its winner was a seeded candidate; the mutation step did not improve it. The seeds were informed by the exploratory runs. This is a feasible demonstration, not a claim of optimality. Reproduce this run with `--iterations 1`.

Earlier `tensor_tuning`, `tensor_tuning_robust`, and `tensor_tuning_verified` folders contain rejected/superseded trials. The accepted result is in `tensor_tuning_final`.

## Full two-state tensor controller

`two_state_tensor.py` removes the valve/turbine interpretation from the controller
states. Both state derivatives and the mechanical-torque output are generated
from the same CP tensor, using the existing `block_from_cp` builder. The native
GENCLS, network, event, initialization, small-signal analysis and `RmsProblemDae`
simulation remain the same.

Define normalized error `e = (1 - omega)/0.005` and independent internal states
`x1, x2`, initially zero. The controller is

```
features = [x1, x2, e, x1*x2, x1*e, x2*e, x1*x2*e]
[dx1/dt, dx2/dt, Tm - Tref] = C @ features
```

All 21 entries of `C` are tuning variables. Each feature is a rank-one product
of affine factors in `[x1, x2, omega]`. Seven components cover the complete
nonconstant multilinear basis; an eighth supplies the held equilibrium torque.
This stored component count is not a claim of minimal CP rank. The factors are
fixed to this complete basis, so tuning `C` already spans every multilinear map
in these three variables; it is not necessary to also optimize redundant factor
parameters. Every feature vanishes at equilibrium, preserving initialization.

This is an abstract dynamic torque controller. It commands the generator directly;
it does not include the earlier physical valve/turbine lags. Consequently an
improvement against that governor does not isolate a benefit from tensor storage
or nonlinear terms. The linear starting controller is included in the search
history as candidate 1. No actuator saturation is implemented.

Search: deterministic differential evolution, seed 23, population 24, two
iterations by default, no polishing. Bounds and initialization are recorded in
`two_state_results/search_config.json`. Target: +5 MW peak frequency deviation.
Constraint scenarios: -5 MW and +7.5 MW. Checks: torque in [0, 1.1] pu, sampled
torque rate <= 0.4 pu/s, internal state magnitudes <= 3, nonneutral initial modes
real part <= -0.05/s, last-five-second frequency variation <= 0.01 Hz, and final
error <= 0.15 Hz scaled by disturbance magnitude. Negative-step peak <= 0.25 Hz.
These are trajectory acceptance criteria, not global stability guarantees.
Search uses 0.01 s steps for 30 s; verification uses 0.005 s, also including a
+2.5 MW holdout. Output `controller.json` contains the actual matrix and ordering;
`comparison.json` and `comparison.png` show verified responses.

Run using the same Python environment as the other exercises:

```bash
python two_state_tensor.py --iterations 2
```

### First full-controller result

The 72-candidate run found a controller passing all four verification scenarios.
At +5 MW, peak deviations were 0.21599 Hz (baseline), 0.19851 Hz (previous tensor
controller), and 0.07923 Hz (full two-state controller). The full controller's
-5 MW, +2.5 MW and +7.5 MW peaks were 0.07866, 0.03957 and 0.11893 Hz.

The ablation is important: the linear seed gives 0.15409 Hz, while removing the
selected controller's nonlinear columns gives 0.07910 Hz, marginally better than
its full 0.07923 Hz response. Thus this experiment demonstrates tuning of the
entire two-state tensor controller, but does **not** demonstrate an advantage
from the nonlinear interactions on the target disturbance. The changed linear
dynamics and direct torque control account for the observed benefit. Ablation
results are target-only, not a four-scenario verification of the linear variant.
Exact equations are saved in `two_state_results/equations.txt`.

Generated testing and tuning output directories have been removed. The results above describe the recorded experiment; rerun the scripts to regenerate output files. The previous-controller comparison coefficients are retained in the source so it does not depend on deleted output files.
