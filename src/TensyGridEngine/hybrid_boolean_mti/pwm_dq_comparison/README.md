# PWM comparison with variable dq commands

This standalone 4 ms PWM test uses a 1 kHz carrier and a 50 Hz reference frame. The normalized d command has a 360 Hz oscillation and a step at 1.37 ms. The q command has a 620 Hz oscillation and a pulse from 2.21 to 2.83 ms. The inverse Park transform produces three modulation commands, clipped to ±1.

The plots cover the dq commands and phase references, the procedural PWM and live comparator, and both MTI variants. The internal-carrier MTI comparison uses held references. The procedural, live comparator, internal-carrier MTI, and scheduled-transition MTI cases yielded phase A/B/C transition counts of 7/8/8, 10/8/8, 7/8/8, and 8/8/9, respectively.

The main harness initializes all sampled PWM variants from the selected input profile at time zero, with the 1 kHz carrier at zero and rising. It uses the same carrier half-period index for sampling and requesting ramp boundaries, with the first peak at 250 µs. Scheduled MTI localization evaluates elapsed trial time from the accepted left endpoint and holds the ramp inputs over that trial. The variable-dq patch supplies the alternate input profile.

## Rerun the PWM comparisons

From the repository root, with the VeraGrid Python dependencies installed:

```sh
case_dir=src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison
run_dir=$(mktemp -d)
cp "$case_dir/../compare_pwm_mti_direct.py" "$run_dir/"
patch -d "$run_dir" -p0 < "$case_dir/variable_dq.patch"
PYTHONPATH="$PWD/src" MPLBACKEND=Agg python "$run_dir/compare_pwm_mti_direct.py"
```

The harness writes the five PWM comparison figures next to the temporary script. The archived input figure shows the `dq_commands` and `pwm_references` functions defined in the patch.

## Compatible soft-sign PWM

After preparing the patched harness above, run the standalone comparator
comparison with:

```sh
PYTHONPATH="$PWD/src" MPLBACKEND=Agg python "$case_dir/compare_soft_pwm.py" "$run_dir/compare_pwm_mti_direct.py"
```

This evaluates the compatible bridge's carrier and gate equations with its
actual sample-and-hold logic, driven by the same phase references as the
hard-PWM test. The 0.1 µs observation grid resolves the soft edges. The plot
covers gate behavior only.

Phase A starts exactly at the comparator threshold. The soft equation returns
0.5 at initialization and the hard scheduler starts low. Both paths now use
the same initial input reference for phase C.

## Filter current and switching ripple

```sh
PYTHONPATH="$PWD/src" MPLBACKEND=Agg python "$case_dir/compare_filter_ripple.py" "$run_dir/compare_pwm_mti_direct.py"
```

The script replays the gate traces into a balanced, floating-neutral RL load.
The assumed DC link is 800 V, with 0.2 Ω and 2 mH per phase and zero initial
current. It uses an exact zero-order-hold RL update at the replay resolution.
It subtracts the current driven by each waveform's carrier-cycle-average
phase voltage to expose the switching ripple. The electrical model is
open-loop and has no grid source, controller feedback, or DC-link dynamics.

The two 1 kHz runs cover the original sinusoidal reference and the variable
dq input. They include procedural hard PWM, compatible soft PWM, the live
comparator, internal-carrier MTI, and scheduled-transition MTI. For MTI,
the replay preserves the gate values at the solver's reported timestamps.
The 20 Hz tie-break plot reproduces the original slow diagnostic variants.
The original standard versus multilinear raw-reference check yielded the
same modulation references to floating-point precision. Both variants
feed the same procedural PWM scheduler, so that check adds no distinct
switching-ripple case under its identical reference conditions.

## Aligned event comparison

```sh
PYTHONPATH="$PWD/src" MPLBACKEND=Agg python "$case_dir/compare_aligned_pwm.py" "$run_dir/compare_pwm_mti_direct.py"
```

The aligned run evaluates the procedural scheduler and both MTI blocks with
the same held samples and ramp boundaries. The carrier starts at zero and
rises to its first peak at 250 µs. A sample is supplied at initialization and
at each carrier extremum. Both MTI blocks use the procedural convention for
saturation and directional equality, with their comparison biases set to zero.
The test finds the internal-carrier guard crossings along the exact linear
carrier ramp and evaluates the scheduled block's algebraic crossing times.
It checks the event polarity and timestamp per phase, then integrates the RL
current analytically between gate changes.

This is an event-equation test. It supplies the ramp boundaries explicitly and
does not use the original BackEulerMTI benchmark adapters.

## Solver timings and full converter grid comparison

Run `benchmark_solvers.py` from the repository root with `PYTHONPATH=src`,
`MPLBACKEND=Agg`, and one BLAS/OpenMP thread. It times four solver-backed PWM
variants and runs the standard, multilinear, and MTI full switched converters
in a power-flow-initialized AC/DC grid. The MTI converter uses the dedicated
MTI problem and integrator. Each case runs in a separate process with a timeout.
For this short grid case, the compiled backend uses its warmup-off policy;
setup and simulation times are recorded separately.

See [the benchmark report](benchmark_results/README.md) for timings, waveform
errors, limitations, raw arrays, and failure logs. Convergence and speed alone
do not imply agreement between converter variants.
