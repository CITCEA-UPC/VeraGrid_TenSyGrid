# Upstream converter initialization investigation (before port)

The averaged-only GridForLoads model was excluded: it does not exercise the three-gate PWM logic.

## Actual three-gate path

The existing sibling checkout provides these diagnostics:

- `/home/pablo/Desktop/eroots/VeraGrid/trunk/dynamics/diagnose_switched_converter_initialization.py`
- `/home/pablo/Desktop/eroots/VeraGrid/trunk/dynamics/validate_switched_converter_no_handover.py`
- `/home/pablo/Desktop/eroots/VeraGrid/src/tests/EMT/test_emt_system_validation.py`

They use the full switched converter and `ThreePhaseCarrierPwmLogic` from
`src/VeraGridEngine/Templates/Emt/converters/bridge_2level_3ph_emt_template.py`.
The existing initialization diagnostic was executed both normally and with
`--direct-pwm`, against the working sibling checkout, without modifying its source.
That checkout contains pre-existing staged and unstaged changes; these runs are
not clean-commit results. Logs are retained alongside this report.

Both diagnostic runs report an algebraic residual of 5.20649079e-11 and a state-equation
residual of zero. With averaged startup, phase-current derivatives match steady
50 Hz rotation (largest discrepancy about 2.3e-9 pu/s). With direct PWM, differences
from the sinusoidal derivatives are approximately -325.67, -4590.16 and +4915.83 pu/s.
These are instantaneous switching responses, not DAE-consistency failures.
A fundamental power-flow seed does not include the carrier-periodic current ripple.
The direct-PWM diagnostic forces effective switching to 1 kHz while the template's
configured frequency is 5 kHz; it is an existing diagnostic variant, not an identical
configuration to the default startup case.

## Same three-bus benchmark on public upstream

Public upstream was fetched from SanPen/VeraGrid at
`3fc0e970c37710a60b9f523a890396f43ca42af3`, exported to an isolated /tmp source snapshot,
and run with the original benchmark topology, controls, 10 us Backward Euler step
and 2 ms horizon. Source paths are asserted by the probe to prevent mixed imports.

| Engine | Explicit algebraic residual | Initial id / id reference | Initial P / filtered P |
| --- | ---: | ---: | ---: |
| TenSyGrid working checkout | 1.702079526 | 0.142010231 | -0.326500349 / 0.300100000 |
| Public upstream snapshot | 6.150067e-9 | 0.463509396 | 0.326500349 / 0.326500349 |

Both runs reported initialization and convergence. The local flag alone therefore
does not demonstrate a physically consistent initial state. Public upstream's
Symbolic attempt crashed during JIT compilation; the successful upstream trajectory
uses StructuralCompiled with warmup off. The successful local trajectory uses Symbolic.
This is an initialization comparison, not a controlled timing or integration-accuracy study.

Upstream includes corrections to power/current per-unit scaling, DC capacitor
operating-point initialization, power-flow mapping, controller filtering, carrier
frequency/phase and startup gain scheduling. The code difference is broader than
an initialization-algorithm change. These runs preceded the port. The applied port and new measurements are in `../after_initialization_port`.

## Earlier trunk scripting attempt

The older `trunk/dynamics_emt/scripting_vsc_switched_emt.py` entry point was also
run from committed sibling snapshot `16efe970f1`, with its original settings.
It failed its positive-DC-voltage validation. Its captured trajectory is retained
under `native_trunk` as a rejected diagnostic, not a validated baseline.
This is distinct from the three-gate validation diagnostics above.

## Reproduction

`probe_upstream_initialization.py` accepts `--source`, `--output`, `--initialization`
and `--backend`. `probe_native_converter.py` accepts `--repo` and `--output` and
captures the older trunk example's validation outcome without suppressing it.
Run with the VeraGrid Python environment, MPLBACKEND=Agg, one BLAS/OpenMP thread,
and VERAGRID_EMT_INIT_CACHE_DIR / VERAGRID_JIT_CACHE_DIR pointing to writable caches.
