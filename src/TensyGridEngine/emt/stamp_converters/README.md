# STAMP GFOR and GFOL converter models

This package contains the full nonlinear STAMP converter control and filter
equations, plus an instantaneous three-phase EMT wrapper. It was ported from
`STAMP_Public/veragrid_stamp` at commit `f132f36` (MPL-2.0). The local modules
have no import dependency on `STAMP_Public`.

- `nonlinear_converters.py` builds the q-d RMS model with measurement delays,
  GFOR droop and voltage control or GFOL PLL and power control, inner current
  control, transformer, and damped LCL filter.
- `emt_nonlinear_converters.py` connects that model to VeraGrid's abc EMT ports.
- `explicit_converters.py` substitutes the EMT wrapper's algebraic voltage,
  power, and current equations into the state equations. It provides explicit
  abc-input ODE blocks for both GFOR and GFOL, plus symbolic output expressions.
- `parameters.py` supplies the converter parameter type and the STAMP GFOR1
  and GFOL2 case values.
- `bases.py` supplies the voltage base conversion.

Build a model with VeraGrid's `VarFactory`:

```python
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from TensyGridEngine.emt.stamp_converters import (
    STAMP_GFOR, STAMP_GFOL, build_stamp_nonlinear_converter_emt,
)

vf = VarFactory()
gfor = build_stamp_nonlinear_converter_emt(vf, STAMP_GFOR, "STAMP_GFOR1_EMT")
gfol = build_stamp_nonlinear_converter_emt(vf, STAMP_GFOL, "STAMP_GFOL2_EMT")
```

Each result is an `EmtModelTemplate` whose `block` can be attached to a
VeraGrid generator device with `set_emt_model`. These are averaged dynamic
converter models; they do not represent individual switches or PWM pulses.
Building the blocks does not require the STAMP IEEE-9 case. Network simulation
still requires a grid, valid terminal mappings, power flow, and initialization.

## Explicit ODEs

Use `build_stamp_explicit_converter_emt(vf, parameters, name)` for a standalone
model with three prescribed phase-voltage inputs and no algebraic unknowns. It
returns `(block, observables)`, where `observables` gives symbolic `p`, `q`,
`current_abc`, and `voltage_rms` expressions. The controller references are
fixed to the original STAMP operating point by default; an experiment can
change them through the resulting problem's runtime-parameter API.

Run both modes and save MATLAB time-series files with:

```bash
MPLCONFIGDIR=/tmp/mpl-stamp-explicit PYTHONPATH=src python -m TensyGridEngine.emt.emt_stamp_demo.stamp_explicit_ode_demo
```

These are explicit nonlinear ODEs. They are not multi-affine lifts, so an exact
`S`/`Phi` extraction is not promised for them.
