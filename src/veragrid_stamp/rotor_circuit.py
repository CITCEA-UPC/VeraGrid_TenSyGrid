"""STAMP synchronous-machine rotor-circuit parameter conversion.

This ports ``generate_parameters_SG.m`` from standard reactance/time-constant
data to the field and d/q damper winding R/L parameters used by STAMP's six
winding-current differential equations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .parameters import OMEGA_BASE, StampSynchronousGeneratorParameters


@dataclass(frozen=True)
class RotorCircuitParameters:
    ll: float
    lmd: float
    lmq: float
    lfd: float
    l1d: float
    l1q: float
    l2q: float
    rf: float
    r1d: float
    r1q: float
    r2q: float


def _axis_winding_parameters(
    x: float, x_transient: float, x_subtransient: float,
    t_open_transient: float, t_open_subtransient: float, xl: float,
) -> tuple[float, float, float, float]:
    """Return the two rotor winding resistances and inductances for one axis."""
    # Normalize with Zbase=1, hence Lbase=1/wb, exactly mirroring MATLAB's
    # intermediate SI calculation without needing the case voltage base.
    x_inductance = x / OMEGA_BASE
    xl_inductance = xl / OMEGA_BASE
    xm = (x - xl) / OMEGA_BASE
    t_transient = t_open_transient * x_transient / x
    t_subtransient = t_open_subtransient * x_subtransient / x_transient
    denominator = t_open_transient + t_open_subtransient - t_transient - t_subtransient
    aa = xm * xm / (x_inductance * denominator)
    a = (x_inductance * (t_transient + t_subtransient) - xl_inductance * (t_open_transient + t_open_subtransient)) / xm
    b = (x_inductance * t_transient * t_subtransient - xl_inductance * t_open_transient * t_open_subtransient) / xm
    cc = (t_open_transient * t_open_subtransient - t_transient * t_subtransient) / denominator
    root = math.sqrt(a * a - 4.0 * b)
    r_a = 2.0 * aa * root / (a - 2.0 * cc + root)
    l_a = r_a * (a + root) / 2.0
    r_b = 2.0 * aa * root / (2.0 * cc - a + root)
    l_b = r_b * (a - root) / 2.0
    # MATLAB first computes SI R/L and then divides by Zbase/Lbase. Since
    # Lbase=Zbase/wb, the per-unit resistance is r and inductance is l*wb.
    return r_a, l_a * OMEGA_BASE, r_b, l_b * OMEGA_BASE


def derive_rotor_circuit(p: StampSynchronousGeneratorParameters) -> RotorCircuitParameters:
    rf, lfd, r1d, l1d = _axis_winding_parameters(
        p.xd, p.xd_prime, p.xd_subtransient, p.td0_prime, p.td0_subtransient, p.xl,
    )
    r1q, l1q, r2q, l2q = _axis_winding_parameters(
        p.xq, p.xq_prime, p.xq_subtransient, p.tq0_prime, p.tq0_subtransient, p.xl,
    )
    return RotorCircuitParameters(
        ll=p.xl, lmd=p.xd - p.xl, lmq=p.xq - p.xl,
        lfd=lfd, l1d=l1d, l1q=l1q, l2q=l2q,
        rf=rf, r1d=r1d, r1q=r1q, r2q=r2q,
    )
