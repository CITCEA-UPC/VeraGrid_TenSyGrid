"""Explicit base conversions at the VeraGrid/STAMP model boundary."""

from __future__ import annotations

import math


RMS_LL_TO_PEAK_LN = math.sqrt(2.0 / 3.0)
PEAK_LN_TO_RMS_LL = math.sqrt(3.0 / 2.0)


def rms_ll_to_peak_ln(value):
    """Convert a per-unit RMS line-line voltage to peak phase-neutral q/d base."""
    return value * RMS_LL_TO_PEAK_LN


def peak_ln_to_rms_ll(value):
    """Convert a per-unit peak phase-neutral voltage to RMS line-line base."""
    return value * PEAK_LN_TO_RMS_LL
