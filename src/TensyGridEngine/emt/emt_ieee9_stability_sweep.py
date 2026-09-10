"""Run one reproducible IEEE9 EMT control-stability sensitivity case."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from TensyGridEngine.emt.emt_ieee9_small_signal import run_small_signal_analysis


CASES: dict[str, dict[str, float]] = {
    "baseline": {},
    "pss_off": {"Ks": 0.0},
    "pss_half": {"Ks": 10.0},
    "governor_original": {"K": 10.0},
    "governor_half": {"K": 5.0},
    "governor_double": {"K": 20.0},
    "machine_damping_1": {"D": 1.0},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    args = parser.parse_args()
    results = run_small_signal_analysis(event_overrides=CASES[args.case])
    dominant = int(np.argmax(np.abs(results.multipliers)))
    mu = results.multipliers[dominant]
    exponent = results.eigenvalues[dominant]
    output = Path(__file__).resolve().parent / f"ieee9_emt_stability_{args.case}.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "mu_real", "mu_imag", "mu_abs", "lambda_real", "lambda_imag", "frequency_hz"))
        writer.writerow((args.case, mu.real, mu.imag, abs(mu), exponent.real, exponent.imag,
                         results.conjugate_frequencies[dominant]))
    print(
        f"case={args.case} mu={mu.real:+.9e}{mu.imag:+.9e}j |mu|={abs(mu):.9e} "
        f"lambda={exponent.real:+.9e}{exponent.imag:+.9e}j "
        f"frequency={results.conjugate_frequencies[dominant]:.6f}Hz"
    )
    print(f"output={output}")


if __name__ == "__main__":
    main()
