#!/usr/bin/env python3
"""Plot only the fastest verified end-to-end EMT SSA run for each grid."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main() -> None:
    """Select minima from the passed verification summary and export PNG/SVG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path,
                        default=Path(__file__).parent / 'verified/summary.json')
    parser.add_argument('--output', type=Path,
                        help='Output filename stem; defaults beside the input summary')
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    if report.get('status') != 'passed':
        raise ValueError('Plot requires a passed end-to-end verification summary')
    selected = []
    for grid in ('ieee39', 'ieee118'):
        candidates = [(key, record) for key, record in report['cases'].items()
                      if key.split('/')[0] == grid]
        if not candidates or any(not math.isfinite(row['total_seconds'])
                                 or row['total_seconds'] <= 0 for _, row in candidates):
            raise ValueError(f'{grid}: missing or invalid verified timings')
        selected.append(min(candidates, key=lambda pair: pair[1]['total_seconds']))
    labels = []
    times = []
    for key, record in selected:
        grid, backend = key.split('/')
        backend_label = 'Multilinear (S/Phi)' if backend == 'Multilinear' else 'Native DAE'
        labels.append(f'{grid.upper()}\n{backend_label}')
        times.append(record['total_seconds'])
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'fastest-verified-emt'}):
        figure, axis = plt.subplots(figsize=(7, 4.7))
        figure.subplots_adjust(left=.12, right=.97, top=.80, bottom=.26)
        bars = axis.bar(labels, times, width=.52, color=['#2463A7', '#2A8C7A'])
        axis.set_ylim(0, max(times)*1.22)
        axis.set_ylabel('Total wall time (s)')
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color='#E5E7EB', linewidth=.8)
        axis.tick_params(axis='x', length=0, pad=10)
        axis.bar_label(bars, labels=[f'{value:.2f} s' for value in times],
                       padding=7, fontsize=13, fontweight='bold')
        figure.text(.12, .94, 'Fastest verified EMT SSA runs', fontsize=17, fontweight='bold')
        figure.text(.12, .875, '6 modes · 600 steps per period · one captured period',
                    fontsize=10, color='#475569')
        figure.text(.12, .075, 'Minimum observed total per grid among the two verified backends.\n'
                    'Includes loading, power flows, construction, capture and mode extraction.',
                    fontsize=9, color='#475569', linespacing=1.5)
        output = args.output or args.summary.with_name('fastest_runs')
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output.with_suffix('.png'), dpi=180, facecolor='white')
        svg = output.with_suffix('.svg')
        figure.savefig(svg, facecolor='white', metadata={'Date': None})
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
        plt.close(figure)
    for key, record in selected:
        print(f'{key}: {record["total_seconds"]:.2f} s')


if __name__ == '__main__':
    main()
