#!/usr/bin/env python3
"""Plot measured Numba compilation and remaining run time for four RMS/EMT cases."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path,
        default=Path(__file__).parent/'rms_ssa/phase_comparison/summary.json')
    parser.add_argument('--output', type=Path, help='Output filename stem')
    args=parser.parse_args()
    report=json.loads(args.summary.read_text())
    if report.get('status')!='passed':
        raise ValueError('All four measured cases must pass validation before plotting')
    keys=['ieee39/rms','ieee39/emt','ieee118/rms','ieee118/emt']
    compilation=np.array([report['cases'][key]['compilation_seconds'] for key in keys])
    calculation=np.array([report['cases'][key]['calculation_seconds'] for key in keys])
    totals=np.array([report['cases'][key]['total_seconds'] for key in keys])
    if not np.all(np.isfinite(totals)) or not np.allclose(compilation+calculation,totals):
        raise ValueError('Phase measurements do not match the total times')
    positions=np.arange(4)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'svg.hashsalt':'verified-rms-emt-phases'}):
        figure,axis=plt.subplots(figsize=(9,5.6))
        figure.subplots_adjust(left=.10,right=.98,bottom=.29,top=.77)
        lower=axis.bar(positions,compilation,width=.62,color='#E9A23B',label='Numba JIT compilation')
        upper=axis.bar(positions,calculation,bottom=compilation,width=.62,color='#2563A6',label='Calculation + setup')
        axis.set_xticks(positions,['IEEE39\nRMS','IEEE39\nEMT','IEEE118\nRMS','IEEE118\nEMT'])
        axis.tick_params(axis='x',length=0,pad=9)
        axis.set_ylabel('Measured wall time (s)')
        axis.set_ylim(0,max(totals)*1.19)
        axis.set_axisbelow(True)
        axis.yaxis.grid(True,color='#E5E7EB',linewidth=.8)
        for i,total in enumerate(totals):
            axis.text(i,total+max(totals)*.025,f'{total:.1f} s',ha='center',va='bottom',fontweight='bold',fontsize=12)
            for height,bottom,color in [(compilation[i],0.,'#172554'),(calculation[i],compilation[i],'white')]:
                if height>max(totals)*.055:
                    axis.text(i,bottom+height/2,f'{height:.1f}',ha='center',va='center',color=color,fontsize=10)
        figure.text(.10,.95,'RMS and EMT SSA: compilation vs calculation',fontsize=17,fontweight='bold')
        figure.text(.10,.895,'Fresh Numba disk cache per case · single-threaded BLAS · sequential runs',fontsize=10,color='#475569')
        figure.legend(handles=[lower,upper],loc='upper right',bbox_to_anchor=(.98,.85),ncol=2,frameon=False)
        figure.text(.10,.12,'Compilation: outermost Numba compile events (cache hits excluded).\n'
            'Calculation + setup: remaining measured pipeline, including loading, PF, assembly, initialization and SSA.\n'
            'RMS: all 80 / 432 finite modes. EMT: 6 modes, 600 steps/period, one period. Interpreter startup excluded.',
            fontsize=8.5,color='#475569',linespacing=1.6)
        output=args.output or args.summary.with_name('rms_emt_compilation_calculation')
        output.parent.mkdir(parents=True,exist_ok=True)
        figure.savefig(output.with_suffix('.png'),dpi=180,facecolor='white')
        svg=output.with_suffix('.svg')
        figure.savefig(svg,facecolor='white',metadata={'Date':None})
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(figure)


if __name__=='__main__':
    main()
