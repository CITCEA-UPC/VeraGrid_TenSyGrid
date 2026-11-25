"""Run the upstream trunk converter example without changing its defaults."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve() / 'src'))
sys.path.insert(0, str(args.repo.resolve() / 'trunk'))

from dynamics_emt.scripting_vsc_switched_emt import (
    build_script_emt_options, SCRIPT_FILTER_R_EQ_PU, SCRIPT_FILTER_L_EQ_PU,
)
from dynamics_emt.support.switched_converter_handover_case import (
    plot_switched_converter_handover, plot_switched_converter_states,
    simulate_switched_converter_handover, validate_switched_converter_handover,
)
import VeraGridEngine.api as gce


def run() -> None:
    """Execute the example's native build, initialization, simulation and assertions."""
    assert Path(gce.__file__).resolve().is_relative_to(args.repo.resolve()), gce.__file__
    args.output.mkdir(parents=True, exist_ok=True)
    trace = simulate_switched_converter_handover(
        emt_options=build_script_emt_options(), t_enable_s=.005,
        r_eq_pu=SCRIPT_FILTER_R_EQ_PU, l_eq_pu=SCRIPT_FILTER_L_EQ_PU)
    validation = 'passed'
    try:
        validate_switched_converter_handover(trace)
    except AssertionError as exc:
        import traceback
        validation = 'failed: ' + str(traceback.extract_tb(exc.__traceback__)[-1].line)
    signals = dict()
    for name in trace.__slots__:
        value = getattr(trace, name)
        if isinstance(value, np.ndarray):
            signals[name] = value
    np.savez_compressed(args.output / 'trajectory.npz', **signals)
    plot_switched_converter_handover(trace, show_plot=False, save_path=args.output / 'handover.png')
    plot_switched_converter_states(trace, show_plot=False, save_path=args.output / 'states.png')
    before = trace.t_arr < trace.t_enable_s - 1e-12
    summary = dict(engine_file=gce.__file__, initialized=bool(trace.well_initialized),
                   converged=bool(trace.converged), native_validation=validation,
                   final_time=float(trace.t_arr[-1]), handover_time=trace.t_enable_s,
                   initialization_requested=str(trace.initialization_requested),
                   initialization_used=str(trace.initialization_used),
                   pre_handover_vdc_min=float(np.min(trace.v_dc_arr[before])),
                   pre_handover_vdc_max=float(np.max(trace.v_dc_arr[before])),
                   pre_handover_idc_min=float(np.min(trace.i_dc_arr[before])),
                   pre_handover_idc_max=float(np.max(trace.i_dc_arr[before])),
                   pre_handover_id_error_max=float(np.max(np.abs(trace.i_d_arr[before] - trace.i_d_ref_arr[before]))),
                   initial=dict(vdc=float(trace.v_dc_arr[0]), idc=float(trace.i_dc_arr[0]),
                                id=float(trace.i_d_arr[0]), id_ref=float(trace.i_d_ref_arr[0]),
                                p_filtered=float(trace.P_f_arr[0])),
                   final=dict(vdc=float(trace.v_dc_arr[-1]), idc=float(trace.i_dc_arr[-1]),
                              id=float(trace.i_d_arr[-1]), id_ref=float(trace.i_d_ref_arr[-1]),
                              p_filtered=float(trace.P_f_arr[-1])))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=True)
    run()
