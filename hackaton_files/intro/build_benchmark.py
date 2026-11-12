"""Rebuild the benchmark with VeraGrid's dynamic RMS grid-forming converter.

Run from the repository root with PYTHONPATH=src:.:
    python -m hackaton_files.intro.build_benchmark

The converter core is the same one used by VscGfmBuild. Here its AC terminal
is attached to a generator injection; no explicit DC network is modeled.
"""
from pathlib import Path
from VeraGridEngine.api import open_file, save_file
from VeraGridEngine.Templates.Rms.vsc_gfm import build_gfm_converter_model
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model
from VeraGridEngine.enumerations import VarPowerFlowReferenceType as Reference


def build_benchmark():
    source = Path(__file__).resolve().parents[1] / "grids/ieee9_rms_mixed_ibr.veragrid"
    grid = open_file(str(source))
    factory = grid.var_factory
    vm = factory.add_var("Vm", reference=Reference.Vm)
    va = factory.add_var("Va", reference=Reference.Va)
    p = factory.add_var("P_gfm", reference=Reference.P)
    q = factory.add_var("Q_gfm", reference=Reference.Q)

    # The core expects VSC terminal powers (positive into the converter).
    # Generator powers are positive into the AC grid, hence the minus signs.
    core, _, _, active_power, reactive_power, _ = build_gfm_converter_model(
        factory, inputs=[vm, va, -p, -q],
    )
    # Only the wrapper owns the external injection mapping. Keep all controller,
    # filter, droop and limiter equations in the unmodified converter core.
    core.external_mapping.clear()
    model = Block(
        name="Dynamic RMS grid-forming converter",
        in_vars=[vm, va], out_vars=[p, q],
        algebraic_vars=[p, q],
        algebraic_eqs=[p - active_power, q - reactive_power],
        external_mapping={Reference.Vm: vm, Reference.Va: va,
                          Reference.P: p, Reference.Q: q},
        children=[core],
    )
    generator = next(generator for generator in grid.generators if generator.name == "gen 1")
    set_rms_model(generator, model, factory)
    return grid


if __name__ == "__main__":
    save_file(build_benchmark(), str(Path(__file__).with_name("benchmark_grid.veragrid")))
