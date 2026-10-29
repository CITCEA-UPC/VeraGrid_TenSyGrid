"""Build and verify additional RMS challenge grids from static test networks."""

from pathlib import Path

from VeraGridEngine.api import open_file, save_file
from VeraGridEngine.Devices.Branches.transformer import Transformer2W
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Templates.Rms.genrow1_rms_template import get_genrow1_rms_template
from VeraGridEngine.Templates.Rms.line_rms_template import get_line_rms_template
from VeraGridEngine.Templates.Rms.load_rms_template import get_load_rms_template
from VeraGridEngine.Templates.Rms.pvd1_rms_template import get_pvd1_rms_template
from VeraGridEngine.Templates.Rms.transformer_rms_template import get_transformer2w_rms
from VeraGridEngine.Templates.Rms.voltage_source_template import VoltageSourceBuild
from VeraGridEngine.Utils.Symbolic.bus_rms_template import initialize_bus_rms
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model

from .rms_tuning import BasicRmsTuner, KUNDUR_CONTROL_RANGES, RmsTunerEvaluator, RmsTuningMetrics


SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "Grids_and_profiles" / "grids"
GRID_DIRECTORY = Path(__file__).with_name("grids")
ADDITIONAL_STABLE_GRID_FILES = (
    GRID_DIRECTORY / "ieee9_rms_transformers.veragrid",
    GRID_DIRECTORY / "ieee9_rms_gfl.veragrid",
    GRID_DIRECTORY / "ieee9_rms_mixed_ibr.veragrid",
    GRID_DIRECTORY / "ieee30_rms.veragrid",
    GRID_DIRECTORY / "ieee39_rms.veragrid",
)


def additional_grid_evaluator() -> RmsTunerEvaluator:
    """Expose controller gains and dispatch without Kundur-specific corridors."""
    return RmsTunerEvaluator(
        control_ranges={
            **KUNDUR_CONTROL_RANGES,
            "dqdv": (-5.0, 0.0),
            "tip": (0.005, 0.1),
            "tiq": (0.005, 0.1),
        },
        line_corridors=(),
        shunt_bus_names=(),
    )


def build_additional_grid(case_name: str) -> MultiCircuit:
    """Attach RMS device models to an IEEE network and set a viable dispatch."""
    sources = {
        "ieee9": "IEEE 9 Bus.gridcal",
        "ieee9_gfl": "IEEE 9 Bus.gridcal",
        "ieee9_mixed_ibr": "IEEE 9 Bus.gridcal",
        "ieee30": "IEEE 30 Bus EMT.veragrid",
        "ieee39": "IEEE39.gridcal",
    }
    if case_name not in sources:
        raise ValueError(f"Unknown additional RMS case: {case_name}")
    grid = open_file(str(SOURCE_DIRECTORY / sources[case_name]))

    if case_name in ("ieee9", "ieee9_gfl", "ieee9_mixed_ibr"):
        # The three zero-resistance, zero-shunt generator branches represent
        # step-up transformers in this source file.
        for line in list(grid.lines):
            if line.R == 0.0 and line.B == 0.0:
                transformer = Transformer2W(
                    bus_from=line.bus_from, bus_to=line.bus_to,
                    name=f"Transformer {line.name}", r=line.R, x=line.X,
                    b=line.B, rate=line.rate,
                )
                grid.delete_line(line)
                grid.add_transformer2w(transformer)
    if case_name == "ieee30":
        for load in grid.loads:
            load.P *= 0.90
            load.Q *= 0.90
        for generator in grid.generators:
            generator.P *= 0.90
    if case_name == "ieee39":
        for load in grid.loads:
            load.P *= 0.95
            load.Q *= 0.95
        for generator in grid.generators:
            generator.P *= 0.95
            generator.Vset = min(generator.Vset + 0.01, 1.045)

    for bus in grid.buses:
        initialize_bus_rms(bus, grid.var_factory)
    for index, generator in enumerate(grid.generators):
        if case_name in ("ieee9_gfl", "ieee9_mixed_ibr") and index == 2:
            model = get_pvd1_rms_template(grid.var_factory).block
        elif case_name == "ieee9_mixed_ibr" and index == 1:
            model = VoltageSourceBuild(grid.var_factory, name="Ideal grid-forming source").block
            model.name = "Ideal grid-forming source"
        else:
            model = get_genrow1_rms_template(grid.var_factory).block
        if case_name in ("ieee30", "ieee39"):
            model.set_parameter_in_model("D", 100.0)
        set_rms_model(generator, model, grid.var_factory)
    for load in grid.loads:
        set_rms_model(load, get_load_rms_template(grid.var_factory).block, grid.var_factory)
    for line in grid.lines:
        set_rms_model(line, get_line_rms_template(grid.var_factory).block, grid.var_factory)
    for transformer in grid.transformers2w:
        set_rms_model(transformer, get_transformer2w_rms(grid.var_factory).block, grid.var_factory)
    return grid


def verify_additional_grid(grid: MultiCircuit) -> RmsTuningMetrics:
    """Require convergence, stable physical modes, and acceptable limits."""
    metrics = BasicRmsTuner().test_tuning(grid, additional_grid_evaluator())
    if not metrics.stable or metrics.voltage_violations or metrics.thermal_violations:
        raise RuntimeError(f"Additional RMS case failed verification: {metrics}")
    return metrics


def save_additional_grids() -> tuple[RmsTuningMetrics, ...]:
    """Regenerate the saved IEEE9, inverter IEEE9, IEEE30, and IEEE39 cases."""
    metrics: list[RmsTuningMetrics] = []
    for case_name, path in zip(("ieee9", "ieee9_gfl", "ieee9_mixed_ibr", "ieee30", "ieee39"), ADDITIONAL_STABLE_GRID_FILES):
        grid = build_additional_grid(case_name)
        metrics.append(verify_additional_grid(grid))
        save_file(grid, str(path))
    return tuple(metrics)
