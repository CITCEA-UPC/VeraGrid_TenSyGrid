"""IEEE 9 data-only grids must receive the reference EMT governor gain."""

import pytest

from TensyGridEngine.emt.emt_ieee9 import attach_emt_models, build_ieee9_grid
from TensyGridEngine.emt.emt_ieee9 import build_emt_options, build_power_flow_options
from VeraGridEngine.Simulations.EMT.emt_problem_factory import build_emt_problem
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Utils.Symbolic.block import Const, find_name_in_block
from VeraGridEngine.api import save_file


def _governor_gains(block):
    gains = [
        float(expr.value)
        for variable, expr in block.event_dict.items()
        if variable.name == "K" and isinstance(expr, Const)
    ]
    for child in block.children:
        gains.extend(_governor_gains(child))
    return gains


@pytest.mark.parametrize("multilinear_machine", [False, True])
def test_reloaded_ieee9_receives_reference_governor_gain(tmp_path, multilinear_machine):
    source = build_ieee9_grid()
    path = tmp_path / "IEEE 9 Bus SSA.gridcal"
    save_file(source, str(path))

    reloaded = build_ieee9_grid(path)
    assert source.fBase == 60.0
    assert reloaded.fBase == source.fBase
    attach_emt_models(reloaded, governor_k=40.0, multilinear_machine=multilinear_machine)

    assert len(reloaded.generators) == 3
    for generator in reloaded.generators:
        assert _governor_gains(generator.emt_model) == [40.0]


def test_ieee9_governors_start_at_equilibrium(tmp_path, monkeypatch):
    monkeypatch.setenv("VERAGRID_EMT_INIT_CACHE_DIR", str(tmp_path / "emt-cache"))
    grid = build_ieee9_grid()
    attach_emt_models(grid, governor_k=40.0, multilinear_machine=False)
    pf_options = build_power_flow_options()
    balanced = PowerFlowDriver(grid=grid, options=pf_options)
    balanced.run()
    three_phase = PowerFlowDriver3Ph(grid=grid, options=pf_options)
    three_phase.run()
    assert balanced.results.converged and three_phase.results.converged

    problem = build_emt_problem(
        grid=grid,
        options=build_emt_options(),
        pf_results=balanced.results,
        pf_results_3ph=three_phase.results,
    )
    runtime_gains = [
        float(problem.event_params_values[index])
        for index, variable in enumerate(problem.get_variable_parameters())
        if variable.name == "K"
    ]
    assert runtime_gains == [40.0, 40.0, 40.0]
    initial_derivatives = problem.get_dx0()
    for generator in grid.generators:
        state = find_name_in_block("y_gov0", generator.emt_model)
        assert state is not None and state.diff_var is not None
        assert abs(initial_derivatives[problem.get_diff_var_idx(state.diff_var)]) < 1e-9
