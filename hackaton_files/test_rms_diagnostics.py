"""Modal helper contracts and real-grid diagnostics, without output artifacts."""
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import eig
from hackaton_files.rms_diagnostics import get_modes_from_unstable_eigenvalues
from VeraGridEngine.api import open_file
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import SmallSignalStabilityRmsDriver
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_options import RmsSmallSignalStabilityOptions
from hackaton_files import BasicRmsTuner, TunableGrid, UNSTABLE_GRID_FILE, UNSTABLE_CONTROL_RANGES

@pytest.fixture(scope='module')
def unstable():
    grid = open_file(str(UNSTABLE_GRID_FILE))
    pf = PowerFlowDriver(grid)
    pf.run()
    ssa = SmallSignalStabilityRmsDriver(grid, pf_results=pf.results, sss_options=RmsSmallSignalStabilityOptions(k=0, ss_assessment_time=0))
    tunable = TunableGrid.from_problem(ssa.problem, UNSTABLE_CONTROL_RANGES)
    tuner = BasicRmsTuner()
    before = ssa.problem._variable_parameters_values.copy()
    analysis = tuner.get_modal_analysis(tunable)
    np.testing.assert_array_equal(before, ssa.problem._variable_parameters_values)
    assert tuner.tests_used == tunable.tests_used == 0
    return (tuner, tunable, analysis)

def test_state_matrix_modes_and_participation_are_aligned(unstable):
    tuner, tunable, analysis = unstable
    assert tuner.get_modal_analysis(analysis) is analysis
    assert tuner.get_state_matrix(analysis) is analysis['state_matrix']
    assert analysis['mass_matrix'] is None
    assert analysis['state_matrix'].shape == (len(analysis['variables']),) * 2
    assert [v['uid'] for v in analysis['variables']] == [v.uid for v in tunable.problem.state_vars]
    values, left, right = eig(analysis['state_matrix'], left=True, right=True)
    np.testing.assert_allclose(values, analysis['eigenvalues'])
    expected = np.abs(left.conj() * right)
    expected /= expected.sum(axis=0)
    np.testing.assert_allclose(expected, analysis['participation_factors'], atol=1e-10)
    for array in (analysis['state_matrix'], analysis['eigenvalues'], analysis['participation_factors']):
        assert not array.flags.writeable

def test_unstable_modes_have_ownership_and_tunable_handles(unstable):
    tuner, tunable, analysis = unstable
    modes = tuner.get_modes_from_unstable_eigenvalues(analysis, top_n=None)
    assert modes
    assert [m['eigenvalue'].real for m in modes] == sorted([m['eigenvalue'].real for m in modes], reverse=True)
    assert any((m['frequency_hz'] == 0 for m in modes))
    assert all((m['damping_ratio'] < 0 for m in modes))
    all_modes = tuner.get_modes_from_unstable_eigenvalues(analysis, include_conjugates=True)
    assert len(all_modes) == np.count_nonzero(analysis['eigenvalues'].real > 1e-08)
    assert len(all_modes) > len(modes)
    for mode in modes:
        shares = [p['participation'] for p in mode['participants']]
        assert shares == sorted(shares, reverse=True)
        assert sum(shares) == pytest.approx(1)
        for entry in mode['participants']:
            v = entry['variable']
            assert v['device_idtag'] in {g.idtag for g in tunable.problem.grid.generators}
            assert v['device_name'] and v['block_path']
            assert v['tunable_parameters']
            assert all((p in tunable.tunable_parameters and p.device_idtag == v['device_idtag'] for p in v['tunable_parameters']))
    assert len(tuner.get_modes_from_unstable_eigenvalues(analysis, top_n=1)[0]['participants']) == 1

def test_raw_stable_grid_is_unchanged():
    grid = open_file(str(Path(__file__).with_name('stable_rms_grid.veragrid')))
    before = [(g.idtag, g.P, str(g.rms_model)) for g in grid.generators]
    tuner = BasicRmsTuner()
    analysis = tuner.get_modal_analysis(grid)
    assert tuner.get_modes_from_unstable_eigenvalues(analysis) == ()
    assert before == [(g.idtag, g.P, str(g.rms_model)) for g in grid.generators]
    assert tuner.tests_used == 0

def test_neutral_infinite_modes_and_argument_validation():
    values = np.array([0, 1e-10, -0.2, np.inf, 0.3 + 2j, 0.3 - 2j, 0.5])
    labels = (dict(index=0, uid=10, name='x', device_idtag=None, device_name=None, block_path=None),)
    analysis = dict(state_matrix=np.zeros((1, 1)), mass_matrix=None, eigenvalues=values, participation_factors=np.ones((1, 7)), variables=labels)
    modes = get_modes_from_unstable_eigenvalues(analysis)
    assert [m['eigenvalue'] for m in modes] == [0.5, 0.3 + 2j]
    for top in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            get_modes_from_unstable_eigenvalues(analysis, top_n=top)
    for tol in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            get_modes_from_unstable_eigenvalues(analysis, real_tolerance=tol)
    with pytest.raises(TypeError):
        BasicRmsTuner().get_modal_analysis(None)

def test_descriptor_analysis_exposes_mass_matrix():
    from types import SimpleNamespace
    from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
    from VeraGridEngine.Devices.multi_circuit import MultiCircuit
    vf = VarFactory()
    x, y = (vf.add_var('x'), vf.add_var('y'))
    problem = SimpleNamespace(grid=MultiCircuit(), state_vars=[x], algebraic_vars=[y], get_x0=lambda: np.zeros(2), get_diff_var_number=lambda: 1, get_E_matrix=lambda x, dx: np.diag([-1.0, 0.0]), get_static_state_matrix=lambda x, dx: np.array([[0.2, 0.0], [1.0, -1.0]]), get_device_vars_dict=lambda: {})
    tunable = TunableGrid(problem, (), (), (), (), ())
    analysis = BasicRmsTuner().get_modal_analysis(tunable)
    np.testing.assert_array_equal(analysis['mass_matrix'], np.diag([1.0, 0.0]))
    assert len(analysis['variables']) == 2
    assert np.count_nonzero(np.isinf(analysis['eigenvalues'])) == 1
    modes = get_modes_from_unstable_eigenvalues(analysis)
    assert len(modes) == 1
    assert modes[0]['eigenvalue'].real == pytest.approx(0.2)
    assert modes[0]['participants'][0]['variable']['name'] == 'x'
