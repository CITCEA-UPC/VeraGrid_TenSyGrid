"""RMS benchmark representation and descriptor-spectrum regression checks."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import linalg, sparse

from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Devices.Substation.bus import Bus
from VeraGridEngine.Devices.Branches.line import Line
from VeraGridEngine.Devices.Injections.generator import Generator
from VeraGridEngine.Devices.Injections.load import Load
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver, PowerFlowOptions
from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
from VeraGridEngine.Simulations.Rms.problems.rms_problem_multilinear import RmsProblemMultilinear
from VeraGridEngine.enumerations import RmsProblemTypes, RmsInitializationMethod
from trunk.benchmarks.multilinear_ssa import attach_rms
from trunk.benchmarks.rms_descriptor import check_symbolic_representation, finite_descriptor_modes


class DescriptorProblem:
    def __init__(self):
        # xdot = -2*x+z, 0=z-0.5*x -> one finite mode -1.5.
        self.a = sparse.csc_matrix([[-2., 1.], [-.5, 1.]])
        self.e = sparse.csc_matrix([[1., 0.], [0., 0.]])

    def get_static_state_matrix(self, initial, derivatives):
        return self.a

    def get_E_matrix(self, initial, derivatives):
        return -self.e


def test_algebraic_elimination_preserves_full_pencil_eigenpairs():
    problem = DescriptorProblem()
    values, vectors, report = finite_descriptor_modes(problem, np.zeros(2), np.zeros(1))
    full = linalg.eigvals(problem.a.toarray(), problem.e.toarray())
    np.testing.assert_allclose(values, full[np.isfinite(full)], atol=1e-14)
    np.testing.assert_allclose(problem.a @ vectors, (problem.e @ vectors)*values, atol=1e-14)
    assert report['descriptor_regularization'] == 0.
    assert report['eigenpair_backward_error_max'] < 1e-14


def test_multilinear_generator_current_ports_preserve_pf_equilibrium_and_export():
    grid = MultiCircuit(Sbase=100.)
    source, target = Bus(name='source', Vnom=110., is_slack=True), Bus(name='target', Vnom=110.)
    grid.add_bus(source); grid.add_bus(target)
    grid.add_line(Line(bus_from=source, bus_to=target, r=.01, x=.1, b=.01))
    grid.add_generator(bus=source, api_obj=Generator(P=50., vset=1.02))
    grid.add_load(bus=target, api_obj=Load(P=50., Q=10.))
    attach_rms(grid)
    power_flow = PowerFlowDriver(grid, PowerFlowOptions(tolerance=1e-10))
    power_flow.run()
    assert power_flow.results.converged
    problem = RmsProblemMultilinear(grid=grid,
        options=RmsOptions(problem_type=RmsProblemTypes.Multilinear,
                          initialization_method=RmsInitializationMethod.Explicit),
        pf_results=power_flow.results)
    problem.build_multilinear_matrices()
    initial = problem.get_x0()
    report = check_symbolic_representation(problem, initial)
    assert report['symbolic_initial_residual_inf'] < 1e-7
    assert report['representation_error_inf'] < 1e-10
    values, vectors, result = finite_descriptor_modes(problem, initial, np.zeros(len(initial)))
    assert len(values) == problem.get_diff_var_number()
    assert result['eigenpair_backward_error_max'] < 1e-10


def test_representation_guard_rejects_omitted_rational_current_term():
    from VeraGridEngine.Utils.Symbolic.symbolic import Var, Const
    voltage = Var('voltage')
    problem = SimpleNamespace(_constant_parameters=[], _constant_params=[],
        _variable_parameters=[], _variable_parameters_values=[],
        _glob_time=Var('time'), _state_eqs=[],
        _algebraic_eqs=[Const(1.)/voltage], _uid2idx_vars={voltage.uid: 0}, _diff_vars=[])
    problem.get_diff_var_number = lambda: 0
    problem.exact_residual_from_s_phi = lambda values: np.zeros(1)
    with pytest.raises(RuntimeError, match='does not reproduce'):
        check_symbolic_representation(problem, np.ones(1))
