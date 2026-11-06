# SPDX-License-Identifier: MPL-2.0
"""Check the native symbolic controller and actual VeraGrid DAE studies."""
import unittest

import numpy as np
from cp_governor import (Parameters, VarFactory, governor, run_study, RmsProblemDae,
                         SymbolicCPTensor, block_from_cp, Const)


class GovernorExerciseTests(unittest.TestCase):
    def test_cp_builder_with_different_state_count_and_rank(self):
        # A one-state rank-two controller: dx/dt = -2*x + 3*u; y = x.
        vf = VarFactory()
        x, u, y = [vf.add_var(name) for name in ('x', 'u', 'y')]
        tensor = SymbolicCPTensor.empty(n_variables=2, n_equations=2, rank=2)
        tensor.factors[0][:] = [[0, 1], [1, 0]]
        tensor.factors[1][:] = [[1, 0], [0, 1]]
        tensor.output_factor[:] = [[-2, 3], [1, 0]]
        block = block_from_cp(name='one-state controller', n_states=1, states=[x],
                              inputs=[u], outputs=[y], tensor_variables=[x, u], tensor=tensor,
                              parameters={}, initial_values={x: Const(0), y: Const(0)})
        bindings = {x.uid: 4., u.uid: 5., y.uid: 4.}
        self.assertEqual(block.state_eqs[0].eval_uid(bindings), 7.)
        self.assertEqual(block.algebraic_eqs[0].eval_uid(bindings), 0.)

    def test_cp_rejects_invalid_shapes_and_dependent_coefficients(self):
        x = VarFactory().add_var('x')
        tensor = SymbolicCPTensor.empty(n_variables=1, n_equations=1, rank=1)
        tensor.factors[0] = np.ones((3, 1), dtype=object)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            tensor.expressions([x])
        tensor.factors[0] = np.array([[x], [1]], dtype=object)
        with self.assertRaisesRegex(ValueError, 'independent'):
            tensor.expressions([x])

    def test_symbolic_controller_equations_and_derivatives(self):
        rng = np.random.default_rng(4)
        for p in (Parameters(alpha=0.), Parameters(),
                  Parameters(c_ev=.02, c_et=-.01, c_evt=.01, c_vt=-.005)):
            alpha = p.alpha
            block = governor(VarFactory(), p)
            v, tau = block.state_vars
            omega, te = block.in_vars
            for speed in (1., .99, 1.01):
                valve, torque = rng.uniform(.7, .9, 2)
                bindings = {var.uid: expr.value for var, expr in block.event_dict.items()
                            if var.name != 'Tref'}
                tref = next(var for var in block.event_dict if var.name == 'Tref')
                bindings.update({v.uid: valve, tau.uid: torque, omega.uid: speed,
                                 te.uid: .8, tref.uid: .8})
                actual = [eq.eval_uid(bindings) for eq in block.state_eqs]
                e, a, b = (1-speed)/.005, (valve-.8)/.1, (torque-.8)/.1
                interaction = p.c_ev*e*a + p.c_et*e*b + p.c_evt*e*a*b + p.c_vt*a*b
                expected = [(0.8-valve-p.K*(speed-1)*(1+alpha*(valve-.8))+interaction)/p.Tg,
                            (valve-torque)/p.Tt]
                np.testing.assert_allclose(actual, expected, atol=1e-12)
                # The zero speed-error factor must still give the correct derivative.
                derivative = block.state_eqs[0].diff(omega).eval_uid(bindings)
                self.assertAlmostEqual(derivative, (-p.K*(1+alpha*(valve-.8))
                                       -(p.c_ev*a+p.c_et*b+p.c_evt*a*b)/.005)/p.Tg)

    def test_interactions_preserve_initial_jacobian(self):
        jacobians = []
        for p in (Parameters(), Parameters(c_ev=.2, c_et=.1, c_evt=-.15, c_vt=-.05)):
            block = governor(VarFactory(), p)
            v, tau = block.state_vars
            omega, te = block.in_vars
            bindings = {var.uid: (0.8 if var.name == 'Tref' else expr.value)
                        for var, expr in block.event_dict.items()}
            bindings.update({v.uid: .8, tau.uid: .8, omega.uid: 1., te.uid: .8})
            np.testing.assert_allclose([eq.eval_uid(bindings) for eq in block.state_eqs], 0., atol=1e-12)
            jacobians.append([[eq.diff(var).eval_uid(bindings) for var in (v, tau, omega)]
                              for eq in block.state_eqs])
        np.testing.assert_allclose(jacobians[0], jacobians[1], atol=1e-12)

    def test_native_connections_initialization_event_and_modes(self):
        for alpha in (0., 1.):
            grid, machine, controller, pf, modal, rms = run_study(
                Parameters(alpha=alpha), duration=6., dt=.01)
            self.assertIsInstance(rms.problem, RmsProblemDae)
            self.assertEqual(machine.in_vars[2].uid, controller.out_vars[0].uid)
            self.assertEqual(machine.out_vars[2].uid, controller.in_vars[0].uid)
            self.assertEqual(machine.out_vars[3].uid, controller.in_vars[1].uid)
            x0 = modal.problem.get_x0()
            dx0 = modal.problem.get_dx0()
            np.testing.assert_allclose(modal.problem.rhs_state(x0, dx0), 0., atol=1e-8)
            np.testing.assert_allclose(modal.problem.rhs_algebraic(x0, dx0), 0., atol=1e-8)
            values = rms.results.values[:, :, 0]
            self.assertTrue(np.all(np.isfinite(values)))
            speed = values[:, rms.results.uid2idx[machine.out_vars[2].uid]]
            np.testing.assert_allclose(speed[:199], 1., atol=1e-8)
            self.assertLess(speed.min(), .999)
            # Test the actual load injection, not merely the presence of an event.
            pl = grid.loads[0].rms_model.out_vars[0]
            load = values[:, rms.results.uid2idx[pl.uid]]
            self.assertAlmostEqual(load[0], -.8, places=8)
            self.assertAlmostEqual(load[-1], -.85, places=8)
            modes = modal.results.eigenvalues
            self.assertEqual(len(modes), 4)
            self.assertEqual(np.count_nonzero(np.abs(modes) < 1e-7), 1)
            self.assertTrue(np.all(modes[np.abs(modes) >= 1e-7].real < 0.))

    def test_no_disturbance_preserves_equilibrium(self):
        *_, rms = run_study(step=0., duration=3., dt=.02)
        y = rms.results.values[:, :, 0]
        np.testing.assert_allclose(y, np.broadcast_to(y[0], y.shape), atol=1e-8)

    def test_multilinear_interactions_reduce_target_peak(self):
        from tune_tensor_governor import evaluate
        baseline, _ = evaluate(Parameters())
        candidate, _ = evaluate(Parameters(c_ev=.08, c_et=-.08, c_evt=-.01, c_vt=-.001))
        self.assertTrue(candidate['feasible'])
        self.assertLess(candidate['peak_hz'], .95 * baseline['peak_hz'])
        self.assertLess(candidate['late_oscillation_hz'], .01)


if __name__ == '__main__':
    unittest.main()
