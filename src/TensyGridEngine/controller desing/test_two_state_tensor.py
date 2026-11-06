import unittest
import numpy as np
from cp_governor import VarFactory, Parameters, run_study
from two_state_tensor import factory, SEED


class FullTensorTests(unittest.TestCase):
    def test_all_equations_match_independent_multilinear_polynomial(self):
        C = np.random.default_rng(12).normal(size=(3, 7))
        block = factory(C)(VarFactory(), Parameters())
        x, y = block.state_vars
        omega, te = block.in_vars
        tm = block.out_vars[0]
        tref = next(iter(block.event_dict))
        for a, b, e in [(0, 0, 0), (.3, -.2, .8), (-.7, .1, -.5)]:
            target = C @ np.array([a, b, e, a*b, a*e, b*e, a*b*e])
            bindings = {x.uid:a, y.uid:b, omega.uid:1-.005*e,
                        te.uid:.8, tref.uid:.8, tm.uid:.8+target[2]}
            np.testing.assert_allclose([eq.eval_uid(bindings) for eq in block.state_eqs], target[:2], atol=1e-11)
            self.assertAlmostEqual(block.algebraic_eqs[0].eval_uid(bindings), 0)

    def test_zero_disturbance_equilibrium_and_native_connection(self):
        _, machine, ctrl, _, _, rms = run_study(step=0, duration=3,
            controller_factory=factory(SEED))
        self.assertEqual(machine.in_vars[2].uid, ctrl.out_vars[0].uid)
        values = rms.results.values[:, :, 0]
        np.testing.assert_allclose(values, np.broadcast_to(values[0], values.shape), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
