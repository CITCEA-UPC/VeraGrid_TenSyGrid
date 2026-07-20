import unittest
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Assuming these modules are in the correct fully qualified path as per eRoots rules
from PolynomialMatrixBuilder import PolynomialMatrixBuilder
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import SmallSignalStabilityRmsDriver
import VeraGridEngine.api as vge
from MTI_problem import MTI_Problem

from small_signal_tensygrid import define_problem, solve_problem


def _run_driver(problem) -> Tuple[np.ndarray, Any]:
    """Run the SmallSignalStabilityRmsDriver on the given problem and return (eigenvalues, results)."""
    rms_options = vge.RmsOptions(
        time_step=0.01, simulation_time=1.0,
        tolerance=1e-6, max_iter=20,
        problem_type=vge.RmsProblemTypes.Multilinear,
    )
    pf_results = vge.power_flow(grid=vge.MultiCircuit())
    driver = vge.SmallSignalStabilityRmsDriver(
        grid=vge.MultiCircuit(),
        rms_options=rms_options,
        sss_options=vge.RmsSmallSignalStabilityOptions(ss_assessment_time=0, verbose=0),
        pf_results=pf_results,
    )
    setattr(driver, 'problem', problem)
    setattr(driver, 'sss_options', vge.RmsSmallSignalStabilityOptions(ss_assessment_time=0, verbose=0))
    setattr(driver, 'assessment_time', 0)
    setattr(driver, 'k', problem.get_states_number() + problem.get_diff_var_number() + 1)
    setattr(driver, 'rms_options', rms_options)

    driver.run_small_signal_stability()
    r = driver.results
    evals = r.eigenvalues if r is not None else np.array([])
    return evals, r


class TestPolynomialStability(unittest.TestCase):
    """
    Test suite to validate the stability of polynomial systems using
    the iMTI / CPN representation builder.
    """
    # Final class with no persistent internal state between tests
    __slots__: List[str] = list()

    def test_unstable_20x20_system(self) -> None:
        """
        Test 1: Large-Scale Linear State-Space System.
        Validates the parsing, matrix allocation, and eigenvalue computation for a
        20-dimensional linear system. Ensures that the underlying sparse matrix builders
        correctly identify a naturally unstable system with a margin of 0.489.

        :return: None
        :rtype: None
        """
        eqs: List[str] = list([
            '-xp1 + 0.3440*x10 + 0.4474',
            '-xp2 + 0.0038*x2 + 0.6549*x10 + 0.7845*x13 + 0.0040*x14 + 0.2282*x19 + 0.9734',
            '-xp3 + 0.6267*u5 + 0.0651*x12 + 0.3473*x16 + 0.8588',
            '-xp4 + 0.0598*u5 + 0.2887*x17 + 0.0327*x18 + 0.8685',
            '-xp5 + 0.0831*u4 + 0.38984*x15 + 0.1381*x18 + 0.2472',
            '-xp6 + 0.4737*x8 + 0.0856*x16 + 0.2286*x18 + 1.6085',
            '-xp7 + 0.0838*u5 + 0.2301*x11 + 0.2002*x12 + 0.8884',
            '-xp8 + 0.0998*u5 + 0.66*x9 + 0.0515*x10 + 0.5883',
            '-xp9 + 0.6785*u2 + 0.0110*x7 + 0.4883*x9 + 0.2754*x12 + 0.6573',
            '-xp10 + 0.0276*x1 + 0.2263*x19 + 0.3483',
            '-xp11 + 0.0549*u2 + 0.2598*x3 + 0.2733*x12 + 0.0994*x16 + 0.1905*x20 + 1.9020',
            '-xp12 + 0.0732*u5 + 0.0466*u6 + 0.8391',
            '-xp13 + 0.1264*u1 + 0.5706*x5 + 0.0839*x13 + 0.2854*x20 + 1.4471',
            '-xp14 + 0.4188*x8 + 0.4638',
            '-xp15 + 0.7539*u1 + 0.4065*x15 + 0.5479',
            '-xp16 + 0.0105*u2 + 0.1236*x1 + 0.1916*x4 + 0.0859*x6 + 0.1613*x18 + 1.0436',
            '-xp17 + 0.2431*u3 + 0.0820',
            '-xp18 + 0.0578*x2 + 0.0867*x4 + 0.0267*x5 + 0.0526*x8 + 2.5279',
            '-xp19 + 0.2644*u5 + 0.4517*x12 + 1.1073',
            '-xp20 + 0.5101*u4 + 0.3917'
        ])

        xOP: np.ndarray = np.array([-0.1668, -0.459, 0.5462, -1.3023, -0.4115, -0.5401, 1.3665, 0.7068, 0.6076, 0.9198,
                                    0.8994, 0.0327, -1.4159, -1.6809, 0.0378, 1.6635, 0.831, 1.0477, 1.7938, -0.4851])
        uOP: np.ndarray = np.array([-0.6336, 1.1019, -0.7170, 2.2000, 0.8906, -0.6354])

        # Initialize dictionary properly
        v_dict: Dict[str, float] = dict()

        # Populate operational point variables
        i: int
        for i in range(20):
            v_dict[f'x{i + 1}'] = float(xOP[i])
            v_dict[f'xp{i + 1}'] = 0.0
            v_dict[f'dx{i + 1}'] = 0.0

        j: int
        for j in range(len(uOP)):
            v_dict[f'u{j + 1}'] = float(uOP[j])

        # Instantiate compiler and driver
        math_problem: MTI_Problem = MTI_Problem(eqs_str=eqs, v_dict=v_dict)

        evals, _ = _run_driver(math_problem)
        print(f"  eigenvalues: {evals}")
        finite_evals = evals[np.isfinite(evals)]
        max_real = float(np.max(np.real(finite_evals))) if len(finite_evals) > 0 else 0.0
        print(f"  max_real(finite): {max_real:.6f}")
        self.assertGreater(max_real, 0, "The 20x20 system should be unstable.")

    def test_stable_nonlinear_5x5_system(self) -> None:
        """
        Test 2: Nonlinear Differential-Algebraic System (3 DAEs + 2 Algebraic).
        Validates the handling of mixed state and algebraic equations with complex
        float coefficients. Ensures the generalized eigenvalue solver correctly
        processes algebraic constraints to find a stable margin of -0.00584.

        :return: None
        :rtype: None
        """
        eqs: List[str] = list([
            '-xp1 + (x3 + 0.2732*1/3*((-2*u1+u2+u3)*x2 + 1.73205081*(u2-u3)*x1) + 314.159265)*(-y2)',
            '-xp2 + (x3 + 0.2732*1/3*((-2*u1+u2+u3)*x2 + 1.73205081*(u2-u3)*x1) + 314.159265)*y1',
            '-xp3 + (0.022508948*1/3*((-2*u1+u2+u3)*x2 + 1.73205081*(u2-u3)*x1))',
            'y1 - x1',
            'y2 - x2'
        ])

        xOP: np.ndarray = np.array([-0.5737, -0.8193, -0.1597])
        uOP: np.ndarray = np.array([-100.6109, -217.5716, 318.1824])
        yOP: np.ndarray = np.array([-0.5737, -0.8193])

        v_dict: Dict[str, float] = dict()

        i: int
        for i in range(len(xOP)):
            v_dict[f'x{i + 1}'] = float(xOP[i])
            v_dict[f'xp{i + 1}'] = 0.0
            v_dict[f'dx{i + 1}'] = 0.0

        j: int
        for j in range(len(uOP)):
            v_dict[f'u{j + 1}'] = float(uOP[j])

        k: int
        for k in range(len(yOP)):
            v_dict[f'y{k + 1}'] = float(yOP[k])

        math_problem: MTI_Problem = MTI_Problem(eqs_str=eqs, v_dict=v_dict)

        evals, _ = _run_driver(math_problem)
        print(f"  eigenvalues: {evals}")
        finite_evals = evals[np.isfinite(evals)]
        max_real = float(np.max(np.real(finite_evals))) if len(finite_evals) > 0 else 0.0
        print(f"  max_real(finite): {max_real:.6f}")
        self.assertLess(max_real, 0, "The nonlinear 5x5 system should be stable.")

    def test_nonlinear_unstable_manual_system(self) -> None:
        """
        Test 3: Implicit Descriptor System (State-Dependent E-Matrix).
        Verifies the builder's capability to handle implicit formulations where the
        state derivative is multiplied by an algebraic variable (e.g., dx1*y1).
        This tests the proper assembly of a non-identity E matrix.
        Expects an unstable margin of 0.400.

        :return: None
        :rtype: None
        """
        eqs: List[str] = list([
            "3*dx1*y1 + 6*(1/2+1/2*z1)*(2/3-1/3*u1)*x1",
            "y1 - dx1"
        ])
        v_dict: Dict[str, float] = dict({
            'xp1': 0.0,
            'x1': 2.0,
            'u1': 3.0,
            'y1': 4.0,
            'z1': 5.0
        })

        math_problem: MTI_Problem = MTI_Problem(eqs_str=eqs, v_dict=v_dict)
        builder: PolynomialMatrixBuilder = PolynomialMatrixBuilder(problem=math_problem, verbose=False)

        builder.linearize(v_dict)
        res: Tuple[Any, Any, Any, Any, bool, float] = builder.compute_stability()
        stable: bool = res[4]
        margin: float = res[5]

        if stable:
            self.fail("Expected the implicit system to be unstable based on the eigenvalues.")
        else:
            self.assertAlmostEqual(margin, 0.400, places=3, msg=f"Expected stability margin 0.4, got {margin:.3f}")

    def test_quadratic_system(self) -> None:
        """
        Test 4: Quadratic System (Multilinear Lifting of Powers).
        Validates that the algorithm correctly parses the Python power operator (**2),
        triggers the automatic multilinear lifting process, and successfully
        augments the matrices without truncating equations.
        Expects an unstable margin of 3.436.

        :return: None
        :rtype: None
        """
        eqs: List[str] = list([
            "xp1 - x1**2 - x2",
            "xp2 - x1*x2 - 1"
        ])
        v_dict: Dict[str, float] = dict({
            'xp1': 0.0,
            'xp2': 0.0,
            'x1': 1.0,
            'x2': 3.5
        })

        math_problem: MTI_Problem = MTI_Problem(eqs_str=eqs, v_dict=v_dict)

        evals, _ = _run_driver(math_problem)
        print(f"  eigenvalues: {evals}")
        finite_evals = evals[np.isfinite(evals)]
        max_real = float(np.max(np.real(finite_evals))) if len(finite_evals) > 0 else 0.0
        print(f"  max_real(finite): {max_real:.6f}")
        self.assertGreater(max_real, 0, "The quadratic system should be unstable.")

    def test_inverse_system(self) -> None:
        """
        Test 5: Inverse Nonlinear System (Fractional Lifting).
        Verifies the automatic handling of inverse states (e.g., 1/x2). Tests if the
        builder correctly generates the internal multiplier auxiliary variables and
        their algebraic coupling constraints (x2 * y1 - 1 = 0) to yield a stable
        margin of -1.000.

        :return: None
        :rtype: None
        """
        eqs: List[str] = list([
            "xp1 + x1 - y1",
            "xp2 + x2 - 2",
            "y1 - 1/x2"
        ])
        v_dict: Dict[str, float] = dict({
            'xp1': -1.0,
            'xp2': 1.0,
            'x1': 2.0,
            'x2': 1.0
        })

        math_problem: MTI_Problem = MTI_Problem(eqs_str=eqs, v_dict=v_dict)

        # Driver path
        evals, _ = _run_driver(math_problem)
        print(f"  driver eigenvalues: {evals}")
        finite_evals = evals[np.isfinite(evals)]
        max_real = float(np.max(np.real(finite_evals))) if len(finite_evals) > 0 else 0.0
        print(f"  max_real(finite): {max_real:.6f}")

        # Builder path (kept because xp1=-1.0, xp2=1.0 are non-zero)
        builder: PolynomialMatrixBuilder = PolynomialMatrixBuilder(problem=math_problem, verbose=False)
        builder.linearize(v_dict)
        res: Tuple[Any, Any, Any, Any, bool, float] = builder.compute_stability()
        stable: bool = res[4]
        margin: float = res[5]

        if not stable:
            self.fail("Expected the fractional system to be stable based on the eigenvalues.")
        else:
            self.assertAlmostEqual(margin, -1.000, places=3, msg=f"Expected stability margin -1.0, got {margin:.3f}")


    def test_sqrt(self) -> None:
        """
        Test 6:

        :return: None
        :rtype: None
        """
        eqs: List[str] = list([
            "sqrt(x1) + x1*y1+1",
            "2*xp1+x1",
            "y1-x1**2-1"
        ])
        v_dict: Dict[str, float] = dict({
            'xp1': 0.5,
            'x1': 1.0
        })

        math_problem: MTI_Problem = MTI_Problem(eqs_str=eqs, v_dict=v_dict)

        # Driver path
        evals, _ = _run_driver(math_problem)
        print(f"  driver eigenvalues: {evals}")
        finite_evals = evals[np.isfinite(evals)]
        max_real = float(np.max(np.real(finite_evals))) if len(finite_evals) > 0 else 0.0
        print(f"  max_real(finite): {max_real:.6f}")

        # Builder path (kept because xp1=0.5 is non-zero)
        builder: PolynomialMatrixBuilder = PolynomialMatrixBuilder(problem=math_problem, verbose=False)
        builder.linearize(v_dict)
        res: Tuple[Any, Any, Any, Any, bool, float] = builder.compute_stability()
        stable: bool = res[4]
        margin: float = res[5]

        if not stable:
            self.fail("Expected the fractional system to be stable based on the eigenvalues.")
        else:
            self.assertAlmostEqual(margin, -0.500, places=3, msg=f"Expected stability margin -0.5, got {margin:.3f}")

    def test_imti_paper_nonlinear_rl_circuit(self) -> None:
        """
        Test 7: Non-linear RL Circuit with Magnitude Feedback (iMTI Paper inspired).

        This test implements a physical use-case modeled after the iMTI CPN
        methodology described in the Small-Signal Stability paper.
        It evaluates a 2-state system where the algebraic magnitude of the
        states feeds back into the derivatives, requiring exact analytic
        linearization of the non-polynomial 'sqrt' function to find the
        true system eigenvalues.

        :return: None
        :rtype: None
        """
        # 1. ALGORITHM: Define the system equations
        # Eq 1 & 2: Differential equations (dx/dt) with cross-coupled non-linear feedback
        # Eq 3: The algebraic constraint for the magnitude (Euclidean norm)
        eqs: List[str] = list([
            "xp1 + x1 + 0.1*y1 - 3.5",
            "xp2 + x2 - 0.2*y1 - 3.0",
            "y1 - sqrt(x1**2 + x2**2)"
        ])

        # 2. ALGORITHM: Define the exact steady-state operating point
        # We use a 3-4-5 Pythagorean triple to ensure clean, rational analytics.
        # sqrt(3^2 + 4^2) = 5.0
        v_dict: Dict[str, float] = dict({
            'x1': 3.0,
            'x2': 4.0,
            'y1': 5.0,
            'xp1': 0.0,
            'xp2': 0.0
        })

        # 3. Build the mathematical problem mapping
        math_problem: MTI_Problem = MTI_Problem(eqs_str=eqs, v_dict=v_dict)

        # 4. Execute via driver
        evals, _ = _run_driver(math_problem)
        print(f"  driver eigenvalues: {evals}")
        finite_evals = evals[np.isfinite(evals)]
        max_real = float(np.max(np.real(finite_evals))) if len(finite_evals) > 0 else 0.0
        print(f"  max_real(finite): {max_real:.6f}")
        self.assertLess(max_real, 0, "Expected the RL circuit to be stable.")

    def test_PLL_example_5_2_automatic_lifting(self) -> None:
        """
        Test: Evaluates the automatic Polynomial-Matrix conversion for the PLL.
        The user provides pure trigonometric non-linearities, and the engine
        MUST perform the multilinear lifting automatically under the hood. This example can be found in
        Small-Signal Stability Analysis of Power Systems
        by Implicit Multilinear Models
        by C. Kaufmann et all.
        """
        kp_val: float = 0.5
        ki_val: float = 9.0


        eqs: List[str] = [
            f"xp1 - x2 - {kp_val}*(-u1*sin(x1) + u2*cos(x1))",
            f"xp2 - {ki_val}*(-u1*sin(x1) + u2*cos(x1))"
        ]

        import math
        theta_eq: float = 4.0 * math.pi / 180.0

        v_dict: Dict[str, float] = {
            'x1': theta_eq,
            'x2': 0.0,
            'xp1': 0.0,
            'xp2': 0.0,
            'u1': 325.2059,
            'u2': 22.7406,
        }


        math_problem: MTI_Problem = MTI_Problem(eqs_str=eqs, v_dict=v_dict)

        # Driver path
        evals, _ = _run_driver(math_problem)
        print(f"  driver eigenvalues: {evals}")
        finite_evals = evals[np.isfinite(evals)]
        found_dominant: bool = False
        found_secondary: bool = False
        for ev in finite_evals:
            if abs(ev.real - (-20.6046)) < 0.01:
                found_dominant = True
            if abs(ev.real - (-142.3954)) < 0.01:
                found_secondary = True

        self.assertTrue(found_dominant, "Dominant pole at -20.6046 not found.")
        self.assertTrue(found_secondary, "Secondary pole at -142.3954 not found.")

    '''def test_vg_1(self) -> None:
        """
        Test:
        """
        problem = define_problem()
        calculated_evs = []
        if problem:
            calculated_evs = solve_problem(problem)

        reference_evs = [complex(x) for x in
                         ["(-20.000000001+0j)", "(-10.000000001+0j)", "(-50.00000013636582+0j)",
                          "(-49.9999998638391+0j)",
                          "(-12.168789431034414+0j)", "(-10.142655446318065+0j)", "(-6.854035294550814+0j)",
                          "(-5.029702766536969+0j)", "(-3.6156680497265823+0j)", "(0.7258649757244431+0j)",
                          "(-1.8851882786542178+0j)", "(-1.719135993985083+0.32085026247445086j)",
                          "(-1.7191359939850832-0.3208502624744508j)", "(-0.49201422052160765+0.8707873679911879j)",
                          "(-0.49201422052160765-0.870787367991188j)", "(-0.48381848641530006+0.465928959924564j)",
                          "(-0.48381848641530006-0.46592895992456396j)", "(-0.10000000099999705+0j)",
                          "(-0.11161483998140341+0j)", "(-0.9999999991690246+0j)", "(-1.0000000028309748+0j)",
                          "(-0.39635072300328184+0j)", "(-10.00000000099956+0j)", "(-1.0000000000000023e-09+0j)",
                          "(-9.999999999999996e-10+0j)", "(-9.999999999999992e-10+0j)"]]

        finite_evs = calculated_evs[np.isfinite(calculated_evs) & (np.abs(calculated_evs) < 1e6)] if len(
            calculated_evs) > 0 else []

        tol = 0.06  # Tolerancia para considerar que coinciden
        matched_calc = []
        unmatched_calc = []
        unmatched_ref = list(reference_evs)
        if len(finite_evs) > 0 and len(reference_evs) > 0:
            # 1. Crear matriz de distancias (costos)
            cost_matrix = np.zeros((len(finite_evs), len(reference_evs)))
            for i, c_ev in enumerate(finite_evs):
                for j, r_ev in enumerate(reference_evs):
                    cost_matrix[i, j] = np.abs(c_ev - r_ev)

            # 2. Asignación óptima
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 3. Filtrar por tolerancia
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < tol:
                    matched_calc.append(finite_evs[i])
                    unmatched_ref[j] = None  # Marcamos como encontrado
                else:
                    unmatched_calc.append(finite_evs[i])

            # 4. Limpiar listas de no emparejados
            unmatched_ref = [x for x in unmatched_ref if x is not None]

            # Añadir los calculados que ni siquiera entraron en la asignación
            for i in range(len(finite_evs)):
                if i not in row_ind:
                    unmatched_calc.append(finite_evs[i])
        self.assertTrue(len(unmatched_ref)==0, f"# unmatched eigenvalues: {len(unmatched_ref)}")'''

if __name__ == "__main__":
    # Standard unittest execution
    unittest.main(buffer=True)
