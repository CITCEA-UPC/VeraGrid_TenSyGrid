from scipy.sparse import csr_matrix, csc_matrix
import time
import concurrent.futures
import numpy as np
from scipy.sparse.linalg import splu, LinearOperator, eigs

from trunk.tensygrid.PolynomialMatrixBuilder import PolynomialMatrixBuilder

class EigenvalueComputation(PolynomialMatrixBuilder.PolynomialMatrixBuilder):
    def compute_stability_sparse(self, sigma=2.0, k_modes=200):
        """
            Perform stability analysis for large-scale systems using sparse ARPACK.

            This method solves the generalized eigenvalue problem :math:`Av = \lambda Ev`
            using the Shift-and-Invert spectral transformation. It computes both right
            and left eigenvectors to derive the Participation Factor matrix, which
            quantifies the influence of state variables on specific system modes.

            The method employs LU factorization (SuperLU) for the linear operator and
            executes right and left eigensolvers in parallel to optimize performance.

            Parameters
            ----------
            sigma : float or complex, optional
                The shift point for the Shift-and-Invert transformation. Eigenvalues
                near this point will be found first. Defaults to 2.0.
            k_modes : int, optional
                The number of eigenvalues and eigenvectors to compute. Defaults to 200.

            Returns
            -------
            evals : numpy.ndarray
                Array of found physical eigenvalues (poles) :math:`\lambda`.
            V : numpy.ndarray
                Matrix where each column is a right eigenvector corresponding to `evals`.
            W : numpy.ndarray
                Matrix where each column is a matched left eigenvector.
            participation_matrix : numpy.ndarray
                Matrix of participation factors where entry :math:`P_{ji}` represents
                the participation of state :math:`j` in mode :math:`i`.
            stable : bool
                Boolean flag indicating system stability (True if all :math:`Re(\lambda) \leq 1e-5`).
            margin : float
                The maximum real part among all computed eigenvalues (stability margin).

            Notes
            -----
            - **Structural Regularization:** The method automatically patches empty rows/columns
              in the shifted matrix to prevent singularity during LU factorization.
            - **Mode Matching:** Since ARPACK may return left and right modes in different
              orders, the Hungarian Algorithm (Linear Sum Assignment) is used to align
              them based on eigenvalue proximity.
            - **Participation Factors:** Calculated as :math:`P_{ji} = \\frac{w_{ji} v_{ji}}{w_i^H v_i}`.
        """
        print(f"\n[+] Starting sparse ARPACK analysis (Shift-and-Invert, sigma={sigma}, k={k_modes})...")
        start = time.time()

        A_csr = csr_matrix(self.A)
        E_csr = csr_matrix(self.E)
        n_dims = A_csr.shape[0]

        print(f"    State-space dimension: {n_dims}")

        M_shift = A_csr - sigma * E_csr
        M_shift_lil = M_shift.tolil()

        # Structural regularization
        row_norms = np.array(np.abs(M_shift).max(axis=1).todense()).flatten()
        col_norms = np.array(np.abs(M_shift).max(axis=0).todense()).flatten()

        zero_rows = np.where(row_norms < 1e-10)[0]
        zero_cols = np.where(col_norms < 1e-10)[0]

        if len(zero_rows) > 0 or len(zero_cols) > 0:
            print(f"    Patching {len(zero_rows)} empty rows and {len(zero_cols)} empty columns...")
            for r in zero_rows: M_shift_lil[r, r] = -1.0
            for c in zero_cols: M_shift_lil[c, c] = -1.0

        # 1. LU factorization (the heavy step)
        print(f"    Factoring (A - {sigma}*E) using SuperLU...")
        try:
            lu_solver = splu(M_shift_lil.tocsc())
        except Exception as e:
            print(f"    [!] SuperLU factorization error: {e}")
            return [], None, None, None, False, 0.0

        # 2. Right operator (V eigenvectors)
        def matvec_right(v):
            y = E_csr.dot(v)
            return lu_solver.solve(y.real) + 1j * lu_solver.solve(y.imag)

        OP_R = LinearOperator((n_dims, n_dims), matvec=matvec_right, dtype=complex)

        print(f"    Solving right/left modes in parallel...")

        def matvec_left(v):
            y = E_csr.T.dot(v)
            # Use trans='T' to solve the transposed system almost for free
            return lu_solver.solve(y.real, trans='T') + 1j * lu_solver.solve(y.imag, trans='T')

        OP_L = LinearOperator((n_dims, n_dims), matvec=matvec_left, dtype=complex)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_r = executor.submit(eigs, OP_R, k=k_modes, which='LM')
                future_l = executor.submit(eigs, OP_L, k=k_modes, which='LM')
                evals_nu_R, evecs_right_raw = future_r.result()
                evals_nu_L, evecs_left_raw = future_l.result()
        except Exception as e:
            print(f"    [!] Parallel ARPACK solve failed: {e}")
            return [], None, None, None, False, 0.0

        evals_R = sigma + (1.0 / evals_nu_R)
        evals_L = sigma + (1.0 / evals_nu_L)

        # 4. Mode matching
        # ARPACK can return modes in a different order. Use the Hungarian algorithm to align them.
        from scipy.optimize import linear_sum_assignment
        dist_matrix = np.abs(evals_R[:, None] - evals_L[None, :])
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        evecs_left_matched = evecs_left_raw[:, col_ind]

        # 5. Transform and filter to physical poles
        valid_nu = np.abs(evals_nu_R) > 1e-10
        evals = evals_R[valid_nu]
        V = evecs_right_raw[:, valid_nu]
        W = evecs_left_matched[:, valid_nu]

        finite_mask = np.isfinite(evals)
        evals = evals[finite_mask]
        V = V[:, finite_mask]
        W = W[:, finite_mask]

        # 6. Participation matrix calculation: P_ji = (W_ji * V_ji) / (W_i^T * V_i)
        dot_products = np.sum(W * V, axis=0)
        participation_matrix = (W * V) / dot_products

        # 7. Stability evaluation
        margin = float(np.max(evals.real)) if len(evals) > 0 else 0.0
        stable = margin <= 1e-5

        print(f"[+] Sparse ARPACK completed in {time.time() - start:.2f} seconds.")
        print(f"    Found {len(evals)} physical poles with their participation factors.")

        return evals, V, W, participation_matrix, stable, margin