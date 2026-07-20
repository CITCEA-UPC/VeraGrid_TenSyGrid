import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, LinearOperator, eigs
import time

# 1. Load matrices
A_csr = sp.load_npz("precomputed_matrices/ieee118_grid_A.npz")
E_csr = sp.load_npz("precomputed_matrices/ieee118_grid_E.npz")
n_dims = A_csr.shape[0]

print(f"Loaded Grid State-Space with {n_dims} dimensions.")
start = time.time()

# 2. Shift parameter (Zooming in on the unstable right-half plane)
sigma = 2.0
k_modes = 200

# 3. Form the Shift Matrix: M_shift = A - sigma * E
M_shift = A_csr - sigma * E_csr
M_shift_lil = M_shift.tolil()

# 4. Strict Structural Regularization (Prevent SuperLU from crashing on Ghost states)
# If a row or column in M_shift is completely zero, the matrix is structurally singular.
row_norms = np.array(np.abs(M_shift).max(axis=1).todense()).flatten()
col_norms = np.array(np.abs(M_shift).max(axis=0).todense()).flatten()

zero_rows = np.where(row_norms < 1e-10)[0]
zero_cols = np.where(col_norms < 1e-10)[0]

if len(zero_rows) > 0 or len(zero_cols) > 0:
    print(f"Patching {len(zero_rows)} dead rows and {len(zero_cols)} dead columns...")

for r in zero_rows:
    M_shift_lil[r, r] = -1.0  # Force dummy equation: -1.0 * x = 0
for c in zero_cols:
    M_shift_lil[c, c] = -1.0

M_shift_csc = M_shift_lil.tocsc()

# 5. Factorize (A - sigma*E) using robust SuperLU
print(f"Factorizing (A - {sigma}*E) using SuperLU...")
try:
    lu_solver = splu(M_shift_csc)
except Exception as e:
    print(f"SuperLU Factorization failed! {e}")
    exit()

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, LinearOperator, eigs
import time

# 1. Load matrices
A_csr = sp.load_npz("precomputed_matrices/ieee118_grid_A.npz")
E_csr = sp.load_npz("precomputed_matrices/ieee118_grid_E.npz")
n_dims = A_csr.shape[0]

print(f"Loaded Grid State-Space with {n_dims} dimensions.")
start = time.time()

# 2. Shift parameter (Zooming in on the unstable right-half plane)
sigma = 2.0
k_modes = 200

# 3. Form the Shift Matrix: M_shift = A - sigma * E
M_shift = A_csr - sigma * E_csr
M_shift_lil = M_shift.tolil()

# 4. Strict Structural Regularization (Prevent SuperLU from crashing on Ghost states)
row_norms = np.array(np.abs(M_shift).max(axis=1).todense()).flatten()
col_norms = np.array(np.abs(M_shift).max(axis=0).todense()).flatten()

zero_rows = np.where(row_norms < 1e-10)[0]
zero_cols = np.where(col_norms < 1e-10)[0]

if len(zero_rows) > 0 or len(zero_cols) > 0:
    print(f"Patching {len(zero_rows)} dead rows and {len(zero_cols)} dead columns...")

for r in zero_rows:
    M_shift_lil[r, r] = -1.0
for c in zero_cols:
    M_shift_lil[c, c] = -1.0

M_shift_csc = M_shift_lil.tocsc()

# 5. Factorize (A - sigma*E) using robust SuperLU
print(f"Factorizing (A - {sigma}*E) using SuperLU...")
try:
    lu_solver = splu(M_shift_csc)
except Exception as e:
    print(f"SuperLU Factorization failed! {e}")
    exit()

# 6. Define the Custom Linear Operator
def matvec(v):
    y = E_csr.dot(v)
    sol_real = lu_solver.solve(y.real)
    sol_imag = lu_solver.solve(y.imag)
    return sol_real + 1j * sol_imag

OP = LinearOperator((n_dims, n_dims), matvec=matvec, dtype=complex)

# 7. Solve for the largest magnitude eigenvalues of the Operator
print(f"Solving for top {k_modes} poles near shift={sigma}...")
evals_nu, _ = eigs(OP, k=k_modes, which='LM')

# 8. Transform back to original physical eigenvalues: lambda = sigma + 1 / nu
valid_nu = np.abs(evals_nu) > 1e-10
evals_nu = evals_nu[valid_nu]
evals = sigma + (1.0 / evals_nu)

# 9. Filter and Sort
evals = evals[np.isfinite(evals)]
unstable_poles = evals[evals.real > 1e-5]

print("\n" + "="*40)
print(f"EXECUTION TIME: {time.time() - start:.2f} seconds")
print(f"Found {len(evals)} physical poles near {sigma}.")

if len(unstable_poles) > 0:
    print(f"🚨 SYSTEM IS UNSTABLE. Found {len(unstable_poles)} unstable poles:")
    for ev in sorted(unstable_poles, key=lambda x: x.real, reverse=True)[:15]:
        print(f"  {ev.real:.4f} + {ev.imag:.4f}j")
else:
    print("✅ System is stable near this shift.")