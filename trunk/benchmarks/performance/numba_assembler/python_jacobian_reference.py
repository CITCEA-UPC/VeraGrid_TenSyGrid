# SPDX-License-Identifier: MPL-2.0
# Previous Python implementation retained only as a benchmark reference.
import numpy as np
from scipy import sparse
def _compute_jacobian_sparse_from_S(S: sparse.csc_matrix, v: np.ndarray) -> sparse.csc_matrix:
    """Sparse multilinear Jacobian equivalent to PolynomialMatrixBuilder logic."""
    n_vars, n_mon = S.shape
    data = S.data
    indices = S.indices
    indptr = S.indptr
    col_prod = np.ones(n_mon, dtype=float)
    for j in range(n_mon):
        start = indptr[j]
        end = indptr[j + 1]
        prod = 1.0
        for p in range(start, end):
            i = indices[p]
            s_ij = data[p]
            prod *= s_ij * v[i] + (1.0 - abs(s_ij))
        col_prod[j] = prod
    f_row: list[int] = []
    f_col: list[int] = []
    f_val: list[float] = []
    for j in range(n_mon):
        start = indptr[j]
        end = indptr[j + 1]
        if start == end:
            continue
        zero_hits = 0
        zero_pos = -1
        for p in range(start, end):
            i = indices[p]
            s_ij = data[p]
            x_ij = s_ij * v[i] + (1.0 - abs(s_ij))
            if abs(x_ij) <= 1e-12:
                zero_hits += 1
                zero_pos = p
        for p in range(start, end):
            i = indices[p]
            s_ij = data[p]
            x_ij = s_ij * v[i] + (1.0 - abs(s_ij))
            if abs(x_ij) > 1e-12:
                val = s_ij * col_prod[j] / x_ij
                f_row.append(i)
                f_col.append(j)
                f_val.append(val)
            elif zero_hits == 1 and p == zero_pos:
                prod_other = 1.0
                for q in range(start, end):
                    if q == p:
                        continue
                    iq = indices[q]
                    s_iq = data[q]
                    x_iq = s_iq * v[iq] + (1.0 - abs(s_iq))
                    prod_other *= x_iq
                val = s_ij * prod_other
                f_row.append(i)
                f_col.append(j)
                f_val.append(val)
    return sparse.csc_matrix((f_val, (f_row, f_col)), shape=(n_vars, n_mon))
