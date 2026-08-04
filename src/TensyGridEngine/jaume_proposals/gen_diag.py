import sys
import numpy as np
from scipy import sparse, linalg as spla
import matplotlib.pyplot as plt


def _unique_pairs(n, k, rng, forbidden=None):
    """Generate k unique (row, col) pairs in [0, n-1] avoiding forbidden set."""
    max_possible = n * n
    k = min(k, max_possible)
    if forbidden is not None and len(forbidden) > 0:
        forbidden_arr = np.array(list(forbidden), dtype=np.int64)
        available = np.setdiff1d(np.arange(max_possible, dtype=np.int64), forbidden_arr)
        k = min(k, len(available))
        flat = rng.choice(available, size=k, replace=False)
    else:
        flat = rng.choice(max_possible, size=k, replace=False)
    return flat // n, flat % n


def build_matrices(n, density, rng=None):
    """Build two n x n sparse matrices E and A with given density.

    Both have exactly ~density * n^2 nonzeros (unique coordinates).
    E: has at least density*n ones on the diagonal, rest off-diagonal random.
    A: fully random sparse matrix with values in [-1, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    total_nnz = max(1, min(n * n, int(np.round(density * n * n))))
    n_diag = max(1, min(total_nnz, int(np.round(density * n))))

    # E: diagonal ones + unique off-diagonal random entries
    diag_idx = rng.choice(n, size=n_diag, replace=False)
    all_diag_flat = set(int(i * n + i) for i in range(n))

    remaining = total_nnz - n_diag
    if remaining > 0:
        od_rows, od_cols = _unique_pairs(n, remaining, rng, forbidden=all_diag_flat)
        rows_E = np.concatenate([diag_idx, od_rows])
        cols_E = np.concatenate([diag_idx, od_cols])
        vals_E = np.concatenate([np.ones(n_diag), rng.uniform(-1, 1, size=remaining)])
    else:
        rows_E = diag_idx
        cols_E = diag_idx
        vals_E = np.ones(n_diag)

    E = sparse.csr_matrix((vals_E, (rows_E, cols_E)), shape=(n, n))

    # A: fully random unique pairs
    rows_A, cols_A = _unique_pairs(n, total_nnz, rng)
    vals_A = rng.uniform(-1, 1, size=total_nnz)
    A = sparse.csr_matrix((vals_A, (rows_A, cols_A)), shape=(n, n))

    return E, A


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    density = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    # Semilla fijada para reproducibilidad
    rng = np.random.default_rng(42)
    E, A = build_matrices(n, density, rng=rng)

    print(f"{'='*60}")
    print(f"  n={n}, density={density}")
    print(f"{'='*60}")
    print(f"  E: {E.shape}, nnz={E.nnz}, diag ones={E.diagonal().sum():.0f}")
    print(f"  A: {A.shape}, nnz={A.nnz}, density={A.nnz / (n * n):.4f}")

    A_dense = A.toarray()
    E_dense = E.toarray()

    # --- APROXIMACIÓN POR SERIE DE NEUMANN PARA E^-1 A ---
    # E = I - R  =>  R = I - E
    # E^-1 ≈ I + R + R^2 + ...
    I = np.eye(n)
    R = I - E_dense

    t0 = A_dense
    t1 = R @ A_dense
    t2 = (R @ R) @ A_dense

    # Operador aproximado a 2º orden: T2 ≈ E^-1 A
    T2 = t0 + t1 + t2

    # Autovalores calculados DIRECTAMENTE sobre la matriz sumada T2
    lambd_T2 = spla.eigvals(T2)

    # Autovalores generalizados exactos (Ax = lambda Ex)
    gen = spla.eigvals(A_dense, E_dense)

    # Ordenamiento lexicográfico para comparar errores
    idx_T2 = np.lexsort((lambd_T2.imag, lambd_T2.real))
    idx_gen = np.lexsort((gen.imag, gen.real))
    mismatch_t2 = np.max(np.abs(lambd_T2[idx_T2] - gen[idx_gen]))

    print(f"\n  tr(T0)                = {np.trace(t0):+.4e}")
    print(f"  tr(T1)                = {np.trace(t1):+.4e}")
    print(f"  tr(T2_term)           = {np.trace(t2):+.4e}")
    print(f"  sum eig(T2)           = {lambd_T2.sum():+.4e}")
    print(f"  sum eig(A, E)         = {gen.sum():+.4e}")
    print(f"  max|eig(T2)-eig(A,E)| = {mismatch_t2:.4e}")

    # --- GRÁFICA ---
    plt.figure(figsize=(8, 6))
    plt.scatter(
        lambd_T2.real,
        lambd_T2.imag,
        marker="o",
        label="T2 = (I + R + R^2)A",
        alpha=0.7,
        color="tab:blue",
    )
    plt.scatter(
        gen.real,
        gen.imag,
        marker="x",
        label="eig(A, E) Exacto",
        alpha=0.8,
        color="tab:orange",
        s=70,
    )
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.xlabel("Real")
    plt.ylabel("Imaginario")
    plt.title(f"Autovalores: Serie Neumann vs Exacto Generalizado (n={n}, density={density})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()