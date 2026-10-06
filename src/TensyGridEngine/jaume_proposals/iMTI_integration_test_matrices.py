import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ==============================================================================
# 1. GENERADOR DEL ARCHIVO .MAT DE PRUEBA
# ==============================================================================
def generar_mat_prueba(nombre_archivo="sistema_cpn1_prueba.mat"):
    # Matriz E: Singular (Fila 3 de ceros -> Variable algebraica y)
    E = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]
    ])

    # Matriz A0: Parte lineal base
    A0 = np.array([
        [-2.0, 1.0, 1.0],
        [1.0, -3.0, -1.0],
        [1.0, 2.0, -4.0]  # Nota: A22 = -4.0 (Garantiza Indice-1 / det != 0)
    ])

    # Formulación CPN1 con 1 sola matriz Phi:
    # Usamos la identidad x1*x2 = 0.25*(x1+x2)^2 - 0.25*(x1-x2)^2
    # Términos proyectados (Phi * z): [ (x1 + x2), (x1 - x2), y ]
    Phi = np.array([
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Matriz S: Selecciona y pondera los términos al cuadrado
    S = np.array([
        [0.25, -0.25, 0.0],  # Ecuación 1: +0.25*(x1+x2)^2 - 0.25*(x1-x2)^2
        [0.00, 0.00, 1.0],  # Ecuación 2: + 1.0*y^2
        [0.00, 0.00, 0.0]  # Ecuación 3: Pura algebraica lineal
    ])

    # Estado inicial consistente con la restricción algebraica (0 = x1 + 2*x2 - 4*y)
    # Si x1(0)=1.0, x2(0)=0.5 -> y(0) = (1 + 2*0.5)/4 = 0.5
    z0 = np.array([1.0, 0.5, 0.5])

    # Guardar en archivo .mat
    sio.savemat(nombre_archivo, {
        'E': E,
        'A0': A0,
        'S': S,
        'Phi': Phi,
        'z0': z0,
        'h': 0.01,
        't_final': 2.0
    })
    print(f"Archivo de prueba '{nombre_archivo}' generado correctamente.")


# ==============================================================================
# 2. CLASE INTEGRADORA CPN1 (TRAPECIO IMPLÍCITO)
# ==============================================================================
class CPN1TrapezoidalIntegrator:
    def __init__(self, E, A0, S, Phi):
        self.E = E
        self.A0 = A0
        self.S = S
        self.Phi = Phi
        self.is_sparse = sp.issparse(E) or sp.issparse(A0)

    def eval_f(self, z):
        p = self.Phi @ z
        return self.A0 @ z + self.S @ (p * p)

    def eval_jacobian(self, z):
        p = self.Phi @ z
        diag_p = sp.diags(p) if self.is_sparse else np.diag(p)
        return self.A0 + 2.0 * (self.S @ (diag_p @ self.Phi))

    def step(self, z_k, h, tol=1e-8, max_iter=15):
        f_k = self.eval_f(z_k)
        b_k = self.E @ z_k + (0.5 * h) * f_k
        z_next = z_k.copy()

        for _ in range(max_iter):
            f_next = self.eval_f(z_next)
            residual = self.E @ z_next - (0.5 * h) * f_next - b_k

            if np.linalg.norm(residual, ord=np.inf) < tol:
                return z_next

            J_f = self.eval_jacobian(z_next)
            J_trap = self.E - (0.5 * h) * J_f

            delta_z = spla.spsolve(J_trap, -residual) if self.is_sparse else np.linalg.solve(J_trap, -residual)
            z_next += delta_z

        raise RuntimeError("Newton-Raphson no convergió.")


# ==============================================================================
# 3. PRUEBA DE EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":
    archivo_mat = "sistema_cpn1_prueba.mat"

    # 1. Crear el .mat
    generar_mat_prueba(archivo_mat)

    # 2. Cargar variables del .mat
    mat_data = sio.loadmat(archivo_mat)
    E = mat_data['E']
    A0 = mat_data['A0']
    S = mat_data['S']
    Phi = mat_data['Phi']
    z_actual = mat_data['z0'].flatten()
    h = float(mat_data['h'].squeeze())
    t_final = float(mat_data['t_final'].squeeze())

    # 3. Inicializar e integrar
    solver = CPN1TrapezoidalIntegrator(E, A0, S, Phi)
    num_pasos = int(t_final / h)

    print(f"\nEjecutando integración ({num_pasos} pasos temporal)...")
    print(f"Estado Inicial t=0.00s  -> z = {z_actual}")

    for k in range(1, num_pasos + 1):
        z_actual = solver.step(z_actual, h)
        if k % 50 == 0:
            print(f"Estado en t={k * h:.2f}s     -> z = {np.round(z_actual, 4)}")

    print("\n¡Simulación finalizada exitosamente sin fallos por matriz singular!")