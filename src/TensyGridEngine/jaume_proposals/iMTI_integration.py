import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class CPN1TrapezoidalIntegrator:
    """
    Motor de integración Trapecio Implícito optimizado para modelos CPN1
    con una sola matriz Phi.
    """

    def __init__(self, E, A0, S, Phi):
        self.E = E
        self.A0 = A0
        self.S = S
        self.Phi = Phi
        # Detectar si las matrices son dispersas (sparse)
        self.is_sparse = sp.issparse(E) or sp.issparse(A0)

    def eval_f(self, z):
        """Evalúa f(z) = A0*z + S * ((Phi*z) ^ 2) mediante producto de Hadamard."""
        p = self.Phi @ z
        return self.A0 @ z + self.S @ (p * p)

    def eval_jacobian(self, z):
        """Calcula el Jacobiano analítico df/dz = A0 + 2 * S * diag(Phi * z) * Phi."""
        p = self.Phi @ z

        if self.is_sparse:
            diag_p = sp.diags(p)
        else:
            diag_p = np.diag(p)

        return self.A0 + 2.0 * (self.S @ (diag_p @ self.Phi))

    def step(self, z_k, h, tol=1e-8, max_iter=15):
        """
        Calcula el siguiente paso de tiempo t_{k+1} mediante Newton-Raphson.
        """
        # Términos conocidos del lado derecho en t_k
        f_k = self.eval_f(z_k)
        b_k = self.E @ z_k + (0.5 * h) * f_k

        # Semilla inicial
        z_next = z_k.copy()

        for i in range(max_iter):
            f_next = self.eval_f(z_next)

            # Residual F(z_{k+1}) = E*z_{k+1} - (h/2)*f(z_{k+1}) - b_k
            residual = self.E @ z_next - (0.5 * h) * f_next - b_k

            # Criterio de parada
            if np.linalg.norm(residual, ord=np.inf) < tol:
                return z_next

            # Matriz del sistema J = E - (h/2) * (df/dz)
            J_f = self.eval_jacobian(z_next)
            J_trap = self.E - (0.5 * h) * J_f

            # Resolver el sistema lineal J * delta_z = -residual
            if self.is_sparse:
                delta_z = spla.spsolve(J_trap, -residual)
            else:
                delta_z = np.linalg.solve(J_trap, -residual)

            z_next += delta_z

        raise RuntimeError(f"El método de Newton-Raphson no convergió en {max_iter} iteraciones.")


def cargar_archivo_mat(ruta_mat):
    """
    Carga las variables y matrices desde el archivo .mat de MATLAB.
    """
    mat_data = sio.loadmat(ruta_mat)

    # Extraer matrices (soporta nombres comunes con o sin guión bajo)
    E = mat_data['E']
    A0 = mat_data['A0'] if 'A0' in mat_data else mat_data['A_0']
    S = mat_data['S']
    Phi = mat_data['Phi'] if 'Phi' in mat_data else mat_data['PHI']

    # Vector de estado inicial (garantizar que sea un array 1D)
    z0 = mat_data['z0'].flatten() if 'z0' in mat_data else mat_data['z_0'].flatten()

    # Parámetros del tiempo de simulación (si están en el .mat, si no asigna valores por defecto)
    h = float(mat_data['h'].squeeze()) if 'h' in mat_data else 0.001
    t_final = float(mat_data['t_final'].squeeze()) if 't_final' in mat_data else 1.0

    return E, A0, S, Phi, z0, h, t_final


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":

    # 1. Nombre de tu archivo .mat
    archivo_mat = "sistema_cpn1.mat"

    try:
        print(f"Cargando datos desde '{archivo_mat}'...")
        E, A0, S, Phi, z0, h, t_final = cargar_archivo_mat(archivo_mat)

        print(f"  - Dimensión del sistema (n) : {len(z0)}")
        print(f"  - Paso de tiempo (h)         : {h} s")
        print(f"  - Tiempo final (t_final)     : {t_final} s")

        # 2. Inicializar el integrador CPN1
        integrador = CPN1TrapezoidalIntegrator(E, A0, S, Phi)

        # 3. Configurar simulación
        num_pasos = int(np.ceil(t_final / h))
        vector_tiempo = np.linspace(0, t_final, num_pasos + 1)

        # Matriz para almacenar la trayectoria (Filas: Tiempo, Columnas: Variables)
        resultados = np.zeros((num_pasos + 1, len(z0)))
        resultados[0, :] = z0

        z_actual = z0.copy()

        print(f"\nIniciando simulación ({num_pasos} pasos de tiempo)...")

        # 4. Bucle principal de simulación
        for k in range(num_pasos):
            z_actual = integrador.step(z_actual, h)
            resultados[k + 1, :] = z_actual

        print("¡Simulación completada con éxito!")

        # 5. Opcional: Guardar los resultados devueltos a un nuevo archivo .mat
        sio.savemat("resultados_simulacion.mat", {
            "t": vector_tiempo,
            "z": resultados
        })
        print("Resultados guardados en 'resultados_simulacion.mat'")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{archivo_mat}'. Revisa la ruta.")
    except KeyError as e:
        print(f"Error: Falta la variable {e} en el archivo .mat.")