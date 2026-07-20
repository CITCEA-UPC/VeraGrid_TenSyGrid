import sympy as sp
import numpy as np
from typing import List, Dict, Optional, Any
import time
import re

from VeraGridEngine.Utils.Symbolic.symbolic import (Var, Const, Expr)
from VeraGridEngine.Utils.Symbolic.compiled_functions import SymbolicDerivative
from VeraGridEngine.Utils.Symbolic.jit_compiler import RMSCompiler
import VeraGridEngine.api as vge
from VeraGridEngine.Simulations.Rms.problems.rms_problem_phasor import RmsProblemPhasor

dummy_grid = vge.MultiCircuit()
rms_options = vge.RmsOptions()
mti_results = vge.power_flow(dummy_grid)


class MTI_Problem(RmsProblemPhasor):
    """
    Mathematical DAE/ODE problem builder for TenSyGrid.

    This class handles the parsing of symbolic equations and maps operating points
    from an optionally provided dictionary (v_dict). It ensures that state, algebraic,
    and derivative variables (dx/xp) are correctly synchronized for stable
    linearization in the iMTI framework.
    """
    __slots__ = (
        "init_guess", "_state_vars", "_state_eqs", "_algebraic_vars",
        "_algebraic_eqs", "_diff_vars", "_variable_parameters",
        "_event_parameters_eqs", "_constant_parameters", "_parameters_values",
        "_compiler_names_dict", "_alias_names_dict", "_uid2idx_vars",
        "_uid2idx_params", "_uid2idx_diff", "_uid2idx_t", "_uid2idx_event_params",
        "_glob_time", "_n_event_params", "_dt", "_delta", "_state_algeb_vars",
        "_n_state", "_n_alg", "_n_algebraic", "_n_diff", "timings",
        "_derivative_fn", "_event_params_fn", "_rhs_algeb_fn", "_rhs_state_fn",
        # JIT/Runtime jacobian function placeholders used by RMS engine
        "_j11_fn", "_j12_fn", "_j21_fn", "_j22_fn"
    )

    VARS_NAME: str = "vars"
    VARIABLE_PARAMS_NAME: str = "vprms"
    CONSTANT_PARAMS_NAME: str = "cprms"
    DIFF_NAME: str = "diff"
    TIME_NAME: str = "glob_time"

    def __init__(self, eqs_str: List[str], v_dict: Optional[Dict[str, float]] = None):
        """
        Initializes the problem using a flat dictionary of operating points.

        :param eqs_str: List of mathematical equations as strings.
        :param v_dict: Optional dictionary mapping variable names (e.g., 'x1', 'dx1', 'y1')
                       to their numerical values at the operating point. If omitted,
                       all values default to 0.0 and can be set later via set_operating_point().
        """
        super().__init__(grid=dummy_grid, pf_results=mti_results, options=rms_options)

        # Default to an empty dictionary if no initial guess is provided yet
        if v_dict is None:
            v_dict = dict()

        all_eqs = " ".join(eqs_str)

        # --- 1. Dimension Discovery via Regex ---
        # Scan equations to find the highest index used for each variable category.
        # This ensures the internal vectors (vars, diff, params) are sized correctly.
        max_x = self._get_max_idx(r'\b(?:xp|dx|x)(\d+)\b', all_eqs, 0)
        max_y = self._get_max_idx(r'\by(\d+)\b', all_eqs, 0)
        max_z = self._get_max_idx(r'\bz(\d+)\b', all_eqs, 0)
        max_u = self._get_max_idx(r'\bu(\d+)\b', all_eqs, 0)

        # --- 2. Initialize Containers ---
        self.init_guess = {}
        self._state_vars, self._state_eqs = [], []
        self._algebraic_vars, self._algebraic_eqs = [], []
        self._diff_vars, self._constant_parameters = [], []
        # Backwards-compatible container for numeric constant parameter values
        self._constant_params = []
        self._variable_parameters, self._event_parameters_eqs = [], []
        self._parameters_values = []

        self._compiler_names_dict, self._alias_names_dict = {}, {}
        self._uid2idx_vars, self._uid2idx_params = {}, {}
        self._uid2idx_diff, self._uid2idx_t = {}, {}
        self._uid2idx_event_params = {}

        self._glob_time = Var(self.TIME_NAME)
        self._register_time_var()

        sym_dict: Dict[str, sp.Symbol] = {}
        var_dict: Dict[str, Var] = {}

        # --- 3. Build State and Differential Variables ---
        for i in range(max_x):
            idx = i + 1
            # Register state variable (x)
            v_x = Var(f"x{idx}")
            self._state_vars.append(v_x)
            self.init_guess[v_x.uid] = float(v_dict.get(f"x{idx}", 0.0))
            self._register_var(v_x, self.VARS_NAME, i, sym_dict, var_dict)

            # Register derivative variable (xp/dx)
            # This is critical for non-linear stability where f depends on x_dot
            v_xp = Var(f"xp{idx}")
            v_xp.base_var = v_x
            self._diff_vars.append(v_xp)

            # Extract derivative value: supports both 'xpN' and 'dxN' aliases
            val_xp = v_dict.get(f"xp{idx}", v_dict.get(f"dx{idx}", 0.0))
            self.init_guess[v_xp.uid] = float(val_xp)

            # JIT Compiler mapping for the diff vector
            self._compiler_names_dict[v_xp.uid] = f"{self.DIFF_NAME}[{i}]"
            self._alias_names_dict[v_xp.uid] = f"{self.DIFF_NAME}_{i}"
            self._uid2idx_diff[v_xp.uid] = i

            # Allow SymPy to recognize both notations during parsing
            for prefix in ["xp", "dx"]:
                name = f"{prefix}{idx}"
                sym_dict[name] = sp.Symbol(name)
                var_dict[name] = v_xp

        # --- 4. Build Algebraic and Output Variables ---
        # Map 'y' variables
        for i in range(max_y):
            idx = i + 1
            v_y = Var(f"y{idx}")
            self._algebraic_vars.append(v_y)
            self.init_guess[v_y.uid] = float(v_dict.get(f"y{idx}", 0.0))
            self._register_var(v_y, self.VARS_NAME, max_x + i, sym_dict, var_dict)

        # Map 'z' variables
        for i in range(max_z):
            idx = i + 1
            v_z = Var(f"z{idx}")
            self._algebraic_vars.append(v_z)
            self.init_guess[v_z.uid] = float(v_dict.get(f"z{idx}", 0.0))
            # Offset by max_x + max_y to keep the 'vars' vector contiguous
            self._register_var(v_z, self.VARS_NAME, max_x + max_y + i, sym_dict, var_dict)

        # --- 5. Build Constant Parameters (u) ---
        for i in range(max_u):
            idx = i + 1
            v_u = Var(f"u{idx}")
            self._constant_parameters.append(v_u)
            val_u = float(v_dict.get(f"u{idx}", 0.0))
            self._parameters_values.append(Const(val_u))
            # Maintain a plain float list for compatibility with older API expectations
            self._constant_params.append(val_u)
            self._register_param(v_u, i, sym_dict, var_dict)

        # --- 6. Equation Parsing and JIT Compilation ---
        self._parse_and_compile(eqs_str, sym_dict, var_dict)
        # Ensure constant params is a numpy array to match engine expectations
        try:
            self._constant_params = np.array(self._constant_params, dtype=float)
        except Exception:
            # Fallback to empty array if conversion fails
            self._constant_params = np.zeros(0, dtype=float)

        # Ensure the counts used by downstream code (inherited RMSProblemPhasor) are set
        # Correctly set counts expected by the RMS engine
        self._n_state = len(getattr(self, '_state_vars', []))
        self._n_alg = len(getattr(self, '_algebraic_vars', []))
        # Compute number of equations (should match number of algebraic+state eqs)
        n_eqs = len(getattr(self, '_state_eqs', [])) + len(getattr(self, '_algebraic_eqs', []))
        # Prefer uid2idx length, but ensure consistency: _n_vars should match the number of equations
        uid2len = len(getattr(self, '_uid2idx_vars', {})) if getattr(self, '_uid2idx_vars', None) is not None else 0
        listlen = self._n_state + self._n_alg
        self._n_vars = max(uid2len, listlen) if listlen > 0 else max(uid2len, n_eqs)
        self._n_algebraic = len(getattr(self, '_algebraic_eqs', []))

    def set_operating_point(self, v_dict: Dict[str, float]) -> None:
        """
        Updates the initial guess and parameter values dynamically.
        Use this to inject a v_dict after the problem has been instantiated.

        :param v_dict: Dictionary mapping variable names to their values.
        """
        # Update states (x)
        for v_x in self._state_vars:
            if v_x.name in v_dict:
                self.init_guess[v_x.uid] = float(v_dict[v_x.name])

        # Update derivatives (xp / dx)
        for v_xp in self._diff_vars:
            alt_name = v_xp.name.replace("xp", "dx")
            if v_xp.name in v_dict:
                self.init_guess[v_xp.uid] = float(v_dict[v_xp.name])
            elif alt_name in v_dict:
                self.init_guess[v_xp.uid] = float(v_dict[alt_name])

        # Update algebraic variables (y, z)
        for v_alg in self._algebraic_vars:
            if v_alg.name in v_dict:
                self.init_guess[v_alg.uid] = float(v_dict[v_alg.name])

        # Update constant parameters (u)
        for i, v_u in enumerate(self._constant_parameters):
            if v_u.name in v_dict:
                self._parameters_values[i] = Const(float(v_dict[v_u.name]))

    def get_E_matrix(self, x, dx):
        """
        Defensive override that ensures a valid E matrix is returned,
        sized (n_vars, n_vars) for compatibility with la.eig(A, -E).
        """
        try:
            if not isinstance(self._constant_params, np.ndarray):
                self._constant_params = np.asarray(self._constant_params, dtype=float)
        except Exception:
            self._constant_params = np.zeros(0, dtype=float)

        try:
            n_vars = getattr(self, '_n_vars', 0)
            if not isinstance(n_vars, int) or n_vars <= 0:
                n_from_lists = len(getattr(self, '_state_vars', [])) + len(getattr(self, '_algebraic_vars', []))
                self._n_vars = n_from_lists if n_from_lists > 0 else 0
                n_vars = self._n_vars
        except Exception:
            n_vars = 0

        if n_vars == 0:
            import scipy.sparse as sp
            return sp.csc_matrix((0, 0), dtype=np.float64)

        all_eqs = getattr(self, '_state_eqs', []) + getattr(self, '_algebraic_eqs', [])
        xdot = getattr(self, '_diff_vars', [])
        n_eqs = len(all_eqs)

        if n_eqs == 0:
            import scipy.sparse as sp
            return sp.eye(n_vars, format="csc", dtype=np.float64)

        try:
            from VeraGridEngine.Utils.Symbolic.compiled_functions import SymbolicJacobian
            E_call = SymbolicJacobian(
                eqs=all_eqs, variables=xdot,
                compiler_names_dict=self._compiler_names_dict,
                alias_names_dict=self._alias_names_dict,
                VARS_NAME=self.VARS_NAME, DIFF_NAME=self.DIFF_NAME,
                EVENT_PARAMS_NAME=self.VARIABLE_PARAMS_NAME,
                PARAMS_NAME=self.CONSTANT_PARAMS_NAME,
                static=True,
            )
            vp = getattr(self, '_variable_parameters_values', np.zeros(0, dtype=float))
            cp = getattr(self, '_constant_params', np.zeros(0, dtype=float))
            E_partial = E_call(x, dx, vp, cp, h=0).tocsc()
        except Exception:
            import scipy.sparse as sp
            return sp.eye(n_vars, format="csc", dtype=np.float64)

        uid2idx = getattr(self, '_uid2idx_vars', {}) or {}
        import scipy.sparse as sp
        E_value = sp.lil_matrix((n_vars, n_vars), dtype=np.float64)

        for j, dvar in enumerate(xdot):
            base_var = getattr(dvar, 'base_var', None)
            if base_var is None:
                continue
            col_idx = uid2idx.get(getattr(base_var, 'uid', None), None)
            if col_idx is not None and 0 <= col_idx < n_vars:
                col = E_partial[:, j]
                if col.shape[0] <= n_vars:
                    E_value[:col.shape[0], col_idx] += col

        return E_value.tocsc()

    def get_static_state_matrix(self, x, dx):
        """Return (n_vars, n_vars) square matrix compatible with la.eig(A, -E)."""
        n_vars = getattr(self, '_n_vars', 0)
        n_eqs = len(getattr(self, '_state_eqs', [])) + len(getattr(self, '_algebraic_eqs', []))
        if n_vars == 0 or n_eqs == 0:
            import scipy.sparse as sp
            return sp.eye(max(n_vars, n_eqs, 1), format="csc", dtype=np.float64)
        try:
            parent_res = super().get_static_state_matrix(x, dx)
            if parent_res.shape == (n_vars, n_vars):
                return parent_res
            if parent_res.shape[0] == n_vars:
                return parent_res
            # Pad rows to make (n_vars, n_vars)
            import scipy.sparse as sp
            if parent_res.shape[1] != n_vars:
                # Rebuild with correct column count
                return self._build_square_static_matrix(x, dx, n_vars)
            extra_rows = n_vars - parent_res.shape[0]
            if extra_rows > 0:
                padding = sp.csc_matrix((extra_rows, n_vars), dtype=np.float64)
                return sp.vstack([parent_res, padding], format="csc")
            return parent_res[:n_vars, :]
        except Exception:
            return self._build_square_static_matrix(x, dx, n_vars)

    def _build_square_static_matrix(self, x, dx, n_vars):
        """Build (n_vars, n_vars) static matrix by padding the parent's rectangular result."""
        try:
            parent_res = super().get_static_state_matrix(x, dx)
            import scipy.sparse as sp
            n_rows, n_cols = parent_res.shape
            if n_rows >= n_vars and n_cols >= n_vars:
                return parent_res[:n_vars, :n_vars]
            result = sp.lil_matrix((n_vars, n_vars), dtype=np.float64)
            result[:min(n_rows, n_vars), :min(n_cols, n_vars)] = parent_res[:min(n_rows, n_vars), :min(n_cols, n_vars)]
            return result.tocsc()
        except Exception:
            import scipy.sparse as sp
            return sp.eye(n_vars, format="csc", dtype=np.float64)

    def _get_max_idx(self, p: str, text: str, d: int) -> int:
        """Finds the maximum numeric index referenced in equations via regex."""
        found = [int(m) for m in re.findall(p, text)]
        return max(max(found, default=0), d)

    def _register_var(self, v: Var, base: str, idx: int, s_d: dict, v_d: dict) -> None:
        """Helper to register a variable in the compiler mapping dictionaries."""
        self._compiler_names_dict[v.uid] = f"{base}[{idx}]"
        self._alias_names_dict[v.uid] = f"{base}_{idx}"
        self._uid2idx_vars[v.uid] = idx
        s_d[v.name] = sp.Symbol(v.name)
        v_d[v.name] = v

    def _register_param(self, v: Var, idx: int, s_d: dict, v_d: dict) -> None:
        """Helper to register a constant parameter (u) in mapping dictionaries."""
        self._compiler_names_dict[v.uid] = f"{self.CONSTANT_PARAMS_NAME}[{idx}]"
        self._alias_names_dict[v.uid] = f"{self.CONSTANT_PARAMS_NAME}_{idx}"
        self._uid2idx_params[v.uid] = idx
        s_d[v.name] = sp.Symbol(v.name)
        v_d[v.name] = v

    def _register_time_var(self) -> None:
        """Maps the global time variable for the JIT compiler."""
        self._compiler_names_dict[self._glob_time.uid] = self.TIME_NAME
        self._alias_names_dict[self._glob_time.uid] = self.TIME_NAME
        self._uid2idx_t[self._glob_time.uid] = 0

    def _parse_and_compile(self, eqs: List[str], s_d: dict, v_d: dict) -> None:
        """
        Parses strings to symbolic residues and triggers JIT compilation.
        Converts SymPy expressions into VeraGrid Var structures safely.

        :param eqs: List of equation strings.
        :type eqs: List[str]
        :param s_d: Dictionary mapping strings to SymPy Symbols.
        :type s_d: dict
        :param v_d: Dictionary mapping strings to VeraGrid Var objects.
        :type v_d: dict
        :return: None
        :rtype: None
        """
        # Rule: Explicit initialization and no lambdas.
        # We pass our explicit function to handle the 'sqrt' logic.
        custom_math: dict = dict()
        custom_math["sqrt"] = _convert_sqrt_to_pow

        # ALGORITHM: Dynamically map native symbolic transcendental functions if supported.
        # This prevents numpy's ufunc from crashing when applied to VeraGrid Var objects.
        import VeraGridEngine.Utils.Symbolic.symbolic as vg_sym
        if hasattr(vg_sym, 'sin'):
            custom_math["sin"] = vg_sym.sin
        if hasattr(vg_sym, 'cos'):
            custom_math["cos"] = vg_sym.cos
        if hasattr(vg_sym, 'tan'):
            custom_math["tan"] = vg_sym.tan
        if hasattr(vg_sym, 'exp'):
            custom_math["exp"] = vg_sym.exp
        if hasattr(vg_sym, 'log'):
            custom_math["log"] = vg_sym.log

        eq_str: str
        for eq_str in eqs:
            # Convert string to SymPy expression using locals for variable mapping
            expr_sympy: sp.Expr = sp.sympify(eq_str, locals=s_d)
            eq_symbols: list = list(expr_sympy.free_symbols)

            # Extract the actual Var objects matching the SymPy symbols
            eq_vars: list = list()
            s: sp.Symbol
            for s in eq_symbols:
                eq_vars.append(v_d[s.name])

            # ALGORITHM: Lambdify maps the SymPy tree to a callable Python function.
            # By injecting custom_math, transcendentals and 'sqrt' are intercepted safely.
            func: Any = sp.lambdify(
                eq_symbols,
                expr_sympy,
                modules=[custom_math, "numpy"]
            )
            expr_var: Any = func(*eq_vars)

            # Categorize residue: If it contains a derivative, it's a state equation
            is_state: bool = False
            s_check: sp.Symbol
            for s_check in eq_symbols:
                # Using standard if/else logic
                if re.match(r'^(xp|dx)\d+', s_check.name):
                    is_state = True
                else:
                    pass

            if is_state:
                self._state_eqs.append(expr_var)
            else:
                self._algebraic_eqs.append(expr_var)

        # Rule: Explicit list building instead of adding lists (no 'list1 + list2')
        self._state_algeb_vars = list()

        v_state: Var
        for v_state in self._state_vars:
            self._state_algeb_vars.append(v_state)

        v_alg: Var
        for v_alg in self._algebraic_vars:
            self._state_algeb_vars.append(v_alg)

        self._n_state = len(self._state_vars)
        self._n_alg = len(self._algebraic_vars)
        self._n_algebraic = len(self._algebraic_eqs)
        self._n_diff = len(self._diff_vars)

        # Standard simulation parameters (internal to the RMS solver)
        self._n_event_params = 0
        self._dt = Var('dt')
        self._delta = Var('delta')

        self._variable_parameters.append(self._dt)
        self._variable_parameters.append(self._delta)

        self._event_parameters_eqs.append(Const(1e-3))
        self._event_parameters_eqs.append(Const(1.0))

        # Register event parameters explicitly
        p_list: list = list([self._dt, self._delta])
        p: Var
        for p in p_list:
            self._compiler_names_dict[p.uid] = str(f"{self.VARIABLE_PARAMS_NAME}[{self._n_event_params}]")
            self._alias_names_dict[p.uid] = str(f"{self.VARIABLE_PARAMS_NAME}_{self._n_event_params}")
            self._uid2idx_event_params[p.uid] = self._n_event_params

            # Explicit counter increment
            self._n_event_params = self._n_event_params + 1

        self.timings = dict()

        # Placeholders so attribute access is safe even if compilation fails
        self._j11_fn = None
        self._j12_fn = None
        self._j21_fn = None
        self._j22_fn = None

        self._compile_system()

    def _compile_system(self) -> None:
        """Invokes the RMSCompiler to generate optimized binary residue functions."""
        t0 = time.time()
        self._derivative_fn = SymbolicDerivative(self._state_algeb_vars, self._uid2idx_vars, self._diff_vars,
                                                 self._compiler_names_dict)
        rms_compiler = RMSCompiler(self._state_algeb_vars, self._diff_vars, self._variable_parameters,
                                   self._constant_parameters, self._dt, self._compiler_names_dict)

        self._rhs_algeb_fn = rms_compiler.compile_rhs(self._algebraic_eqs, "rhs_algeb")
        if self._n_state > 0:
            self._rhs_state_fn = rms_compiler.compile_rhs(self._state_eqs, "rhs_state")

            # Compile sparse jacobians used by the RMS engine for small-signal analysis
            try:
                self._j11_fn = rms_compiler.compile_sparse_jacobian(self._state_eqs, self._state_vars, "j11")
                self._j12_fn = rms_compiler.compile_sparse_jacobian(self._state_eqs, self._algebraic_vars, "j12")
                self._j21_fn = rms_compiler.compile_sparse_jacobian(self._algebraic_eqs, self._state_vars, "j21")
                self._j22_fn = rms_compiler.compile_sparse_jacobian(self._algebraic_eqs, self._algebraic_vars, "j22")
            except Exception as e:
                # If compilation fails, provide a runtime Python fallback using SymbolicJacobian
                print(f"[MTI_problem] Sparse jacobian compilation failed ({e}), installing runtime fallbacks.")
                try:
                    from VeraGridEngine.Utils.Symbolic.compiled_functions import SymbolicJacobian
                    import numpy as _np

                    def _make_runtime_jac(eqs, variables):
                        def _jfn(x, dx, vp=None, cp=None, h=0):
                            vp_arr = vp if vp is not None else _np.zeros(0, dtype=float)
                            cp_arr = cp if cp is not None else _np.zeros(0, dtype=float)
                            sj = SymbolicJacobian(eqs=eqs,
                                                 variables=variables,
                                                 compiler_names_dict=self._compiler_names_dict,
                                                 alias_names_dict=self._alias_names_dict,
                                                 VARS_NAME=self.VARS_NAME,
                                                 DIFF_NAME=self.DIFF_NAME,
                                                 EVENT_PARAMS_NAME=self.VARIABLE_PARAMS_NAME,
                                                 PARAMS_NAME=self.CONSTANT_PARAMS_NAME,
                                                 static=True)
                            return sj(x, dx, vp_arr, cp_arr, h=0).tocsc()
                        return _jfn

                    self._j11_fn = _make_runtime_jac(self._state_eqs, self._state_vars)
                    self._j12_fn = _make_runtime_jac(self._state_eqs, self._algebraic_vars)
                    self._j21_fn = _make_runtime_jac(self._algebraic_eqs, self._state_vars)
                    self._j22_fn = _make_runtime_jac(self._algebraic_eqs, self._algebraic_vars)
                except Exception:
                    # If even the runtime fallback fails, raise the original compilation error
                    raise
        self.timings["Compilation"] = time.time() - t0


def _convert_sqrt_to_pow(x: Any) -> Any:
    """
    ALGORITHM: Maps the square root operation to a fractional power.
    This allows SymPy's lambdify to evaluate 'sqrt' using the native
    dunder method __pow__ of the custom 'Var' object, avoiding the
    need for a dedicated .sqrt() attribute.

    :param x: The base variable or expression (typically a Var object).
    :type x: Any
    :return: The resulting expression object representing x raised to 0.5.
    :rtype: Any
    """
    return x ** 0.5
