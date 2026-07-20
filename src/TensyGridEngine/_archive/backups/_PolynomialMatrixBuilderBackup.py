import sympy as sp
from scipy import sparse, linalg
from enum import Enum
from typing import Optional, Any, Tuple
import re, math
import threading
import sys
import itertools
import concurrent.futures
import time
import numpy as np

from scipy.sparse.linalg import splu, LinearOperator, eigs



class LoadingSpinner:
    """A threaded terminal spinner to show activity during blocking C/Fortran calls."""
    def __init__(self, message="Computing"):
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.message = message
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.start_time = 0.0

    def start(self):
        self.start_time = time.time()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.message) + 15) + '\r') # Clear line
        sys.stdout.flush()
        return time.time() - self.start_time

    def _spin(self):
        while not self.stop_event.is_set():
            sys.stdout.write(f'\r{self.message} {next(self.spinner)} ')
            sys.stdout.flush()
            time.sleep(0.1)


# ------------------------------------------------------------------
# Enums for Options
# ------------------------------------------------------------------

class ConstraintType(Enum):
    """
    Enum representing the type of constraint to process.
    Replaces the usage of string options or ambiguous booleans.
    """
    EQUALITY = 1
    INEQUALITY = 2


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

def _sort_symbol(sym: sp.Symbol) -> tuple:
    """
    Evaluates a SymPy symbol to determine if it represents a derivative or a standard state variable,
    generating a sorting key tuple.

    From an algorithmic perspective, this function acts as a custom sorting key. In symbolic system
    formulation, it is strictly necessary to segregate time-derivatives from standard algebraic variables
    to properly structure Differential Algebraic Equations (DAEs) or state-space representations.

    The function extracts the string name of the symbol and checks it against standard derivative
    prefixes ('dx', 'xp', 'd_', 'dt_'). By returning a tuple of `(integer_priority, name)`, it forces
    Python sorting algorithms to group all derivatives first (priority 0) and standard variables second
    (priority 1). The inclusion of the string name in the tuple ensures that within those two categories,
    the symbols are sorted alphabetically.

    :param sym: The symbolic variable to be evaluated and categorized.
    :type sym: sp.Symbol
    :return: A tuple used for sorting, containing the priority integer (0 if it is a derivative,
             1 otherwise) and the string name of the symbol.
    :rtype: tuple
    """
    name: str = sym.name
    if name.startswith('dx') or name.startswith('xp') or name.startswith('d_') or name.startswith('dt_'):
        return 0, name
    else:
        return 1, name

def _clean_name_fn(n):
    """
    Standardizes the string representation of a symbolic variable name by normalizing
    various time-derivative prefixes into a single canonical format.

    From an algorithmic perspective, different symbolic operations, external libraries,
    or intermediate parsing steps may generate inconsistent first-derivative prefixes
    (such as 'dt_1_', 'd_1_', or simply 'd_'). This cleaning step is critical in the
    broader algorithm because downstream processes—such as state-space formulation,
    variable mapping, or the previous sorting logic—rely on strict, predictable string
    matching to correctly identify and manipulate time derivatives.

    The function executes the following sequence:
    1. Casts the input to a string to guarantee that string manipulation methods are available,
       allowing it to gracefully handle SymPy symbols or raw strings.
    2. Explicitly replaces known anomalous first-derivative notations ('dt_1_' and 'd_1_')
       with the standard 'dt_' prefix.
    3. Utilizes a regular expression to find word-boundary instances of 'd_' (matching the
       start of the string) and replaces them with 'dt_'. The word boundary is essential to
       prevent the accidental replacement of 'd_' if it occurs in the middle or end of a
       variable's base name.

    :param n: The variable or object whose name requires standardizing.
    :type n: Any (typically str or sp.Symbol)
    :return: The cleaned and standardized string name with a unified 'dt_' derivative prefix.
    :rtype: str
    """
    c = str(n)
    c = c.replace('dt_1_', 'dt_').replace('d_1_', 'dt_')
    c = re.sub(r'\bd_', 'dt_', c)
    return c


def _get_real_part(val: complex) -> float:
    return float(val.real)


def _safe_to_string(item: Any) -> str:
    """ Safely extracts string representation without getattr/hasattr """
    try:
        return str(item.expr)
    except AttributeError:
        try:
            return item.to_string()
        except AttributeError:
            return str(item)

class PolynomialMatrixBuilder:
    """
    Builds the S_H / Phi_H (and S_W / Phi_W) matrices from a list of
    polynomial equations, and performs analytic linearization using the
    iMTI / CPN representation described in the TenSyGrid papers.
    """

    __slots__ = (
        "verbose", "eqs", "ineqs", "state_vars", "algebraic_vars", "has_problem",
        "_param_map", "_protected_params", "_explicit_deriv_map", "all_symbols",
        "sym_to_idx", "_argaux_map", "S_H", "Phi_H", "S_W", "Phi_W",
        "E", "A", "B", "EABC"
    )

    def __init__(
            self,
            problem: Optional[Any] = None,
            eqs: Optional[list] = None,
            ineqs: Optional[list] = None,
            state_vars: Optional[list] = None,
            algebraic_vars: Optional[list] = None,
            verbose: bool = False
    ) -> None:

        self.verbose: bool = verbose
        self.eqs: list = list()
        self.ineqs: list = list()
        self.state_vars: list = list()
        self.algebraic_vars: list = list()
        self.has_problem: bool = problem is not None

        # Unconditional initialization
        self._param_map: dict = {}
        self._protected_params: list = []
        self._explicit_deriv_map: dict = {}

        # --- 0. RESOLVE DUPLICATE VARIABLE NAMES DYNAMICALLY ---
        # This prevents SymPy from physically fusing distinct limiters together.
        if self.has_problem:
            seen_vars = set()
            name_counters = {}
            all_vg_vars = []

            try:
                if problem._state_vars is not None:
                    all_vg_vars.extend(problem._state_vars)
            except AttributeError:
                pass

            try:
                if problem._algebraic_vars is not None:
                    all_vg_vars.extend(problem._algebraic_vars)
            except AttributeError:
                pass

            for var in all_vg_vars:
                try:
                    base_name = var.name
                except AttributeError:
                    continue  # Skip if it has no name

                vid = id(var)
                if vid not in seen_vars:
                    seen_vars.add(vid)

                    if base_name not in name_counters:
                        name_counters[base_name] = 0
                    else:
                        name_counters[base_name] += 1
                        new_name = f"{base_name}_{name_counters[base_name]}"
                        try:
                            var.name = new_name
                        except Exception as e:
                            print("Exception on for var in all_vg_vars", e)
                            pass

                        # Also rename the corresponding derivative variable to avoid mapping breaks
                        try:
                            d_var = var.diff_var
                            if d_var is not None:
                                d_vid = id(d_var)
                                if d_vid not in seen_vars:
                                    try:
                                        d_base = d_var.name
                                        seen_vars.add(d_vid)
                                        d_new = d_base.replace(base_name, new_name)
                                        if d_new == d_base:
                                            d_new = f"{d_base}_{name_counters[base_name]}"
                                        try:
                                            d_var.name = d_new
                                        except Exception:
                                            pass
                                    except AttributeError:
                                        pass
                        except AttributeError:
                            pass

        # 1. PARAMETER EXTRACTION
        if self.has_problem:
            try:
                v_params = problem._variable_parameters or []
            except AttributeError:
                v_params = []

            try:
                v_param_vals = problem._variable_parameters_values
            except AttributeError:
                try:
                    v_param_vals = problem._variable_parameter_values
                except AttributeError:
                    v_param_vals = []

            for i, vp in enumerate(v_params):
                try:
                    p_name = vp.name
                except AttributeError:
                    p_name = str(vp)

                self._protected_params.append(p_name)
                if i < len(v_param_vals):
                    self._param_map[p_name] = v_param_vals[i]

            try:
                c_params = problem._constant_parameters or []
                c_vals = problem._constant_params if problem._constant_params is not None else []
                for i, cp in enumerate(c_params):
                    try:
                        p_name = cp.name
                    except AttributeError:
                        p_name = str(cp)

                    self._protected_params.append(p_name)
                    if i < len(c_vals):
                        self._param_map[p_name] = c_vals[i]
            except AttributeError:
                pass

        processed_eqs = list()
        processed_ineqs = list()
        processed_state_vars = list()
        processed_algebraic_vars = list()

        if eqs is not None:
            processed_eqs = list(eqs)
        elif self.has_problem:
            try:
                for s_eq in problem._state_eqs:
                    processed_eqs.append(_safe_to_string(s_eq))
            except AttributeError:
                pass
            try:
                for a_eq in problem._algebraic_eqs:
                    processed_eqs.append(_safe_to_string(a_eq))
            except AttributeError:
                pass

        if ineqs is not None:
            processed_ineqs = list(ineqs)



        def _extract_mappings(var_list: list) -> None:
            """
            Extracts and maps base variables to their explicit time-derivative counterparts
            from a list of variable objects.

            From an algorithmic perspective, this function builds a required lookup table
            (`self._explicit_deriv_map`) linking state variables to their corresponding derivatives.
            It iterates over the provided list, bypassing raw strings, and inspects objects for
            `diff_var` or `base_var` properties. By applying `_clean_name_fn` to the extracted names,
            it guarantees standardized string matching for downstream differential equation assembly.

            :param var_list: A collection of variable objects to inspect for derivative relationships.
            :type var_list: list
            :return: Modifies the internal `_explicit_deriv_map` in place.
            :rtype: None
            """
            if not var_list: return
            for var_obj in var_list:
                if isinstance(var_obj, str):
                    continue

                try:
                    base_name_mappings = var_obj.name
                except AttributeError:
                    continue  # We need a name to map it

                try:
                    diff_var = var_obj.diff_var
                    if diff_var is not None:
                        try:
                            self._explicit_deriv_map[_clean_name_fn(base_name_mappings)] = _clean_name_fn(diff_var.name)
                        except AttributeError:
                            pass
                except AttributeError:
                    pass

                try:
                    base_var = var_obj.base_var
                    if base_var is not None:
                        try:
                            self._explicit_deriv_map[_clean_name_fn(base_var.name)] = _clean_name_fn(base_name_mappings)
                        except AttributeError:
                            pass
                except AttributeError:
                    pass

        if state_vars is not None:
            _extract_mappings(state_vars)
            processed_state_vars = [_safe_to_string(v) for v in state_vars]
        elif self.has_problem:
            try:
                prob_state_vars = problem._state_vars
                _extract_mappings(prob_state_vars)
                for s_var in prob_state_vars:
                    processed_state_vars.append(_safe_to_string(s_var))
            except AttributeError:
                pass

        if algebraic_vars is not None:
            _extract_mappings(algebraic_vars)
            processed_algebraic_vars = [_safe_to_string(v) for v in algebraic_vars]
        elif self.has_problem:
            try:
                prob_alg_vars = problem._algebraic_vars
                _extract_mappings(prob_alg_vars)
                for a_var in prob_alg_vars:
                    processed_algebraic_vars.append(_safe_to_string(a_var))
            except AttributeError:
                pass

        parsing_context = dict()
        parsing_context['cos'] = sp.cos
        parsing_context['sin'] = sp.sin
        parsing_context['tan'] = sp.tan
        parsing_context['exp'] = sp.exp
        parsing_context['sqrt'] = sp.sqrt
        parsing_context['pi'] = sp.pi

        parsed_predefined: list[sp.Symbol] = []

        # 2. Symbols Extraction
        for names, target_list in [(processed_state_vars, self.state_vars),
                                   (processed_algebraic_vars, self.algebraic_vars)]:
            for name in names:
                clean_name = name.replace('dt_1_', 'dt_').replace('d_1_', 'dt_')
                clean_name = re.sub(r'\bd_', 'dt_', clean_name)

                if clean_name not in parsing_context:
                    sym = sp.Symbol(clean_name)
                    parsing_context[clean_name] = sym
                    target_list.append(sym)
                    parsed_predefined.append(sym)

        # 3. Expression parsing
        for raw_exprs, target_list in [(processed_eqs, self.eqs), (processed_ineqs, self.ineqs)]:
            for raw in raw_exprs:
                s = raw if isinstance(raw, str) else str(raw)
                s = s.replace('dt_1_', 'dt_').replace('d_1_', 'dt_')
                s = re.sub(r'\bd_', 'dt_', s)

                expr = sp.parsing.sympy_parser.parse_expr(s, local_dict=parsing_context, evaluate=False)

                if self._param_map:
                    expr = expr.subs(self._param_map)

                target_list.append(expr)

        # 4. Symbols' inizialization
        self.all_symbols = []
        self.sym_to_idx = {}
        self.extract_symbols(self.eqs + self.ineqs, predefined=parsed_predefined)
        self._argaux_map = dict()

        self._flatten_nonlinear_arguments()
        self._apply_automatic_trig_lifting()

        self.all_symbols.sort(key=_sort_symbol)
        self.sym_to_idx.clear()
        for idx, sym in enumerate(self.all_symbols):
            self.sym_to_idx[sym.name] = idx

        # --- ALGORITHM: TWO-PASS COMPILATION FOR AUTOMATIC LIFTING ---
        self.matrix_creation(self.eqs)
        self._generate_lifting_equations(problem)

        S_H_sp, Phi_H_sp = self.matrix_creation(self.eqs)
        S_W_sp, Phi_W_sp = self.matrix_creation(self.ineqs)

        self.S_H: sparse.csc_matrix = S_H_sp
        self.Phi_H: sparse.csr_matrix = Phi_H_sp
        self.S_W: sparse.csc_matrix = S_W_sp
        self.Phi_W: sparse.csr_matrix = Phi_W_sp

        self.E: Optional[sparse.csr_matrix] = None
        self.A: Optional[sparse.csr_matrix] = None
        self.B: Optional[sparse.csr_matrix] = None
        self.EABC: Optional[sparse.csr_matrix] = None

    # ------------------------------------------------------------------
    # Symbol extraction
    # ------------------------------------------------------------------
    def _is_protected_param(self, var_name: str) -> bool:
        return var_name in self._protected_params

    def extract_symbols(self, all_exprs: list, predefined: list | None = None) -> list:
        """
        Extracts and sorts unique symbolic variables from a list of equations.

        Algorithmically, this function builds the definitive sequence of state variables.
        It uses a set for O(1) deduplication, appending `predefined` symbols first.
        Then, it extracts `free_symbols` from all expressions, actively filtering out
        protected parameters. Finally, it applies the derivative-first sorting logic
        to guarantee a deterministic variable order for matrix and state-space assembly.

        :param all_exprs: List of symbolic expressions or equations to parse.
        :type all_exprs: list
        :param predefined: Optional list of symbols to force-include first.
        :type predefined: list | None
        :return: A sorted list of all unique, non-protected symbols.
        :rtype: list
        """
        seen: set = set()
        extracted_symbols: list = list()

        if predefined is not None:
            for sym_p in predefined:
                if sym_p not in seen:
                    seen.add(sym_p)
                    extracted_symbols.append(sym_p)

        for eq in all_exprs:
            for sym in list(eq.free_symbols):
                if sym not in seen and not self._is_protected_param(sym.name):
                    seen.add(sym)
                    extracted_symbols.append(sym)

        extracted_symbols.sort(key=_sort_symbol)
        self.all_symbols = extracted_symbols
        return extracted_symbols

    # ------------------------------------------------------------------
    # Matrix creation
    # ------------------------------------------------------------------
    def matrix_creation(self, all_exprs: list) -> tuple:
        """
        Assembles the sparse structural matrices S (monomial definitions) and Phi (coefficients)
        from a list of symbolic expressions.

        Algorithmically, this translates symbolic equations into a vectorized numeric format
        required for high-performance evaluation. It parses each equation into terms, extracts
        unique monomials to avoid redundancies, and maps their respective coefficients.
        To optimize memory and conform to pure sparse operations from the ground up, it natively
        constructs Coordinate (COO) arrays before compiling S in CSC format and Phi in CSR format.

        :param all_exprs: A list of symbolic expressions representing the system equations.
        :type all_exprs: list
        :return: A tuple containing the (S, Phi) sparse matrices.
        :rtype: tuple
        """
        S_cols: list = []
        Phi_data: list = []
        monom_to_idx: dict = dict()

        for eq in all_exprs:
            terms = list(eq.args) if eq.is_Add else [eq]
            current_eq_coeffs: dict = dict()
            for term in terms:
                current_eq_coeffs = self._get_monomial_weights(term, S_cols, monom_to_idx, current_eq_coeffs)
            Phi_data.append(current_eq_coeffs)

        n_vars = len(self.all_symbols)
        n_monoms = len(S_cols)
        n_eqs = len(all_exprs)

        if n_monoms > 0:
            s_row, s_col, s_val = [], [], []
            for col_idx, sparse_col in enumerate(S_cols):
                for row_idx, val in sparse_col:
                    s_row.append(row_idx)
                    s_col.append(col_idx)
                    s_val.append(val)

            S = sparse.csc_matrix((s_val, (s_row, s_col)), shape=(n_vars, n_monoms))

            p_row, p_col, p_val = [], [], []
            for row_idx, eq_dict in enumerate(Phi_data):
                for col_idx, val in eq_dict.items():
                    p_row.append(row_idx)
                    p_col.append(col_idx)
                    p_val.append(val)

            Phi = sparse.csr_matrix((p_val, (p_row, p_col)), shape=(n_eqs, n_monoms))
        else:
            S = sparse.csc_matrix((n_vars, 0))
            Phi = sparse.csr_matrix((n_eqs, 0))

        return S, Phi

    def _get_monomial_weights(self, term: sp.Expr, S_cols: list, monom_to_idx: dict, current_eq_coeffs: dict) -> dict:
        """
        Parses a symbolic term to extract its structural weights for the S matrix and
        its scalar coefficient for the Phi matrix.

        Algorithmically, this function is the core algebraic reducer for the TenSyGrid engine.
        It breaks down complex terms into standardized multilinear monomials. If it detects
        higher-order degrees, inverses, or square roots, it dynamically generates and injects
        auxiliary variables (`mulaux`, `invaux`, `sqrtaux`) to linearize the term's structure.
        During this process, it updates the global symbol registry and recursively builds
        sparse coordinate tuples to instantly accommodate growing matrix dimensions without dense padding.

        :param term: The specific symbolic algebraic term to be evaluated.
        :type term: sp.Expr
        :param S_cols: List of sparse tuples representing the structural footprint of known monomials.
        :type S_cols: list
        :param monom_to_idx: Dictionary mapping unique monomial structures to their column index.
        :type monom_to_idx: dict
        :param current_eq_coeffs: Dictionary accumulating the coefficient values for the current equation.
        :type current_eq_coeffs: dict
        :return: The updated dictionary of coefficients mapping monomial indices to their values.
        :rtype: dict
        """
        coeff_sym, symbolic_part = term.as_coeff_mul()
        factors: tuple = sp.Mul.make_args(sp.Mul(*symbolic_part))
        requires_expansion: bool = False

        for f in factors:
            if len(list(f.free_symbols)) > 1:
                requires_expansion = True

        if requires_expansion:
            expanded_expr: sp.Expr = term.expand()
            expanded_sub_terms: tuple = sp.Add.make_args(expanded_expr)
            if len(expanded_sub_terms) == 1 and expanded_sub_terms[0] == term:
                raise ValueError(f"Irreducible non-linear term detected: {term}")

            for sub_t in expanded_sub_terms:
                current_eq_coeffs = self._get_monomial_weights(sub_t, S_cols, monom_to_idx, current_eq_coeffs)
            return current_eq_coeffs

        else:
            global_phi: float = float(coeff_sym)
            sparse_weights: dict = {}

            for f in factors:
                f_vars_inner = list(f.free_symbols)
                if len(f_vars_inner) == 1:
                    s: sp.Symbol = f_vars_inner[0]
                    idx_val: int | None = self.sym_to_idx.get(s.name, None)

                    if idx_val is not None:
                        b_val_sym: sp.Expr = sp.diff(f, s)
                        deg: float = float(f.exp) if isinstance(f, sp.Pow) else float(sp.degree(f, s))
                        is_sqrt: bool = abs(deg - 0.5) < 1e-6
                        is_protected = self._is_protected_param(s.name)

                        if (deg > 1.0 or deg < 0.0) and not is_sqrt and not is_protected:
                            is_positive: bool = deg > 1.0
                            num_aux: int = int(deg - 1.0) if is_positive else int(-deg + 1.0)
                            prefix: str = "mulaux" if is_positive else "invaux"

                            new_vars_names = [f"{s.name}_{prefix}{i_aux}" for i_aux in range(num_aux)]
                            new_symbols_list = []

                            for v_name in new_vars_names:
                                if v_name not in self.sym_to_idx:
                                    new_sym_obj = sp.Symbol(v_name)
                                    self.all_symbols.append(new_sym_obj)
                                    self.sym_to_idx[v_name] = len(self.all_symbols) - 1
                                    new_symbols_list.append(new_sym_obj)
                                else:
                                    new_symbols_list.append(self.all_symbols[self.sym_to_idx[v_name]])

                            multilinear_f = s
                            for sym_aux in new_symbols_list:
                                multilinear_f = multilinear_f * sym_aux

                            new_term = (term / f) * multilinear_f
                            return self._get_monomial_weights(new_term, S_cols, monom_to_idx, current_eq_coeffs)

                        elif is_sqrt and not is_protected:
                            new_vars_names_sqrt = [f"{s.name}_sqrtaux0", f"{s.name}_sqrtaux1"]
                            new_symbols_list_sqrt = []

                            for v_name_sqrt in new_vars_names_sqrt:
                                if v_name_sqrt not in self.sym_to_idx:
                                    new_sym_obj_sqrt = sp.Symbol(v_name_sqrt)
                                    self.all_symbols.append(new_sym_obj_sqrt)
                                    self.sym_to_idx[v_name_sqrt] = len(self.all_symbols) - 1
                                    new_symbols_list_sqrt.append(new_sym_obj_sqrt)
                                else:
                                    new_symbols_list_sqrt.append(self.all_symbols[self.sym_to_idx[v_name_sqrt]])

                            multilinear_f_sqrt = new_symbols_list_sqrt[0]
                            new_term_sqrt = (term / f) * multilinear_f_sqrt
                            return self._get_monomial_weights(new_term_sqrt, S_cols, monom_to_idx, current_eq_coeffs)

                        b_val = float(b_val_sym)
                        a_val = float(f.subs(s, 0))
                        scale = abs(a_val) + abs(b_val)
                        if scale == 0.0: scale = 1.0

                        sparse_weights[idx_val] = b_val / scale
                        global_phi *= scale

                elif len(f_vars_inner) == 0:
                    global_phi *= float(f)

            monom_tuple = tuple(sorted(sparse_weights.items()))
            existing_idx = monom_to_idx.get(monom_tuple, None)

            if existing_idx is None:
                final_idx = len(S_cols)
                monom_to_idx[monom_tuple] = final_idx
                S_cols.append(monom_tuple)
            else:
                final_idx = existing_idx

            current_eq_coeffs[final_idx] = current_eq_coeffs.get(final_idx, 0.0) + global_phi
            return current_eq_coeffs

    def _apply_automatic_trig_lifting(self) -> None:
        """
        Applies automatic trigonometric lifting to eliminate transcendental functions
        by replacing them with polynomial proxy variables.

        Algorithmically, this step converts transcendental non-linearities into a purely
        polynomial Differential Algebraic Equation (DAE) system. It searches for sine and
        cosine operations, replaces them with auxiliary state variables (e.g., `x_sin_arg`),
        and automatically appends their analytical time-derivative constraints to the system
        equations. This is strictly required to ensure all terms can be subsequently parsed
        into standard multilinear monomial matrices.

        :return: Modifies the internal `eqs` and `all_symbols` lists in place.
        :rtype: None
        """
        new_eqs, subs_dict, processed_args = [], {}, set()

        for eq in self.eqs:
            for term in eq.atoms(sp.sin, sp.cos):
                arg = term.args[0]
                if arg not in processed_args:
                    arg_name = arg.name
                    if self._is_protected_param(arg_name): continue

                    s_var = sp.Symbol(f"x_sin_{arg.name}")
                    c_var = sp.Symbol(f"x_cos_{arg.name}")
                    sp_var = sp.Symbol(f"xp_sin_{arg.name}")
                    cp_var = sp.Symbol(f"xp_cos_{arg.name}")

                    arg_p = sp.Symbol(self._explicit_deriv_map.get(arg_name, arg_name.replace('x', 'xp')))

                    new_eqs.extend([sp_var - (c_var * arg_p), cp_var + (s_var * arg_p)])
                    subs_dict[sp.sin(arg)] = s_var
                    subs_dict[sp.cos(arg)] = c_var

                    for new_sym in [s_var, c_var, sp_var, cp_var]:
                        if new_sym not in self.all_symbols:
                            self.all_symbols.append(new_sym)

                    processed_args.add(arg)

        self.eqs = [eq.subs(subs_dict) for eq in self.eqs] + new_eqs

    def _get_symbol_by_name(self, name: str) -> Optional[sp.Symbol]:
        for target_sym in self.all_symbols:
            if target_sym.name == name: return target_sym
        return None

    def _generate_lifting_equations(self, problem) -> None:
        """
        Generates and appends exact algebraic constraint equations for auxiliary variables
        introduced during the structural reduction phase.

        Algorithmically, prior steps inject proxy variables (`_mulaux`, `_invaux`, `_sqrtaux`)
        to convert non-linearities (such as inverses and square roots) into purely multilinear terms.
        To ensure the resulting Differential Algebraic Equation (DAE) system remains mathematically
        closed and well-posed, this function automatically formulates their structural polynomial
        definitions (e.g., enforcing `base * invaux - 1 = 0` or `base - sqrtaux**2 = 0`) and
        appends them to the global system equations.

        :param problem: The overarching problem instance, used to filter out predefined variable parameters.
        :type problem: Any
        :return: Modifies the internal `eqs` list in place with the new algebraic constraints.
        :rtype: None
        """
        new_constraints = []

        try:
            prob_v_params = problem._variable_parameters if problem._variable_parameters else []
        except AttributeError:
            prob_v_params = []

        for sym in self.all_symbols:
            if sym not in prob_v_params:
                s_name = sym.name
                if "_mulaux" in s_name:
                    base_sym = self._get_symbol_by_name(s_name.split("_mulaux")[0])
                    if base_sym and (base_sym - sym) not in self.eqs:
                        new_constraints.append(base_sym - sym)
                elif "_invaux" in s_name:
                    base_sym = self._get_symbol_by_name(s_name.split("_invaux")[0])
                    if base_sym and (base_sym * sym - 1.0) not in self.eqs:
                        new_constraints.append(base_sym * sym - 1.0)
                elif s_name.endswith("_sqrtaux0"):
                    base_sym = self._get_symbol_by_name(s_name.split("_sqrtaux0")[0])
                    sym_aux1 = self._get_symbol_by_name(f"{s_name.split('_sqrtaux0')[0]}_sqrtaux1")
                    if base_sym and sym_aux1:
                        if (base_sym - (sym * sym_aux1)) not in self.eqs:
                            new_constraints.append(base_sym - (sym * sym_aux1))
                        if (sym - sym_aux1) not in self.eqs:
                            new_constraints.append(sym - sym_aux1)

        self.eqs.extend(new_constraints)

    def _flatten_nonlinear_arguments(self) -> None:
        """
        Flattens complex expressions inside non-integer powers by introducing proxy state variables.

        Algorithmically, downstream structural parsers require the bases of fractional powers
        (e.g., square roots) to be single state variables rather than composite polynomials.
        This function scans all equations and inequalities for non-integer exponents applied
        to complex expressions. It replaces the complex base with an auxiliary variable (`argaux`)
        and automatically appends its explicit structural definition (`argaux - base_expr = 0`)
        to the global system constraints.

        :return: Modifies the internal `eqs`, `ineqs`, and `all_symbols` lists in place.
        :rtype: None
        """
        new_constraints, arg_counter = [], 0
        for eq_list in [self.eqs, self.ineqs]:
            for i in range(len(eq_list)):
                current_eq = eq_list[i]
                for p in current_eq.find(sp.Pow):
                    if not p.exp.is_integer:
                        base_expr = p.base
                        if isinstance(base_expr, sp.Symbol) and self._is_protected_param(base_expr.name): continue
                        if not isinstance(base_expr, sp.Symbol):
                            new_sym = sp.Symbol(f"argaux{arg_counter}")
                            arg_counter += 1
                            if new_sym not in self.all_symbols: self.all_symbols.append(new_sym)
                            self._argaux_map[new_sym] = base_expr
                            current_eq = current_eq.subs(p, new_sym ** p.exp)
                            new_constraints.append(new_sym - base_expr)
                eq_list[i] = current_eq
        self.eqs.extend(new_constraints)

    # ------------------------------------------------------------------
    # Linearization – public entry point
    # ------------------------------------------------------------------
    def linearize(self, v_dict: dict, c_type: ConstraintType = ConstraintType.EQUALITY) -> sparse.csr_matrix:
        """
        Evaluates the numerical linearization of the system around a specified operating point
        to extract the linear state-space matrices (E, A, B).

        Algorithmically, this function dynamically injects the exact numerical values for
        trigonometric proxy states to guarantee consistency at the operating point. It then
        computes the analytical Jacobian (F) of the target structural matrix (S) evaluated
        at the state vector. By projecting this Jacobian through the coefficient matrix (Phi),
        it algebraically computes the global linear state-space block `EABC` (Phi @ F^T),
        which dictates the local dynamic behavior of the system.

        :param v_dict: Dictionary containing the numerical values defining the operating point.
        :type v_dict: dict
        :param c_type: Specifies whether to linearize equality or inequality constraints.
        :type c_type: ConstraintType
        :return: The combined, linearized state-space matrix block EABC.
        :rtype: np.ndarray
        """
        for key in list(v_dict.keys()):
            if key.startswith('x') and 'p' not in key:
                val = float(v_dict[key])
                v_dict[f"x_sin_{key}"] = math.sin(val)
                v_dict[f"x_cos_{key}"] = math.cos(val)
                v_dict[f"xp_sin_{key}"] = 0.0
                v_dict[f"xp_cos_{key}"] = 0.0

        S, Phi = (self.S_W, self.Phi_W) if c_type == ConstraintType.INEQUALITY else (self.S_H, self.Phi_H)
        v = self._build_v_vector(v_dict)
        F = self._compute_jacobian_sparse(S, v)

        self.EABC = Phi @ F.T
        self.E, self.A, self.B = self._split_EABC(self.EABC)

        return self.EABC

    def _build_v_vector(self, v_dict_local: dict) -> np.ndarray:
        """
        Constructs the complete dense numerical state vector from a dictionary of operating point values.

        Algorithmically, this function maps user-defined inputs to the strict internal array layout
        required for Jacobian evaluation. It executes three critical completion passes: First, it applies
        a robust bridging logic to mirror derivative values across different canonical prefixes
        ('xp', 'dx', 'dt_', 'd_'). Second, it numerically evaluates complex base expressions to populate
        'argaux' proxy variables. Finally, it calculates the exact states of structural auxiliary variables
        ('mulaux', 'invaux', 'sqrtaux') directly from their parent values, ensuring the final vector
        is mathematically closed and fully consistent for matrix operations.

        :param v_dict_local: Dictionary mapping symbolic variable names to their scalar numerical values.
        :type v_dict_local: dict
        :return: A 1D NumPy array representing the fully populated state vector.
        :rtype: np.ndarray
        """
        v = np.zeros(len(self.all_symbols), dtype=float)
        for k, val in v_dict_local.items():
            if k in self.sym_to_idx:
                v[self.sym_to_idx[k]] = float(val)

            # Robust Bridge: Ensures manual derivative inputs are mirrored properly across models
            if k.startswith('xp'):
                base = k[2:]
                for alt in [f'dx{base}', f'dt_{base}', f'd_{base}']:
                    if alt in self.sym_to_idx: v[self.sym_to_idx[alt]] = float(val)
            elif k.startswith('dx'):
                base = k[2:]
                for alt in [f'xp{base}', f'dt_{base}', f'd_{base}']:
                    if alt in self.sym_to_idx: v[self.sym_to_idx[alt]] = float(val)
            elif k.startswith('dt_'):
                base = k[3:]
                for alt in [f'xp{base}', f'dx{base}', f'd_{base}']:
                    if alt in self.sym_to_idx: v[self.sym_to_idx[alt]] = float(val)
            elif k.startswith('d_'):
                base = k[2:]
                for alt in [f'xp{base}', f'dx{base}', f'dt_{base}']:
                    if alt in self.sym_to_idx: v[self.sym_to_idx[alt]] = float(val)

        for sym_arg in self.all_symbols:
            if sym_arg in self._argaux_map:
                base_expr = self._argaux_map[sym_arg]
                subs_dict = {s: float(v[self.sym_to_idx[s.name]]) for s in base_expr.free_symbols if
                             s.name in self.sym_to_idx}
                if sym_arg.name in self.sym_to_idx:
                    v[self.sym_to_idx[sym_arg.name]] = float(base_expr.subs(subs_dict))

        for sym in self.all_symbols:
            s_name = sym.name
            if s_name not in v_dict_local:
                idx_current = self.sym_to_idx[s_name]
                if "_mulaux" in s_name:
                    base_idx = self.sym_to_idx.get(s_name.split("_mulaux")[0])
                    if base_idx is not None: v[idx_current] = v[base_idx]
                elif "_invaux" in s_name:
                    base_idx = self.sym_to_idx.get(s_name.split("_invaux")[0])
                    if base_idx is not None:
                        val_base = float(v[base_idx])
                        v[idx_current] = 1.0 / val_base if abs(val_base) > 1e-12 else 1.0
                elif "_sqrtaux" in s_name:
                    base_idx = self.sym_to_idx.get(s_name.split("_sqrtaux")[0])
                    if base_idx is not None and float(v[base_idx]) > 0.0:
                        v[idx_current] = float(np.sqrt(float(v[base_idx])))

        return v

    def _compute_jacobian(self, S: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Computes the analytical Jacobian matrix of the system's structural monomials evaluated
        at a specific state vector.

        Algorithmically, this function exploits the purely multilinear nature of the TenSyGrid
        formulation. It calculates all partial derivatives simultaneously using a highly optimized,
        vectorized inverse-product rule (where the derivative of a product is the product divided
        by the target variable). Crucially, it implements an algebraic safeguard for zero-crossings:
        if exactly one variable in a monomial evaluates to zero, it bypasses the resulting 0/0
        singularity by directly computing the product of the remaining non-zero factors.

        :param S: The structural matrix defining the variable compositions of each monomial.
        :type S: np.ndarray
        :param v: The 1D numerical array representing the populated state vector.
        :type v: np.ndarray
        :return: The numerically evaluated dense Jacobian matrix F.
        :rtype: np.ndarray
        """
        X = (S.T * v) + (1.0 - np.abs(S.T))
        Y = np.prod(X, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            invX = np.where(np.abs(X) > 1e-12, 1.0 / X, 0.0)
            F = S * (Y[:, np.newaxis] * invX).T
            zeros_per_col = np.sum(np.abs(X) < 1e-12, axis=1)
            single_zero_cols = np.where(zeros_per_col == 1)

            for col in single_zero_cols[0]:
                zero_row = int(np.where(np.abs(X[col, :]) < 1e-12)[0][0])
                other_factors = np.delete(X[col, :], zero_row)
                prod_val: float = float(np.prod(other_factors))
                s_val: float = float(S[zero_row, col])

                F[zero_row, col] = s_val * prod_val

        return F

    def _compute_jacobian_sparse(self, S: sparse.csc_matrix, v: np.ndarray) -> sparse.csc_matrix:
        """
        Computes the analytical Jacobian matrix of the system's structural monomials evaluated
        at a specific state vector.

        Algorithmically, this function exploits the purely multilinear nature of the TenSyGrid
        formulation using a highly efficient Sparse Compressed Column (CSC) traversal.
        It accurately evaluates the iMTI / CPN structured contracted product exclusively on the
        non-zero structural elements. Crucially, it implements an algebraic safeguard for
        zero-crossings: if exactly one variable in a monomial evaluates to zero, it bypasses
        the resulting 0/0 singularity by directly computing the product of the remaining
        non-zero factors.

        :param S: The structural matrix defining the variable compositions of each monomial.
        :type S: sparse.csc_matrix
        :param v: The 1D numerical array representing the populated state vector.
        :type v: np.ndarray
        :return: The numerically evaluated sparse Jacobian matrix F.
        :rtype: sparse.csc_matrix
        """
        n_vars, n_monom = S.shape

        F_data = []
        F_indices = []
        F_indptr = [0]

        for r in range(n_monom):
            col_start = S.indptr[r]
            col_end = S.indptr[r + 1]

            row_indices = S.indices[col_start:col_end]
            S_values = S.data[col_start:col_end]

            prod_val = 1.0
            zero_count = 0
            zero_idx = -1

            X_vals = []
            for idx, i in enumerate(row_indices):
                S_ir = S_values[idx]

                X_i = (1.0 - abs(S_ir)) + (S_ir * v[i])
                X_vals.append(X_i)

                if abs(X_i) < 1e-12:
                    zero_count += 1
                    zero_idx = i
                else:
                    prod_val *= X_i

            for idx, i in enumerate(row_indices):
                S_ir = S_values[idx]

                if zero_count > 1:
                    d_val = 0.0
                elif zero_count == 1:
                    if i == zero_idx:
                        d_val = S_ir * prod_val
                    else:
                        d_val = 0.0
                else:
                    d_val = S_ir * (prod_val / X_vals[idx])

                F_indices.append(i)
                F_data.append(d_val)

            F_indptr.append(len(F_data))

        return sparse.csc_matrix((F_data, F_indices, F_indptr), shape=(n_vars, n_monom))

    def _split_EABC(self, EABC: sparse.csr_matrix) -> tuple:
        """
        Partitions the combined linearized Jacobian block matrix into E, A, and B.
        Zero-RAM overhead implementation using Sparse Projection Matrices instead of LIL.
        """
        idx_dx, idx_vars, idx_u = [], [], []

        for idx_counter, s in enumerate(self.all_symbols):
            s_name = s.name
            if s_name.startswith('dx') or s_name.startswith('xp') or s_name.startswith('d_') or s_name.startswith(
                    'dt_'):
                idx_dx.append(idx_counter)
            elif s_name.startswith('u') and not s_name.startswith('u_'):
                idx_u.append(idx_counter)
            else:
                idx_vars.append(idx_counter)

        n_eqs = int(EABC.shape[0])
        cols_to_take = len(idx_vars)
        dim = max(n_eqs, cols_to_take)

        # 1. Build A directly by slicing EABC and resizing (Ultra-fast, minimal RAM)
        A = EABC[:, idx_vars[:cols_to_take]].copy()
        A.resize((dim, dim))

        # 2. Build E using a Sparse Projection Matrix
        P_row, P_col, P_val = [], [], []

        for i in range(cols_to_take):
            state_sym = self.all_symbols[idx_vars[i]]
            state_name = state_sym.name

            deriv_names = []
            if state_name.startswith('x'):
                deriv_names.extend([f"xp{state_name[1:]}", f"dx{state_name[1:]}"])
            deriv_names.extend([f"dt_{state_name}", f"d_{state_name}"])

            deriv_idx = -1

            if state_name in self._explicit_deriv_map:
                exact_d_name = self._explicit_deriv_map[state_name]
                if exact_d_name in self.sym_to_idx:
                    deriv_idx = self.sym_to_idx[exact_d_name]

            if deriv_idx == -1:
                for d_name in deriv_names:
                    if d_name in self.sym_to_idx:
                        deriv_idx = self.sym_to_idx[d_name]
                        break

            if deriv_idx == -1:
                for dx_idx in idx_dx:
                    dx_name = self.all_symbols[dx_idx].name
                    if dx_name.endswith(f"_{state_name}") and (dx_name.startswith("dt_") or dx_name.startswith("d_")):
                        deriv_idx = dx_idx
                        break

            if deriv_idx != -1 and deriv_idx in idx_dx:
                P_row.append(deriv_idx)
                P_col.append(i)
                P_val.append(-1.0)  # Extract and negate the derivative column

        # Build Projection matrix P in pure CSC format
        P = sparse.csc_matrix((P_val, (P_row, P_col)), shape=(EABC.shape[1], dim))

        # E is instantly computed by C++ backend (EABC @ P)
        E = (EABC @ P).tocsr()
        E.resize((dim, dim))

        # 3. Build B directly
        if len(idx_u) > 0:
            B = EABC[:, idx_u].copy()
            B.resize((dim, len(idx_u)))
        else:
            B = sparse.csr_matrix((dim, 0))

        # --- FLOATING POINT NOISE CLEANUP ---
        A.data[np.abs(A.data) < 1e-10] = 0.0
        A.eliminate_zeros()

        E.data[np.abs(E.data) < 1e-10] = 0.0
        E.eliminate_zeros()

        return E, A, B

    def save_state_space(self, file_prefix: str = "system") -> None:
        """
        Serializes the compiled sparse state-space matrices (A and E) to disk.
        This allows for instant loading and eigenvalue analysis in future sessions
        without re-running the power flow or symbolic linearizer.

        :param file_prefix: The base name for the saved files.
        """
        if self.A is not None and self.E is not None:
            sparse.save_npz(f"{file_prefix}_A.npz", self.A)
            sparse.save_npz(f"{file_prefix}_E.npz", self.E)

            # Optional: Save the symbol names so you know what the columns mean!
            if hasattr(self, 'all_symbols'):
                import json
                symbol_names = [s.name for s in self.all_symbols]
                with open(f"{file_prefix}_symbols.json", "w") as f:
                    json.dump(symbol_names, f)

            if self.verbose:
                print(f"[+] Sparse state-space saved to {file_prefix}_A.npz and {file_prefix}_E.npz")
        else:
            print("[-] Cannot save matrices: Linearization has not been performed yet.")

    def compute_stability(self) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool, Optional[float]]:
        """
        Computes the generalized eigenvalues, eigenvectors, and participation factors
        to evaluate the small-signal stability of the linearized system.

        Algorithmically, this function evaluates the generalized eigenvalue problem (A*v = lambda*E*v).
        Before calling the solver, it performs 'Singular Pencil Patching' strictly using
        sparse Coordinate data to neutralize structural null-spaces.

        It natively uses ARPACK sparse eigenvalue decomposition (`eigs`) to find the most
        dominant modes for large systems. For small systems (or strictly singular DAEs where
        ARPACK fails), it seamlessly drops down to dense QZ decomposition to guarantee the
        complete eigenvalue spectrum is found.
        """

        current_a, current_e = self.A, self.E
        if current_a is None or current_e is None: return None, None, None, None, False, None

        try:
            n_dims = current_a.shape[0]
            patched_rows, patched_cols = 0, 0

            # --- SPARSE SINGULAR PENCIL PATCHING ---
            A_patched = current_a.tolil()
            A_csr = current_a.tocsr()
            E_csr = current_e.tocsr()

            for i in range(n_dims):
                a_data = A_csr.data[A_csr.indptr[i]:A_csr.indptr[i + 1]]
                e_data = E_csr.data[E_csr.indptr[i]:E_csr.indptr[i + 1]]
                a_max = np.max(np.abs(a_data)) if len(a_data) > 0 else 0.0
                e_max = np.max(np.abs(e_data)) if len(e_data) > 0 else 0.0
                if a_max < 1e-10 and e_max < 1e-10:
                    A_patched[i, i] = -1.0
                    patched_rows += 1

            A_csc = current_a.tocsc()
            E_csc = current_e.tocsc()
            for j in range(n_dims):
                a_data = A_csc.data[A_csc.indptr[j]:A_csc.indptr[j + 1]]
                e_data = E_csc.data[E_csc.indptr[j]:E_csc.indptr[j + 1]]
                a_max = np.max(np.abs(a_data)) if len(a_data) > 0 else 0.0
                e_max = np.max(np.abs(e_data)) if len(e_data) > 0 else 0.0
                if a_max < 1e-10 and e_max < 1e-10:
                    A_patched[j, j] = -1.0
                    patched_cols += 1

            if self.verbose and (patched_rows > 0 or patched_cols > 0):
                print(
                    f"\n[+] Sparse QZ Pre-conditioning: Neutralized {patched_rows} empty rows and {patched_cols} empty columns.")

            # --- EIGENVALUE DECOMPOSITION ROUTING ---
            SPARSE_THRESHOLD = 2000
            used_sparse = False

            if n_dims >= SPARSE_THRESHOLD:
                k_modes = min(100, n_dims - 2)
                if k_modes > 0:
                    try:
                        sparse_res = self.compute_stability_sparse(sigma=2.0, k_modes=k_modes)
                        evals_s, evecs_right_s, evecs_left_s, p_s, stable_s, margin_s = sparse_res
                    except (AttributeError, NotImplementedError) as e:
                        if self.verbose:
                            print(f"[*] Sparse method not available or error: {e}")
                        sparse_res = (None, None, None, None, False, 0.0)

                    if (
                            evals_s is not None
                            and evecs_right_s is not None
                            and evecs_left_s is not None
                            and len(evals_s) > 0
                    ):
                        return evals_s, evecs_left_s, evecs_right_s, p_s, stable_s, margin_s

                if self.verbose:
                    print("[*] Dedicated sparse solve failed or returned empty. Falling back to dense QZ...")

            if not used_sparse:
                A_dense = A_patched.toarray()
                E_dense = current_e.toarray()

                if self.verbose:
                    spinner = LoadingSpinner(f"[*] Diagonalizing Dense Matrix (N={n_dims})")
                    spinner.start()

                evals, evecs_left_raw, evecs_right_raw = linalg.eig(A_dense, b=E_dense, left=True, right=True)

                if self.verbose:
                    elapsed = spinner.stop()
                    print(f"[+] Dense QZ Diagonalization finished in {elapsed:.2f} seconds.")

            # --- MODE FILTERING AND PARTICIPATION ---
            valid_idx = np.isfinite(evals)
            evals = evals[valid_idx]
            evecs_left = evecs_left_raw[:, valid_idx].astype(complex)
            evecs_right = evecs_right_raw[:, valid_idx].astype(complex)

            if len(evals) == 0: return evals, evecs_left, evecs_right, None, True, -np.inf

            n_states, n_modes = evecs_right.shape
            participation_matrix = np.zeros((n_states, n_modes), dtype=float)

            # DAE FORMULA FIX: Mapping by Physical States (Columns), NOT Equations
            for m in range(n_modes):
                # P_k = V_k * (w^H * E)_k
                w_H = evecs_left[:, m].conj()
                w_H_E = w_H @ self.E  # This yields a row vector mapping to states

                participation_raw = evecs_right[:, m] * w_H_E
                normalization = np.sum(participation_raw)

                if abs(normalization) > 1e-16:
                    participation_matrix[:, m] = np.abs(participation_raw / normalization)
                else:
                    participation_matrix[:, m] = 0.0

            max_real = float(np.max(np.real(evals)))
            is_stable = max_real < 1e-6

            if self.verbose:
                print(f"  Found {len(evals)} finite eigenvalues (Physical Modes)\n")

            return evals, evecs_left, evecs_right, participation_matrix, is_stable, max_real

        except linalg.LinAlgError as e:
            if self.verbose and 'spinner' in locals(): spinner.stop()
            print(f"Error computing eigenvalues: {e}")
            return None, None, None, None, False, None

    def compute_stability_sparse(self, sigma=2.0, k_modes=200):
        '''
        Perform stability analysis for large-scale systems using sparse ARPACK.

        This method solves the generalized eigenvalue problem :math:`Av = \\lambda Ev`
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
            Array of found physical eigenvalues (poles) :math:`\\lambda`.
        V : numpy.ndarray
            Matrix where each column is a right eigenvector corresponding to `evals`.
        W : numpy.ndarray
            Matrix where each column is a matched left eigenvector.
        participation_matrix : numpy.ndarray
            Matrix of participation factors where entry :math:`P_{ji}` represents
            the participation of state :math:`j` in mode :math:`i`.
        stable : bool
            Boolean flag indicating system stability (True if all :math:`Re(\\lambda) \\leq 1e-5`).
        margin : float
            The maximum real part among all computed eigenvalues (stability margin).

        Notes
        -----
        - **Structural Regularization:** The method automatically patches empty rows/columns
          in the shifted matrix to prevent singularity during LU factorization.
        - **Mode Matching:** Since ARPACK may return left and right modes in different
          orders, the Hungarian Algorithm (Linear Sum Assignment) is used to align
          them based on eigenvalue proximity.
        - **Participation Factors:** Calculated as :math:`P_{ji} = \frac{w_{ji} v_{ji}}{w_i^H v_i}`.
        '''
        print(f"\n[+] Starting sparse ARPACK analysis (Shift-and-Invert, sigma={sigma}, k={k_modes})...")
        start = time.time()

        A_csr = sparse.csr_matrix(self.A)
        E_csr = sparse.csr_matrix(self.E)
        n_dims = A_csr.shape[0]

        print(f"    State-space dimension: {n_dims}")

        # (Asegúrate de tener esta parte dentro de compute_stability_sparse)
        M_shift = A_csr - sigma * E_csr

        # Structural regularization (WITHOUT LIL MATRIX)
        row_norms = np.array(np.abs(M_shift).max(axis=1).todense()).flatten()
        col_norms = np.array(np.abs(M_shift).max(axis=0).todense()).flatten()

        zero_rows = np.where(row_norms < 1e-10)[0]
        zero_cols = np.where(col_norms < 1e-10)[0]
        all_zeros = np.unique(np.concatenate((zero_rows, zero_cols)))

        if len(all_zeros) > 0:
            print(f"    Patching {len(all_zeros)} empty structural nodes (Zero-RAM Diagonal Addition)...")
            # Create a diagonal array filled with zeros, and put -1.0 only at empty spots
            patch_diag = np.zeros(n_dims)
            patch_diag[all_zeros] = -1.0

            # Sparse diagonal matrix
            D = sparse.diags(patch_diag, format='csr')

            # Add directly without mutating sparsity structure manually
            M_shift = M_shift + D

        print(f"    Factoring (A - {sigma}*E) using SuperLU...")
        try:
            lu_solver = splu(M_shift.tocsc())
        except Exception as e:
            print(f"    [!] SuperLU factorization error: {e}")
            return [], None, None, None, False, 0.0

        def matvec_right(v):
            y = E_csr.dot(v)
            return lu_solver.solve(y.real) + 1j * lu_solver.solve(y.imag)

        OP_R = LinearOperator((n_dims, n_dims), matvec=matvec_right, dtype=complex)

        print(f"    Solving right/left modes in parallel...")

        def matvec_left(v):
            y = E_csr.T.dot(v)
            return lu_solver.solve(y.real, trans='T') + 1j * lu_solver.solve(y.imag, trans='T')

        OP_L = LinearOperator((n_dims, n_dims), matvec=matvec_left, dtype=complex)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future_r = executor.submit(eigs, OP_R, k=k_modes, which='LM')
                future_l = executor.submit(eigs, OP_L, k=k_modes, which='LM')
                evals_nu_R, evecs_right_raw = future_r.result()
                evals_nu_L, evecs_left_raw = future_l.result()
        except Exception as e:
            print(f"    [!] Parallel ARPACK solve failed: {e}")
            return [], None, None, None, False, 0.0

        evals_R = sigma + (1.0 / evals_nu_R)
        evals_L = sigma + (1.0 / evals_nu_L)

        from scipy.optimize import linear_sum_assignment
        dist_matrix = np.abs(evals_R[:, None] - evals_L[None, :])
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        evecs_left_matched = evecs_left_raw[:, col_ind]

        valid_nu = np.abs(evals_nu_R) > 1e-10
        evals = evals_R[valid_nu]
        V = evecs_right_raw[:, valid_nu]
        W = evecs_left_matched[:, valid_nu]

        finite_mask = np.isfinite(evals)
        evals = evals[finite_mask]
        V = V[:, finite_mask]
        W = W[:, finite_mask]

        # --- DAE FORMULA FIX FOR SPARSE MATRICES ---
        # Formula: P_ki = V_ki * (W_i^H * E)_k
        E_T = E_csr.T
        W_conj = np.conj(W)

        # (W^H * E)^T = E^T * conj(W)
        E_T_W_conj = E_T.dot(W_conj)

        # Element-wise product ensures we map onto the states correctly
        participation_raw = V * E_T_W_conj

        dot_products = np.sum(participation_raw, axis=0)
        dot_products[np.abs(dot_products) < 1e-16] = 1.0  # Prevent div-by-zero

        participation_matrix = np.abs(participation_raw / dot_products)

        margin = float(np.max(evals.real)) if len(evals) > 0 else 0.0
        stable = margin <= 1e-5

        print(f"[+] Sparse ARPACK completed in {time.time() - start:.2f} seconds.")
        print(f"    Found {len(evals)} physical poles with their participation factors.")

        return evals, V, W, participation_matrix, stable, margin

    def get_participation_mapping(self, evals: np.ndarray, participation_matrix: np.ndarray,
                                  threshold: float = 0.05) -> dict:
        """
        Maps the raw participation matrix to the exact physical state variables for each dynamic mode.
        Filters out derivative and input tokens to match the true dimensions of the state-space.
        """
        mapping: dict = dict()
        if participation_matrix is None or evals is None: return mapping

        state_symbols: list = list()
        for s in self.all_symbols:
            s_name: str = s.name
            is_deriv: bool = s_name.startswith('dx') or s_name.startswith('xp') or s_name.startswith(
                'd_') or s_name.startswith('dt_')
            is_input: bool = s_name.startswith('u') and not s_name.startswith('u_')

            if not is_deriv and not is_input:
                state_symbols.append(s)

        n_modes: int = int(participation_matrix.shape[1])
        n_states: int = int(participation_matrix.shape[0])

        for m in range(n_modes):
            mode_eval: complex = evals[m]
            mode_pfs: list = list()

            limit: int = min(n_states, len(state_symbols))
            for i in range(limit):
                pf_val: float = float(abs(participation_matrix[i, m]))
                if pf_val >= threshold:
                    mode_pfs.append((state_symbols[i].name, pf_val))

            mode_pfs.sort(key=lambda x: x[1], reverse=True)
            mapping[mode_eval] = mode_pfs

        return mapping

    '''def report(self, eigenvalues=None, is_stable=False, max_real=None, print_matrices=True, save_path=None):
        """
        Outputs or exports the internal structural matrices and eigenvalue results
        for diagnostic inspection and external validation.

        Algorithmically, this function provides observability into the compiled Differential
        Algebraic Equation (DAE) system. It sequentially extracts and formats the sparse
        matrices (S, Phi, E, A, B). If eigenvalues are provided, it actively filters for
        finite values and sorts them by their real parts in descending order, immediately
        highlighting the most dynamically critical (or unstable) modes. It optionally serializes
        this system state to a specified file path for persistent logging.

        :param eigenvalues: Array of computed system eigenvalues.
        :type eigenvalues: np.ndarray | None
        :param is_stable: Boolean flag indicating if the system is stable.
        :type is_stable: bool
        :param max_real: The maximum real part found among the eigenvalues.
        :type max_real: float | None
        :param print_matrices: Flag to enable console printing of the matrices.
        :type print_matrices: bool
        :param save_path: Absolute or relative file path to save the exported text report.
        :type save_path: str | None
        :return: Modifies no state; performs I/O operations.
        :rtype: None
        """
        if print_matrices:
            for label, mat in [("S", self.S_H), ("Phi", self.Phi_H), ("E", self.E), ("A", self.A), ("B", self.B)]:
                if mat is not None:
                    print(f"\n--- {label} Matrix ---")
                    print(sparse.csc_matrix(mat))
            if eigenvalues is not None:
                for ev in sorted(list(eigenvalues[np.isfinite(eigenvalues)]), key=_get_real_part, reverse=True):
                    print(f"  {ev.real:.6f} + {ev.imag:.6f}j")

        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                for label, mat in [("S Matrix", self.S_H), ("Phi Matrix", self.Phi_H), ("E Matrix", self.E),
                                   ("A Matrix", self.A)]:
                    if mat is not None:
                        f.write(f"\n--- {label} ---\n")
                        f.write(str(sparse.csc_matrix(mat)) + "\n")'''