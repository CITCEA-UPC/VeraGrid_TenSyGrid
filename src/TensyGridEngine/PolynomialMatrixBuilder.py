# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

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
import json
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import splu, LinearOperator, eigs


def _parallel_executor_cls():
    """
    Select executor class with Windows-safe default.

    On Windows, ProcessPoolExecutor uses spawn and re-imports module-level code
    in each child process. For these preprocessing stages, ThreadPoolExecutor
    avoids repeated interpreter startup and recursive import storms.
    """
    if sys.platform.startswith("win"):
        return concurrent.futures.ThreadPoolExecutor
    return concurrent.futures.ProcessPoolExecutor


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
        sys.stdout.write('\r' + ' ' * (len(self.message) + 15) + '\r')  # Clear line
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


def _parse_single_equation(raw_eq, parsing_context, param_map):
    """Isolated parser to run equation parsing in process workers."""

    s = raw_eq if isinstance(raw_eq, str) else str(raw_eq)
    s = s.replace('dt_1_', 'dt_').replace('d_1_', 'dt_')
    s = re.sub(r'\bd_', 'dt_', s)

    expr = sp.parsing.sympy_parser.parse_expr(s, local_dict=parsing_context, evaluate=False)

    if param_map:
        expr = expr.subs(param_map)

    return expr


def _process_equation_worker(raw_eq, parsing_context, param_map, sym_to_idx_snap, protected_params_snap):
    """
    Worker that combines equation parsing and monomial weight extraction.
    Returns only primitive data (lists/tuples) to avoid SymPy pickling issues
    when used with ProcessPoolExecutor.
    """
    s = raw_eq if isinstance(raw_eq, str) else str(raw_eq)
    s = s.replace('dt_1_', 'dt_').replace('d_1_', 'dt_')
    s = re.sub(r'\bd_', 'dt_', s)
    expr = sp.parsing.sympy_parser.parse_expr(s, local_dict=parsing_context, evaluate=False)
    if param_map:
        expr = expr.subs(param_map)

    protected_set = set(protected_params_snap)
    phi_row: list = []
    new_aux_names: list = []

    def _extract_term(term):
        coeff_sym, symbolic_part = term.as_coeff_mul()
        factors = sp.Mul.make_args(sp.Mul(*symbolic_part))

        if any(len(list(f.free_symbols)) > 1 for f in factors):
            expanded = term.expand()
            sub_terms = sp.Add.make_args(expanded)
            if len(sub_terms) == 1 and sub_terms[0] == term:
                raise ValueError(f"Irreducible non-linear term detected: {term}")
            for sub_t in sub_terms:
                _extract_term(sub_t)
            return

        global_phi = float(coeff_sym)
        sparse_weights: dict = {}

        for f in factors:
            f_vars = list(f.free_symbols)
            if len(f_vars) == 1:
                sym = f_vars[0]
                idx_val = sym_to_idx_snap.get(sym.name)

                if idx_val is None:
                    new_aux_names.append(sym.name)
                    return

                b_val_sym = sp.diff(f, sym)
                deg = float(f.exp) if isinstance(f, sp.Pow) else float(sp.degree(f, sym))
                is_sqrt = abs(deg - 0.5) < 1e-6
                is_protected = sym.name in protected_set

                if (deg > 1.0 or deg < 0.0) and not is_sqrt and not is_protected:
                    is_positive = deg > 1.0
                    num_aux = int(deg - 1.0) if is_positive else int(-deg + 1.0)
                    prefix = "mulaux" if is_positive else "invaux"
                    aux_names = [f"{sym.name}_{prefix}{i_aux}" for i_aux in range(num_aux)]
                    missing = [a for a in aux_names if a not in sym_to_idx_snap]
                    if missing:
                        new_aux_names.extend(missing)
                        return
                    new_syms = [sp.Symbol(a) for a in aux_names]
                    multilinear_f = sym
                    for s_aux in new_syms:
                        multilinear_f = multilinear_f * s_aux
                    _extract_term((term / f) * multilinear_f)
                    return

                elif is_sqrt and not is_protected:
                    aux_names_sqrt = [f"{sym.name}_sqrtaux0", f"{sym.name}_sqrtaux1"]
                    missing = [a for a in aux_names_sqrt if a not in sym_to_idx_snap]
                    if missing:
                        new_aux_names.extend(missing)
                        return
                    _extract_term((term / f) * sp.Symbol(aux_names_sqrt[0]))
                    return

                b_val = float(b_val_sym)
                a_val = float(f.subs(sym, 0))
                scale = abs(a_val) + abs(b_val)
                if scale == 0.0:
                    scale = 1.0
                sparse_weights[idx_val] = b_val / scale
                global_phi *= scale

            elif len(f_vars) == 0:
                global_phi *= float(f)

        monom_tuple = tuple(sorted(sparse_weights.items()))
        phi_row.append((monom_tuple, global_phi))

    terms = list(expr.args) if expr.is_Add else [expr]
    for term in terms:
        _extract_term(term)

    return phi_row, list(set(new_aux_names))


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
            """
            if not var_list: return
            for var_obj in var_list:
                if isinstance(var_obj, str):
                    continue

                try:
                    base_name_mappings = var_obj.name
                except AttributeError:
                    continue

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
        print("Processing equations")
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

        # 3. Expression parsing (parallelized)
        executor_cls = _parallel_executor_cls()
        with executor_cls() as executor:
            self.eqs = list(
                executor.map(
                    _parse_single_equation,
                    processed_eqs,
                    itertools.repeat(parsing_context),
                    itertools.repeat(self._param_map),
                    chunksize=50,
                )
            )

            self.ineqs = list(
                executor.map(
                    _parse_single_equation,
                    processed_ineqs,
                    itertools.repeat(parsing_context),
                    itertools.repeat(self._param_map),
                    chunksize=50,
                )
            )

        # =====================================================================
        # TRUE PARAMETER RESOLUTION (Ghost Pole Eliminator)
        # If VeraGrid forgot to register a constant/parameter (like 'In'),
        # it looks like a dynamic state. This catches it, looks up its numeric
        # value in var_factory, and physically substitutes it into the math.
        # =====================================================================
        if self.has_problem:
            allowed_vars = set()
            try:
                for v in (problem._state_vars or []): allowed_vars.add(_clean_name_fn(v.name))
                for v in (problem._algebraic_vars or []): allowed_vars.add(_clean_name_fn(v.name))
            except AttributeError:
                try:
                    for v in problem.state_and_algebraic_vars: allowed_vars.add(_clean_name_fn(v.name))
                except AttributeError:
                    pass
            for s in parsed_predefined:
                allowed_vars.add(s.name)

            if allowed_vars:
                v_dict_full = self.op_extraction(problem)
                subs_map = {}

                all_current_syms = set()
                for eq in self.eqs + self.ineqs:
                    all_current_syms.update(eq.free_symbols)

                for sym in all_current_syms:
                    s_name = sym.name
                    if s_name not in allowed_vars and not s_name.startswith('dt_') and not s_name.startswith('d_'):
                        val = None
                        if s_name in v_dict_full:
                            val = v_dict_full[s_name]
                        elif f"{s_name}_aux" in v_dict_full:
                            val = v_dict_full[f"{s_name}_aux"]
                        elif re.sub(r'_\d+$', '', s_name) in v_dict_full:
                            val = v_dict_full[re.sub(r'_\d+$', '', s_name)]

                        if val is not None:
                            subs_map[sym] = float(val)

                if subs_map:
                    if self.verbose: print(f"[*] Substituting {len(subs_map)} unregistered constants/parameters...")
                    self.eqs = [eq.subs(subs_map) for eq in self.eqs]
                    self.ineqs = [ineq.subs(subs_map) for ineq in self.ineqs]
        # =====================================================================

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

        S_H_sp, Phi_H_sp = self.matrix_creation(self.eqs, parallel=True)
        S_W_sp, Phi_W_sp = self.matrix_creation(self.ineqs, parallel=True)

        self.S_H: sparse.csc_matrix = S_H_sp
        self.Phi_H: sparse.csr_matrix = Phi_H_sp
        self.S_W: sparse.csc_matrix = S_W_sp
        self.Phi_W: sparse.csr_matrix = Phi_W_sp

        self.E: Optional[sparse.csr_matrix] = None
        self.A: Optional[sparse.csr_matrix] = None
        self.B: Optional[sparse.csr_matrix] = None
        self.EABC: Optional[sparse.csr_matrix] = None

    @staticmethod
    def op_extraction(problem: Any) -> dict:
        """
        Returns the operating point dictionary from a defined VeraGrid problem.
        This upgraded method parses the VarFactory directly, ensuring intermediate ports
        and aliases (like 'In', 'Ir', limiters) are preserved for the linearizer.
        """
        v_dict: dict = dict()

        def safe_get_value(val):
            try:
                return float(val)
            except:
                if hasattr(val, 'value') and val.value is not None:
                    try:
                        return float(val.value)
                    except:
                        pass
                elif hasattr(val, 'v') and val.v is not None:
                    try:
                        return float(val.v)
                    except:
                        pass
            return None

        # 1. BEST METHOD: Extract directly from the dynamic VarFactory.
        # This captures EVERY variable evaluated during initialization (states, limiters, aliases, etc.)
        try:
            vf = getattr(problem, 'var_factory', None)
            if vf is None and hasattr(problem, 'grid'):
                vf = getattr(problem.grid, 'var_factory', None)

            if vf is not None and hasattr(vf, 'vars'):
                for var_obj in vf.vars:
                    val = safe_get_value(var_obj)
                    if val is not None:
                        name = str(var_obj.name) if hasattr(var_obj, 'name') else str(var_obj)
                        v_dict[name] = val
        except Exception as e:
            print(f"[!] Warning: VarFactory extraction failed: {e}")

        # 2. TRADITIONAL FALLBACK: Extract x0 state vector directly
        try:
            x0 = problem.get_x0()
            all_vars = []

            # Safely combine the variable lists directly from the engine's backend
            if hasattr(problem, '_state_vars') and problem._state_vars is not None:
                all_vars.extend(problem._state_vars)
            if hasattr(problem, '_algebraic_vars') and problem._algebraic_vars is not None:
                all_vars.extend(problem._algebraic_vars)

            # Fallback for older engine structures
            if not all_vars and hasattr(problem, 'state_and_algebraic_vars'):
                all_vars = problem.state_and_algebraic_vars

            for i, var_obj in enumerate(all_vars):
                if i < len(x0):
                    s_name = str(var_obj.name) if hasattr(var_obj, 'name') else str(var_obj)
                    if s_name not in v_dict:
                        v_dict[s_name] = float(x0[i])
        except Exception as e:
            pass

        # 3. EXTRACTION FROM THE PARAMETER PATHS
        if hasattr(problem, '_variable_parameters') and hasattr(problem, '_variable_parameters_values'):
            names = problem._variable_parameters
            values = problem._variable_parameters_values
            if names is not None and values is not None:
                for name_obj, val_obj in zip(names, values):
                    s_name = str(name_obj.name) if hasattr(name_obj, 'name') else str(name_obj)
                    val = safe_get_value(val_obj)
                    if val is not None:
                        v_dict[s_name] = val

        # 4. Backup just in case there are also fixed constants
        if hasattr(problem, '_constant_parameters') and hasattr(problem, '_constant_params'):
            for n, v in zip(problem._constant_parameters, problem._constant_params):
                s_name = str(n.name) if hasattr(n, 'name') else str(n)
                val = safe_get_value(v)
                if val is not None:
                    v_dict[s_name] = val

        return v_dict

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
    def matrix_creation(self, all_exprs: list, parallel: bool = False) -> tuple:
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
        :param parallel: If True, use ProcessPoolExecutor via _process_equation_worker.
                         Only safe to use after full symbol extraction and lifting (stable sym_to_idx).
        :type parallel: bool
        :return: A tuple containing the (S, Phi) sparse matrices.
        :rtype: tuple
        """
        if parallel and len(all_exprs) > 0:
            return self._matrix_creation_parallel(all_exprs)
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

    def _matrix_creation_parallel(self, all_exprs: list) -> tuple:
        """
        Parallel version of matrix_creation using _process_equation_worker.

        Converts each equation to its string form, dispatches workers via
        ProcessPoolExecutor, and merges primitive results into sparse matrices.
        Called only when sym_to_idx is fully stable (after lifting and symbol extraction).

        Equations whose worker signals new_aux_names (unexpected symbols) fall back
        automatically to the sequential _get_monomial_weights path in the main thread.

        :param all_exprs: List of SymPy expressions (post-lifting).
        :type all_exprs: list
        :return: A tuple containing the (S, Phi) sparse matrices.
        :rtype: tuple
        """
        parsing_context_snap: dict = {sym.name: sym for sym in self.all_symbols}
        parsing_context_snap.update({
            'cos': sp.cos, 'sin': sp.sin, 'tan': sp.tan,
            'exp': sp.exp, 'sqrt': sp.sqrt, 'pi': sp.pi,
        })
        sym_to_idx_snap: dict = dict(self.sym_to_idx)
        protected_snap: list = list(self._protected_params)
        raw_strings: list = [str(e) for e in all_exprs]

        executor_cls = _parallel_executor_cls()
        with executor_cls() as executor:
            worker_results = list(executor.map(
                _process_equation_worker,
                raw_strings,
                itertools.repeat(parsing_context_snap),
                itertools.repeat({}),
                itertools.repeat(sym_to_idx_snap),
                itertools.repeat(protected_snap),
                chunksize=50,
            ))

        S_cols: list = []
        monom_to_idx: dict = {}
        Phi_data: list = [None] * len(all_exprs)
        fallback_indices: list = []

        for eq_idx, (phi_row_prims, new_aux_names) in enumerate(worker_results):
            if new_aux_names:
                fallback_indices.append(eq_idx)
                continue
            eq_dict: dict = {}
            for monom_tuple, coeff in phi_row_prims:
                existing = monom_to_idx.get(monom_tuple)
                if existing is None:
                    final_idx = len(S_cols)
                    monom_to_idx[monom_tuple] = final_idx
                    S_cols.append(monom_tuple)
                else:
                    final_idx = existing
                eq_dict[final_idx] = eq_dict.get(final_idx, 0.0) + coeff
            Phi_data[eq_idx] = eq_dict

        for eq_idx in fallback_indices:
            eq = all_exprs[eq_idx]
            terms = list(eq.args) if eq.is_Add else [eq]
            current_eq_coeffs: dict = {}
            for term in terms:
                current_eq_coeffs = self._get_monomial_weights(term, S_cols, monom_to_idx, current_eq_coeffs)
            Phi_data[eq_idx] = current_eq_coeffs

        n_vars = len(self.all_symbols)
        n_monoms = len(S_cols)
        n_eqs = len(all_exprs)

        if n_monoms > 0:
            s_row, s_col, s_val = [], [], []
            for col_idx, monom_tuple in enumerate(S_cols):
                for row_idx, val in monom_tuple:
                    s_row.append(row_idx)
                    s_col.append(col_idx)
                    s_val.append(val)

            S = sparse.csc_matrix((s_val, (s_row, s_col)), shape=(n_vars, n_monoms))

            p_row, p_col, p_val = [], [], []
            for row_idx, eq_dict in enumerate(Phi_data):
                if eq_dict:
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

        # Debug: dump symbol mapping and v vector for troubleshooting ordering issues
        try:
            print("[DEBUG] all_symbols:")
            for i, s in enumerate(self.all_symbols):
                print(f"  {i}: {s.name}")
            print("[DEBUG] sym_to_idx:")
            for k, vv in sorted(self.sym_to_idx.items(), key=lambda x: x[1]):
                print(f"  {vv}: {k}")
            print("[DEBUG] v vector:")
            print(v)
        except Exception:
            pass

        F = self._compute_jacobian_sparse(S, v)

        # Debug: basic stats on S, Phi, F
        try:
            print(f"[DEBUG] S.shape={S.shape}, Phi.shape={Phi.shape}, F.shape={F.shape}")
            print(f"[DEBUG] S.nnz={getattr(S, 'nnz', None)}, Phi.nnz={getattr(Phi, 'nnz', None)}, F.nnz={getattr(F, 'nnz', None)}")
            # For small matrices show dense forms
            if S.shape[0] <= 20 and S.shape[1] <= 20:
                try:
                    print("[DEBUG] S dense:\n", S.toarray())
                except Exception:
                    pass
            if Phi.shape[0] <= 20 and Phi.shape[1] <= 20:
                try:
                    print("[DEBUG] Phi dense:\n", Phi.toarray())
                except Exception:
                    pass
            try:
                print("[DEBUG] F dense:\n", F.toarray())
            except Exception:
                pass
        except Exception:
            pass

        self.EABC = Phi @ F.T

        # Debug: print EABC shape and a dense sample
        try:
            print(f"[DEBUG] EABC.shape={self.EABC.shape}")
            print(self.EABC.toarray())
        except Exception:
            pass

        self.E, self.A, self.B = self._split_EABC(self.EABC)

        # Debug: print idx_vars/idx_dx lists used in splitting
        try:
            idx_dx, idx_vars, idx_u = [], [], []
            for idx_counter, s in enumerate(self.all_symbols):
                s_name = s.name
                if s_name.startswith('dx') or s_name.startswith('xp') or s_name.startswith('d_') or s_name.startswith('dt_'):
                    idx_dx.append(idx_counter)
                elif s_name.startswith('u') and not s_name.startswith('u_'):
                    idx_u.append(idx_counter)
                else:
                    idx_vars.append(idx_counter)
            print(f"[DEBUG] idx_vars={idx_vars}")
            print(f"[DEBUG] idx_dx={idx_dx}")
            print(f"[DEBUG] idx_u={idx_u}")
        except Exception:
            pass

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
                P_val.append(-1.0)  # Extract and negate the derivative column (original convention)

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
            SPARSE_THRESHOLD = 5000
            used_sparse = False

            if n_dims >= SPARSE_THRESHOLD:
                k_modes = min(100, n_dims - 2)
                if k_modes > 0:
                    try:
                        sparse_res = self.compute_stability_sparse(sigma=2.0 + 1e-6j, k_modes=k_modes)
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

            # DAE FORMULA FIX: Element-wise product of left/right evec vectors mapping to Physical States
            for m in range(n_modes):
                w_H = evecs_left[:, m].conj()
                w_H_E = w_H @ E_dense

                participation_raw = w_H_E * evecs_right[:, m]

                # NORMALIZATION FIX: Sum the absolute magnitudes to guarantee 1.0 (100%) distribution limit
                p_abs = np.abs(participation_raw)
                normalization = np.sum(p_abs)

                if normalization > 1e-12:
                    participation_matrix[:, m] = p_abs / normalization
                else:
                    # ALGEBRAIC FALLBACK: If E completely masks out the mode (purely algebraic instability),
                    # use the raw right eigenvector magnitude to reveal the responsible variables.
                    v_abs = np.abs(evecs_right[:, m])
                    v_norm = np.sum(v_abs)
                    if v_norm > 1e-16:
                        participation_matrix[:, m] = v_abs / v_norm
                    else:
                        participation_matrix[:, m] = 0.0

            max_real = float(np.max(np.real(evals)))
            is_stable = max_real < -1e-6

            if self.verbose:
                print(f"  Found {len(evals)} finite eigenvalues (Physical Modes)\n")

            return evals, evecs_left, evecs_right, participation_matrix, is_stable, max_real

        except linalg.LinAlgError as e:
            if self.verbose and 'spinner' in locals(): spinner.stop()
            print(f"Error computing eigenvalues: {e}")
            return None, None, None, None, False, None

    def compute_stability_sparse(self, sigma=2.0, k_modes=200):
        """
        Perform stability analysis for large-scale systems using sparse ARPACK.

        This method solves the generalized eigenvalue problem Av = λEv
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
            Array of found physical eigenvalues (poles) λ.
        V : numpy.ndarray
            Matrix where each column is a right eigenvector corresponding to evals.
        W : numpy.ndarray
            Matrix where each column is a matched left eigenvector.
        participation_matrix : numpy.ndarray
            Matrix of participation factors where entry P_ji represents
            the participation of state j in mode i.
        stable : bool
            Boolean flag indicating system stability (True if all Re(λ) <= 1e-5).
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

        A_csr = sparse.csr_matrix(self.A)
        E_csr = sparse.csr_matrix(self.E)
        n_dims = A_csr.shape[0]

        print(f"    State-space dimension: {n_dims}")

        # Guard against invalid operating points triggering NaNs
        if not np.all(np.isfinite(A_csr.data)):
            print("    [!] Warning: State-Space 'A' matrix contains NaN values! Cleaning...")
            A_csr.data = np.nan_to_num(A_csr.data)
        if not np.all(np.isfinite(E_csr.data)):
            print("    [!] Warning: State-Space 'E' matrix contains NaN values! Cleaning...")
            E_csr.data = np.nan_to_num(E_csr.data)

        # --- ROBUST LU FACTORIZATION WITH SIGMA RETRY ---
        lu_solver = None
        sigma_candidates = [sigma, 0.0, 1.0j, -1.0j, 1.0, -1.0]
        for s_val in sigma_candidates:
            M_shift = A_csr - s_val * E_csr

            # Structural patching for M_shift (Using safe CSR/CSC operations over LIL to avoid __abs__ AttributeErrors)
            M_shift_csr = M_shift.tocsr()
            M_shift_csc = M_shift.tocsc()

            M_abs_csr = M_shift_csr.copy()
            M_abs_csr.data = np.abs(M_abs_csr.data)
            row_sums = np.array(M_abs_csr.sum(axis=1)).flatten()
            zero_rows = np.where(row_sums < 1e-12)[0]

            M_abs_csc = M_shift_csc.copy()
            M_abs_csc.data = np.abs(M_abs_csc.data)
            col_sums = np.array(M_abs_csc.sum(axis=0)).flatten()
            zero_cols = np.where(col_sums < 1e-12)[0]

            M_shift_patched = M_shift.tolil()
            for idx in np.unique(np.concatenate((zero_rows, zero_cols))):
                M_shift_patched[idx, idx] = -1.0  # Patch empty diagonal structure

            M_shift_patched = M_shift_patched.tocsc()  # Convert back to optimized CSC for splu

            try:
                lu_solver = splu(M_shift_patched)
                sigma = s_val  # Update sigma to the successful value
                print(f"    Successfully factored (A - {sigma}*E) using SuperLU.")
                break
            except Exception as e:
                print(f"    [!] SuperLU factorization failed for sigma={s_val}: {e}")

                # Attempt structural regularization for exactly singular blocks (like redundant constraints or isolated subgrids)
                print(f"    [*] Attempting structural regularization for sigma={s_val}...")
                try:
                    # Append a very small, non-interfering term onto the diagonal to cure perfect matrix singularity
                    reg_matrix = M_shift_patched + sparse.eye(n_dims, format='csc') * 1e-8
                    lu_solver = splu(reg_matrix)
                    sigma = s_val
                    print(f"    Successfully factored with structural regularization.")
                    break
                except Exception as e2:
                    print(f"    [!] Regularized SuperLU failed: {e2}")

        if lu_solver is None:
            print("    [!!!] All SuperLU factorization attempts failed. Aborting sparse analysis.")
            return [], None, None, None, False, 0.0

        def matvec_right(v):
            return lu_solver.solve(E_csr.dot(v))

        def matvec_left(v):
            return lu_solver.solve(E_csr.T.dot(v), trans='T')

        OP_R = LinearOperator((n_dims, n_dims), matvec=matvec_right, dtype=complex)
        OP_L = LinearOperator((n_dims, n_dims), matvec=matvec_left, dtype=complex)

        print(f"    Solving right/left modes in parallel...")
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
        E_T = E_csr.T

        # W from ARPACK (trans='T') satisfies W^T * A = \lambda W^T * E.
        # The correct row vector for participation is W^T * E.
        E_T_W = E_T.dot(W)

        # Element-wise product maps participation to physical states: P_ki = V_ki * (W_i^T * E)_k
        participation_raw = V * E_T_W

        # NORMALIZATION FIX: Sum the absolute magnitudes to guarantee 1.0 (100%) distribution limit
        p_abs = np.abs(participation_raw)
        dot_products = np.sum(p_abs, axis=0)

        # ALGEBRAIC FALLBACK: If E completely masks out a mode (dot_product ~ 0),
        # use the raw right eigenvector magnitude to reveal the responsible variables.
        for m in range(len(dot_products)):
            if dot_products[m] < 1e-12:
                v_abs = np.abs(V[:, m])
                p_abs[:, m] = v_abs
                dot_products[m] = np.sum(v_abs)

        dot_products[dot_products < 1e-16] = 1.0  # Prevent div-by-zero
        participation_matrix = p_abs / dot_products

        # Align left eigenvectors with standard definition (W_left^H A = \lambda W_left^H E)
        # Since W^T A = \lambda W^T E, taking the conjugate yields W^H A = \lambda^* W^H E.
        # Thus, W_left = np.conj(W) corresponds to the true left eigenvector for \lambda.
        W_left = np.conj(W)

        margin = float(np.max(evals.real)) if len(evals) > 0 else 0.0
        stable = margin <= 1e-5

        print(f"[+] Sparse ARPACK completed in {time.time() - start:.2f} seconds.")
        print(f"    Found {len(evals)} physical poles with their participation factors.")

        return evals, V, W_left, participation_matrix, stable, margin

    def get_participation_mapping(self, evals: np.ndarray, participation_matrix: np.ndarray,
                                  threshold: float = 0.02) -> dict:
        """
        Maps the raw participation matrix to the exact physical state variables for each dynamic mode.
        Filters out derivative and input tokens to match the true dimensions of the state-space.
        """
        mapping: dict = dict()
        if participation_matrix is None or evals is None: return mapping

        n_modes: int = int(participation_matrix.shape[1])
        n_states: int = int(participation_matrix.shape[0])

        # 1. Reconstruct the mapping of the physical state variables (identical to _split_EABC)
        idx_vars = []
        for idx_counter, s in enumerate(self.all_symbols):
            s_name = s.name
            is_deriv = s_name.startswith('dx') or s_name.startswith('xp') or s_name.startswith(
                'd_') or s_name.startswith('dt_')
            is_input = s_name.startswith('u') and not s_name.startswith('u_')

            if not is_deriv and not is_input:
                idx_vars.append(idx_counter)

        for m in range(n_modes):
            mode_eval: complex = evals[m]
            mode_pfs: list = list()
            all_valid_pfs: list = list()

            # Iterate strictly up to the available physical states
            for i in range(min(n_states, len(idx_vars))):
                # Retrieve the EXACT symbol associated with this row in the A/E matrices
                sym_idx = idx_vars[i]
                s_name = self.all_symbols[sym_idx].name

                # Since idx_vars inherently filters out derivatives/inputs, we just pull the value
                pf_val = float(abs(participation_matrix[i, m]))
                all_valid_pfs.append((s_name, pf_val))

                if pf_val >= threshold:
                    mode_pfs.append((s_name, pf_val))

            mode_pfs.sort(key=lambda x: x[1], reverse=True)

            # If participations were highly dispersed, grab the top 5 largest ones unconditionally.
            if not mode_pfs:
                all_valid_pfs.sort(key=lambda x: x[1], reverse=True)
                mode_pfs = [x for x in all_valid_pfs[:5] if x[1] > 1e-6]

            mapping[mode_eval] = mode_pfs

        return mapping

    def plot_evolution(self, t_f: float, timestep: float, v_dict: dict, evals: np.ndarray, evecs: np.ndarray,
                       variables: list = None):
        """
        Plots the time evolution of specified variables based on the linear state-space modes.

        :param t_f: Final simulation time.
        :param timestep: Time step for the plot.
        :param v_dict: Dictionary of the initial physical states (absolute or deviation).
        :param evals: 1D array of eigenvalues.
        :param evecs: 2D array of right eigenvectors (shape: n_states x n_modes).
        :param variables: List of variable names (str) to plot.
        """
        if variables is None:
            variables = []

        time_steps = np.arange(0, t_f, timestep)

        # 1. Reconstruct the state variable map
        idx_vars = []
        if hasattr(self, 'all_symbols'):
            for idx_counter, s in enumerate(self.all_symbols):
                s_name = s.name
                is_deriv = s_name.startswith('dx') or s_name.startswith('xp') or s_name.startswith(
                    'd_') or s_name.startswith('dt_')
                is_input = s_name.startswith('u') and not s_name.startswith('u_')
                if not is_deriv and not is_input:
                    idx_vars.append(idx_counter)

        var_name_to_ss_idx = {self.all_symbols[sym_idx].name: i for i, sym_idx in enumerate(idx_vars)}

        # 2. Build the actual initial condition vector (x0)
        x0 = np.zeros(evecs.shape[0])
        for k, val in v_dict.items():
            if k in var_name_to_ss_idx:
                x0[var_name_to_ss_idx[k]] = float(val)

        # 3. Project physical states to the modal domain: c = pinv(V) * x0
        c = np.linalg.pinv(evecs) @ x0

        plt.figure(figsize=(10, 6))

        # 4. Iterate over the requested variables
        for var in variables:
            if var not in var_name_to_ss_idx:
                print(f"[-] Warning: Variable '{var}' not found in the state space. Skipping.")
                continue

            var_idx = var_name_to_ss_idx[var]

            # Evaluate the incremental response: Delta_x_i(t) = Sum_m (c_m * exp(lambda_m * t) * V_{i,m})
            evolution = np.zeros(len(time_steps))
            mode_coeffs = c * evecs[var_idx, :]

            for i, t in enumerate(time_steps):
                # We take the real part since complex conjugate pairs cancel each other out
                evolution[i] = np.real(np.sum(mode_coeffs * np.exp(evals * t)))

            plt.plot(time_steps, evolution, label=f"$\\Delta$ {var}", linewidth=2)

        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Deviation from Operating Point ($\\Delta x$)", fontsize=12)
        plt.title("Linear Dynamic Evolution (Small Signal)", fontsize=14)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()

    def compute_bifurcation_stability(self, pf_map: dict, evals: np.ndarray,
                                      evecs_right: np.ndarray, evecs_left: np.ndarray,
                                      E_matrix: sparse.csc_matrix, margin: float = 0.1,
                                      tolerance: float = 1e-3, t_margin: float = 5.0) -> dict:
        """
        Mathematically evaluates if perturbations in critical modes (near bifurcation)
        decay within a tolerance limit after a time t_margin using bi-orthogonal projection.

        :param pf_map: Dictionary returned by `get_participation_mapping`.
        :param evals: 1D array of eigenvalues.
        :param evecs_right: 2D matrix of right eigenvectors (V).
        :param evecs_left: 2D matrix of left eigenvectors (W).
        :param E_matrix: The sparse descriptor mass matrix (E).
        :param margin: Real part threshold to consider a mode as "critical" (near the imaginary axis).
        :param tolerance: Maximum allowed deviation at t = t_margin for the perturbation to be considered decayed.
        :param t_margin: Time in the future (seconds) where the envelope is analytically evaluated.
        :return: Dictionary containing the report of evaluated variables and their alarm status.
        """
        # 1. Map variable names to state-space indices
        idx_vars = []
        for idx_counter, s in enumerate(self.all_symbols):
            s_name = s.name
            is_deriv = any(s_name.startswith(pre) for pre in ['dx', 'xp', 'd_', 'dt_'])
            is_input = s_name.startswith('u') and not s_name.startswith('u_')

            if not is_deriv and not is_input:
                idx_vars.append(idx_counter)

        var_name_to_ss_idx = {self.all_symbols[sym_idx].name: i for i, sym_idx in enumerate(idx_vars)}

        # 2. Enforce descriptor bi-orthogonal normalization: w_i^H * E * v_i = 1
        # Clone matrices to avoid modifying external state
        V = evecs_right.copy()
        W = evecs_left.copy()

        for i in range(len(evals)):
            # Calculate scaling factor for the i-th mode
            scale_factor = W[:, i].conj().T @ E_matrix @ V[:, i]
            # Normalize the left eigenvector accordingly
            W[:, i] = W[:, i] / scale_factor.conj()

        if self.verbose:
            print(f"\n[ Bifurcation & Stability Analysis | Margin: +/-{margin} | Time: {t_margin}s ]")

        report = {}

        # 3. Iterate over the participation map
        for ev, participations in pf_map.items():
            # Identify if the eigenvalue is critically close to the imaginary axis (Non-Hyperbolic point)
            if -margin < np.real(ev) <= margin:
                ev_str = f"{ev:.4f}"
                report[ev_str] = []

                if self.verbose:
                    freq = np.abs(np.imag(ev)) / (2 * np.pi)
                    damping = -np.real(ev) / np.abs(ev) * 100 if np.abs(ev) > 0 else 0.0
                    print(f"\n--- Critical Eigenvalue Detected: {ev:.4f} ---")
                    print(f"    (Frequency: {freq:.2f} Hz, Damping: {damping:.2f}%)")

                for var_name, pf in participations:
                    if not str(var_name).startswith("u_"):

                        if var_name not in var_name_to_ss_idx:
                            continue

                        var_idx = var_name_to_ss_idx[var_name]

                        # A. Construct descriptor initial perturbation profile (E * x0)
                        # Instead of an independent x0 vector, we extract the column from E.
                        # If the variable is purely algebraic, its corresponding column in E is zero,
                        # automatically forcing a consistent initial algebraic response.
                        E_x0 = E_matrix[:, var_idx].toarray().flatten() * 0.05

                        # B. Project to modal domain using bi-orthogonality: c_i = w_i^H * (E * x0)
                        # This replaces the mathematically incorrect and slow 'np.linalg.pinv(V)'
                        c = np.zeros(len(evals), dtype=complex)
                        for i in range(len(evals)):
                            c[i] = W[:, i].conj().T @ E_x0

                        # C. Calculate EXACT linear deviation at t = t_margin inside the computed manifold
                        mode_coeffs = c * V[var_idx, :]
                        delta_at_t = np.real(np.sum(mode_coeffs * np.exp(evals * t_margin)))

                        # D. Evaluate bifurcation and stability criteria
                        passed = abs(delta_at_t) < tolerance

                        # Programmatic Alerts tailored for critical points
                        if not passed and np.real(ev) < -1e-4:
                            print(f"Programmatic alert: Poorly damped mode causing slow decay at {var_name}")
                        elif not passed and abs(np.real(ev)) <= 1e-4:
                            print(f"Programmatic alert: True bifurcation detected at {var_name}. "
                                  f"Linear simulation cannot guarantee long-term boundedness.")

                        report[ev_str].append({
                            "variable": var_name,
                            "delta_t": abs(delta_at_t),
                            "passed": passed
                        })

                        if self.verbose:
                            status = "[OK]" if passed else "[DANGER]"
                            msg = "decays" if passed else "DOES NOT damp/converge"
                            symbol = "<" if passed else ">="
                            print(f"  {status} Variable {var_name}: Perturbation {msg}. "
                                  f"Delta(t={t_margin}) = {abs(delta_at_t):.6f} {symbol} {tolerance}")

        return report
