"""Compact colored AD for large arithmetic EMT residual systems.

The numerical kernels are compiled once, independently of model size. Model
expressions become instruction arrays rather than a large generated function.
Unsupported symbolic nodes retain the existing generated AD backend.
"""

import numpy as np
from numba import njit

from VeraGridEngine.Utils.Symbolic.symbolic import BinOp, Const, Func, UnOp, Var
from VeraGridEngine.enumerations import DynamicIntegrationMethod


# Instruction codes are private to the tape builder and its numerical kernel.
_CONST, _STATE, _PARAM, _DIFF, _STEP = range(5)
_ADD, _SUB, _MUL, _DIV, _NEG, _POWER = range(5, 11)
_SIN, _COS, _EXP, _LOG, _SQRT, _ABS = range(11, 17)


@njit(cache=True)
def _evaluate_values(ops, left, right, literals, states, params, history,
                     d_history, history2, h, method, values, partial_left, partial_right):
    for i in range(len(ops)):
        op = ops[i]
        a = left[i]
        b = right[i]
        da = 0.0
        db = 0.0
        if op == _CONST:
            value = literals[i]
        elif op == _STATE:
            value = states[a]
        elif op == _PARAM:
            value = params[a]
        elif op == _DIFF:
            if method == 0:
                value = (2.0 / h) * (states[a] - history[a]) - d_history[b]
            elif method == 1:
                value = (states[a] - history[a]) / h
            else:
                value = (1.5 * states[a] - 2.0 * history[a] + 0.5 * history2[a]) / h
        elif op == _STEP:
            value = h
        elif op == _ADD:
            value = values[a] + values[b]
            da, db = 1.0, 1.0
        elif op == _SUB:
            value = values[a] - values[b]
            da, db = 1.0, -1.0
        elif op == _MUL:
            value = values[a] * values[b]
            da, db = values[b], values[a]
        elif op == _DIV:
            value = values[a] / values[b]
            da, db = 1.0 / values[b], -value / values[b]
        elif op == _NEG:
            value = -values[a]
            da = -1.0
        elif op == _POWER:
            exponent = literals[i]
            value = values[a] ** exponent
            if exponent != 0.0:
                da = exponent * values[a] ** (exponent - 1.0)
        elif op == _SIN:
            value = np.sin(values[a])
            da = np.cos(values[a])
        elif op == _COS:
            value = np.cos(values[a])
            da = -np.sin(values[a])
        elif op == _EXP:
            value = np.exp(values[a])
            da = value
        elif op == _LOG:
            value = np.log(values[a])
            da = 1.0 / (values[a] + 1e-15)
        elif op == _SQRT:
            value = np.sqrt(values[a])
            da = 0.5 / (value + 1e-15)
        else:  # _ABS
            value = abs(values[a])
            da = np.sign(values[a])
        values[i] = value
        partial_left[i] = da
        partial_right[i] = db


@njit(cache=True)
def _evaluate_jacobian(ops, left, right, roots, colors, indptr, indices,
                       partial_left, partial_right, h, method, tangents, data):
    tangents.fill(0.0)
    for i in range(len(ops)):
        op = ops[i]
        if op == _STATE:
            tangents[i, colors[left[i]]] = 1.0
        elif op == _DIFF:
            factor = 2.0 if method == 0 else (1.0 if method == 1 else 1.5)
            tangents[i, colors[left[i]]] = factor / h
        elif op >= _ADD:
            a = left[i]
            b = right[i]
            for color in range(tangents.shape[1]):
                value = partial_left[i] * tangents[a, color]
                if b >= 0:
                    value += partial_right[i] * tangents[b, color]
                tangents[i, color] = value

    for col in range(len(colors)):
        for pos in range(indptr[col], indptr[col + 1]):
            data[pos] = tangents[roots[indices[pos]], colors[col]]


class StructuralADTape:
    """Problem-local instruction arrays and reusable residual/Jacobian buffers."""

    def __init__(self, instructions, roots, colors, matrix, method):
        self.ops = np.asarray([item[0] for item in instructions], dtype=np.int32)
        self.left = np.asarray([item[1] for item in instructions], dtype=np.int32)
        self.right = np.asarray([item[2] for item in instructions], dtype=np.int32)
        self.literals = np.asarray([item[3] for item in instructions], dtype=np.float64)
        self.roots = np.asarray(roots, dtype=np.int32)
        self.colors = colors
        self.matrix = matrix
        self.method = method
        self.values = np.empty(len(instructions))
        self.partial_left = np.empty(len(instructions))
        self.partial_right = np.empty(len(instructions))
        self.tangents = np.empty((len(instructions), int(max(colors, default=-1)) + 1))

    def _values(self, states, params, history, d_history, h, history2):
        _evaluate_values(self.ops, self.left, self.right, self.literals,
                         states, params, history, d_history, history2, h, self.method,
                         self.values, self.partial_left, self.partial_right)

    def residual(self, states, params, history, d_history, h, residual_out, history2):
        self._values(states, params, history, d_history, h, history2)
        residual_out[:] = self.values[self.roots]

    def jacobian(self, states, params, history, d_history, h, history2):
        self._values(states, params, history, d_history, h, history2)
        _evaluate_jacobian(self.ops, self.left, self.right, self.roots, self.colors,
                           self.matrix.indptr, self.matrix.indices, self.partial_left,
                           self.partial_right, h, self.method, self.tangents, self.matrix.data)
        return self.matrix


def build_structural_ad_tape(equations, variables, parameters, colors, matrix, method,
                             diff_vars=None):
    """Build a tape when every expression has supported scalar AD semantics.

    Dynamic powers, binary functions and nested derivatives use the generated
    backend. Bound tangent storage to 128 MiB before allocating model buffers.
    """
    methods = {DynamicIntegrationMethod.DaeTrapezoidal: 0,
               DynamicIntegrationMethod.DaeBackEuler: 1,
               DynamicIntegrationMethod.DaeBDF2: 2}
    if method not in methods:
        return None
    var_map = {var.uid: i for i, var in enumerate(variables)}
    param_map = {var.uid: i for i, var in enumerate(parameters)}
    diff_map = {var.base_var.uid: i for i, var in enumerate(diff_vars or [])
                if var.base_var is not None}
    binary_ops = {'+': _ADD, '-': _SUB, '*': _MUL, '/': _DIV}
    functions = {'sin': _SIN, 'cos': _COS, 'exp': _EXP,
                 'log': _LOG, 'sqrt': _SQRT, 'abs': _ABS}
    instructions = []
    slots = {}
    n_colors = int(max(colors, default=-1)) + 1
    for equation in equations:
        stack = [(equation, False)]
        while stack:
            node, expanded = stack.pop()
            key = id(node)
            if key in slots:
                continue
            if not expanded:
                stack.append((node, True))
                if isinstance(node, BinOp):
                    if node.op == '**':
                        if not isinstance(node.right, Const) or not np.isrealobj(node.right.value):
                            return None
                        stack.append((node.left, False))
                    elif node.op in binary_ops:
                        stack.append((node.right, False))
                        stack.append((node.left, False))
                    else:
                        return None
                elif isinstance(node, UnOp) and node.op in ('+', '-'):
                    stack.append((node.operand, False))
                elif isinstance(node, Func) and node.op in functions:
                    stack.append((node.arg, False))
                elif not isinstance(node, (Const, Var)):
                    return None
                continue

            a, b, literal = -1, -1, 0.0
            if isinstance(node, Const):
                if not np.isrealobj(node.value):
                    return None
                op, literal = _CONST, float(node.value)
            elif isinstance(node, Var):
                if node.uid in var_map:
                    op, a = _STATE, var_map[node.uid]
                elif node.uid in param_map:
                    op, a = _PARAM, param_map[node.uid]
                elif node.base_var is not None:
                    if node.base_var.base_var is not None or node.base_var.uid not in var_map:
                        return None
                    op, a = _DIFF, var_map[node.base_var.uid]
                    b = diff_map.get(node.base_var.uid, a)
                elif node.name == 'h':
                    op = _STEP
                else:
                    return None
            elif isinstance(node, BinOp):
                a = slots[id(node.left)]
                if node.op == '**':
                    op, literal = _POWER, float(node.right.value)
                else:
                    op, b = binary_ops[node.op], slots[id(node.right)]
            elif isinstance(node, UnOp):
                if node.op == '+':
                    slots[key] = slots[id(node.operand)]
                    continue
                op, a = _NEG, slots[id(node.operand)]
            else:
                op, a = functions[node.op], slots[id(node.arg)]
            slots[key] = len(instructions)
            instructions.append((op, a, b, literal))
            if len(instructions) * n_colors * 8 > 128 * 1024 * 1024:
                return None
    roots = [slots[id(equation)] for equation in equations]
    return StructuralADTape(instructions, roots, colors, matrix, methods[method])
