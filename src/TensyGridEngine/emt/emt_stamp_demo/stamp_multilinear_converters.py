"""Multilinear lift of the public STAMP GFOR/GFOL EMT converter models."""

from __future__ import annotations

import numpy as np

from TensyGridEngine.emt.emt_stamp_demo.stamp_common import add_stamp_public_to_path

add_stamp_public_to_path()

from veragrid_stamp.emt_converters import build_stamp_converter_emt
from veragrid_stamp.parameters import OMEGA_BASE


def build_stamp_converter_emt_multilinear(vf, parameters, name: str,
                                           reference_omega=None):
    """Build STAMP's converter and replace its rotating-frame trig functions.

    ``u_cos`` and ``u_sin`` follow the exact Lie derivative of the original
    ``theta_grid`` state.  Park and inverse-Park expressions then contain only
    products of distinct variables and are therefore multi-affine.
    """
    from VeraGridEngine.Utils.Symbolic import symbolic as sym

    template = build_stamp_converter_emt(
        vf, parameters, name=name, reference_omega=reference_omega)
    block = template.block
    theta = next(variable for variable in block.state_vars
                 if variable.name == f"{name}.theta_grid")
    u_cos = vf.add_var(f"{name}.u_cos")
    u_sin = vf.add_var(f"{name}.u_sin")
    d_u_cos = vf.add_diff_var(f"d_{name}.u_cos", base_var=u_cos)
    d_u_sin = vf.add_diff_var(f"d_{name}.u_sin", base_var=u_sin)
    shift = 2.0*np.pi/3.0

    from VeraGridEngine.Utils.Symbolic.symbolic import BinOp, Func, UnOp

    def trig_nodes(expression):
        found = []
        def visit(node):
            if isinstance(node, Func):
                if node.op in ("sin", "cos"):
                    found.append(node)
                visit(node.arg)
            elif isinstance(node, BinOp):
                visit(node.left); visit(node.right)
            elif isinstance(node, UnOp):
                visit(node.operand)
        visit(expression)
        return found

    replacements_by_key = {
        ("sin", str(theta)): u_sin,
        ("cos", str(theta)): u_cos,
        ("sin", f"({theta}) - ({shift})"): -0.5*u_sin-(np.sqrt(3.0)/2.0)*u_cos,
        ("cos", f"({theta}) - ({shift})"): -0.5*u_cos+(np.sqrt(3.0)/2.0)*u_sin,
        ("sin", f"({theta}) + ({shift})"): -0.5*u_sin+(np.sqrt(3.0)/2.0)*u_cos,
        ("cos", f"({theta}) + ({shift})"): -0.5*u_cos-(np.sqrt(3.0)/2.0)*u_sin,
    }
    def rewrite(equation):
        mapping = {}
        for function in trig_nodes(equation):
            replacement = replacements_by_key.get((function.op, str(function.arg)))
            if replacement is None:
                raise RuntimeError(f"Unrecognized STAMP converter trig term: {function}")
            mapping[function] = replacement
        return equation.subs(mapping).simplify()

    block.state_eqs = [rewrite(equation) for equation in block.state_eqs]
    block.algebraic_eqs = [rewrite(equation) for equation in block.algebraic_eqs]
    block.state_vars.extend([u_cos, u_sin])
    block.state_eqs.extend([
        -vf.add_const(OMEGA_BASE)*u_sin,
        vf.add_const(OMEGA_BASE)*u_cos,
    ])
    block.diff_vars.extend([d_u_cos, d_u_sin])
    block.init_eqs[u_cos] = vf.add_const(1.0)
    block.init_eqs[u_sin] = vf.add_const(0.0)
    block.diff_init_eqs[d_u_cos] = vf.add_const(0.0)
    block.diff_init_eqs[d_u_sin] = vf.add_const(OMEGA_BASE)
    block.reformulated_vars.extend([u_cos, u_sin])
    return template
