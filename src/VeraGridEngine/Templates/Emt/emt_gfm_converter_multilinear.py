"""Multilinear trigonometric reformulation for aggregated EMT GFM models."""

from __future__ import annotations

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Utils.Symbolic import symbolic as sym
from VeraGridEngine.Utils.Symbolic.block import Block, Var, find_name_in_block
from VeraGridEngine.Utils.Symbolic.symbolic import BinOp, Expr, Func, UnOp


def _trig_functions(expression: Expr) -> list[Func]:
    found: list[Func] = []

    def visit(node: Expr) -> None:
        if isinstance(node, Func):
            if node.op in {"sin", "cos"}:
                found.append(node)
            visit(node.arg)
        elif isinstance(node, BinOp):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, UnOp):
            visit(node.operand)

    visit(expression)
    return found


def make_gfm_trigonometry_multilinear(block: Block, vf: VarFactory) -> Block:
    """Replace every runtime sin(theta)/cos(theta) by differential auxiliaries.

    This operates on both VeraGrid's aggregated GFM builder and the equivalent
    Deliverable 3.3 combined-model builder. Initialization retains exact
    trigonometric expressions; only runtime equations are reformulated.
    """
    theta = find_name_in_block("theta", block)
    if theta is None:
        theta = next((var for var in block.get_all_vars() if var.name.startswith("theta_")), None)
    omega = find_name_in_block("omega", block)
    if omega is None:
        omega = next((var for var in block.get_all_vars() if var.name.startswith("omega_")), None)
    omega_base = next(
        (var for owner in block.get_all_blocks() for var in owner.event_dict
         if var.name == "omega_base" or var.name.startswith("omega_base_")),
        None,
    )
    if theta is None or omega is None or omega_base is None:
        raise RuntimeError("GFM model is missing theta, omega, or omega_base")

    u_cos = vf.add_var("u_cos_gfm")
    u_sin = vf.add_var("u_sin_gfm")
    d_cos = vf.add_diff_var("d_u_cos_gfm", base_var=u_cos)
    d_sin = vf.add_diff_var("d_u_sin_gfm", base_var=u_sin)

    replaced = 0
    for owner in block.get_all_blocks():
        rewritten = []
        for equation in owner.algebraic_eqs:
            mapping = {}
            for function in _trig_functions(equation):
                if str(function.arg) == str(theta):
                    mapping[function] = u_cos if function.op == "cos" else u_sin
            replaced += len(mapping)
            rewritten.append(equation.subs(mapping))
        owner.algebraic_eqs = rewritten
    if replaced == 0:
        raise RuntimeError("GFM multilinear reformulation found no runtime theta trigonometry")

    # Mirror the production GFM state equation theta_dot=+omega_b*omega.
    theta_rate = omega_base * omega
    block.state_vars.extend([u_cos, u_sin])
    block.diff_vars.extend([d_cos, d_sin])
    block.state_eqs.extend([-theta_rate * u_sin, theta_rate * u_cos])
    block.init_eqs.update({u_cos: sym.cos(theta), u_sin: sym.sin(theta)})
    block.diff_init_eqs.update({d_cos: -theta_rate * u_sin, d_sin: theta_rate * u_cos})
    block.reformulated_vars.extend([u_cos, u_sin])
    return block
