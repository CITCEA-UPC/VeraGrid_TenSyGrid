"""Exact multi-affine lift of the public STAMP EMT synchronous generator."""

from __future__ import annotations

import sys
from pathlib import Path


STAMP_ROOT = Path("/home/pablo/Desktop/eroots/STAMP_Public")
if str(STAMP_ROOT) not in sys.path:
    sys.path.insert(0, str(STAMP_ROOT))

from veragrid_stamp.emt_generator import build_stamp_generator_emt


def _nodes(expression):
    from VeraGridEngine.Utils.Symbolic.symbolic import BinOp, Func, UnOp
    result = []
    def visit(node):
        result.append(node)
        if isinstance(node, BinOp):
            visit(node.left); visit(node.right)
        elif isinstance(node, Func):
            visit(node.arg)
        elif isinstance(node, UnOp):
            visit(node.operand)
    visit(expression)
    return result


def _plus_terms(expression):
    from VeraGridEngine.Utils.Symbolic.symbolic import BinOp
    if isinstance(expression, BinOp) and expression.op == "+":
        return _plus_terms(expression.left) + _plus_terms(expression.right)
    return [expression]


def build_stamp_generator_emt_multilinear(vf, parameters, name="STAMP_SG1_EMT_ML"):
    """Lift terminal coordinates, exciter magnitude, and turbine division."""
    from VeraGridEngine.Utils.Symbolic import symbolic as sym
    from VeraGridEngine.Utils.Symbolic.symbolic import BinOp, Func, Var

    template = build_stamp_generator_emt(vf, parameters, name=name)
    block = template.block
    vm = next(variable for variable in block.algebraic_vars if variable.name.endswith(".Vm"))
    va = next(variable for variable in block.algebraic_vars if variable.name.endswith(".Va"))
    omega = next(variable for variable in block.state_vars if variable.name.endswith(".w_pu"))

    # Replace the two repeated Vm*cos(Va)/Vm*sin(Va) expressions everywhere
    # by the actual fixed-frame q/d terminal-voltage coordinates.  Their two
    # original wrapper equations then become direct linear definitions from
    # the live abc voltage, eliminating Vm/Va from the runtime system.
    candidates = {}
    for equation in block.state_eqs + block.algebraic_eqs:
        for node in _nodes(equation):
            variables = {variable.uid for variable in node.get_vars()}
            if variables != {vm.uid, va.uid}:
                continue
            functions = [item for item in _nodes(node) if isinstance(item, Func)]
            for function in functions:
                if function.arg is va and function.op in ("sin", "cos"):
                    current = candidates.get(function.op)
                    if current is None or len(str(node)) > len(str(current)):
                        candidates[function.op] = node
    if set(candidates) != {"sin", "cos"}:
        raise RuntimeError(f"Could not identify STAMP terminal q/d expressions: {candidates}")
    v_terminal_q = vf.add_var(f"{name}.v_terminal_q")
    v_terminal_d = vf.add_var(f"{name}.v_terminal_d")
    expression_targets = {
        str(candidates["cos"]): v_terminal_q,
        str(candidates["sin"]): v_terminal_d,
    }

    def terminal_mapping(equation):
        mapping = {}
        for node in _nodes(equation):
            replacement = expression_targets.get(str(node))
            if replacement is not None:
                mapping[node] = replacement
        return equation.subs(mapping).simplify()

    block.state_eqs = [terminal_mapping(equation) for equation in block.state_eqs]
    block.algebraic_eqs = [terminal_mapping(equation) for equation in block.algebraic_eqs]
    # The RMS model also owns symbolic initialization formulas for its current
    # and controller states.  They contain the same terminal-voltage
    # coordinates and must follow the runtime replacement as well.
    block.init_eqs = {
        variable: terminal_mapping(equation)
        for variable, equation in block.init_eqs.items()
    }
    block.diff_init_eqs = {
        variable: terminal_mapping(equation)
        for variable, equation in block.diff_init_eqs.items()
    }
    block.algebraic_vars = [variable for variable in block.algebraic_vars
                            if variable not in (vm, va)] + [v_terminal_q, v_terminal_d]

    # Obtain explicit initialization expressions from the two rewritten
    # voltage-interface equations (aux - live_voltage = 0).
    for auxiliary in (v_terminal_q, v_terminal_d):
        definition = next(
            equation for equation in block.algebraic_eqs
            if isinstance(equation, BinOp) and equation.op == "-"
            and isinstance(equation.left, Var)
            and equation.left.uid == auxiliary.uid
        )
        block.init_eqs[auxiliary] = definition.right
    block.init_eqs.pop(vm, None); block.init_eqs.pop(va, None)

    # Lift sqrt(vq^2+vd^2+eps) with duplicated factors so every monomial is
    # affine in each individual variable.
    sqrt_nodes = []
    for equation in block.state_eqs:
        sqrt_nodes.extend(node for node in _nodes(equation)
                          if isinstance(node, Func) and node.op == "sqrt")
    if len(sqrt_nodes) != 1:
        raise RuntimeError(f"Expected one runtime STAMP voltage magnitude, found {len(sqrt_nodes)}")
    sqrt_node = sqrt_nodes[0]
    squared_terms = [term for term in _plus_terms(sqrt_node.arg)
                     if isinstance(term, BinOp) and term.op == "*"
                     and str(term.left) == str(term.right)]
    if len(squared_terms) != 2:
        raise RuntimeError(f"Could not split STAMP voltage magnitude: {sqrt_node}")
    vsg_q_expression, vsg_d_expression = squared_terms[0].left, squared_terms[1].left
    vsg_q = vf.add_var(f"{name}.vsg_q_aux")
    vsg_d = vf.add_var(f"{name}.vsg_d_aux")
    vsg_q_copy = vf.add_var(f"{name}.vsg_q_copy")
    vsg_d_copy = vf.add_var(f"{name}.vsg_d_copy")
    magnitude = vf.add_var(f"{name}.vsg_mag_aux")
    magnitude_copy = vf.add_var(f"{name}.vsg_mag_copy")
    block.state_eqs = [equation.subs({sqrt_node: magnitude}).simplify()
                       for equation in block.state_eqs]
    block.algebraic_vars.extend([
        vsg_q, vsg_d, vsg_q_copy, vsg_d_copy, magnitude, magnitude_copy])
    block.algebraic_eqs.extend([
        vsg_q-vsg_q_expression,
        vsg_d-vsg_d_expression,
        vsg_q_copy-vsg_q,
        vsg_d_copy-vsg_d,
        magnitude_copy-magnitude,
        magnitude*magnitude_copy-vsg_q*vsg_q_copy-vsg_d*vsg_d_copy-vf.add_const(1e-12),
    ])
    block.init_eqs.update({
        vsg_q: vsg_q_expression, vsg_d: vsg_d_expression,
        vsg_q_copy: vsg_q, vsg_d_copy: vsg_d,
        magnitude: sym.sqrt(vsg_q*vsg_q+vsg_d*vsg_d+vf.add_const(1e-12)),
        magnitude_copy: magnitude,
    })

    # Replace tm/omega by tm*omega_inverse with an exact bilinear reciprocal
    # constraint.  The operating speed is nonzero, so this branch is regular.
    division_nodes = []
    for equation in block.state_eqs:
        division_nodes.extend(node for node in _nodes(equation)
                              if isinstance(node, BinOp) and node.op == "/"
                              and isinstance(node.right, Var) and node.right is omega)
    if len(division_nodes) != 1:
        raise RuntimeError(f"Expected one turbine/speed division, found {len(division_nodes)}")
    division = division_nodes[0]
    omega_inverse = vf.add_var(f"{name}.omega_inverse")
    block.state_eqs = [equation.subs({division: division.left*omega_inverse}).simplify()
                       for equation in block.state_eqs]
    block.algebraic_vars.append(omega_inverse)
    block.algebraic_eqs.append(omega*omega_inverse-vf.add_const(1.0))
    block.init_eqs[omega_inverse] = vf.add_const(1.0)/omega

    # Lift the EMT wrapper's network-frame angle exactly, as for the STAMP
    # converters.  The angle advances at the constant system base frequency.
    theta_grid = next(variable for variable in block.state_vars
                      if variable.name == f"{name}.theta_grid")
    u_cos_grid = vf.add_var(f"{name}.u_cos_grid")
    u_sin_grid = vf.add_var(f"{name}.u_sin_grid")
    d_u_cos_grid = vf.add_diff_var(f"d_{name}.u_cos_grid", base_var=u_cos_grid)
    d_u_sin_grid = vf.add_diff_var(f"d_{name}.u_sin_grid", base_var=u_sin_grid)
    shift = 2.0*3.141592653589793/3.0
    trig_replacements = {
        ("sin", str(theta_grid)): u_sin_grid,
        ("cos", str(theta_grid)): u_cos_grid,
        ("sin", f"({theta_grid}) - ({shift})"): -0.5*u_sin_grid-(3.0**0.5/2.0)*u_cos_grid,
        ("cos", f"({theta_grid}) - ({shift})"): -0.5*u_cos_grid+(3.0**0.5/2.0)*u_sin_grid,
        ("sin", f"({theta_grid}) + ({shift})"): -0.5*u_sin_grid+(3.0**0.5/2.0)*u_cos_grid,
        ("cos", f"({theta_grid}) + ({shift})"): -0.5*u_cos_grid-(3.0**0.5/2.0)*u_sin_grid,
    }
    def rewrite_grid_trig(equation):
        mapping = {}
        for node in _nodes(equation):
            if isinstance(node, Func) and node.op in ("sin", "cos"):
                replacement = trig_replacements.get((node.op, str(node.arg)))
                if replacement is not None:
                    mapping[node] = replacement
        return equation.subs(mapping).simplify()
    block.state_eqs = [rewrite_grid_trig(equation) for equation in block.state_eqs]
    block.algebraic_eqs = [rewrite_grid_trig(equation) for equation in block.algebraic_eqs]
    omega_base = vf.add_const(2.0*3.141592653589793*50.0)
    block.state_vars.extend([u_cos_grid, u_sin_grid])
    block.state_eqs.extend([-omega_base*u_sin_grid, omega_base*u_cos_grid])
    block.diff_vars.extend([d_u_cos_grid, d_u_sin_grid])
    block.init_eqs[u_cos_grid] = vf.add_const(1.0)
    block.init_eqs[u_sin_grid] = vf.add_const(0.0)
    block.diff_init_eqs[d_u_cos_grid] = vf.add_const(0.0)
    block.diff_init_eqs[d_u_sin_grid] = omega_base
    block.reformulated_vars.extend([
        v_terminal_q, v_terminal_d, vsg_q, vsg_d, vsg_q_copy,
        vsg_d_copy, magnitude, magnitude_copy, omega_inverse,
        u_cos_grid, u_sin_grid])
    return template
