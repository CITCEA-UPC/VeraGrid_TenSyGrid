"""Multilinear-trigonometric variants of the Sauer-Pai EMT generator.

The validated non-multilinear machine remains in
``generator_emt_type_template.py``.  This module derives an equivalent model
whose runtime Park transforms use the states ``u_cos = cos(theta_abs)`` and
``u_sin = sin(theta_abs)`` instead of symbolic trigonometric calls.
"""

from __future__ import annotations

import numpy as np

from VeraGridEngine.Devices.Dynamic.emt_template import EmtModelTemplate
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Templates.Emt.generator_emt_type_template import (
    get_complete_generator_template_emt,
    get_generator_sauer_pai_type_emt_template,
)
from VeraGridEngine.Utils.Symbolic import symbolic as sym
from VeraGridEngine.Utils.Symbolic.block import Block, find_name_in_block
from VeraGridEngine.Utils.Symbolic.symbolic import BinOp, Expr, Func, UnOp
from VeraGridEngine.enumerations import ParamPowerFlowReferenceType


def _trig_functions(expression: Expr) -> list[Func]:
    """Return every sine/cosine node in one symbolic expression."""
    found: list[Func] = []

    def visit(node: Expr) -> None:
        if isinstance(node, Func):
            if node.op in ("sin", "cos"):
                found.append(node)
            visit(node.arg)
        elif isinstance(node, BinOp):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, UnOp):
            visit(node.operand)

    visit(expression)
    return found


def _make_sauer_pai_trig_multilinear(block: Block, vf: VarFactory) -> None:
    """Replace the Sauer-Pai runtime trigonometric calls in place."""
    theta = find_name_in_block("theta_abs_", block)
    omega = find_name_in_block("omega_", block)
    omega_b = block.api_obj_mapping[ParamPowerFlowReferenceType.omega_base]
    if theta is None or omega is None:
        raise RuntimeError("Sauer-Pai generator is missing theta_abs or omega")

    u_cos = vf.add_var("u_cos_sauer_pai")
    u_sin = vf.add_var("u_sin_sauer_pai")
    d_u_cos = vf.add_diff_var("d_u_cos_sauer_pai", base_var=u_cos)
    d_u_sin = vf.add_diff_var("d_u_sin_sauer_pai", base_var=u_sin)

    c120 = float(np.cos(2.0 * np.pi / 3.0))
    s120 = float(np.sin(2.0 * np.pi / 3.0))
    replacements = {
        ("cos", str(theta)): u_cos,
        ("sin", str(theta)): u_sin,
        ("cos", f"({theta}) - (2.0943951023931953)"): u_cos * c120 + u_sin * s120,
        ("sin", f"({theta}) - (2.0943951023931953)"): u_sin * c120 - u_cos * s120,
        ("cos", f"({theta}) + (2.0943951023931953)"): u_cos * c120 - u_sin * s120,
        ("sin", f"({theta}) + (2.0943951023931953)"): u_sin * c120 + u_cos * s120,
    }

    rewritten: list[Expr] = []
    replaced_count = 0
    for equation in block.algebraic_eqs:
        mapping: dict[Expr, Expr] = {}
        for function in _trig_functions(equation):
            replacement = replacements.get((function.op, str(function.arg)))
            if replacement is not None:
                mapping[function] = replacement
        replaced_count += len(mapping)
        rewritten.append(equation.subs(mapping))
    if replaced_count != 12:
        raise RuntimeError(
            f"Expected 12 Sauer-Pai runtime trig terms, replaced {replaced_count}"
        )
    block.algebraic_eqs = rewritten

    block.state_vars.extend([u_cos, u_sin])
    block.diff_vars.extend([d_u_cos, d_u_sin])
    block.state_eqs.extend([
        -(omega_b * omega) * u_sin,
        (omega_b * omega) * u_cos,
    ])
    block.init_eqs[u_cos] = sym.cos(theta)
    block.init_eqs[u_sin] = sym.sin(theta)
    block.diff_init_eqs[d_u_cos] = -(omega_b * omega) * u_sin
    block.diff_init_eqs[d_u_sin] = (omega_b * omega) * u_cos


def get_generator_sauer_pai_type_emt_multilinear_template(
    vf: VarFactory,
    name: str = "sauer_pai_generator_emt_multilinear",
    conventional_three_phase_base: bool = False,
    mechanical_damping: float = 0.0,
    freeze_e_qp: bool = False,
) -> EmtModelTemplate:
    """Return the standalone Sauer-Pai machine with multilinear Park transforms."""
    template = get_generator_sauer_pai_type_emt_template(
        vf=vf,
        name=name,
        conventional_three_phase_base=conventional_three_phase_base,
        mechanical_damping=mechanical_damping,
        freeze_e_qp=freeze_e_qp,
    )
    _make_sauer_pai_trig_multilinear(template.block, vf)
    return template


def get_complete_generator_template_emt_multilinear(
    vf: VarFactory,
    name: str = "complete_generator_emt_multilinear",
    conventional_three_phase_base: bool = False,
    mechanical_damping: float = 0.0,
    frozen_controls: bool = False,
    frozen_excitation: bool = False,
    freeze_e_qp: bool = False,
    multilinear_controls: bool = True,
) -> EmtModelTemplate:
    """Return the complete controlled generator with a multilinear Sauer-Pai core."""
    template = get_complete_generator_template_emt(
        vf=vf,
        name=name,
        conventional_three_phase_base=conventional_three_phase_base,
        mechanical_damping=mechanical_damping,
        frozen_controls=frozen_controls,
        frozen_excitation=frozen_excitation,
        freeze_e_qp=freeze_e_qp,
        multilinear_controls=multilinear_controls,
    )
    if not template.block.children:
        raise RuntimeError("Complete Sauer-Pai generator has no machine child")
    _make_sauer_pai_trig_multilinear(template.block.children[0], vf)
    return template
