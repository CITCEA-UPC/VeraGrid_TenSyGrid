"""VeraGrid RMS model factories for the STAMP WSCC case."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .parameters import NOMINAL_FREQUENCY_HZ, OMEGA_BASE
from .parameters import STAMP_GFOL, STAMP_GFOR, STAMP_SG


def _children(block: Any) -> list[Any]:
    return list(getattr(block, "children", None) or [])


def _replace_event_constants(block: Any, vfactory: Any, values: Mapping[str, float]) -> set[str]:
    changed: set[str] = set()
    event_dict = getattr(block, "event_dict", None)
    if event_dict:
        for variable in list(event_dict):
            name = getattr(variable, "name", "")
            if name in values:
                event_dict[variable] = vfactory.add_const(values[name])
                changed.add(name)
    for child in _children(block):
        changed.update(_replace_event_constants(child, vfactory, values))
    return changed


def _disable_unconfigured_exciter_limiter(root: Any) -> bool:
    """Remove the stock exciter limiter that is absent from STAMP AC4A data."""
    from VeraGridEngine.Utils.Symbolic.symbolic import get_expression_vars

    for block in [root, *root.get_all_blocks()]:
        equations = getattr(block, "algebraic_eqs", None) or []
        for index, equation in enumerate(equations):
            variables = {variable.name: variable for variable in get_expression_vars(equation)}
            if "y_exciter4" in variables and "Efe" in variables and "heaviside" in str(equation):
                equations[index] = variables["y_exciter4"] - variables["Efe"]
                return True
    return False


def get_stamp_synchronous_generator_rms(vfactory: Any, name: str = "STAMP_SG1") -> Any:
    """Return the named-equation STAMP SG + AC4A + IEEEG1 model."""
    from .nonlinear_generator import build_stamp_generator_rms

    template = build_stamp_generator_rms(vfactory, STAMP_SG, name)
    p = STAMP_SG
    # VeraGrid uses M in the speed equation. STAMP stores H; its generated
    # machine equations use the same per-unit mechanical convention.
    values = {
        "fn": NOMINAL_FREQUENCY_HZ,
        "ws": OMEGA_BASE,
        "M": 2.0 * p.inertia_h,
        "D": p.damping_d,
        "Rs": p.rs,
        "Ra": p.rs,
        "Xl": p.xl,
        "Xd": p.xd,
        "Xq": p.xq,
        "Xd_prime": p.xd_prime,
        "Xq_prime": p.xq_prime,
        "Xd_2prime": p.xd_subtransient,
        "Xq_2prime": p.xq_subtransient,
        "Td0_prime": p.td0_prime,
        "Tq0_prime": p.tq0_prime,
        "Td0_2prime": p.td0_subtransient,
        "Tq0_2prime": p.tq0_subtransient,
        "TR": p.exciter_tr,
        "KA": p.exciter_ka,
        "TA": p.exciter_ta,
        "TB": p.exciter_tb,
        "TC": p.exciter_tc,
        "R": p.governor_r,
        "T1": p.governor_t1,
        "T2": p.governor_t2,
        "T3": p.governor_t3,
    }
    _replace_event_constants(template.block, vfactory, values)
    return template


def get_stamp_gfor_rms(vfactory: Any, name: str = "STAMP_GFOR1",
                       reference_omega: Any | None = None) -> Any:
    """Return the named-equation STAMP GFOR model."""
    from .source_linear_converters import build_stamp_source_linear_converter
    return build_stamp_source_linear_converter(vfactory, STAMP_GFOR, name, reference_omega)


def get_stamp_gfol_rms(vfactory: Any, name: str = "STAMP_GFOL2",
                       reference_omega: Any | None = None) -> Any:
    """Return the named-equation STAMP GFOL model."""
    from .source_linear_converters import build_stamp_source_linear_converter
    return build_stamp_source_linear_converter(vfactory, STAMP_GFOL, name, reference_omega)
