"""IEEE9 EMT load-event case using the classic trigonometric synchronous generator."""

from __future__ import annotations

from typing import Any

from VeraGridEngine.Templates.Emt.generator_emt_type_template import (
    get_complete_generator_template_emt,
)
from VeraGridEngine.Utils.Symbolic.block import Block

from ieee9_emt_from_scratch import USE_CONVENTIONAL_THREE_PHASE_BASE
from ieee9_emt_multilinear_simulation import run_case


def get_ieee9_classic_generator(vf: Any, name: str) -> Block:
    """Build the controller-complete Sauer-Pai machine with sine/cosine Park transforms."""
    return get_complete_generator_template_emt(
        vf=vf,
        name=name,
        conventional_three_phase_base=USE_CONVENTIONAL_THREE_PHASE_BASE,
        frozen_excitation=True,
        multilinear_controls=False,
    ).block


def main() -> None:
    run_case(
        generator_builder=get_ieee9_classic_generator,
        case_title="classic trigonometric synchronous generator",
        output_stem="ieee9_emt_classic",
    )


if __name__ == "__main__":
    main()
