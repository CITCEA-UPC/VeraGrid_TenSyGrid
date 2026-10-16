#!/usr/bin/env python3
"""EMT grid-forming converter model helpers for validation scripts.

This module mirrors the RMS combined GFM validation topology, but represents
the GFM source as a DC-fed EMT VSC using the phase-domain GFM template.
"""

from __future__ import annotations

import os
from typing import Any

import VeraGridEngine.api as vge
from VeraGridEngine.Utils.Symbolic.block import Block

from VeraGridEngine.Templates.Emt.emt_gfm_upc import build_emt_gfm_aggregated_model


def find_name_in_block(name: str, block: Block) -> Any:
    """Recursively find a variable/event parameter by exact name or prefix."""

    def find_match(search_block: Block, allow_prefix: bool):
        block_vars = (
            search_block.algebraic_vars
            + search_block.state_vars
            + search_block.diff_vars
            + search_block.in_vars
            + search_block.out_vars
            + list(search_block.event_dict.keys())
            + list(search_block.parameters.keys())
        )
        for var in block_vars:
            var_name = getattr(var, "name", None)
            if var_name is None:
                continue
            if name == var_name or (allow_prefix and var_name.startswith(f"{name}_")):
                return var

        for child in search_block.children:
            found = find_match(child, allow_prefix)
            if found is not None:
                return found
        return None

    exact = find_match(block, allow_prefix=False)
    if exact is not None:
        return exact
    return find_match(block, allow_prefix=True)


def build_emt_gfm_model(vf: vge.VarFactory, name: str = "GFM_EMT") -> Block:
    """Build the detailed RMS-adapted EMT GFM block used by validation.

    This is the EMT counterpart of the RMS combined GFM source: it keeps the
    RMS filter/capacitor/droop/voltage-loop/current-loop structure, while using
    instantaneous abc inputs, full EMT angle dynamics, Park/inverse-Park phase
    transforms, EMT power scaling, and phase-current injections.
    """

    return build_emt_gfm_aggregated_model(vf=vf, name=name)


def create_grid() -> tuple[vge.MultiCircuit, dict[str, Any]]:
    """Create the same AC topology as ``rms_gfm_combinedmodels.py``."""

    grid = vge.MultiCircuit(name="EMT GFM combined model", Sbase=100.0, fbase=50.0)

    bus1_ac = vge.Bus(name="Bus 1 (AC)", Vnom=230.0, is_slack=True)
    bus3_ac = vge.Bus(name="Bus 1_1 (AC)", Vnom=230.0)
    bus4_ac = vge.Bus(name="Bus 4 (AC)", Vnom=230.0)
    for bus in (bus1_ac, bus3_ac, bus4_ac):
        grid.add_bus(bus)

    gen_slack = vge.Generator(name="Generator", P=10.0, vset=0.98, Snom=grid.Sbase, freq=50.0)
    gen_gfm = vge.Generator(name="GFM Generator", P=4.0, vset=0.98, Snom=grid.Sbase, freq=50.0)
    ac_load = vge.Load(name="AC Load", P=3.0, Q=0.0)

    line13 = vge.Line(
        name="Line13",
        bus_from=bus3_ac,
        bus_to=bus1_ac,
        rate=100.0,
        r=0.01,
        x=0.05,
        b=0.0,
    )
    line34 = vge.Line(
        name="Line 3-4",
        bus_from=bus3_ac,
        bus_to=bus4_ac,
        r=0.02,
        x=0.08,
        b=0.0,
        rate=100.0,
    )

    grid.add_generator(bus=bus1_ac, api_obj=gen_slack)
    grid.add_generator(bus=bus3_ac, api_obj=gen_gfm)
    grid.add_load(bus=bus4_ac, api_obj=ac_load)
    grid.add_line(line13)
    grid.add_line(line34)
    grid.apply_all_branch_types()

    devices = {
        "bus_slack": bus1_ac,
        "bus_gfm": bus3_ac,
        "bus_load": bus4_ac,
        "gen_slack": gen_slack,
        "gen_gfm": gen_gfm,
        "load": ac_load,
        "line13": line13,
        "line34": line34,
    }
    return grid, devices
