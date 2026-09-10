# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Standalone EMT simulation of the Kundur two-area system.

Workflow
--------
1. Build the 11-bus Kundur network.
2. Give all 230-kV lines a three-phase tower-backed ABC representation while
   preserving the original Kundur positive-sequence R/X/B exactly.
3. Use PF3-compatible transformer winding connections for the static
   three-phase power flow.
4. Attach three-wire EMT transformer/load models because the original Kundur
   benchmark does not define a neutral or zero-sequence grounding network.
5. Run the balanced positive-sequence power flow.
6. Build EmtProblemDae from the converged balanced operating point.
7. Run the selected EMT solver.
8. Plot representative bus voltages, generator speeds, and tie-line currents.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import VeraGridEngine.api as gce
from VeraGridEngine.enumerations import (
    DynamicIntegrationMethod,
    EmtInitializationMethod,
    EmtSolverTypes,
    ShuntConnectionType,
    WindingType,
)
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.EMT.solvers.StructuralVectorizedSolver import StructuralVectorizedSolver
from VeraGridEngine.Simulations.EMT.solvers.jit_symbolic_solver import JitSymbolicSolver
from VeraGridEngine.Simulations.EMT.solvers.solver_AD import JitAdSolver
from VeraGridEngine.Simulations.EMT.solvers.structural_compiled_solver import StructuralCompiledSolver
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Templates.Emt.generator_emt_type_template import get_complete_generator_template_emt
from VeraGridEngine.Templates.Emt.load_RLC_emt_template import get_shunt_rlc_combo_emt_template
from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
from VeraGridEngine.Templates.Emt.transformer_emt_template import get_series_transformer_emt_template
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Var
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model


# ==============================================================================
# USER SETTINGS
# ==============================================================================

CHOSEN_SOLVER = "SYMBOLIC"  # "SYMBOLIC", "AD", "VECTORIZED", "COMPILED"
TIME_STEP = float(os.environ.get("VERAGRID_KUNDUR_EMT_TIME_STEP", "5e-6"))
SIMULATION_TIME = float(os.environ.get("VERAGRID_KUNDUR_EMT_SIMULATION_TIME", "1.2"))
INITIALIZATION_METHOD_NAME = os.environ.get(
    "VERAGRID_KUNDUR_EMT_INITIALIZATION_METHOD", "Explicit"
)
FIX_PF_BUS_VOLTAGES_DURING_INITIALIZATION = os.environ.get(
    "VERAGRID_KUNDUR_EMT_FIX_PF_BUS_VOLTAGES", "0"
) == "1"
try:
    INITIALIZATION_METHOD = EmtInitializationMethod[INITIALIZATION_METHOD_NAME]
except KeyError as exc:
    valid_initialization_methods = ", ".join(method.name for method in EmtInitializationMethod)
    raise ValueError(
        f"Unknown initialization method '{INITIALIZATION_METHOD_NAME}'. "
        f"Use one of: {valid_initialization_methods}."
    ) from exc
EMT_TOLERANCE = 1e-6
ENABLE_PLOTS = os.environ.get("VERAGRID_KUNDUR_EMT_PLOTS", "1") != "0"
SHOW_PLOTS = os.environ.get("VERAGRID_KUNDUR_EMT_SHOW_PLOTS", "0") == "1"
PLOT_DIRECTORY = Path(os.environ.get(
    "VERAGRID_KUNDUR_EMT_PLOT_DIR",
    str(Path(__file__).with_name("kundur_emt_plots")),
))
SOLVER_VERBOSE = True
DENSE_THRESHOLD = 0

SOLVER_TYPE_MAP = {
    "SYMBOLIC": EmtSolverTypes.Symbolic,
    "AD": EmtSolverTypes.Automatic,
    "VECTORIZED": EmtSolverTypes.StructuralAD,
    "COMPILED": EmtSolverTypes.StructuralCompiled,
}

if CHOSEN_SOLVER not in SOLVER_TYPE_MAP:
    raise ValueError(
        f"Unknown solver '{CHOSEN_SOLVER}'. "
        "Use SYMBOLIC, AD, VECTORIZED or COMPILED."
    )


def find_name_in_block(name: str, block: Block) -> Var | None:
    """Find one symbolic variable by name in a block hierarchy."""
    for var in (
        block.algebraic_vars
        + block.state_vars
        + list(block.event_dict.keys())
        + block.diff_vars
    ):
        if name == var.name:
            return var

    for child in block.children:
        result = find_name_in_block(name, child)
        if result is not None:
            return result

    return None


def get_series(problem: EmtProblemDae, y: np.ndarray, block: Block, var_name: str) -> np.ndarray | None:
    """Return one simulated variable series if the variable exists."""
    var = find_name_in_block(var_name, block)
    if var is None:
        return None

    return y[:, problem.get_var_idx(var)]


#####################################################################################
#                    Kundur two-area grid
#####################################################################################

# The original rms_kundur.py is a positive-sequence benchmark.  Floquet/EMT,
# however, needs a phase-domain network and the 3-phase power flow needs phase
# information on every AC branch.  The 230-kV lines below therefore receive an
# OverheadLineType tower and explicit NABC matrices.
#
# Important modelling choice:
#   * the Kundur R/X/B values remain the authoritative positive-sequence data;
#   * the tower supplies the A/B/C architecture and a physically reasonable
#     zero-sequence ratio, because the original Kundur script contains no
#     zero-sequence line data;
#   * the stored phase-domain matrices are converted to a transposed-equivalent
#     line whose positive sequence is EXACTLY the original Kundur R/X/B.
#
# This keeps the balanced RMS benchmark and the EMT/Floquet network electrically
# consistent instead of replacing the Kundur line impedances by arbitrary tower
# impedances.

grid = gce.MultiCircuit(name="Kundur two-area RMS/EMT", Sbase=100.0, fbase=60.0)

KUNDUR_LINE_VNOM_KV = 230.0
KUNDUR_GEN_VNOM_KV = 20.0
KUNDUR_TRANSFORMER_SN_MVA = 900.0
KUNDUR_TRANSFORMER_X_PU = 0.15 * (grid.Sbase / KUNDUR_TRANSFORMER_SN_MVA)

# Symmetrical-component transform, sequence order [0, 1, 2].
_a = np.exp(1j * 2.0 * np.pi / 3.0)
_T_012_TO_ABC = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, _a ** 2, _a],
        [1.0, _a, _a ** 2],
    ],
    dtype=complex,
)
_T_ABC_TO_012 = np.linalg.inv(_T_012_TO_ABC)


def _safe_positive_ratio(numerator: float, denominator: float, default: float) -> float:
    """Return a finite positive ratio, otherwise a conservative fallback."""
    if abs(denominator) <= 1e-14:
        return default

    value = float(numerator / denominator)
    if not np.isfinite(value) or value <= 0.0:
        return default

    return value


def _build_kundur_230kv_tower() -> tuple[gce.OverheadLineType, gce.Wire]:
    """
    Build the physical three-phase tower used by all Kundur 230-kV lines.

    The original Kundur data set contains no conductor geometry.  This tower is
    therefore a reference physical architecture, not an attempt to reconstruct a
    historical tower.  Its zero-sequence relationship is used while the published
    positive-sequence R/X/B values are retained exactly.
    """
    wire = gce.Wire(
        name="Kundur 230-kV phase conductor",
        diameter=30.0,          # mm
        diameter_internal=0.0, # solid equivalent conductor
        is_tube=False,
        r=0.040,                # ohm/km; only used to establish physical ratios
        max_current=2.5,        # kA
    )

    tower = gce.OverheadLineType(
        name="Kundur 230-kV reference tower",
        Vnom=KUNDUR_LINE_VNOM_KV,
        earth_resistivity=100.0,
        frequency=grid.fBase,
    )

    # Three-wire overhead line: no explicit neutral conductor.  Carson earth
    # return inside OverheadLineType provides the ground-return contribution.
    tower.add_wire_relationship(wire=wire, xpos=-8.0, ypos=25.0, phase=1)  # A
    tower.add_wire_relationship(wire=wire, xpos=0.0, ypos=32.0, phase=2)   # B
    tower.add_wire_relationship(wire=wire, xpos=8.0, ypos=25.0, phase=3)   # C
    tower.compute()

    if not tower.is_computed():
        raise RuntimeError("The Kundur 230-kV overhead-line tower could not be computed.")

    return tower, wire


def _set_transposed_kundur_line_matrices(
    line: gce.Line,
    r1: float,
    x1: float,
    b1: float,
    rate: float,
) -> None:
    """
    Re-scale one tower-backed line to the Kundur positive-sequence values.

    ``line.apply_template()`` first creates valid NABC phase flags and physical
    matrices from the tower.  We retain the tower-derived zero/positive sequence
    ratios, but rebuild a transposed ABC equivalent so the positive-sequence
    impedance/shunt of the 3-phase network exactly matches the RMS benchmark.
    """
    tower_r1 = float(line.R)
    tower_x1 = float(line.X)
    tower_b1 = float(line.B)
    tower_r0 = float(line.R0)
    tower_x0 = float(line.X0)
    tower_b0 = float(line.B0)

    r0_ratio = _safe_positive_ratio(tower_r0, tower_r1, default=3.0)
    x0_ratio = _safe_positive_ratio(tower_x0, tower_x1, default=3.0)
    b0_ratio = _safe_positive_ratio(tower_b0, tower_b1, default=1.0)

    r0 = r1 * r0_ratio
    x0 = x1 * x0_ratio
    b0 = b1 * b0_ratio

    # Passive transposed line: negative sequence equals positive sequence.
    z012 = np.diag(
        np.array(
            [
                complex(r0, x0),
                complex(r1, x1),
                complex(r1, x1),
            ],
            dtype=complex,
        )
    )
    zabc = _T_012_TO_ABC @ z012 @ _T_ABC_TO_012
    yabc_series = np.linalg.inv(zabc)

    # OverheadLineType stores line.ysh with the historical 1e6 scaling.  Keep
    # that storage convention so both PowerFlow3Ph and the EMT static mapper
    # interpret the matrix exactly as they do for a normal tower-backed line.
    y012_shunt_stored = np.diag(
        1j * 1e6 * np.array([b0, b1, b1], dtype=float)
    )
    yabc_shunt_stored = _T_012_TO_ABC @ y012_shunt_stored @ _T_ABC_TO_012

    ys_nabc = np.zeros((4, 4), dtype=complex)
    ysh_nabc = np.zeros((4, 4), dtype=complex)
    ys_nabc[1:4, 1:4] = yabc_series
    ysh_nabc[1:4, 1:4] = yabc_shunt_stored

    line.ys.values = ys_nabc
    line.ysh.values = ysh_nabc

    for admittance_matrix in (line.ys, line.ysh):
        admittance_matrix.phN = False
        admittance_matrix.phA = True
        admittance_matrix.phB = True
        admittance_matrix.phC = True

    # Scalars used by the ordinary positive-sequence PF/RMS path.
    line.R = r1
    line.X = x1
    line.B = b1
    line.R0 = r0
    line.X0 = x0
    line.B0 = b0
    line.R2 = r1
    line.X2 = x1
    line.B2 = b1
    line.rate = rate


def _make_kundur_line(
    name: str,
    bus_from: gce.Bus,
    bus_to: gce.Bus,
    r: float,
    x: float,
    b: float,
    rate: float,
    tower: gce.OverheadLineType,
) -> gce.Line:
    """Create one Kundur 230-kV tower-backed, three-phase transmission line."""
    zbase_ohm = KUNDUR_LINE_VNOM_KV ** 2 / grid.Sbase
    _, tower_x1_ohm_per_km, _, _ = tower.get_sequence_values(circuit_idx=1, seq=1)

    if tower_x1_ohm_per_km > 1e-12:
        length_km = x * zbase_ohm / tower_x1_ohm_per_km
    else:
        length_km = 1.0

    length_km = max(float(length_km), 1e-3)

    line = gce.Line(
        name=name,
        bus_from=bus_from,
        bus_to=bus_to,
        r=r,
        x=x,
        b=b,
        rate=rate,
        length=length_km,
    )

    line.apply_template(tower, grid.Sbase, grid.fBase)
    _set_transposed_kundur_line_matrices(
        line=line,
        r1=r,
        x1=x,
        b1=b,
        rate=rate,
    )
    return line


def _make_kundur_transformer(
    name: str,
    bus_hv: gce.Bus,
    bus_lv: gce.Bus,
) -> gce.Transformer2W:
    """Create one 230/20-kV Kundur generator step-up transformer."""
    trafo = gce.Transformer2W(
        name=name,
        bus_from=bus_hv,
        bus_to=bus_lv,
        HV=KUNDUR_LINE_VNOM_KV,
        LV=KUNDUR_GEN_VNOM_KV,
        nominal_power=KUNDUR_TRANSFORMER_SN_MVA,
        design_rate=KUNDUR_TRANSFORMER_SN_MVA,
        rate=KUNDUR_TRANSFORMER_SN_MVA,
        r=0.0,
        x=KUNDUR_TRANSFORMER_X_PU,
        g=0.0,
        b=0.0,
        r0=0.0,
        x0=KUNDUR_TRANSFORMER_X_PU,
        b0=0.0,
        r2=0.0,
        x2=KUNDUR_TRANSFORMER_X_PU,
        b2=0.0,
    )

    # PF3 and EMT need slightly different representations here.
    #
    # The current VeraGrid 3-phase Transformer2W admittance implementation
    # supports grounded/neutral star, delta and zig-zag combinations, but not
    # FloatingStar.  Use GroundedStar/GroundedStar for the STATIC PF3 model so
    # the transformer is stamped into the 3-phase network.
    #
    # The original Kundur benchmark has no zero-sequence/neutral data.  The EMT
    # symbolic model is therefore attached separately as a three-wire ABC model
    # (see the EMT-model section below), avoiding an invented external neutral.
    trafo.conn_f = WindingType.GroundedStar
    trafo.conn_t = WindingType.GroundedStar
    trafo.vector_group_number = 0
    trafo.phases = np.array([1, 2, 3], dtype=int)
    return trafo


def _validate_kundur_line_positive_sequence(lines: list[gce.Line]) -> None:
    """Verify that each stored ABC matrix reproduces its scalar Kundur R/X/B."""
    max_z_error = 0.0
    max_b_error = 0.0

    for line in lines:
        zabc = np.linalg.inv(np.asarray(line.ys.values[1:4, 1:4], dtype=complex))
        z012 = _T_ABC_TO_012 @ zabc @ _T_012_TO_ABC

        # Undo the OverheadLineType historical shunt-storage factor.
        yabc = np.asarray(line.ysh.values[1:4, 1:4], dtype=complex) / 1e6
        y012 = _T_ABC_TO_012 @ yabc @ _T_012_TO_ABC

        max_z_error = max(max_z_error, abs(z012[1, 1] - complex(line.R, line.X)))
        max_b_error = max(max_b_error, abs(y012[1, 1].imag - line.B))

    if max_z_error > 1e-10 or max_b_error > 1e-10:
        raise RuntimeError(
            "Kundur line ABC conversion is inconsistent with the positive-sequence data: "
            f"max |dZ1|={max_z_error:.3e}, max |dB1|={max_b_error:.3e}."
        )

    print(
        "Kundur 3ph line check: "
        f"max |dZ1|={max_z_error:.3e}, max |dB1|={max_b_error:.3e}"
    )


#####################################################################################
#                    Buses
#####################################################################################

bus1 = gce.Bus(name="Bus1", Vnom=20.0)
bus2 = gce.Bus(name="Bus2", Vnom=20.0)
bus3 = gce.Bus(name="Bus3", Vnom=20.0, is_slack=True)
bus4 = gce.Bus(name="Bus4", Vnom=20.0)
bus5 = gce.Bus(name="Bus5", Vnom=230.0)
bus6 = gce.Bus(name="Bus6", Vnom=230.0)
bus7 = gce.Bus(name="Bus7", Vnom=230.0)
bus8 = gce.Bus(name="Bus8", Vnom=230.0)
bus9 = gce.Bus(name="Bus9", Vnom=230.0)
bus10 = gce.Bus(name="Bus10", Vnom=230.0)
bus11 = gce.Bus(name="Bus11", Vnom=230.0)

for bus in [bus1, bus2, bus3, bus4, bus5, bus6, bus7, bus8, bus9, bus10, bus11]:
    grid.add_bus(bus)



#####################################################################################
#                    Three-phase overhead-line architecture
#####################################################################################

kundur_tower, kundur_wire = _build_kundur_230kv_tower()
grid.add_wire(kundur_wire)
grid.add_overhead_line(kundur_tower)

line0 = _make_kundur_line("line 5-6-1", bus5, bus6, 0.00500, 0.05000, 0.02187, 750.0, kundur_tower)
line1 = _make_kundur_line("line 5-6-2", bus5, bus6, 0.00500, 0.05000, 0.02187, 750.0, kundur_tower)
line2 = _make_kundur_line("line 6-7-1", bus6, bus7, 0.00300, 0.03000, 0.00583, 700.0, kundur_tower)
line3 = _make_kundur_line("line 6-7-2", bus6, bus7, 0.00300, 0.03000, 0.00583, 700.0, kundur_tower)
line4 = _make_kundur_line("line 6-7-3", bus6, bus7, 0.00300, 0.03000, 0.00583, 700.0, kundur_tower)
line5 = _make_kundur_line("line 7-8-1", bus7, bus8, 0.01100, 0.11000, 0.19250, 400.0, kundur_tower)
line6 = _make_kundur_line("line 7-8-2", bus7, bus8, 0.01100, 0.11000, 0.19250, 400.0, kundur_tower)
line7 = _make_kundur_line("line 8-9-1", bus8, bus9, 0.01100, 0.11000, 0.19250, 400.0, kundur_tower)
line8 = _make_kundur_line("line 8-9-2", bus8, bus9, 0.01100, 0.11000, 0.19250, 400.0, kundur_tower)
line9 = _make_kundur_line("line 9-10-1", bus9, bus10, 0.00300, 0.03000, 0.00583, 700.0, kundur_tower)
line10 = _make_kundur_line("line 9-10-2", bus9, bus10, 0.00300, 0.03000, 0.00583, 700.0, kundur_tower)
line11 = _make_kundur_line("line 9-10-3", bus9, bus10, 0.00300, 0.03000, 0.00583, 700.0, kundur_tower)
line12 = _make_kundur_line("line 10-11-1", bus10, bus11, 0.00500, 0.05000, 0.02187, 750.0, kundur_tower)
line13 = _make_kundur_line("line 10-11-2", bus10, bus11, 0.00500, 0.05000, 0.02187, 750.0, kundur_tower)

transmission_lines = [
    line0, line1, line2, line3, line4, line5, line6,
    line7, line8, line9, line10, line11, line12, line13,
]

_validate_kundur_line_positive_sequence(transmission_lines)

for line in transmission_lines:
    grid.add_line(line)


#####################################################################################
#                    Generator step-up transformers
#####################################################################################

trafo_G1 = _make_kundur_transformer("trafo 5-1", bus5, bus1)
trafo_G2 = _make_kundur_transformer("trafo 6-2", bus6, bus2)
trafo_G3 = _make_kundur_transformer("trafo 11-3", bus11, bus3)
trafo_G4 = _make_kundur_transformer("trafo 10-4", bus10, bus4)
transformers = [trafo_G1, trafo_G2, trafo_G3, trafo_G4]

for transformer in transformers:
    grid.add_transformer2w(transformer)


#####################################################################################
#                    Loads and generators
#####################################################################################

load_p = float(os.environ.get("VERAGRID_KUNDUR_LOAD_P", "9.999999"))
load_q = float(os.environ.get("VERAGRID_KUNDUR_LOAD_Q", "0.999999"))

load1 = gce.Load(
    name="load1",
    P=load_p,
    Q=load_q,
    P1=load_p / 3.0,
    P2=load_p / 3.0,
    P3=load_p / 3.0,
    Q1=load_q / 3.0,
    Q2=load_q / 3.0,
    Q3=load_q / 3.0,
)
load2 = gce.Load(
    name="load2",
    P=load_p,
    Q=load_q,
    P1=load_p / 3.0,
    P2=load_p / 3.0,
    P3=load_p / 3.0,
    Q1=load_q / 3.0,
    Q2=load_q / 3.0,
    Q3=load_q / 3.0,
)
load1.conn = ShuntConnectionType.FloatingStar
load2.conn = ShuntConnectionType.FloatingStar

xd = 0.3 * grid.Sbase / KUNDUR_TRANSFORMER_SN_MVA

gen1 = gce.Generator(name="Gen1", P=10.0, vset=1.03, Snom=900.0, x1=xd, r1=0.0, freq=60.0)
gen2 = gce.Generator(name="Gen2", P=10.0, vset=1.01, Snom=900.0, x1=xd, r1=0.0, freq=60.0)
gen3 = gce.Generator(name="Gen3", P=10.0, vset=1.03, Snom=900.0, x1=xd, r1=0.0, freq=60.0)
gen4 = gce.Generator(name="Gen4", P=10.0, vset=1.01, Snom=900.0, x1=xd, r1=0.0, freq=60.0)

generators = [gen1, gen2, gen3, gen4]
loads = [load1, load2]

grid.add_load(bus=bus7, api_obj=load1)
grid.add_load(bus=bus9, api_obj=load2)

grid.add_generator(bus=bus1, api_obj=gen1)
grid.add_generator(bus=bus2, api_obj=gen2)
grid.add_generator(bus=bus3, api_obj=gen3)
grid.add_generator(bus=bus4, api_obj=gen4)

# EMT bus models must be present before device EMT wiring is created.
for bus in grid.buses:
    get_bus_emt_template(grid, bus)

print(
    "Kundur network built: "
    f"{len(grid.buses)} buses, {len(grid.lines)} 230-kV lines, "
    f"{len(grid.transformers2w)} transformers, {len(grid.generators)} generators, "
    f"{len(grid.loads)} loads."
)

print(
    "Kundur transformer architecture: "
    "PF3=GroundedStar/GroundedStar, EMT=three-wire ABC."
)


#####################################################################################
#                    EMT models
#####################################################################################

for generator in generators:
    generator_emt_model = get_complete_generator_template_emt(
        vf=grid.var_factory,
        conventional_three_phase_base=True,
    ).block
    set_emt_model(device=generator, model=generator_emt_model, var_factory=grid.var_factory)

for line_idx, line in enumerate(transmission_lines):
    line_emt_model = get_pi_line_emt_template(
        vf=grid.var_factory,
        phN=False,
        phA=True,
        phB=True,
        phC=True,
        name=f"Pi_kundur_line_{line_idx}",
        numerical_damping_conductance=0.0,
    ).block
    set_emt_model(device=line, model=line_emt_model, var_factory=grid.var_factory)

for transformer_idx, transformer in enumerate(transformers):
    # Kundur's step-up transformers contain only series R/X and a fixed tap.
    transformer_emt_model = get_series_transformer_emt_template(
        vf=grid.var_factory,
        name=f"Kundur_transformer_{transformer_idx}_emt",
        r=transformer.R,
        x=transformer.X,
        tap_module=transformer.tap_module,
    ).block
    set_emt_model(device=transformer, model=transformer_emt_model, var_factory=grid.var_factory)

for load in loads:
    load_emt_model = get_shunt_rlc_combo_emt_template(
        vf=grid.var_factory,
        include_r=True,
        include_l=abs(load_q) > 1.0e-15,
        include_c=False,
        phA=True,
        phB=True,
        phC=True,
        connection_type=ShuntConnectionType.FloatingStar,
        name=f"{load.name}_RL_emt",
    ).block
    set_emt_model(device=load, model=load_emt_model, var_factory=grid.var_factory)


def _validate_three_wire_emt_interfaces() -> None:
    """
    Verify that the Kundur EMT network exposes only A/B/C terminals externally.

    The source Kundur benchmark contains positive-sequence data only. Its
    phase-domain realization therefore uses a three-wire ABC external network;
    floating star points remain internal to transformer/load models.
    """
    neutral_ref_names = {
        "v_N", "i_N",
        "vf_N", "if_N",
        "vt_N", "it_N",
    }
    offenders: list[str] = []

    devices = list(transformers) + list(loads) + list(transmission_lines) + list(generators)

    for device in devices:
        model = device.emt_model
        for block in model.get_all_blocks():
            mapping = block.external_mapping
            if mapping is None:
                continue

            for ref, mapped_var in mapping.items():
                if mapped_var is None:
                    continue

                ref_name = getattr(ref, "value", str(ref))
                if ref_name in neutral_ref_names:
                    offenders.append(
                        f"{device.name}: {ref_name} -> {mapped_var.name}"
                    )

    if offenders:
        raise RuntimeError(
            "Kundur EMT three-wire interface validation failed:\n  - "
            + "\n  - ".join(offenders)
        )

    print("Kundur EMT interface check: three-wire ABC, no external neutral ports.")


_validate_three_wire_emt_interfaces()




# ==============================================================================
# THREE-PHASE POWER FLOW
# ==============================================================================

pf_options = gce.PowerFlowOptions(
    solver_type=gce.SolverType.NR,
    retry_with_other_methods=False,
    verbose=0,
    initialize_with_existing_solution=True,
    tolerance=1e-6,
    max_iter=25,
    control_q=False,
    control_taps_modules=True,
    control_taps_phase=True,
    control_remote_voltage=True,
    orthogonalize_controls=True,
    apply_temperature_correction=True,
    branch_impedance_tolerance_mode=gce.BranchImpedanceMode.Specified,
    distributed_slack=False,
    ignore_single_node_islands=False,
    trust_radius=1.0,
    backtracking_parameter=0.05,
    use_stored_guess=False,
    initialize_angles=False,
    generate_report=False,
)

power_flow = PowerFlowDriver(grid, pf_options)
power_flow.run()
pf_results = power_flow.results
if not bool(pf_results.converged):
    raise RuntimeError("Kundur balanced power flow did not converge")
print("Balanced PF converged: True")


# ==============================================================================
# EMT PROBLEM
# ==============================================================================

emt_options = EmtOptions(
    time_step=TIME_STEP,
    simulation_time=SIMULATION_TIME,
    tolerance=EMT_TOLERANCE,
    solver_type=SOLVER_TYPE_MAP[CHOSEN_SOLVER],
    integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
    initialization_method=INITIALIZATION_METHOD,
    init_fix_pf_bus_voltages=FIX_PF_BUS_VOLTAGES_DURING_INITIALIZATION,
    conventional_three_phase_base=True,
    verbose=1,
)


def create_problem() -> EmtProblemDae:
    """Build the balanced Kundur EMT DAE from its positive-sequence PF."""
    return EmtProblemDae(
        grid=grid,
        options=emt_options,
        pf_results_3ph=None,
        pf_results=pf_results,
    )


def create_solver(solver_key: str, problem: EmtProblemDae) -> Any:
    """Create the selected EMT solver."""
    common = dict(
        problem=problem,
        t0=0.0,
        t_end=emt_options.simulation_time,
        h=emt_options.time_step,
        method=emt_options.integration_method,
        pred_method=DynamicIntegrationMethod.OdeEuler,
        dense_threshold=DENSE_THRESHOLD,
        verbose=SOLVER_VERBOSE,
    )

    if solver_key == "SYMBOLIC":
        return JitSymbolicSolver(**common)

    if solver_key == "AD":
        return JitAdSolver(**common)

    if solver_key == "VECTORIZED":
        return StructuralVectorizedSolver(
            **common,
            auto_vectorization=False,
        )

    if solver_key == "COMPILED":
        return StructuralCompiledSolver(
            **common,
            auto_build=False,
        )

    raise ValueError(
        f"Unknown solver '{solver_key}'. "
        "Use SYMBOLIC, AD, VECTORIZED or COMPILED."
    )


def build_solver_backend(solver_key: str, solver: Any, problem: EmtProblemDae) -> None:
    """Build the backend required by the selected solver."""
    use_dense_solver = problem.get_all_vars_number() <= DENSE_THRESHOLD
    use_sparse = not use_dense_solver

    if solver_key == "SYMBOLIC":
        solver.build_jit_kernel(emt_options.integration_method)
        solver._build_jit_symbolic_hybrid(
            emt_options.integration_method,
            use_sparse=use_sparse,
        )

        if emt_options.integration_method == DynamicIntegrationMethod.DaeTrapezoidal:
            solver.build_jit_kernel(DynamicIntegrationMethod.DaeBackEuler)
            solver._build_jit_symbolic_hybrid(
                DynamicIntegrationMethod.DaeBackEuler,
                use_sparse=use_sparse,
            )
        return

    if solver_key == "AD":
        solver.build_jit_ad()
        return

    if solver_key == "VECTORIZED":
        solver.auto_detect_vectorization(emt_options.integration_method)
        return

    if solver_key == "COMPILED":
        solver._build_vectorized_backend(emt_options.integration_method)
        return

    raise ValueError(
        f"Unknown solver '{solver_key}'. "
        "Use SYMBOLIC, AD, VECTORIZED or COMPILED."
    )


problem_build_t0 = perf_counter()
problem = create_problem()
problem_build_s = perf_counter() - problem_build_t0

print("\nEMT problem built")
print(f"  build time: {problem_build_s:.6f} s")
print(f"  states: {problem.get_states_number()}")
print(f"  algebraic vars: {problem.get_algebraic_var_number()}")
print(f"  differential vars: {problem.get_diff_var_number()}")
print(f"  runtime parameters: {problem.get_variable_parameter_number()}")
if problem.initialization_report is not None:
    initialization_report = problem.initialization_report
    print("  initialization:")
    print(f"    requested: {initialization_report.method_requested}")
    print(f"    used: {initialization_report.method_used}")
    print(f"    status: {initialization_report.status}")
    print(f"    unknown variables: {initialization_report.unknown_var_count}")
    print(f"    initial residual inf: {initialization_report.initial_residual_inf:.12e}")
    print(f"    final residual inf: {initialization_report.final_residual_inf:.12e}")
    print(f"    Newton iterations: {initialization_report.newton_iterations}")
    print(f"    message: {initialization_report.message}")

init_info = problem.get_init_guess_info()
print(f"\nEMT initialization guesses\n{init_info}")

print("\n" + "=" * 72)
print(f"Initializing EMT solver: {CHOSEN_SOLVER}")
print("=" * 72)

solver = create_solver(CHOSEN_SOLVER, problem)

backend_build_t0 = perf_counter()
build_solver_backend(CHOSEN_SOLVER, solver, problem)
backend_build_s = perf_counter() - backend_build_t0

boundary_updater = cast(Any, problem)

simulation_t0 = perf_counter()
t, y, dy, well_initialized, converged = solver.simulate(
    boundary_updater=boundary_updater
)
simulation_s = perf_counter() - simulation_t0

print("\nEMT simulation finished")
print(f"  well_initialized: {well_initialized}")
print(f"  converged: {converged}")
print(f"  backend build: {backend_build_s:.6f} s")
print(f"  simulation: {simulation_s:.6f} s")
print(f"  time samples: {len(t)}")
print(f"  solution shape: {y.shape}")
print(f"  derivative shape: {dy.shape}")

if not well_initialized or not converged:
    raise RuntimeError(
        "Kundur EMT steady-state validation failed: "
        f"well_initialized={well_initialized}, converged={converged}"
    )

build_stats = solver.get_backend_build_stats()
if build_stats:
    print("\nBackend build stats")
    for key, value in build_stats.items():
        print(f"  {key}: {value:.6f} s")


def get_initial_derivative(block: Block, variable_name: str) -> float | None:
    """Return the stored t=0 derivative of a state in one device model."""
    variable = find_name_in_block(variable_name, block)
    if variable is None:
        return None
    problem_variable = next(
        (candidate for candidate in problem.get_state_vars() if candidate.uid == variable.uid),
        None,
    )
    if problem_variable is None:
        return None
    differential = next(
        (candidate for candidate in problem.get_diff_vars()
         if candidate.base_var is not None and candidate.base_var.uid == problem_variable.uid),
        None,
    )
    if differential is None:
        return None
    return float(dy[0, problem.get_diff_var_idx(differential)])


print("\nInitial derivative audit")
stationary_state_names = (
    "omega_", "psi_d_", "psi_q_", "psi_0_", "e_qp_", "e_dp_",
    "psi_pp_d_", "psi_pp_q_", "y_gov0", "y2_3_gov", "Vf",
    "y_exciter1", "y_exciter2", "y_exciter3", "y_exciter4",
)
for generator in generators:
    stationary_derivatives = []
    for state_name in stationary_state_names:
        derivative = get_initial_derivative(generator.emt_model, state_name)
        if derivative is not None:
            stationary_derivatives.append((abs(derivative), state_name, derivative))
    stationary_derivatives.sort(reverse=True)
    magnitude, state_name, derivative = stationary_derivatives[0]
    print(
        f"  {generator.name} largest nominally-zero derivative: "
        f"{state_name}={derivative:+.12e}"
    )

omega_base = 2.0 * np.pi * grid.fBase
rotating_residuals = []
for device in list(transmission_lines) + list(transformers) + list(loads):
    if device in loads:
        phase_names = ("iL_A", "iL_B", "iL_C")
    else:
        phase_names = ("i_ser_A", "i_ser_B", "i_ser_C")
    phase_vars = [find_name_in_block(name, device.emt_model) for name in phase_names]
    if any(variable is None for variable in phase_vars):
        continue
    problem_phase_vars = [next(
        (candidate for candidate in problem.get_state_vars() if candidate.uid == variable.uid),
        None,
    ) for variable in phase_vars]
    if any(variable is None for variable in problem_phase_vars):
        continue
    phase_diff_vars = [next(
        (candidate for candidate in problem.get_diff_vars()
         if candidate.base_var is not None and candidate.base_var.uid == variable.uid),
        None,
    ) for variable in problem_phase_vars]
    if any(variable is None for variable in phase_diff_vars):
        continue
    values = np.asarray([y[0, problem.get_var_idx(variable)] for variable in phase_vars])
    derivatives = np.asarray([
        dy[0, problem.get_diff_var_idx(variable)] for variable in phase_diff_vars
    ])
    expected = (omega_base / np.sqrt(3.0)) * np.asarray([
        values[2] - values[1],
        values[0] - values[2],
        values[1] - values[0],
    ])
    for phase_index, phase_name in enumerate(phase_names):
        rotating_residuals.append((
            abs(float(derivatives[phase_index] - expected[phase_index])),
            device.name,
            phase_name,
            float(derivatives[phase_index]),
            float(expected[phase_index]),
        ))
rotating_residuals.sort(reverse=True)
print("  Largest rotating-ABC derivative residuals:")
for residual, device_name, state_name, derivative, expected in rotating_residuals[:12]:
    print(
        f"    {device_name}.{state_name}: residual={residual:.12e}, "
        f"stored={derivative:+.12e}, expected={expected:+.12e}"
    )

if not ENABLE_PLOTS:
    raise SystemExit(0)


# ==============================================================================
# PLOTS
# ==============================================================================

# Representative bus voltages:
#   Bus1: generator terminal
#   Bus7: left-area load bus
#   Bus9: right-area load bus
selected_buses = [bus1, bus7, bus9]

fig, axes = plt.subplots(len(selected_buses), 1, figsize=(12, 9), sharex=True)
if len(selected_buses) == 1:
    axes = [axes]

for ax, bus in zip(axes, selected_buses):
    v_a = get_series(problem, y, bus.emt_model, "v_A")
    v_b = get_series(problem, y, bus.emt_model, "v_B")
    v_c = get_series(problem, y, bus.emt_model, "v_C")

    if v_a is not None:
        ax.plot(t, v_a, label="A")
    if v_b is not None:
        ax.plot(t, v_b, label="B")
    if v_c is not None:
        ax.plot(t, v_c, label="C")

    ax.set_title(f"{bus.name} instantaneous voltage")
    ax.set_ylabel("Voltage [p.u.]")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time [s]")
fig.tight_layout()
PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
bus_voltage_plot = PLOT_DIRECTORY / "kundur_bus_voltages.png"
fig.savefig(bus_voltage_plot, dpi=180, bbox_inches="tight")
plt.close(fig)


# Representative synchronous-machine states and outputs.
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
generator_signals = (
    ("omega_", "Rotor speed", "Speed [p.u.]"),
    ("theta_abs_", "Rotor electrical angle", "Angle [rad]"),
    ("psi_d_", "Direct-axis stator flux", "Flux [p.u.]"),
    ("psi_q_", "Quadrature-axis stator flux", "Flux [p.u.]"),
)
for ax, (variable_name, title, ylabel) in zip(axes.flat, generator_signals):
    for generator in generators:
        values = get_series(problem, y, generator.emt_model, variable_name)
        if values is not None:
            ax.plot(t, values, label=generator.name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
for ax in axes[-1, :]:
    ax.set_xlabel("Time [s]")
fig.tight_layout()
generator_state_plot = PLOT_DIRECTORY / "kundur_generator_states.png"
fig.savefig(generator_state_plot, dpi=180, bbox_inches="tight")
plt.close(fig)

# Steady-state diagnostics: speed deviation and accelerating torque.
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
print("\nGenerator steady-state diagnostics")
for generator in generators:
    omega = get_series(problem, y, generator.emt_model, "omega_")
    mechanical_torque = get_series(problem, y, generator.emt_model, "Tm")
    electrical_torque = get_series(problem, y, generator.emt_model, "Te")
    if omega is None or mechanical_torque is None or electrical_torque is None:
        raise KeyError(f"Missing speed or torque trajectory for {generator.name}")
    speed_deviation = omega - 1.0
    accelerating_torque = mechanical_torque - electrical_torque
    axes[0].plot(t, 1.0e6 * speed_deviation, label=generator.name)
    axes[1].plot(t, accelerating_torque, label=generator.name)
    print(
        f"  {generator.name}: "
        f"dw0={speed_deviation[0]:+.12e}, "
        f"dw_end={speed_deviation[-1]:+.12e}, "
        f"max|dw|={np.max(np.abs(speed_deviation)):.12e}, "
        f"Tm0={mechanical_torque[0]:+.12e}, "
        f"Tm_end={mechanical_torque[-1]:+.12e}, "
        f"Te0={electrical_torque[0]:+.12e}, "
        f"Te_end={electrical_torque[-1]:+.12e}, "
        f"dT0={accelerating_torque[0]:+.12e}, "
        f"dT_end={accelerating_torque[-1]:+.12e}, "
        f"max|dT|={np.max(np.abs(accelerating_torque)):.12e}"
    )
axes[0].set_title("Generator speed deviation from synchronous speed")
axes[0].set_ylabel(r"$10^6(\omega-1)$ [p.u.]")
axes[1].set_title("Generator accelerating torque")
axes[1].set_ylabel(r"$T_m-T_e$ [p.u.]")
axes[1].set_xlabel("Time [s]")
for ax in axes:
    ax.grid(True)
    ax.legend()
fig.tight_layout()
generator_balance_plot = PLOT_DIRECTORY / "kundur_generator_balance_diagnostics.png"
fig.savefig(generator_balance_plot, dpi=180, bbox_inches="tight")
plt.close(fig)


# Tie-line current: use one of the two 7-8 circuits.
tie_line = line5
fig, ax = plt.subplots(figsize=(12, 5))

for phase in ("A", "B", "C"):
    current = get_series(problem, y, tie_line.emt_model, f"if_Pi_{phase}")
    if current is None:
        current = get_series(problem, y, tie_line.emt_model, f"if_{phase}")

    if current is not None:
        ax.plot(t, current, label=f"Phase {phase}")

ax.set_title(f"{tie_line.name} from-side current")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Current [p.u.]")
ax.grid(True)
ax.legend()
fig.tight_layout()
tie_line_current_plot = PLOT_DIRECTORY / "kundur_tie_line_currents.png"
fig.savefig(tie_line_current_plot, dpi=180, bbox_inches="tight")
plt.close(fig)

print("Saved EMT plots:")
print(f"  {bus_voltage_plot.resolve()}")
print(f"  {generator_state_plot.resolve()}")
print(f"  {generator_balance_plot.resolve()}")
print(f"  {tie_line_current_plot.resolve()}")
if SHOW_PLOTS:
    plt.show()
