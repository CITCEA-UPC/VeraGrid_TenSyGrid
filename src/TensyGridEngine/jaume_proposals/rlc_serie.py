"""
VeraGrid RMS - Circuito RLC serie
===================================

Topologia:

    Bus_Slack (V=1∠0) ─── R + j(XL - XC) ─── Bus_Carga ─── R_load a tierra

donde XL = ωL y XC = 1/(ωC).

Parametros:
  R  = 10 Ω
  L  = 50 mH
  C  = 100 μF
  f  = 50 Hz
  Sb = 100 MVA, Vb = 100 kV
  R_load = 200 Ω  (≈ 0.5 pu)

La reactancia neta: X = ωL - 1/(ωC) = 15.708 - 31.831 = -16.123 Ω
Como X < 0 predomina el condensador (circuito capacitivo).
"""

import os, sys

_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "src")
if not os.path.isdir(os.path.join(_SRC_DIR, "VeraGridEngine")):
    _SRC_DIR = _SCRIPT_DIR
    for _ in range(5):
        if os.path.isdir(os.path.join(_SRC_DIR, "VeraGridEngine")):
            break
        _SRC_DIR = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

import VeraGridEngine as vge
from VeraGridEngine.IO.file_save import FileSave

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
for _ in range(10):
    if os.path.isfile(os.path.join(_PROJECT_ROOT, "requirements.txt")):
        break
    _PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)

# ==========================================================================
#  PARAMETROS DEL RLC (valores reales)
# ==========================================================================
f  = 50.0
omega = 2 * 3.141592653589793 * f

R  = 10.0     # Ω
L  = 50e-3    # H (50 mH)
C  = 100e-6   # F (100 μF)

XL = omega * L        # 15.708 Ω
XC = 1 / (omega * C)  # 31.831 Ω
X_net = XL - XC       # -16.123 Ω (capacitivo)

Vnom = 100.0  # kV
Sbase = 100.0  # MVA
Zb = Vnom**2 / Sbase  # 100 Ω

R_pu  = R  / Zb   # 0.1 pu
X_pu  = X_net / Zb  # -0.16123 pu
PLOAD = 0.5  # pu (R_load=200 Ω a Vnom=100 kV → 50 MW = 0.5 pu en base 100 MVA)

print(f"f={f} Hz, omega={omega:.4f} rad/s")
print(f"R={R} ohm, L={L*1000:.1f} mH, C={C*1e6:.0f} uF")
print(f"XL={XL:.4f} ohm, XC={XC:.4f} ohm, X_net={X_net:.4f} ohm")
print(f"R_pu={R_pu:.6f} pu, X_pu={X_pu:.6f} pu, P_load={PLOAD:.4f} pu")

# ==========================================================================
#  CIRCUITO
# ==========================================================================
grid = vge.MultiCircuit(Sbase=Sbase, fbase=f)

bus_s = vge.Bus(name="Fuente", Vnom=Vnom)  # ya no es slack, lo impone la ExternalGrid
grid.add_bus(bus_s)

bus_l = vge.Bus(name="Carga", Vnom=Vnom)
grid.add_bus(bus_l)

# Fuente ideal de tension AC (VD -> slack interno en PF)
eg = vge.ExternalGrid(name="IdealAC", Vm=1.0, Va=0.0, mode=vge.ExternalGridMode.VD)
grid.add_external_grid(bus=bus_s, api_obj=eg)

grid.add_line(vge.Line(name="RLC", bus_from=bus_s, bus_to=bus_l, r=R_pu, x=X_pu, b=0.0))
grid.add_load(bus=bus_l, api_obj=vge.Load(P=PLOAD, Q=0.0))

# ==========================================================================
#  FLUJO DE CARGAS
# ==========================================================================
pf_opts = vge.PowerFlowOptions(solver_type=vge.SolverType.NR, tolerance=1e-6, max_iter=20)
pf_res = vge.power_flow(grid, options=pf_opts)

print("\n=== FLUJO DE CARGAS ===")
print(pf_res.get_bus_df())
print(f"Converged: {pf_res.converged}")

# ==========================================================================
#  GUARDAR
# ==========================================================================
out_dir = os.path.join(_PROJECT_ROOT, "Grids_and_profiles", "grids")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "rlc_serie.gridcal")
FileSave(circuit=grid, file_name=out_path).save()
print(f"Red guardada en: {out_path}")
