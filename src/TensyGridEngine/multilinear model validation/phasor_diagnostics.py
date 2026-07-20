from __future__ import annotations

import numpy as np

import VeraGridEngine.api as vge
from VeraGridEngine.Utils.Symbolic.block import find_name_in_block


def _pf_to_current(S: complex, V: complex) -> tuple[float, float]:
    v2 = (V.real * V.real) + (V.imag * V.imag)
    if v2 <= 1e-14:
        return 0.0, 0.0
    Ir = (S.real * V.real + S.imag * V.imag) / v2
    Ii = (S.real * V.imag - S.imag * V.real) / v2
    return Ir, Ii


def check_pf_line_equations(grid: vge.MultiCircuit, pf_results, tol: float = 1e-8, max_report: int = 20) -> bool:
    if len(grid.lines) == 0:
        return True

    bus_idx = {bus: i for i, bus in enumerate(grid.buses)}
    all_branches = list(grid.get_branches(add_hvdc=False, add_vsc=False, add_switch=True))
    branch_idx = {br: i for i, br in enumerate(all_branches)}
    nc = vge.compile_numerical_circuit_at(grid)

    Sf = pf_results.Sf / grid.Sbase
    St = pf_results.St / grid.Sbase
    If_pf = pf_results.If
    It_pf = pf_results.It

    rows = []
    max_efr = max_efi = max_etr = max_eti = 0.0
    max_epf = max_eqf = max_ept = max_eqt = 0.0

    for line in grid.lines:
        if line not in branch_idx:
            continue

        k = branch_idx[line]
        active = bool(nc.passive_branch_data.active[k])

        Vf = pf_results.voltage[bus_idx[line.bus_from]]
        Vt = pf_results.voltage[bus_idx[line.bus_to]]
        Vrf, Vif = Vf.real, Vf.imag
        Vrt, Vit = Vt.real, Vt.imag

        Irf = If_pf[k].real
        Iif = If_pf[k].imag
        Irt = It_pf[k].real
        Iit = It_pf[k].imag

        Pf, Qf = Sf[k].real, Sf[k].imag
        Pt, Qt = St[k].real, St[k].imag

        if active:
            R = float(nc.passive_branch_data.R[k])
            X = float(nc.passive_branch_data.X[k])
            B = float(nc.passive_branch_data.B[k])
            m = float(nc.active_branch_data.tap_module[k])
            phi = float(nc.active_branch_data.tap_angle[k])
            vtap_f = float(nc.passive_branch_data.virtual_tap_f[k])
            vtap_t = float(nc.passive_branch_data.virtual_tap_t[k])

            z2 = R * R + X * X
            if z2 <= 1e-14:
                continue

            g = R / z2
            b = -X / z2
            bsh = B

            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            k_from = m * vtap_f
            k_cross = m * vtap_f * vtap_t
            k_to = vtap_t

            gff = g / (k_from ** 2)
            bff = (bsh / 2.0 + b) / (k_from ** 2)
            gtt = g / (k_to ** 2)
            btt = (bsh / 2.0 + b) / (k_to ** 2)

            gft = -(g * cos_phi - b * sin_phi) / k_cross
            bft = -(g * sin_phi + b * cos_phi) / k_cross
            gtf = -(g * cos_phi + b * sin_phi) / k_cross
            btf = (g * sin_phi - b * cos_phi) / k_cross

            efr = Irf - (gff * Vrf - bff * Vif + gft * Vrt - bft * Vit)
            efi = Iif - (bff * Vrf + gff * Vif + bft * Vrt + gft * Vit)
            etr = Irt - (gtf * Vrf - btf * Vif + gtt * Vrt - btt * Vit)
            eti = Iit - (btf * Vrf + gtf * Vif + btt * Vrt + gtt * Vit)

            If_calc = complex(Irf - efr, Iif - efi)
            It_calc = complex(Irt - etr, Iit - eti)
            Sfc = Vf * np.conj(If_calc)
            Stc = Vt * np.conj(It_calc)

            epf = Pf - Sfc.real
            eqf = Qf - Sfc.imag
            ept = Pt - Stc.real
            eqt = Qt - Stc.imag
        else:
            efr = Irf
            efi = Iif
            etr = Irt
            eti = Iit
            epf = Pf
            eqf = Qf
            ept = Pt
            eqt = Qt

        max_err = max(abs(efr), abs(efi), abs(etr), abs(eti), abs(epf), abs(eqf), abs(ept), abs(eqt))
        max_efr = max(max_efr, abs(efr))
        max_efi = max(max_efi, abs(efi))
        max_etr = max(max_etr, abs(etr))
        max_eti = max(max_eti, abs(eti))
        max_epf = max(max_epf, abs(epf))
        max_eqf = max(max_eqf, abs(eqf))
        max_ept = max(max_ept, abs(ept))
        max_eqt = max(max_eqt, abs(eqt))
        rows.append((k, line.name, active, max_err, efr, efi, etr, eti, epf, eqf, ept, eqt))

    rows.sort(key=lambda x: abs(x[3]), reverse=True)
    ok = True

    if len(rows) > 0:
        print("\n[2.5] PF -> line equation consistency check...")
        print(f"  Checked lines: {len(rows)}, tol={tol:.1e}")
        print(
            "  Current equation max residuals: "
            f"Irf={max_efr:.6e}, Iif={max_efi:.6e}, Irt={max_etr:.6e}, Iit={max_eti:.6e}"
        )
        print(
            "  Power equation max residuals: "
            f"Pf={max_epf:.6e}, Qf={max_eqf:.6e}, Pt={max_ept:.6e}, Qt={max_eqt:.6e}"
        )
        worst = rows[0]
        print(f"  Worst line: {worst[1]}, active={worst[2]}, max|err|={worst[3]:.6e}")

        for idx, name, active, max_err, efr, efi, etr, eti, epf, eqf, ept, eqt in rows[:max_report]:
            print(f"\n    line[{idx}] {name}: active={active}, max={max_err:.6e}")
            print(f"      Irf - (...) = {efr:.6e}")
            print(f"      Iif - (...) = {efi:.6e}")
            print(f"      Irt - (...) = {etr:.6e}")
            print(f"      Iit - (...) = {eti:.6e}")
            print(f"      Pf  - (...) = {epf:.6e}")
            print(f"      Qf  - (...) = {eqf:.6e}")
            print(f"      Pt  - (...) = {ept:.6e}")
            print(f"      Qt  - (...) = {eqt:.6e}")
            if max_err > tol:
                ok = False

    return ok


def check_equation_consistency(problem: vge.RmsProblemPhasor, tol: float = 1e-6, max_report: int = 200) -> bool:
    x0 = problem.get_x0()
    dx0 = np.zeros(problem.get_diff_var_number(), dtype=float)

    state_res = np.array([], dtype=float)
    if problem.get_states_number() > 0:
        state_res = np.asarray(problem.rhs_state(x0, dx0), dtype=float)

    algeb_res = np.asarray(problem.rhs_algebraic(x0, dx0), dtype=float)
    rhs = np.r_[state_res, algeb_res]

    max_abs = float(np.max(np.abs(rhs))) if rhs.size > 0 else 0.0
    ok = max_abs <= tol

    print("\n[3.05] Consistency check at initialization (x0, dx=0)...")
    print(f"  max|residual| = {max_abs:.6e}, tol = {tol:.1e}, ok = {ok}")

    if not ok and rhs.size > 0:
        all_eqs = problem._state_eqs + problem._algebraic_eqs
        order = np.argsort(np.abs(rhs))[::-1]
        n = min(max_report, len(order))
        print(f"  Top {n} residual equations:")
        for idx in order[:n]:
            print(f"    [{idx:4d}] err={rhs[idx]: .6e} :: {all_eqs[idx]}")

    return ok


def debug_exciter_equations(problem: vge.RmsProblemPhasor, max_report: int = 120) -> None:
    x0 = problem.get_x0()
    dx0 = np.zeros(problem.get_diff_var_number(), dtype=float)
    state_res = np.array([], dtype=float)
    if problem.get_states_number() > 0:
        state_res = np.asarray(problem.rhs_state(x0, dx0), dtype=float)
    algeb_res = np.asarray(problem.rhs_algebraic(x0, dx0), dtype=float)
    rhs = np.r_[state_res, algeb_res]

    all_eqs = problem._state_eqs + problem._algebraic_eqs
    all_vars = problem._state_vars + problem._algebraic_vars
    name_to_uids: dict[str, list[int]] = {}
    for var in all_vars:
        name_to_uids.setdefault(var.name, []).append(var.uid)

    focus_names = {"y_subexciter1", "Efe", "f_input", "f_output", "u_subexciter1", "Vf"}
    matches = []
    for idx, eq in enumerate(all_eqs):
        eq_str = str(eq)
        if any(name in eq_str for name in focus_names):
            matches.append((idx, rhs[idx], eq_str))

    matches.sort(key=lambda t: abs(t[1]), reverse=True)
    n = min(max_report, len(matches))

    print("\n[3.06] Exciter-targeted diagnostics...")
    print(f"  Matching equations: {len(matches)} (showing {n})")
    for i, err, eq_str in matches[:n]:
        print(f"    [{i:4d}] err={err: .6e} :: {eq_str}")

    efe_vals = []
    vf_vals = []
    vemax_vals = []
    idx_map = problem.uid2idx_vars
    for var in all_vars:
        if var.name == "Efe":
            idx = idx_map.get(var.uid, None)
            if idx is not None:
                efe_vals.append((var.uid, x0[idx]))
        if var.name == "Vf":
            idx = idx_map.get(var.uid, None)
            if idx is not None:
                vf_vals.append((var.uid, x0[idx]))
        if var.name == "VeMaxPu":
            idx = idx_map.get(var.uid, None)
            if idx is not None:
                vemax_vals.append((var.uid, x0[idx]))
    for uid, val in efe_vals:
        print(f"  Efe (uid={uid}) = {val: .6e}")
    for uid, val in vf_vals:
        print(f"  Vf (uid={uid}) = {val: .6e}")
    for uid, val in vemax_vals:
        print(f"  VeMaxPu (uid={uid}) = {val: .6e}")

    print("  Targeted Gen29/Gen36 values:")
    for gen_idx, gen in enumerate(problem.grid.generators):
        if gen.rms_model is None:
            continue
        if gen_idx not in {27, 29, 36}:
            continue

        print(f"    Gen{gen_idx} -> {gen.name} @ {gen.bus.name}:")
        for key in ("VeMaxPu" ,"Vf", "y_subexciter1", "f_output", "Efe", "Vf"):
            var = find_name_in_block(key, gen.rms_model)
            if var is None:
                continue
            idx = idx_map.get(var.uid, None)
            if idx is not None:
                print(f"      {key} = {x0[idx]: .6e}")




def check_generator_current_residuals(problem: vge.RmsProblemPhasor, pf_results) -> None:
    x0 = problem.get_x0()
    all_vars = problem._state_vars + problem._algebraic_vars
    uid_to_idx = problem.get_uid_to_idx_dict()
    print("\n[3.07] Generator current vs PF bus residual...")
    for gen in problem.grid.generators:
        model = gen.rms_model
        if model is None:
            continue
        bus = gen.bus
        if bus is None:
            continue
        for cur_name_r, cur_name_i in [("Ir", "Ii")]:
            r_uid = None
            i_uid = None
            for var in model.block.external_mapping.values():
                if var.name == cur_name_r:
                    r_uid = var.uid
                elif var.name == cur_name_i:
                    i_uid = var.uid
            if r_uid is None or i_uid is None:
                continue
            r_idx = uid_to_idx.get(r_uid)
            i_idx = uid_to_idx.get(i_uid)
            if r_idx is None or i_idx is None:
                continue
            print(
                f"  gen {gen.name} @ {bus.name}: Iinit=({x0[r_idx]:.6e},{x0[i_idx]:.6e})"
            )


def check_shunt_pf_currents(grid: vge.MultiCircuit, pf_results, tol: float = 1e-8, max_report: int = 50) -> bool:
    if len(grid.shunts) == 0:
        return True

    bus_idx = {bus: i for i, bus in enumerate(grid.buses)}
    shunt_q = getattr(pf_results, "shunt_q", None)
    rows = []
    max_err = 0.0

    for idx, sh in enumerate(grid.shunts):
        if sh.bus is None:
            continue

        bi = bus_idx[sh.bus]
        V = pf_results.voltage[bi]
        Y = complex(float(sh.G) / grid.Sbase, float(sh.B) / grid.Sbase)
        I = Y * V
        S = V * np.conj(I)

        q_pf = float(shunt_q[idx]) / grid.Sbase if shunt_q is not None else 0.0
        i_q_r, i_q_i = _pf_to_current(complex(0.0, q_pf), V)

        er = I.real - i_q_r
        ei = I.imag - i_q_i
        es = S.imag - q_pf
        e = max(abs(er), abs(ei), abs(es))
        max_err = max(max_err, e)
        rows.append((sh.name, sh.bus.name, V, I, S, er, ei, es, e))

    rows.sort(key=lambda r: abs(r[-1]), reverse=True)
    ok = max_err <= tol

    print("\n[2.6] Shunt PF current consistency check...")
    print(f"  Checked shunts: {len(rows)}, tol={tol:.1e}, max|err|={max_err:.6e}, ok={ok}")
    for name, bus_name, V, I, S, er, ei, es, e in rows[:max_report]:
        print(
            f"  {name} @ {bus_name}: V={V.real:.6e}+j{V.imag:.6e}, "
            f"I={I.real:.6e}+j{I.imag:.6e}, errI=({er:.6e},{ei:.6e}), errQ={es:.6e}"
        )

    return ok


def print_buses_with_many_injections(grid: vge.MultiCircuit, min_devices: int = 3) -> None:
    bus_to_devices = {bus: [] for bus in grid.buses}
    for dev in grid.get_injection_devices_iter():
        if dev.bus is not None:
            bus_to_devices[dev.bus].append(dev)

    found = False
    for bus, devices in bus_to_devices.items():
        if len(devices) >= min_devices:
            found = True
            names = ", ".join(f"{dev.device_type.value}:{dev.name}" for dev in devices)
            print(f"  {bus.name}: {len(devices)} injection devices -> {names}")

    if not found:
        print(f"  No buses with >= {min_devices} injection devices found.")


def print_bus_current_balance(grid: vge.MultiCircuit, pf_results, tol: float = 1e-8, max_report: int = 50) -> None:
    bus_idx = {bus: i for i, bus in enumerate(grid.buses)}
    n_bus = len(grid.buses)
    branch_sum = np.zeros(n_bus, dtype=complex)
    gen_sum = np.zeros(n_bus, dtype=complex)
    batt_sum = np.zeros(n_bus, dtype=complex)
    load_sum = np.zeros(n_bus, dtype=complex)
    shunt_sum = np.zeros(n_bus, dtype=complex)
    slack_sum = np.zeros(n_bus, dtype=complex)

    all_branches = list(grid.get_branches(add_hvdc=False, add_vsc=False, add_switch=True))
    for br, if_, it_ in zip(all_branches, pf_results.If, pf_results.It):
        if br.bus_from is not None:
            branch_sum[bus_idx[br.bus_from]] += if_
        if br.bus_to is not None:
            branch_sum[bus_idx[br.bus_to]] += it_

    gen_idx_map = {elm.idtag: i for i, elm in enumerate(grid.generators)}
    batt_idx_map = {elm.idtag: i for i, elm in enumerate(grid.batteries)}
    shunt_idx_map = {elm.idtag: i for i, elm in enumerate(grid.shunts)}
    loads_idx_map = {elm.idtag: i for i, elm in enumerate(grid.loads)}

    fixed_sum = np.zeros(n_bus, dtype=complex)
    slack_gen_count = np.zeros(n_bus, dtype=int)

    for dev in grid.get_injection_devices_iter():
        if dev.bus is None or dev.bus.is_dc:
            continue

        bi = bus_idx[dev.bus]
        v = pf_results.voltage[bi]

        if dev.idtag in gen_idx_map:
            if dev.bus.is_slack:
                slack_gen_count[bi] += 1
                continue
            gidx = gen_idx_map[dev.idtag]
            s = complex(float(dev.P), float(pf_results.gen_q[gidx])) / grid.Sbase
            i = np.conj(s / v)
            gen_sum[bi] += i
            fixed_sum[bi] += i
        elif dev.idtag in batt_idx_map:
            bidx = batt_idx_map[dev.idtag]
            s = complex(float(dev.P), float(pf_results.battery_q[bidx])) / grid.Sbase
            i = np.conj(s / v)
            batt_sum[bi] += i
            fixed_sum[bi] += i
        elif dev.idtag in shunt_idx_map:
            y = complex(float(dev.G) / grid.Sbase, float(dev.B) / grid.Sbase)
            # Sign convention: shunt template behaves as load-like current sink.
            i = -y * v
            shunt_sum[bi] += i
            fixed_sum[bi] += i
        elif dev.idtag in loads_idx_map:
            s = complex(-float(dev.P), -float(dev.Q)) / grid.Sbase
            i = np.conj(s / v)
            load_sum[bi] += i
            fixed_sum[bi] += i

    residual = branch_sum - fixed_sum
    for bi in range(n_bus):
        if slack_gen_count[bi] > 0:
            slack_sum[bi] = residual[bi]

    inj_sum = fixed_sum + slack_sum

    rows = []
    for bi, bus in enumerate(grid.buses):
        total = branch_sum[bi] - inj_sum[bi]
        err = abs(total)
        if err > tol:
            rows.append((
                err,
                bus.name,
                branch_sum[bi],
                gen_sum[bi],
                batt_sum[bi],
                load_sum[bi],
                shunt_sum[bi],
                slack_sum[bi],
                total,
            ))

    rows.sort(key=lambda r: r[0], reverse=True)
    print("\n[2.8] Bus current balance report...")
    print(f"  buses over tol={tol:.1e}: {len(rows)}")
    for err, bus_name, b, g, bt, l, s, sl, total in rows[:max_report]:
        print(
            f"  {bus_name:<12} |err|={err:.6e} "
            f"Br=({b.real:.3e},{b.imag:.3e}) "
            f"G=({g.real:.3e},{g.imag:.3e}) "
            f"Bt=({bt.real:.3e},{bt.imag:.3e}) "
            f"L=({l.real:.3e},{l.imag:.3e}) "
            f"Sh=({s.real:.3e},{s.imag:.3e}) "
            f"Sl=({sl.real:.3e},{sl.imag:.3e}) "
            f"R=({total.real:.3e},{total.imag:.3e})"
        )
