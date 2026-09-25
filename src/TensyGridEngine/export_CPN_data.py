# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Export the CPN1 multilinear system (S, Phi, equations, variables) to .mat.

Configuration flags (set in main() call, see __name__ == "__main__" block):

    SIMPLIFY : bool (default True)
        Applies the full simplification pipeline:
          - merge_duplicate_monomials (sums Phi columns with identical S vectors)
          - fixpoint loop: regenerate_eqs_list -> simplify_system -> substitute_pinned_constants
          - handle_free_variables -> reduce_rank_qr -> pin_vars_without_trivial_monomials -> cleanup
        When False, returns the raw system after parameter substitution.

    ZERO_DERIVATIVES : bool (default True)
        Removes derivative columns (dx/dt -> 0 substitution).
        When False, keeps derivative variables in the system.

    EXPORT_EXCEL : bool (default True)
        Writes a .xlsx workbook with sheets for Summary, Variables, Equations,
        S matrix (COO), Phi matrix (COO), and Jacobian evaluated at x0.

    EXPORT_PDF : bool (default True)
        Compiles a LaTeX report (pdflatex) with original equations, pinned
        variables, final equations, S/Phi, and Jacobian as a PDF.

Usage:
    python export_CPN_data.py <grid_filename>
"""

# ── Imports ──

from __future__ import annotations

import sys
from pathlib import Path

import scipy
import numpy as np

project_base = Path(__file__).resolve().parents[2]
src_path = project_base / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from TensyGridEngine.cpn_utils import build_problems, process_cpn_system, compute_jacobian_from_S


# ── Helpers ──

def _export_excel(cpn, problem, x0, n_eqs, n_vars, n_mon, rank_Phi, F, J, grid_stem=""):
    """Write S, Phi, equations, variables, and Jacobian to an .xlsx workbook."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    out_dir = Path('CPN1_computations')
    out_dir.mkdir(exist_ok=True)
    stem = grid_stem or str(problem.grid)
    filepath = out_dir / f'{stem}_stats.xlsx'

    wb = Workbook()
    hdr_font = Font(bold=True, size=11)
    hdr_fill = PatternFill(start_color='D9E1F2', end_color='D9E1F2', fill_type='solid')
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin'),
    )

    def _write_header(ws, headers, row=1):
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=row, column=c, value=h)
            cell.font = hdr_font
            cell.fill = hdr_fill
            cell.border = thin_border
            cell.alignment = Alignment(horizontal='center')

    def _auto_width(ws):
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

    ws = wb.active
    ws.title = 'Summary'
    rows = [
        ('Grid', str(problem.grid)),
        ('n_eqs', n_eqs),
        ('n_vars', n_vars),
        ('n_mon', n_mon),
        ('Rank(Phi)', rank_Phi),
        ('Well-determined', rank_Phi == n_vars),
    ]
    _write_header(ws, ['Property', 'Value'])
    for r, (k, v) in enumerate(rows, 2):
        ws.cell(row=r, column=1, value=k).font = Font(bold=True)
        ws.cell(row=r, column=2, value=v)
    _auto_width(ws)

    ws = wb.create_sheet('Variables')
    _write_header(ws, ['idx', 'name', 'x0'])
    for i, var in enumerate(cpn.vars_list):
        ws.cell(row=i + 2, column=1, value=i)
        ws.cell(row=i + 2, column=2, value=var)
        ws.cell(row=i + 2, column=3, value=float(x0[i]) if i < len(x0) else None)
    _auto_width(ws)

    ws = wb.create_sheet('Equations')
    _write_header(ws, ['idx', 'equation'])
    for i, eq in enumerate(cpn.eqs_list):
        ws.cell(row=i + 2, column=1, value=i)
        ws.cell(row=i + 2, column=2, value=_clean_eq_str(str(eq)))
    _auto_width(ws)

    ws = wb.create_sheet('S_coo')
    S_coo = cpn.S.tocoo()
    _write_header(ws, ['row(var)', 'col(mon)', 'value'])
    for r, c, v in zip(S_coo.row, S_coo.col, S_coo.data):
        row_num = ws.max_row + 1
        ws.cell(row=row_num, column=1, value=int(r))
        ws.cell(row=row_num, column=2, value=int(c))
        ws.cell(row=row_num, column=3, value=float(v))
    _auto_width(ws)

    ws = wb.create_sheet('Phi_coo')
    Phi_coo = cpn.Phi.tocoo()
    _write_header(ws, ['row(eq)', 'col(mon)', 'value'])
    for r, c, v in zip(Phi_coo.row, Phi_coo.col, Phi_coo.data):
        row_num = ws.max_row + 1
        ws.cell(row=row_num, column=1, value=int(r))
        ws.cell(row=row_num, column=2, value=int(c))
        ws.cell(row=row_num, column=3, value=float(v))
    _auto_width(ws)

    ws = wb.create_sheet('Jacobian')
    J_dense = J.toarray() if hasattr(J, 'toarray') else np.asarray(J)
    _write_header(ws, [''] + [f'v{j}' for j in range(J_dense.shape[1])])
    for i in range(J_dense.shape[0]):
        ws.cell(row=i + 2, column=1, value=f'eq{i}').font = Font(bold=True)
        for j in range(J_dense.shape[1]):
            val = J_dense[i, j]
            if abs(val) > 1e-15:
                ws.cell(row=i + 2, column=j + 2, value=round(val, 6))
    _auto_width(ws)

    wb.save(filepath)
    print(f"Excel saved: {filepath}")


# ── LaTeX / PDF export ──

import re as _re


_TOKEN_NUM = _re.compile(r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?')
_TOKEN_VAR = _re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def _tokenize_eq(eq_str: str):
    """Split a CPN equation string into (kind, value) tokens.

    Keeps parentheses, signs and ``*``/``/``/``**`` operators so the original
    expression structure is preserved (no distribution, no expansion).
    """
    s = eq_str
    toks = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == ' ':
            i += 1
            continue
        m = _TOKEN_NUM.match(s, i)
        if m and m.end() > i and (c.isdigit() or (c in '+-' and i + 1 < n and (s[i + 1].isdigit() or s[i + 1] == '.'))):
            toks.append(('num', m.group(0)))
            i = m.end()
            continue
        if c in '()*/-+':
            toks.append(('op', c))
            i += 1
            continue
        m = _TOKEN_VAR.match(s, i)
        if not m:
            i += 1
            continue
        name = m.group(0)
        j = i + len(name)
        if s[j:j + 2] == '**':
            d = _re.compile(r'\d+').match(s, j + 2)
            toks.append(('var', name))
            toks.append(('pow', str(int(d.group(0)))))
            i = d.end()
        else:
            toks.append(('var', name))
            i = j
    return toks


class _ParseErr(Exception):
    pass


def _num_tex(v: float) -> str:
    if v == 0:
        return '0'
    a = abs(v)
    if 1e-7 <= a < 1e8:
        s = f'{v:.6f}'.rstrip('0').rstrip('.')
        return s if s not in ('', '-0', '-') else '0'
    m = f'{v:.6e}'
    mant, _, exp = m.partition('e')
    mant = mant.rstrip('0').rstrip('.')
    return rf'{mant}\times 10^{{{int(exp)}}}'


def _parse_expr(toks):
    """Build a minimal AST from the token stream (no expansion).

    Node types: ('num', float), ('var', name, exp), ('neg', node),
    ('mul', [nodes]), ('div', left, right), ('add', [(sign, node), ...]).
    """
    pos = 0
    n = len(toks)

    def peek():
        return toks[pos] if pos < n else None

    def take():
        nonlocal pos
        if pos >= n:
            raise _ParseErr('unexpected end')
        t = toks[pos]
        pos += 1
        return t

    def parse_factor():
        nonlocal pos
        sgn = 1
        while True:
            t = peek()
            if t and t[0] == 'op' and t[1] in '+-':
                if t[1] == '-':
                    sgn = -sgn
                pos += 1
            else:
                break
        t = peek()
        if t is None:
            raise _ParseErr('expected operand')
        if t[0] == 'num':
            pos += 1
            return ('num', sgn * float(t[1]))
        if t[0] == 'var':
            pos += 1
            exp = 1
            nxt = peek()
            if nxt and nxt[0] == 'pow':
                pos += 1
                exp = int(nxt[1])
            node = ('var', t[1], exp)
            return ('neg', node) if sgn < 0 else node
        if t[0] == 'op' and t[1] == '(':
            pos += 1
            node = parse_addsub()
            c = peek()
            if c and c[0] == 'op' and c[1] == ')':
                pos += 1
            else:
                raise _ParseErr('missing )')
            return ('neg', node) if sgn < 0 else node
        raise _ParseErr(f'unexpected {t}')

    def parse_term():
        nonlocal pos
        node = parse_factor()
        while True:
            t = peek()
            if t and t[0] == 'op' and t[1] == '*':
                pos += 1
                rhs = parse_factor()
                node = ('mul', [node, rhs])
            elif t and t[0] == 'op' and t[1] == '/':
                pos += 1
                rhs = parse_factor()
                node = ('div', node, rhs)
            else:
                break
        return node

    def parse_addsub():
        nonlocal pos
        node = parse_term()
        parts = [(1, node)]
        while True:
            t = peek()
            if t and t[0] == 'op' and t[1] in '+-':
                op = take()[1]
                parts.append((-1 if op == '-' else 1, parse_term()))
            else:
                break
        if len(parts) == 1:
            return parts[0][1]
        return ('add', parts)

    return parse_addsub()


def _render_node(node, alias_map) -> str:
    typ = node[0]
    if typ == 'num':
        return _num_tex(float(node[1]))
    if typ == 'var':
        _, name, exp = node
        v = _var_tex(alias_map.get(name, name))
        return v if exp == 1 else f'{v}^{{{exp}}}'
    if typ == 'neg':
        inner = node[1]
        ctex = _render_node(inner, alias_map)
        if ctex.startswith('-'):
            return ctex[1:].lstrip()
        if inner[0] == 'add':
            return f'-({ctex})'
        return f'-{ctex}'
    if typ == 'mul':
        parts = []
        for child in node[1]:
            ctex = _render_node(child, alias_map)
            if child[0] in ('add', 'neg'):
                ctex = f'({ctex})'
            parts.append(ctex)
        return ' \\, '.join(parts)
    if typ == 'div':
        num = _render_node(node[1], alias_map)
        den = _render_node(node[2], alias_map)
        return f'\\frac{{{num}}}{{{den}}}'
    if typ == 'add':
        out = []
        for i, (sgn, child) in enumerate(node[1]):
            ctex = _render_node(child, alias_map)
            if child[0] == 'add' and sgn < 0:
                ctex = f'({ctex})'
            if i == 0:
                if sgn < 0 and not ctex.startswith('-'):
                    ctex = '-' + ctex
                out.append(ctex)
            else:
                if ctex.startswith('-'):
                    out.append((' - ' if sgn > 0 else ' + ') + ctex[1:].lstrip())
                else:
                    out.append((' + ' if sgn > 0 else ' - ') + ctex)
        return ''.join(out)
    raise _ParseErr(f'unknown node {typ}')


def _add_pieces(node, alias_map) -> list[str]:
    flat: list[tuple[int, object]] = []
    for sgn, child in node[1]:
        if child[0] == 'add':
            for s2, sub in child[1]:
                flat.append((sgn * s2, sub))
        else:
            flat.append((sgn, child))
    parts = []
    for i, (sgn, child) in enumerate(flat):
        ctex = _render_node(child, alias_map)
        neg = sgn < 0
        if ctex.startswith('-'):
            neg = not neg
            ctex = ctex[1:].lstrip()
        if i == 0:
            parts.append(('- ' + ctex) if neg else ctex)
        else:
            parts.append(('- ' + ctex) if neg else (' + ' + ctex))
    return parts


def _collect_var_names(all_eqs, vars_list, pinned_vars) -> dict[str, str]:
    """Build short display names for variables appearing in the report.

    Variables of the reduced system come first and keep their base name (long
    numeric uid suffixes are dropped); other names (pinned vars, variables that
    only appear in the raw equations) get a numeric subscript by order of first
    appearance when their base name collides.
    """
    seen: list[str] = []

    def add(name):
        if name not in seen:
            seen.append(name)

    for name in vars_list:
        add(name)
    for name in pinned_vars:
        add(name)
    for eq in all_eqs:
        for name in _re.findall(r'[A-Za-z_][A-Za-z0-9_]*', eq):
            add(name)

    base_counts: dict[str, int] = {}
    alias: dict[str, str] = {}
    for name in seen:
        base = _re.sub(r'_\d+$', '', name)
        n = base_counts.get(base, 0)
        base_counts[base] = n + 1
        alias[name] = base if n == 0 else f'{base}_{n}'
    return alias


def _var_tex(alias_name: str) -> str:
    """Render a variable name: first char as italic base, the rest as an
    upright subscript suffix, e.g. ``Irt`` -> ``I_\\mathrm{rt}`` and
    ``vr_aux`` -> ``v_\\mathrm{r\\_aux}``."""
    base = alias_name[:1]
    suffix = alias_name[1:]
    if not suffix:
        return base
    return fr'{base}_{{\mathrm{{{suffix.replace("_", r"\_")}}}}}'


def _landscape(content: str) -> str:
    """Wrap content in a landscape page (pdflscape) with wider usable width."""
    return ('\n\\begin{landscape}\n\\newgeometry{landscape, margin=0.8cm}\n'
            '{\\small\n' + content + '}\n'
            '\\restoregeometry\n\\end{landscape}\n')


_NOISE_THRESH = 1e-14


def _term_coeff(node):
    """Absolute constant factor of an additive term (None if not a constant
    factor, i.e. the term also contains symbolic factors of at least unit scale)."""
    typ = node[0]
    if typ == 'num':
        return abs(float(node[1]))
    if typ == 'neg':
        return _term_coeff(node[1])
    if typ == 'mul':
        coeff = 1.0
        undef = False
        zero = False
        for ch in node[1]:
            c = _term_coeff(ch)
            if c is None:
                undef = True
                continue
            if c == 0:
                zero = True
            coeff *= c
        if zero:
            return 0.0
        if undef:
            return coeff if coeff < _NOISE_THRESH else None
        return coeff
    return None


def _is_noise(node) -> bool:
    c = _term_coeff(node)
    return c is not None and c < _NOISE_THRESH


def _filter_noise_ast(node):
    """Drop additive terms whose coefficient magnitude is below the noise
    threshold. Returns a cleaned AST."""
    typ = node[0]
    if typ == 'add':
        kept = [(s, ch) for s, ch in node[1] if not _is_noise(ch)]
        if not kept:
            return ('num', 0.0)
        if len(kept) == 1:
            s, ch = kept[0]
            ch = _filter_noise_ast(ch)
            if s < 0:
                if ch[0] == 'num':
                    return ('num', -ch[1])
                return ('neg', ch)
            return ch
        return ('add', [(s, _filter_noise_ast(ch)) for s, ch in kept])
    if typ == 'mul':
        return ('mul', [_filter_noise_ast(ch) for ch in node[1]])
    if typ == 'neg':
        return ('neg', _filter_noise_ast(node[1]))
    if typ == 'div':
        return ('div', _filter_noise_ast(node[1]), _filter_noise_ast(node[2]))
    return node


def _render_plain(node) -> str:
    """Render an AST back to the original equation string syntax."""
    typ = node[0]
    if typ == 'num':
        return f'({repr(float(node[1]))})'
    if typ == 'var':
        _, name, exp = node
        return f'({name})' if exp == 1 else f'({name}**{exp})'
    if typ == 'neg':
        return f'-{_render_plain(node[1])}'
    if typ == 'mul':
        return ' * '.join(_render_plain(c) for c in node[1])
    if typ == 'div':
        return f'({_render_plain(node[1])}) / ({_render_plain(node[2])})'
    if typ == 'add':
        out = []
        for i, (sgn, ch) in enumerate(node[1]):
            t = _render_plain(ch)
            starts_neg = t.startswith('-')
            if i == 0:
                if sgn < 0 and not starts_neg:
                    t = '-' + t
                out.append(t)
            else:
                neg = (sgn < 0) != starts_neg
                body = t[1:].lstrip() if starts_neg else t
                out.append((' - ' if neg else ' + ') + body)
        return ''.join(out)
    return ''


def _clean_eq_str(eq_str: str) -> str:
    """Return the equation string with sub-1e-14 noise additive terms removed."""
    s = eq_str.strip()
    if s in ('0', '0.0', '-0.0'):
        return s
    try:
        node = _filter_noise_ast(_parse_expr(_tokenize_eq(s)))
    except _ParseErr:
        return eq_str
    return _render_plain(node)


def _eq_to_latex(eq_str, alias_map, max_width: int = 165) -> str:
    """Render a CPN equation as LaTeX with minimal parentheses."""
    s = eq_str.strip()
    if s in ('0', '0.0', '-0.0'):
        return r'\begin{equation}0 = 0\end{equation}'

    node = _filter_noise_ast(_parse_expr(_tokenize_eq(s)))
    body = _render_node(node, alias_map)

    if len(body) <= max_width or node[0] != 'add':
        lines = [body]
    else:
        pieces = _add_pieces(node, alias_map)
        lines = []
        cur = pieces[0]
        for piece in pieces[1:]:
            if len(cur) + len(piece) + 4 > max_width:
                lines.append(cur)
                cur = r'&\quad ' + piece.lstrip()
            else:
                cur = cur + ' ' + piece
        if cur:
            lines.append(cur)

    if len(lines) > 1:
        lines[-1] += ' & = 0'
        body = ' \\\\\n'.join(lines)
    else:
        body = lines[-1] + ' = 0'

    return r'\begin{equation}\begin{aligned}' + '\n' + body + '\n' + r'\end{aligned}\end{equation}'


def _fmt_cell(v, precision: int = 6) -> str:
    return f'{float(v):.{precision}f}'


def _export_latex_pdf(cpn, problem, x0, n_eqs, n_vars, n_mon, rank_Phi, F, J, grid_stem=""):
    """Generate a LaTeX report (equations, pinned vars, S/Phi info) and compile it to PDF."""
    import subprocess

    out_dir = Path(__file__).resolve().parent / 'CPN1_computations'
    out_dir.mkdir(exist_ok=True)
    stem = grid_stem or str(problem.grid)
    tex_path = out_dir / f'{stem}_report.tex'
    pdf_path = out_dir / f'{stem}_report.pdf'

    alias_map = _collect_var_names(cpn.raw_eqs_list + cpn.eqs_list, cpn.vars_list, cpn.pinned_vars)

    pinned_rows = ''.join(
        f'${_var_tex(alias_map.get(var, var))}$ & {_fmt_cell(val)} \\\\\n'
        for var, val in sorted(cpn.pinned_vars.items())
    )
    pinned_count = len(cpn.pinned_vars)

    summary_rows = '\n'.join([
        f'Grid & \\texttt{{{_latex_escape(str(problem.grid))}}} \\\\',
        f'$n_{{\\text{{eqs}}}}$ & {n_eqs} \\\\',
        f'$n_{{\\text{{vars}}}}$ & {n_vars} \\\\',
        f'$n_{{\\text{{mon}}}}$ & {n_mon} \\\\',
        r'$\mathrm{rank}(\Phi)$ & ' + str(rank_Phi) + r' \\',
        r'Well-determined & ' + (r'\textbf{yes}' if rank_Phi == n_vars else r'\textbf{no}') + r' \\',
        r'Pinned variables & ' + str(pinned_count) + r' \\',
    ])

    raw_eqs = '\n'.join(_eq_to_latex(e, alias_map) for e in cpn.raw_eqs_list)
    final_eqs = '\n'.join(_eq_to_latex(e, alias_map) for e in cpn.eqs_list)

    var_rows = ''.join(
        f'{i} & ${_var_tex(alias_map.get(v, v))}$ & \\texttt{{{_latex_escape(v)}}} & {_fmt_cell(x0[i])} \\\\\n'
        for i, v in enumerate(cpn.vars_list)
    )

    S_coo = cpn.S.tocoo()
    S_rows = ''.join(
        f'{int(r)} & {int(c)} & {_fmt_cell(v)} \\\\\n'
        for r, c, v in zip(S_coo.row, S_coo.col, S_coo.data)
    )
    Phi_coo = cpn.Phi.tocoo()
    Phi_rows = ''.join(
        f'{int(r)} & {int(c)} & {_fmt_cell(v)} \\\\\n'
        for r, c, v in zip(Phi_coo.row, Phi_coo.col, Phi_coo.data)
    )

    J_dense = J.toarray() if hasattr(J, 'toarray') else np.asarray(J)
    if J_dense.shape[0] <= 15 and J_dense.shape[1] <= 15:
        jac_body = ' \\\\\n'.join(
            ' & '.join(_fmt_cell(J_dense[i, j]) if abs(J_dense[i, j]) > 1e-15 else '0'
                       for j in range(J_dense.shape[1]))
            for i in range(J_dense.shape[0])
        )
        jac_plain = (
            f'\\subsection*{{Jacobian $J = \\Phi \\, F^T$ evaluated at $x_0$}}'
            f'\n\\[\nJ = \\begin{{bmatrix}}\n{jac_body}\n\\end{{bmatrix}}\n\\]\n'
        )
        jac_section = _landscape(jac_plain) if J_dense.shape[1] > 9 else jac_plain
    else:
        jac_rows = ''.join(
            f'{i} & {j} & {_fmt_cell(J_dense[i, j])} \\\\\n'
            for i in range(J_dense.shape[0])
            for j in range(J_dense.shape[1])
            if abs(J_dense[i, j]) > 1e-15
        )
        jac_section = (
            f'\\subsection*{{Jacobian $J = \\Phi \\, F^T$ evaluated at $x_0$ (nonzero entries)}}'
            f'\n\\begin{{longtable}}{{rrr}}\n\\toprule\n$i$ & $j$ & value \\\\\n\\midrule\n{jac_rows}\\bottomrule\n\\end{{longtable}}\n'
        )

    matrices = 'S' if cpn.S.shape[0] <= 15 and cpn.S.shape[1] <= 15 else None

    def _matrix_latex(name, M):
        Md = M.toarray() if hasattr(M, 'toarray') else np.asarray(M)
        body = ' \\\\\n'.join(
            ' & '.join(_fmt_cell(Md[i, j]) if abs(Md[i, j]) > 1e-15 else '0'
                       for j in range(Md.shape[1]))
            for i in range(Md.shape[0])
        )
        inner = (f'\\subsection*{{Matrix ${name}$}}\n\\[\n{name} = \\begin{{bmatrix}}\n'
                 f'{body}\n\\end{{bmatrix}}\n\\]\n')
        return _landscape(inner) if Md.shape[1] > 9 else inner

    if matrices == 'S':
        S_section = _matrix_latex('S', cpn.S)
        Phi_section = _matrix_latex(r'\Phi', cpn.Phi)
    else:
        S_section = (
            f'\\subsection*{{Matrix $S$ (nonzero entries)}}\n'
            f'\\begin{{longtable}}{{rrr}}\n\\toprule\nrow (var) & col (mon) & value \\\\\n\\midrule\n{S_rows}\\bottomrule\n\\end{{longtable}}\n'
        )
        Phi_section = (
            f'\\subsection*{{Matrix $\\Phi$ (nonzero entries)}}\n'
            f'\\begin{{longtable}}{{rrr}}\n\\toprule\nrow (eq) & col (mon) & value \\\\\n\\midrule\n{Phi_rows}\\bottomrule\n\\end{{longtable}}\n'
        )

    doc = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{pdflscape}}
\usepackage[margin=2.2cm]{{geometry}}
\usepackage[colorlinks=true,linkcolor=blue]{{hyperref}}
\setcounter{{MaxMatrixCols}}{{40}}

\title{{CPN1 System Report\\\large\texttt{{{_latex_escape(stem)}}}}}
\author{{TensyGridEngine}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{Summary}}
\begin{{tabular}}{{ll}}
\toprule
Property & Value \\\\
\midrule
{summary_rows}
\bottomrule
\end{{tabular}}

\section{{Original equations}}
Equations as derived before simplification (after parameter substitution and
$dx/dt = 0$), each equal to zero.

{raw_eqs}

\section{{Pinned (trivial) variables}}
Variables fixed to constant values during simplification and substituted out.
\begin{{tabular}}{{ll}}
\toprule
Variable & Pinned value \\\\
\midrule
{pinned_rows}\bottomrule
\end{{tabular}}

\section{{Final equations}}
Reduced system after simplification, each equal to zero.

{final_eqs}

\section{{Variables of the reduced system}}
\begin{{longtable}}{{rllr}}
\toprule
idx & short & full name & $x_0$ \\\\
\midrule
{var_rows}\bottomrule
\end{{longtable}}

{S_section}

{Phi_section}

{jac_section}

\end{{document}}
"""
    tex_path.write_text(doc, encoding='utf-8')

    print(f"[LATEX] Writing {tex_path}")
    result = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex_path.name],
        cwd=str(out_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[LATEX] pdflatex failed (exit {result.returncode}):")
        print(result.stdout[-4000:])
        print(result.stderr[-2000:])
        return pdf_path
    for ext in ('.aux', '.log', '.out'):
        leftover = out_dir / f'{stem}_report{ext}'
        if leftover.exists():
            leftover.unlink()
    print(f"PDF saved: {pdf_path}")
    return pdf_path


def _latex_escape(s: str) -> str:
    """Escape a plain string for use as LaTeX text."""
    import re
    return s.replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')


def run_small_signal_from_driver(problem, pf_results, rms_options):
    import VeraGridEngine.api as vge
    ss_options = vge.RmsSmallSignalStabilityOptions(ss_assessment_time=0, verbose=0)
    driver = vge.SmallSignalStabilityRmsDriver(
        grid=vge.MultiCircuit(Sbase=problem.grid.Sbase),
        rms_options=rms_options,
        sss_options=ss_options,
        pf_results=pf_results,
    )
    driver.problem = problem
    driver.k = problem.get_states_number()
    driver.run()
    state_var_names = [str(v.name) if hasattr(v, 'name') else f"state_{i}"
                       for i, v in enumerate(problem.state_and_algebraic_vars)]
    return driver.results.eigenvalues, driver.results.participation_factors, state_var_names


# ── Main entry ──

def main(grid_file: str = "red_enana.gridcal",
         SIMPLIFY: bool = True,
         ZERO_DERIVATIVES: bool = True,
         EXPORT_EXCEL: bool = True,
         EXPORT_PDF: bool = True) -> None:
    """Build the CPN1 steady-state system and export to .mat.

    Parameters
    ----------
    grid_file : str, default "red_enana.gridcal"
        Grid file name to load.
    SIMPLIFY : bool, default True
        Apply simplification pipeline (deduplicate, substitute pinned variables).
    ZERO_DERIVATIVES : bool, default True
        Remove derivative columns (dx/dt -> 0).
    EXPORT_EXCEL : bool, default True
        Export an .xlsx workbook alongside the .mat file.
    EXPORT_PDF : bool, default True
        Export a LaTeX-compiled PDF report alongside the .mat file.
    """
    grid_filename = sys.argv[1] if len(sys.argv) > 1 else grid_file
    problem_ml, pf_results, rms_options_ml = build_problems(grid_filename=grid_filename)

    problem_ml.build_multilinear_matrices()

    print(f"[INFO] Original S: {problem_ml.S.shape}, Phi: {problem_ml.Phi.shape}")
    print(f"[INFO] Parameters: {len(problem_ml._constant_parameters)} constant, {len(problem_ml._variable_parameters)} variable")

    cpn = process_cpn_system(problem_ml, jaume_flag=SIMPLIFY, zero_derivatives=ZERO_DERIVATIVES)

    n_eqs = cpn.Phi.shape[0]
    n_vars = cpn.S.shape[0]
    n_mon = cpn.S.shape[1]
    rank_Phi = np.linalg.matrix_rank(cpn.Phi.toarray())

    print(f"[RESULT] Equations: {n_eqs}, Variables: {n_vars}, Monomials: {n_mon}, Rank(Phi): {rank_Phi}")
    print(f"[RESULT] System is {'well-determined' if rank_Phi == n_vars else 'UNDERDETERMINED (rank < n_vars)'}")
    print(f"[PINNED] Trivial variables fixed during simplification: {cpn.pinned_vars}")

    eqs_array = np.array([_clean_eq_str(str(eq)) for eq in cpn.eqs_list], dtype=object)
    [print(f"[EQUATION {i+1}]: {eq}\n") for i, eq in enumerate(eqs_array)]
    vars_array = np.array(cpn.vars_list, dtype=object)
    output_dir: Path = Path(__file__).resolve().parent / "CPN1_computations"
    output_dir.mkdir(exist_ok=True)
    output_path: Path = output_dir / f"CPN1_{Path(grid_filename).stem}.mat"
    scipy.io.savemat(str(output_path), {
        'S': cpn.S,
        'Phi': cpn.Phi,
        'eqs': eqs_array,
        'var': vars_array,
        'pinned_vars': cpn.pinned_vars
    })

    print(f"CPN1 file saved: {output_path}")

    if EXPORT_EXCEL:
        x0 = cpn.x0[cpn.orig_keep_idx]
        F = compute_jacobian_from_S(cpn.S, x0)
        J = cpn.Phi @ F.T
        _export_excel(
            cpn=cpn, problem=problem_ml, x0=x0,
            grid_stem=Path(grid_filename).stem,
            n_eqs=n_eqs, n_vars=n_vars, n_mon=n_mon,
            rank_Phi=rank_Phi, F=F, J=J,
        )

    if EXPORT_PDF:
        x0 = cpn.x0[cpn.orig_keep_idx]
        F = compute_jacobian_from_S(cpn.S, x0)
        J = cpn.Phi @ F.T
        _export_latex_pdf(
            cpn=cpn, problem=problem_ml, x0=x0,
            grid_stem=Path(grid_filename).stem,
            n_eqs=n_eqs, n_vars=n_vars, n_mon=n_mon,
            rank_Phi=rank_Phi, F=F, J=J,
        )


if __name__ == "__main__":
    # Entry point: export CPN1 system for the given grid with default flags
    main("red_enana.gridcal", True, True, True, True)
