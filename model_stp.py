"""原 SKINNY MILP 的 STP 后端版本：构造约束并导出着色报告。

运行入口是 solve_stp.py；局部规则采用 GreenCancellation v4。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import stp_backend as gp
from stp_backend import GRB

from CPMITM_LIN import (
    Cell,
    add_3xor,
    add_copy,
    add_initial,
    add_match_column,
    add_mc_backward,
    add_mc_forward,
    add_sc,
)
from lin_rules import (
    ALPHA,
    MAX_LIN,
    BITS_COLOR,
    PT,
    SHIFT,
    column_indices,
    format_equation,
    match_equations,
    round_partition,
)


class Index:
    def __init__(self):
        self.tr = 0
        self.ini_r = 0
        self.mat_r = 0
        self.forward = []
        self.backward = []
        self.X = []
        self.U = []
        self.Y = []
        self.Z = []
        self.TK = []
        self.ell = []
        self.gammas = []
        self.enc_b = []
        self.enc_r = []
        self.ksa_b = []
        self.ksa_r = []
        self.GObj = None
        self.GDeg1 = None
        self.GDeg2 = None
        self.GMat = None


def _states(model, prefix, rounds):
    return [[Cell(model, f"{prefix}_j{j}_r{r}") for j in range(16)] for r in range(rounds)]


def _record_enc(index, pair):
    d_enc_b, d_enc_r = pair
    index.enc_b.append(d_enc_b)
    index.enc_r.append(d_enc_r)


def _record_art(index, deltas):
    d_enc_b, d_enc_r, d_ksa_b, d_ksa_r = deltas
    index.enc_b.append(d_enc_b)
    index.enc_r.append(d_enc_r)
    index.ksa_b.append(d_ksa_b)
    index.ksa_r.append(d_ksa_r)


def _add_art(model, index, rnd, pre_art, post_art, forward: bool):
    tk = [index.TK[lane][rnd] for lane in range(3)]
    if forward:
        state_in, state_out = pre_art, post_art
    else:
        state_in, state_out = post_art, pre_art
    for i in range(8):
        _record_art(
            index,
            add_3xor(
                model,
                state_in[i],
                tk[0][i],
                tk[1][i],
                tk[2][i],
                state_out[i],
                f"ART_r{rnd}_j{i}",
            ),
        )
    for i in range(8, 16):
        add_copy(model, state_in[i], state_out[i], f"ARTcopy_r{rnd}_j{i}")


def _add_sr(model, rnd, after_art, after_sr):
    for i in range(16):
        add_copy(model, after_art[SHIFT[i]], after_sr[i], f"SR_r{rnd}_j{i}")


def _add_round(model, index, rnd, kind):
    x_state = index.X[rnd]
    u_state = index.U[rnd]
    y_state = index.Y[rnd]
    z_state = index.Z[rnd]
    nxt = (rnd + 1) % index.tr
    if kind in ("F", "M"):
        for i in range(16):
            add_sc(model, x_state[i], u_state[i], index.ell[rnd][i], f"SC_r{rnd}_j{i}")
        _add_art(model, index, rnd, u_state, y_state, forward=True)
    else:
        for i in range(16):
            add_sc(model, u_state[i], x_state[i], index.ell[rnd][i], f"SC_r{rnd}_j{i}")
        _add_art(model, index, rnd, u_state, y_state, forward=False)
    _add_sr(model, rnd, y_state, z_state)
    if kind == "F":
        for pair in add_mc_forward(model, z_state, index.X[nxt], f"MC_r{rnd}"):
            _record_enc(index, pair)
    elif kind == "B":
        for pair in add_mc_backward(model, z_state, index.X[nxt], f"MC_r{rnd}"):
            _record_enc(index, pair)
    else:
        for col in range(4):
            x_cells = [z_state[cell] for cell in column_indices(col)]
            y_cells = [index.X[nxt][cell] for cell in column_indices(col)]
            index.gammas.append(add_match_column(model, x_cells, y_cells, f"MAT_c{col}"))


def build_model(total_round, ini_r, mat_r):
    """建好模型并设好目标，不求解。"""
    forward, mat, backward = round_partition(total_round, ini_r, mat_r)
    model = gp.Model(f"skinny_lin_TR{total_round}_ini{ini_r}_mat{mat}")
    model.ModelSense = GRB.MAXIMIZE
    index = Index()
    index.tr = total_round
    index.ini_r = ini_r
    index.mat_r = mat
    index.forward = forward
    index.backward = backward
    index.X = _states(model, "X", total_round)
    index.U = _states(model, "U", total_round)
    index.Y = _states(model, "Y", total_round)
    index.Z = _states(model, "Z", total_round)
    index.TK = [_states(model, f"TK{lane+1}", total_round + 1) for lane in range(3)]
    index.ell = [
        [model.addVar(vtype=GRB.BINARY, name=f"ell_j{j}_r{r}") for j in range(16)]
        for r in range(total_round)
    ]

    for rnd in forward:
        _add_round(model, index, rnd, "F")
    _add_round(model, index, mat, "M")
    for rnd in backward:
        _add_round(model, index, rnd, "B")

    for rnd in range(total_round):
        for lane in range(3):
            for i in range(16):
                add_copy(
                    model,
                    index.TK[lane][rnd][PT[i]],
                    index.TK[lane][rnd + 1][i],
                    f"PT_t{lane+1}_r{rnd}_j{i}",
                )

    for cell_i, cell in enumerate(index.X[ini_r]):
        add_initial(model, cell, f"S0_X_j{cell_i}")
    for lane in range(3):
        for cell_i, cell in enumerate(index.TK[lane][0]):
            add_initial(model, cell, f"S0_TK{lane+1}_j{cell_i}")

    ell_vars = [index.ell[r][j] for r in range(total_round) for j in range(16)]
    model.addConstr(gp.quicksum(ell_vars) <= MAX_LIN, name="lin_at_most_two")

    lam_enc_b = gp.quicksum(cell.b for cell in index.X[ini_r])
    lam_enc_r = gp.quicksum(cell.red for cell in index.X[ini_r])
    key_cells = [cell for lane in index.TK for cell in lane[0]]
    lam_ksa_b = gp.quicksum(cell.b for cell in key_cells)
    lam_ksa_r = gp.quicksum(cell.red for cell in key_cells)
    model.addConstr(lam_enc_b == gp.quicksum(index.enc_b), name="bal_enc_b")
    model.addConstr(lam_enc_r == gp.quicksum(index.enc_r), name="bal_enc_r")

    index.GDeg1 = model.addVar(lb=1, ub=48, vtype=GRB.INTEGER, name="GDeg1")
    index.GDeg2 = model.addVar(lb=1, ub=48, vtype=GRB.INTEGER, name="GDeg2")
    index.GMat = model.addVar(lb=1, ub=16, vtype=GRB.CONTINUOUS, name="GMat")
    model.addConstr(index.GDeg1 == lam_ksa_b - gp.quicksum(index.ksa_b), name="dB")
    model.addConstr(index.GDeg2 == lam_ksa_r - gp.quicksum(index.ksa_r), name="dR")
    model.addConstr(index.GMat == gp.quicksum(index.gammas), name="dM")
    index.nlin = gp.quicksum(ell_vars)
    return model, index


def _bit(value) -> int:
    rounded = int(round(value))
    if abs(value - rounded) > 1e-5:
        raise ValueError(f"二元变量不是 0/1：{value}")
    return rounded


def _color_of(cell) -> str:
    bits = (_bit(cell.b.X), _bit(cell.w.X), _bit(cell.red.X))
    try:
        return BITS_COLOR[bits]
    except KeyError as exc:
        raise ValueError(f"非法颜色 {bits}") from exc


def _state_colors(cells):
    return [_color_of(cell) for cell in cells]


def extract_solution(index: Index):
    """读当前解。γ 必须落在整数上，否则凸包没有把掩码夹紧。"""
    gammas = []
    equations = []
    z_colors = _state_colors(index.Z[index.mat_r])
    nxt = (index.mat_r + 1) % index.tr
    x_colors = _state_colors(index.X[nxt])
    for col, gamma_var in enumerate(index.gammas):
        gamma = gamma_var.X
        nearest = int(round(gamma))
        if abs(gamma - nearest) > 1e-5:
            raise ValueError(f"第 {col} 列 γ = {gamma}，不是整数")
        gammas.append(nearest)
        x_white = [1 if z_colors[cell] == "W" else 0 for cell in column_indices(col)]
        y_white = [1 if x_colors[cell] == "W" else 0 for cell in column_indices(col)]
        equations.append(
            [format_equation(terms, col) for terms in match_equations(x_white, y_white)]
        )
    lin_at = [
        [r, j]
        for r in range(index.tr)
        for j in range(16)
        if _bit(index.ell[r][j].X) == 1
    ]
    kind = {}
    for rnd in index.forward:
        kind[rnd] = "forward"
    kind[index.mat_r] = "match"
    for rnd in index.backward:
        kind[rnd] = "backward"
    return {
        "TR": index.tr,
        "ini_r": index.ini_r,
        "mat_r": index.mat_r,
        "GObj": index.GObj.X,
        "GDeg1": int(round(index.GDeg1.X)),
        "GDeg2": int(round(index.GDeg2.X)),
        "GMat": sum(gammas),
        "alpha": ALPHA,
        "nLIN": len(lin_at),
        "lin_at": lin_at,
        "kind": {str(r): kind[r] for r in range(index.tr)},
        "X": [_state_colors(index.X[r]) for r in range(index.tr)],
        "U": [_state_colors(index.U[r]) for r in range(index.tr)],
        "Y": [_state_colors(index.Y[r]) for r in range(index.tr)],
        "Z": [_state_colors(index.Z[r]) for r in range(index.tr)],
        "TK": [
            [_state_colors(index.TK[lane][r]) for r in range(index.tr + 1)]
            for lane in range(3)
        ],
        "match_gamma": gammas,
        "match_equations": equations,
        "note": (
            "GDeg1 是剩余蓝密钥单元 dB，GDeg2 是剩余红密钥单元 dR，"
            "GMat 是四列 γ 之和，并且至少为 1。绿色按非白参加匹配。"
            "有效过滤还要按打出来的方程做联合独立性检验。"
        ),
    }


def format_report(sol) -> str:
    from lin_rules import color_grid

    lines = [
        f"TR={sol['TR']}  ini_r={sol['ini_r']}  mat_r={sol['mat_r']}",
        f"GObj={sol['GObj']:.6f}",
        f"dB=GDeg1={sol['GDeg1']}  dR=GDeg2={sol['GDeg2']}  dM=GMat={sol['GMat']}",
        f"nLIN={sol['nLIN']}  alpha={sol['alpha']:.6f}  lin_at={sol['lin_at']}",
        "",
    ]
    for rnd in range(sol["TR"]):
        lines.append(f"round {rnd}  {sol['kind'][str(rnd)]}")
        lines.append("X")
        lines.append(color_grid(sol["X"][rnd]))
        lines.append("Y")
        lines.append(color_grid(sol["Y"][rnd]))
        lines.append("Z")
        lines.append(color_grid(sol["Z"][rnd]))
        lines.append("")
    lines.append("TK1 at round 0")
    lines.append(color_grid(sol["TK"][0][0]))
    lines.append("TK2 at round 0")
    lines.append(color_grid(sol["TK"][1][0]))
    lines.append("TK3 at round 0")
    lines.append(color_grid(sol["TK"][2][0]))
    lines.append("")
    lines.append("match equations")
    for col, (gamma, eqs) in enumerate(zip(sol["match_gamma"], sol["match_equations"])):
        lines.append(f"column {col}  gamma={gamma}")
        if eqs:
            lines.extend(eqs)
        else:
            lines.append("(no equation)")
    lines.append("")
    lines.append(sol["note"])
    return "\n".join(lines) + "\n"

