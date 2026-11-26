"""把五色局部规则写成 Gurobi 约束。

XOR / 2XOR / 3XOR 用 GreenCancellation v4 的闭式，不取凸包。
绿色按蓝、红分量计入 nB、nR，可以降色。S 盒上的 LIN 仍单独处理。
SC 用 SKINNY_SC_LIN_MILP 的三条等式。
匹配列用 lin_rules.MATCH_HULL，那是 (白格掩码, γ) 的凸包。
"""

from __future__ import annotations

import stp_backend as gp
from stp_backend import GRB

from lin_rules import MATCH_HULL


class Cell:
    """一个单元的 (b, w, red)。式 (1)：b+w ≤ 1，red+w ≤ 1。"""

    def __init__(self, model, name):
        self.b = model.addVar(vtype=GRB.BINARY, name=name + "_b")
        self.w = model.addVar(vtype=GRB.BINARY, name=name + "_w")
        self.red = model.addVar(vtype=GRB.BINARY, name=name + "_r")
        model.addConstr(self.b + self.w <= 1, name=name + "_bw")
        model.addConstr(self.red + self.w <= 1, name=name + "_rw")


def add_and(model, left, right, name):
    """二元与。用一般约束，和三条线性不等式是同一个集合。"""
    prod = model.addVar(vtype=GRB.BINARY, name=name)
    model.addGenConstrAnd(prod, [left, right], name=name + "_and")
    return prod


def add_or(model, parts, name):
    """二元或。"""
    flag = model.addVar(vtype=GRB.BINARY, name=name)
    model.addGenConstrOr(flag, list(parts), name=name + "_or")
    return flag


def add_and_not(model, positive, negative, name):
    """A = P ∧ ¬W。"""
    flag = model.addVar(vtype=GRB.BINARY, name=name)
    flipped = model.addVar(vtype=GRB.BINARY, name=name + "_not")
    model.addConstr(flipped + negative == 1, name=name + "_neg")
    model.addGenConstrAnd(flag, [positive, flipped], name=name + "_and")
    return flag


def _presence(model, cells, name):
    """P_B、P_R、P_W 和白色优先后的 A_B、A_R。v4 不再建 P_G。"""
    present_b = add_or(model, [cell.b for cell in cells], name + "_PB")
    present_r = add_or(model, [cell.red for cell in cells], name + "_PR")
    present_w = add_or(model, [cell.w for cell in cells], name + "_PW")
    active_b = add_and_not(model, present_b, present_w, name + "_AB")
    active_r = add_and_not(model, present_r, present_w, name + "_AR")
    return present_w, active_b, active_r


def add_enc_gate(model, inputs, output, name):
    """XOR 或 2XOR。消耗只记 ENC。inputs 至少两个 Cell。

    绿色的蓝、红比特已经在 2δ ≤ nQ 里各算 1。没有 Δ ≤ 1−P_G。
    """
    present_w, active_b, active_r = _presence(model, inputs, name)
    d_enc_b = model.addVar(vtype=GRB.BINARY, name=name + "_dEncB")
    d_enc_r = model.addVar(vtype=GRB.BINARY, name=name + "_dEncR")
    model.addConstr(output.w == present_w, name=name + "_w")
    model.addConstr(output.b + d_enc_b == active_b, name=name + "_b")
    model.addConstr(output.red + d_enc_r == active_r, name=name + "_r")
    model.addConstr(2 * d_enc_b <= gp.quicksum(cell.b for cell in inputs), name=name + "_sb")
    model.addConstr(2 * d_enc_r <= gp.quicksum(cell.red for cell in inputs), name=name + "_sr")
    return d_enc_b, d_enc_r


def add_3xor(model, state, tk1, tk2, tk3, output, name):
    """ART：v = u ⊕ t1 ⊕ t2 ⊕ t3。u 带该色则整笔记 ENC，否则整笔记 KSA。"""
    inputs = [state, tk1, tk2, tk3]
    present_w, active_b, active_r = _presence(model, inputs, name)
    d_enc_b = model.addVar(vtype=GRB.BINARY, name=name + "_dEncB")
    d_enc_r = model.addVar(vtype=GRB.BINARY, name=name + "_dEncR")
    d_ksa_b = model.addVar(vtype=GRB.BINARY, name=name + "_dKsaB")
    d_ksa_r = model.addVar(vtype=GRB.BINARY, name=name + "_dKsaR")
    model.addConstr(output.w == present_w, name=name + "_w")
    model.addConstr(output.b + d_enc_b + d_ksa_b == active_b, name=name + "_b")
    model.addConstr(output.red + d_enc_r + d_ksa_r == active_r, name=name + "_r")
    model.addConstr(
        2 * (d_enc_b + d_ksa_b) <= gp.quicksum(cell.b for cell in inputs),
        name=name + "_sb",
    )
    model.addConstr(
        2 * (d_enc_r + d_ksa_r) <= gp.quicksum(cell.red for cell in inputs),
        name=name + "_sr",
    )
    model.addConstr(d_enc_b <= state.b, name=name + "_eb")
    model.addConstr(d_ksa_b <= 1 - state.b, name=name + "_kb")
    model.addConstr(d_enc_r <= state.red, name=name + "_er")
    model.addConstr(d_ksa_r <= 1 - state.red, name=name + "_kr")
    return d_enc_b, d_enc_r, d_ksa_b, d_ksa_r


def add_copy(model, src, dst, name):
    model.addConstr(dst.b == src.b, name=name + "_b")
    model.addConstr(dst.w == src.w, name=name + "_w")
    model.addConstr(dst.red == src.red, name=name + "_r")


def add_gnd(model, before, name):
    """白格猜为 B/R/G 的着色抽象；具体双向猜测相容性须事后验证。"""
    after = Cell(model, name + "_after")
    gb = model.addVar(vtype=GRB.BINARY, name=name + "_gB")
    gr = model.addVar(vtype=GRB.BINARY, name=name + "_gR")
    gbr = model.addVar(vtype=GRB.BINARY, name=name + "_gBR")
    model.addConstr(gb + gr + gbr <= before.w, name=name + "_white")
    model.addConstr(after.b == before.b + gb + gbr, name=name + "_b")
    model.addConstr(after.red == before.red + gr + gbr, name=name + "_r")
    model.addConstr(after.w == before.w - gb - gr - gbr, name=name + "_w")
    return after, (gb, gr, gbr)


def add_sc(model, src, dst, ell, name):
    """正向时 src 是 S 盒输入；反向时 src 是 S 盒输出。ell 由调用方创建。"""
    green = add_and(model, src.b, src.red, name + "_g")
    model.addConstr(ell <= green, name=name + "_ell")
    model.addConstr(dst.b == src.b - green + ell, name=name + "_b")
    model.addConstr(dst.red == src.red - green + ell, name=name + "_r")
    model.addConstr(dst.w == src.w + green - ell, name=name + "_w")


def add_mc_forward(model, zin, xout, name):
    """(a,b,c,d) → (a⊕c⊕d, a, b⊕c, a⊕c)。α 用一次 2XOR，不拆成两步。"""
    deltas = []
    for index in range(4):
        a = zin[index]
        b = zin[index + 4]
        c = zin[index + 8]
        d = zin[index + 12]
        alpha = xout[index]
        beta = xout[index + 4]
        gamma = xout[index + 8]
        eta = xout[index + 12]
        deltas.append(add_enc_gate(model, [a, c, d], alpha, f"{name}_a{index}"))
        add_copy(model, a, beta, f"{name}_b{index}")
        deltas.append(add_enc_gate(model, [b, c], gamma, f"{name}_g{index}"))
        deltas.append(add_enc_gate(model, [a, c], eta, f"{name}_e{index}"))
    return deltas


def add_mc_backward(model, zin, xout, name):
    """逆列：a=β，b=β⊕γ⊕η，c=β⊕η，d=α⊕η。b 用一次 2XOR。"""
    deltas = []
    for index in range(4):
        a = zin[index]
        b = zin[index + 4]
        c = zin[index + 8]
        d = zin[index + 12]
        alpha = xout[index]
        beta = xout[index + 4]
        gamma = xout[index + 8]
        eta = xout[index + 12]
        add_copy(model, beta, a, f"{name}_a{index}")
        deltas.append(add_enc_gate(model, [beta, gamma, eta], b, f"{name}_b{index}"))
        deltas.append(add_enc_gate(model, [beta, eta], c, f"{name}_c{index}"))
        deltas.append(add_enc_gate(model, [alpha, eta], d, f"{name}_d{index}"))
    return deltas


def add_match_column(model, x_cells, y_cells, name):
    """一列的 γ。x 是 MC 输入（Z），y 是 MC 输出（下一轮 X）。

    白格是未知量。γ 由 MATCH_HULL 与八个 w 位唯一决定。
    """
    gamma = model.addVar(lb=0.0, ub=4.0, vtype=GRB.CONTINUOUS, name=name + "_gamma")
    whites = [cell.w for cell in x_cells] + [cell.w for cell in y_cells]
    for k, coef in enumerate(MATCH_HULL):
        expr = gp.LinExpr(coef[0])
        expr += coef[9] * gamma
        for i in range(8):
            if coef[i + 1]:
                expr += coef[i + 1] * whites[i]
        model.addConstr(expr >= 0, name=f"{name}_h{k}")
    return gamma


def add_initial(model, cell, name):
    """初始结构只允许蓝、红、灰。"""
    model.addConstr(cell.w == 0, name=name + "_nw")
    model.addConstr(cell.b + cell.red <= 1, name=name + "_ng")
