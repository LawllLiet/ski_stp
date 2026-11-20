"""SKINNY MITM + LIN 的局部规则。

这里只放与求解器无关的定义，MILP 在 CPMITM_LIN.py / MITM_Skinny_LIN.py。

来源，按后面的版本覆盖前面的版本：

- Dong、Hua、Sun、Li、Wang、Hu，CRYPTO 2021，ePrint 2021/427。
  轮函数骨架、PT、ShiftRows、闭环切分 (ini_r, mat_r) 与 Dong21 的
  MITM_Skinny_v3_keyrecovery.py 一致。
- SKINNY_XOR_MILP_GreenCancellation v4：在 FourCounters v3 上去掉
  「有绿则禁止降色」。绿色同时计入蓝、红分量，同色分量至少两个即可消耗。
  平衡仍是 v3 式 (18)，在 v4 里编号为式 (20)。
- SKINNY_SC_LIN_MILP v1：绿色穿过 SubCells。全局至多两次 LIN，
  猜测代价 dℓ = α Σℓ ≤ (1/2) log2(3)。
- skinny_mc_dom：匹配轮不做颜色传播，按列算 γ = 4 - rank(A)。
- skinny_LIN_DIS：SKINNY-64 的 4-bit S 盒三片仿射。这些集合不进 MILP，
  只在恢复阶段验收。逆向验收看 y ∈ Y_h。

匹配列的 (白格掩码, γ) 凸包由 cddlib 生成，57 条刻面。导入时会检查
256 个掩码都被夹成唯一的 γ。XOR 与 SC 文档给的是闭式等式，不走凸包。
"""

from __future__ import annotations

import math
from fractions import Fraction

ALPHA = math.log2(3) / 4.0
# 至多猜两个绿色做 LIN。一次代价是 α，两次合计 (1/2) log2(3)。
MAX_LIN = 2

# (b, w, red)。蓝是正向依赖，红是反向依赖，绿是两侧叠在同一格，
# 灰是本 episode 的常数，白是当前模型不再追踪的格。
COLOR_BITS = {
    "C": (0, 0, 0),
    "B": (1, 0, 0),
    "R": (0, 0, 1),
    "G": (1, 0, 1),
    "W": (0, 1, 0),
}
BITS_COLOR = {bits: name for name, bits in COLOR_BITS.items()}
ALL_COLORS = ("C", "B", "R", "G", "W")

# Z[i] = Y[SHIFT[i]]。与 CPMITM.ShiftRow_Skinny 一致。
SHIFT = (0, 1, 2, 3, 7, 4, 5, 6, 10, 11, 8, 9, 13, 14, 15, 12)
# TK_{r+1}[i] = TK_r[PT[i]]。LFSR 不改变颜色，不建。
PT = (9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7)

# 行主序。Y = M X，列向量 (x0, x1, x2, x3) = 下标 (j, j+4, j+8, j+12)。
MC = (
    (1, 0, 1, 1),
    (1, 0, 0, 0),
    (0, 1, 1, 0),
    (1, 0, 1, 0),
)
MC_INV = (
    (0, 1, 0, 0),
    (0, 1, 1, 1),
    (0, 1, 0, 1),
    (1, 0, 0, 1),
)

# SKINNY-64 S 盒。x = x0 + 2 x1 + 4 x2 + 8 x3，x0 是低位。
SBOX = (
    0xC, 0x6, 0x9, 0x0, 0x1, 0xA, 0x2, 0xB,
    0x3, 0x8, 0x5, 0xD, 0x4, 0xE, 0x7, 0xF,
)
# 每个 hint 的列向量：bit i 为 1 时异或进去的半字节，再异或常数。
# 正向只在 x ∈ X_h 时等于 S；逆向只在 y ∈ Y_h 时等于 S^{-1}。
LIN_FORWARD = (
    ((0x8, 0x1, 0x2, 0x4), 0x0),
    ((0x9, 0x6, 0xB, 0xC), 0xF),
    ((0x1, 0x3, 0xD, 0x5), 0xC),
)
LIN_BACKWARD = (
    ((0x2, 0x4, 0x8, 0x1), 0x0),
    ((0xE, 0x5, 0x7, 0xF), 0x3),
    ((0x1, 0x3, 0x9, 0xC), 0x5),
)
X_HINT = (
    frozenset({0x5, 0x7, 0xA, 0xB, 0xD, 0xE, 0xF}),
    frozenset({0x1, 0x2, 0x3, 0x6, 0x8}),
    frozenset({0x0, 0x4, 0x9, 0xC}),
)

# b + a0 w_x0 + ... + a3 w_x3 + a4 w_y0 + ... + a7 w_y3 + a8 γ ≥ 0。
# cddlib 对 256 个点 (w, γ(w)) 的凸包。含变量界。导入时核对过：
# 每个 0-1 掩码上，γ 被夹成唯一实数，且等于 4 - rank(A)。
MATCH_HULL = (
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
    (-2, 1, 0, 1, 0, 0, 1, 0, 1, 1),
    (-3, 1, 1, 1, 0, 0, 1, 1, 1, 1),
    (-1, 0, 0, 0, 1, 1, 0, 0, 1, 1),
    (-3, 1, 0, 1, 1, 1, 1, 0, 1, 1),
    (-4, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    (-1, 0, 1, 1, 0, 0, 0, 1, 0, 1),
    (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
    (-1, 1, 0, 0, 0, 0, 1, 0, 0, 1),
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    (4, 0, 0, -1, 0, -1, -1, 0, -1, -1),
    (4, 0, -1, 0, 0, -1, -1, 0, -1, -1),
    (4, 0, 0, -1, -1, 0, -1, 0, -1, -1),
    (4, 0, -1, 0, -1, 0, -1, 0, -1, -1),
    (4, 0, -1, -1, 0, 0, -1, 0, -1, -1),
    (4, -1, 0, -1, 0, 0, -1, 0, -1, -1),
    (4, -1, -1, 0, 0, 0, -1, 0, -1, -1),
    (4, -1, 0, -1, 0, -1, 0, 0, -1, -1),
    (4, -1, -1, 0, 0, -1, 0, 0, -1, -1),
    (4, -1, 0, -1, -1, 0, 0, 0, -1, -1),
    (4, -1, -1, 0, -1, 0, 0, 0, -1, -1),
    (1, 0, 0, 0, 0, 0, 0, 0, -1, 0),
    (4, -1, -1, -1, 0, 0, 0, 0, -1, -1),
    (4, -1, 0, 0, 0, -1, 0, -1, -1, -1),
    (4, -1, 0, 0, -1, 0, 0, -1, -1, -1),
    (4, -1, 0, -1, 0, 0, 0, -1, -1, -1),
    (4, 0, 0, 0, -1, 0, -1, -1, -1, -1),
    (4, -1, 0, 0, 0, 0, -1, -1, -1, -1),
    (4, 0, 0, -1, 0, 0, -1, -1, -1, -1),
    (4, 0, 0, 0, 0, -1, -1, -1, -1, -1),
    (4, -1, 0, 0, 0, -1, -1, -1, 0, -1),
    (4, 0, 0, -1, 0, -1, -1, -1, 0, -1),
    (4, -1, 0, 0, -1, 0, -1, -1, 0, -1),
    (4, 0, 0, -1, -1, 0, -1, -1, 0, -1),
    (4, -1, 0, -1, 0, -1, 0, -1, 0, -1),
    (4, -1, 0, -1, -1, 0, 0, -1, 0, -1),
    (1, 0, 0, 0, 0, 0, 0, -1, 0, 0),
    (4, 0, -1, -1, 0, -1, -1, 0, 0, -1),
    (4, -1, 0, -1, 0, -1, -1, 0, 0, -1),
    (4, -1, -1, 0, 0, -1, -1, 0, 0, -1),
    (4, 0, -1, -1, -1, 0, -1, 0, 0, -1),
    (4, -1, 0, -1, -1, 0, -1, 0, 0, -1),
    (4, -1, -1, 0, -1, 0, -1, 0, 0, -1),
    (1, 0, 0, 0, 0, 0, -1, 0, 0, 0),
    (4, -1, -1, -1, 0, -1, 0, 0, 0, -1),
    (1, 0, 0, 0, 0, -1, 0, 0, 0, 0),
    (4, -1, -1, -1, -1, 0, 0, 0, 0, -1),
    (1, 0, 0, 0, -1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, -1, 0, 0, 0, 0, 0, 0),
    (1, 0, -1, 0, 0, 0, 0, 0, 0, 0),
    (1, -1, 0, 0, 0, 0, 0, 0, 0, 0),
)


def gf2_rank(columns) -> int:
    """columns 是若干长度为 4 的 0/1 向量，按列排。返回 F2 秩。"""
    if not columns:
        return 0
    cols_n = len(columns)
    mat = [[columns[c][r] & 1 for c in range(cols_n)] for r in range(4)]
    rank = 0
    col0 = 0
    for row in range(4):
        pivot = None
        for col in range(col0, cols_n):
            for rr in range(row, 4):
                if mat[rr][col] == 1:
                    pivot = (rr, col)
                    break
            if pivot is not None:
                break
        if pivot is None:
            continue
        rr, col = pivot
        if rr != row:
            mat[row], mat[rr] = mat[rr], mat[row]
        for rr2 in range(4):
            if rr2 != row and mat[rr2][col] == 1:
                for cc in range(cols_n):
                    mat[rr2][cc] ^= mat[row][cc]
        rank += 1
        col0 = col + 1
    return rank


def active_columns(x_white, y_white):
    """白格进 A 的列。x_white / y_white 各 4 个 0/1，1 表示白。"""
    cols = []
    for i in range(4):
        if x_white[i]:
            cols.append(tuple(MC[r][i] for r in range(4)))
    for j in range(4):
        if y_white[j]:
            cols.append(tuple(1 if r == j else 0 for r in range(4)))
    return cols


def column_gamma(x_white, y_white) -> int:
    return 4 - gf2_rank(active_columns(x_white, y_white))


def _gamma_table():
    table = []
    for mask in range(256):
        x_white = [(mask >> i) & 1 for i in range(4)]
        y_white = [(mask >> (4 + j)) & 1 for j in range(4)]
        table.append(column_gamma(x_white, y_white))
    return tuple(table)


GAMMA = _gamma_table()


def _hull_forces_table(hull=MATCH_HULL, table=GAMMA) -> None:
    for mask, gamma in enumerate(table):
        white = [(mask >> i) & 1 for i in range(8)]
        lower = None
        upper = None
        for coef in hull:
            acc = coef[0]
            for i in range(8):
                acc += coef[i + 1] * white[i]
            ag = coef[9]
            if ag == 0:
                if acc < 0:
                    raise RuntimeError(f"凸包切掉了掩码 {mask}")
                continue
            bound = Fraction(-acc, ag)
            if ag > 0:
                lower = bound if lower is None else max(lower, bound)
            else:
                upper = bound if upper is None else min(upper, bound)
        if lower != gamma or upper != gamma:
            raise RuntimeError(f"掩码 {mask} 的 γ 没有被夹成 {gamma}，得到 [{lower}, {upper}]")


_hull_forces_table()


def compute_match_hull():
    """用 cddlib 重算 MATCH_HULL。没装 cddlib 时抛 ImportError。"""
    import cdd

    rows = []
    for mask, gamma in enumerate(GAMMA):
        white = [(mask >> i) & 1 for i in range(8)]
        rows.append([1, *white, gamma])
    generators = cdd.matrix_from_array(rows, rep_type=cdd.RepType.GENERATOR)
    poly = cdd.polyhedron_from_matrix(generators)
    ineq = cdd.copy_inequalities(poly)
    if list(ineq.lin_set):
        raise RuntimeError("匹配凸包出现了等式行，和满维假设不一致")
    hull = []
    for row in ineq.array:
        coef = []
        for value in row:
            as_int = int(round(float(value)))
            if abs(float(value) - as_int) > 1e-8:
                raise RuntimeError(f"凸包系数不是整数：{value}")
            coef.append(as_int)
        hull.append(tuple(coef))
    _hull_forces_table(tuple(hull))
    return tuple(hull)


def mask_of(x_white, y_white) -> int:
    mask = 0
    for i, bit in enumerate(x_white):
        if bit:
            mask |= 1 << i
    for j, bit in enumerate(y_white):
        if bit:
            mask |= 1 << (4 + j)
    return mask


def dot4(mask, column) -> int:
    acc = 0
    for row in range(4):
        if (mask >> row) & 1 and column[row]:
            acc ^= 1
    return acc


def kernel_basis(columns):
    """{h ∈ F2^4 : h 与每一列正交} 的一组基，h 用 4 个比特表示。"""
    survivors = []
    for h in range(1, 16):
        if all(dot4(h, col) == 0 for col in columns):
            survivors.append(h)
    basis = []
    for vector in survivors:
        reduced = vector
        for pivot in basis:
            bit = pivot & -pivot
            if reduced & bit:
                reduced ^= pivot
        if reduced == 0:
            continue
        bit = reduced & -reduced
        for i, pivot in enumerate(basis):
            if pivot & bit:
                basis[i] = pivot ^ reduced
        basis.append(reduced)
    return basis


def equation_terms(h, x_white, y_white):
    """h^T c = 0 里出现的非白格。返回 ('x', i) 或 ('y', j)。"""
    terms = []
    for i in range(4):
        if x_white[i]:
            continue
        column = tuple(MC[r][i] for r in range(4))
        if dot4(h, column):
            terms.append(("x", i))
    for j in range(4):
        if y_white[j]:
            continue
        if (h >> j) & 1:
            terms.append(("y", j))
    return terms


def match_equations(x_white, y_white):
    columns = active_columns(x_white, y_white)
    return [equation_terms(h, x_white, y_white) for h in kernel_basis(columns)]


def format_equation(terms, column) -> str:
    if not terms:
        return "(空)"
    parts = []
    for side, index in terms:
        cell = column + 4 * index
        parts.append(f"{side}{index}[{cell}]")
    return " ⊕ ".join(parts) + " = 0"


def round_partition(total_round, ini_r, mat_r):
    """返回 (正向轮列表, 匹配轮, 反向轮列表)。三个集合覆盖 0..TR-1 各一次。

    与 Dong21 genConstraints_total 的两个分支一致。正向列表按加密顺序，
    反向列表不保证计算顺序。
    """
    if total_round < 2:
        raise ValueError("至少 2 轮")
    if not (0 <= ini_r < total_round and 0 <= mat_r < total_round):
        raise ValueError("ini_r / mat_r 越界")
    if ini_r == mat_r:
        raise ValueError("起点轮和匹配轮不能是同一轮")
    if mat_r > ini_r:
        forward = list(range(ini_r, mat_r))
        backward = list(range(mat_r + 1, total_round)) + list(range(0, ini_r))
    else:
        forward = list(range(ini_r, total_round)) + list(range(0, mat_r))
        backward = list(range(mat_r + 1, ini_r))
    return forward, mat_r, backward


def gate_outcomes(inputs, ksa: bool):
    """枚举一次 XOR / 2XOR / 3XOR 的可行 (输出, δ)。

    inputs 是 (b, w, red) 的列表。返回集合，元素为
    ((b, w, red), δEncB, δEncR, δKsaB, δKsaR)。
    ksa 为 False 时两个 KSA 消耗固定为 0，对应纯 ENC 的 XOR 与 2XOR。
    """
    present_b = int(any(cell[0] for cell in inputs))
    present_r = int(any(cell[2] for cell in inputs))
    present_w = int(any(cell[1] for cell in inputs))
    active_b = present_b if present_w == 0 else 0
    active_r = present_r if present_w == 0 else 0
    # 绿色的 b、r 都是 1，所以 nB、nR 已经把绿色分量算进去。v4 不再用 P_G 禁止降色。
    sum_b = sum(cell[0] for cell in inputs)
    sum_r = sum(cell[2] for cell in inputs)
    state_b, _, state_r = inputs[0]
    found = set()
    for d_enc_b in (0, 1):
        for d_ksa_b in (0, 1):
            for d_enc_r in (0, 1):
                for d_ksa_r in (0, 1):
                    if not ksa and (d_ksa_b or d_ksa_r):
                        continue
                    delta_b = d_enc_b + d_ksa_b
                    delta_r = d_enc_r + d_ksa_r
                    if delta_b > 1 or delta_r > 1:
                        continue
                    if 2 * delta_b > sum_b or 2 * delta_r > sum_r:
                        continue
                    if ksa:
                        if d_enc_b > state_b or d_ksa_b > 1 - state_b:
                            continue
                        if d_enc_r > state_r or d_ksa_r > 1 - state_r:
                            continue
                    out_b = active_b - delta_b
                    out_r = active_r - delta_r
                    if out_b not in (0, 1) or out_r not in (0, 1):
                        continue
                    found.add(((out_b, present_w, out_r), d_enc_b, d_enc_r, d_ksa_b, d_ksa_r))
    return found


def delta0_output(inputs):
    """不消耗时的唯一输出颜色。"""
    chosen = [
        item[0]
        for item in gate_outcomes(inputs, ksa=(len(inputs) == 4))
        if item[1] == item[2] == item[3] == item[4] == 0
    ]
    if len(chosen) != 1:
        raise RuntimeError(f"δ = 0 的输出不是唯一的：{inputs} -> {chosen}")
    return BITS_COLOR[chosen[0]]


def sc_target(src_bits, ell: int):
    """SC 的三条等式。源不是绿时 ell 只能是 0。非法时返回 None。"""
    b, w, red = src_bits
    green = 1 if b == 1 and red == 1 else 0
    if ell not in (0, 1) or ell > green:
        return None
    out = (b - green + ell, w + green - ell, red - green + ell)
    if out not in BITS_COLOR:
        return None
    return out


def apply_affine(columns, const, value) -> int:
    out = const
    for i in range(4):
        if (value >> i) & 1:
            out ^= columns[i]
    return out


def sbox_inverse():
    inverse = [0] * 16
    for value, image in enumerate(SBOX):
        inverse[image] = value
    return tuple(inverse)


def column_indices(index: int):
    return (index, index + 4, index + 8, index + 12)


def get_column(state, index):
    return tuple(state[cell] for cell in column_indices(index))


def set_column(state, index, values):
    for cell, value in zip(column_indices(index), values):
        state[cell] = value


def shift_rows(state):
    return [state[SHIFT[i]] for i in range(16)]


def inv_shift_rows(state):
    out = [None] * 16
    for i in range(16):
        out[SHIFT[i]] = state[i]
    return out


def permute_tk(state):
    return [state[PT[i]] for i in range(16)]


def mc_forward_colors(column):
    a, b, c, d = column
    alpha = delta0_output((COLOR_BITS[a], COLOR_BITS[c], COLOR_BITS[d]))
    beta = a
    gamma = delta0_output((COLOR_BITS[b], COLOR_BITS[c]))
    eta = delta0_output((COLOR_BITS[a], COLOR_BITS[c]))
    return (alpha, beta, gamma, eta)


def mc_backward_colors(column):
    alpha, beta, gamma, eta = column
    a = beta
    b = delta0_output((COLOR_BITS[beta], COLOR_BITS[gamma], COLOR_BITS[eta]))
    c = delta0_output((COLOR_BITS[beta], COLOR_BITS[eta]))
    d = delta0_output((COLOR_BITS[alpha], COLOR_BITS[eta]))
    return (a, b, c, d)


def _sc_state(state):
    return [BITS_COLOR[sc_target(COLOR_BITS[color], 0)] for color in state]


def _art(state_in, tweakeys):
    out = [None] * 16
    for i in range(8):
        out[i] = delta0_output(
            (
                COLOR_BITS[state_in[i]],
                COLOR_BITS[tweakeys[0][i]],
                COLOR_BITS[tweakeys[1][i]],
                COLOR_BITS[tweakeys[2][i]],
            )
        )
    for i in range(8, 16):
        out[i] = state_in[i]
    return out


def forward_round_colors(x_state, tweakeys):
    """ell = 0 且每次 XOR 都不消耗时的正向一轮。返回 U, Y, Z, X_next。"""
    after_sc = _sc_state(x_state)
    after_art = _art(after_sc, tweakeys)
    after_sr = shift_rows(after_art)
    nxt = [None] * 16
    for index in range(4):
        set_column(nxt, index, mc_forward_colors(get_column(after_sr, index)))
    return after_sc, after_art, after_sr, nxt


def match_round_colors(x_state, tweakeys):
    """匹配轮：正向 SC、ART、ShiftRows，不传播 MC。"""
    after_sc = _sc_state(x_state)
    after_art = _art(after_sc, tweakeys)
    after_sr = shift_rows(after_art)
    return after_sc, after_art, after_sr


def backward_round_colors(x_next, tweakeys):
    """ell = 0 且不消耗时的反向一轮。返回 X, U, Y, Z。"""
    after_sr = [None] * 16
    for index in range(4):
        set_column(after_sr, index, mc_backward_colors(get_column(x_next, index)))
    after_art = inv_shift_rows(after_sr)
    after_sc = _art(after_art, tweakeys)
    x_state = _sc_state(after_sc)
    return x_state, after_sc, after_art, after_sr


def zero_consumption_trail(total_round, ini_r, mat_r, x_ini, tk0):
    """从初始着色出发，ell = 0 且所有 δ = 0，把整条环上的颜色推出来。

    x_ini 是 16 个颜色名。tk0 是三条 tweakey、每条 16 个颜色名。
    返回 X, U, Y, Z（各 TR 个状态）和 TK（3 × (TR+1)）。
    """
    forward, mat, backward = round_partition(total_round, ini_r, mat_r)
    tk = [[None] * (total_round + 1) for _ in range(3)]
    for lane in range(3):
        tk[lane][0] = list(tk0[lane])
        for rnd in range(total_round):
            tk[lane][rnd + 1] = permute_tk(tk[lane][rnd])

    def lanes(rnd):
        return [tk[lane][rnd] for lane in range(3)]

    x_states = [None] * total_round
    u_states = [None] * total_round
    y_states = [None] * total_round
    z_states = [None] * total_round
    x_states[ini_r] = list(x_ini)

    for rnd in forward:
        if x_states[rnd] is None:
            raise RuntimeError(f"正向轮 {rnd} 的 X 还没有颜色")
        u_states[rnd], y_states[rnd], z_states[rnd], nxt = forward_round_colors(
            x_states[rnd], lanes(rnd)
        )
        nxt_r = (rnd + 1) % total_round
        if x_states[nxt_r] is not None:
            raise RuntimeError(f"正向轮 {rnd} 写到了已经有颜色的 X{nxt_r}")
        x_states[nxt_r] = nxt

    if x_states[mat] is None:
        raise RuntimeError("匹配轮的 X 没有被正向 MC 写出来")
    u_states[mat], y_states[mat], z_states[mat] = match_round_colors(x_states[mat], lanes(mat))

    pending = list(backward)
    steps = 0
    while pending:
        steps += 1
        if steps > total_round + 2:
            raise RuntimeError(f"反向轮推不下去：{pending}")
        rest = []
        moved = False
        for rnd in pending:
            nxt_r = (rnd + 1) % total_round
            if x_states[nxt_r] is None:
                rest.append(rnd)
                continue
            if x_states[rnd] is not None:
                raise RuntimeError(f"反向轮 {rnd} 的 X 已经被写过")
            x_states[rnd], u_states[rnd], y_states[rnd], z_states[rnd] = backward_round_colors(
                x_states[nxt_r], lanes(rnd)
            )
            moved = True
        pending = rest
        if pending and not moved:
            raise RuntimeError(f"反向轮没有已知的下游状态：{pending}")
    return x_states, u_states, y_states, z_states, tk


def color_grid(colors) -> str:
    return "\n".join(" ".join(colors[row * 4:(row + 1) * 4]) for row in range(4))
