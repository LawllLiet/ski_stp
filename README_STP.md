# SKINNY-64-192 MITM：STP 搜索代码

此目录独立运行，无需 Gurobi 或 Python 第三方包。它迁移的是提供的 `skinny_LIN_DIS-main` 的当前 MILP 代码：五色、GreenCancellation v4、四类自由度、最多两次 LIN、匹配列凸包、相同轮切分与闭环。`lin_rules.py` 保留原来的匹配秩与凸包检验，`CPMITM_LIN.py` 和 `model_stp.py` 保留原模型的建模逻辑；`stp_backend.py` 将其线性约束与 AND/OR 转为 SMT-LIB2 QF_BV；`solve_stp.py` 用 STP 可满足性查询完成最大化。

**版本差异：** `SKINNY_MITM_STP_Modeling.pdf` 的基线 A 使用绿色不降色、正向仅消耗红色、反向仅消耗蓝色；原代码和 `SKINNY_XOR_MILP_GreenCancellation_v4.pdf` 则允许绿色降色、两方向均可消耗红蓝两色。本实现以用户现有 Gurobi 模型及 v4 为准，属于前一文档中所说的宽松版本 B，**不是基线 A**。若要搜索基线 A，需要另行修改局部规则与对应记账并重新验证。

## 环境和运行

安装带 SMT-LIB2 命令行前端、支持 `get-value` 的 STP，确保 `stp --version` 可用；Python 3.9 或更新版本。STP 路径不在 PATH 时使用 `--stp /path/to/stp`。运行前请先切换到本目录：

```bash
cd /Users/grandkinez/Downloads/codex_skinny_lin/stp_skinny
python3 solve_stp.py --tr 23 --ini 2 --mat 12 --time 3600
python3 solve_stp.py --tr 23 --all --time 3600
```

`--time` 是每一组 `(ini, mat)` 的总秒数，而非每次调用；0 表示无限时。`--all` 枚举全部不同的起始轮与匹配轮。`--outdir` 指定输出目录，默认为当前目录下 `stp_solutions`。23 轮的运行时间和内存取决于 STP 版本及机器，不能保证在给定时限内找到解或证明最优。

只生成不带目标阈值的基础可行性模型：

```bash
python3 solve_stp.py --tr 23 --ini 2 --mat 12 --write-only
stp stp_solutions/TR23_ini2_mat12.smt2
```

正常求解找到解后，输出同名 `.smt2`（该可行解对应的完整阈值查询）、`.stp.out`（求解器原始输出）、`.json`（所有颜色和计数）及 `.txt`（报告）。`Result_TR*.json` 是所有切分的汇总。`optimal` 表示搜索的候选阈值均已通过 SAT/UNSAT 判断证明最优；`feasible_unproven` 表示有可行解但因为超时或 unknown 未证明最优；`unsat` 表示在原模型的正自由度约束下无可行解；`unknown` 表示没有取得可行解且未完成证明。求解器错误会报错，不当作无解。重跑同一输出目录会覆盖同名结果。

## 目标和数值处理

令 α=log2(3)/4，n 为 LIN 次数，搜索目标为 `min(dB-αn, dR-αn, dM)`。三个自由度在原模型中均为整数，0≤n≤2，1≤dM≤16，因此所有候选值为 `k−n0·α`，其中 1≤k≤16、n0∈{0,1,2}。搜索使用二分法，对每个阈值检查 n=0、1、2。对于阈值 `k−n0·α`，条件恰好为 `dM≥k`、`dB,dR≥k+[n>n0]`；因为 0<2α<1，整数边界完全不依赖浮点取整。显示的目标值才使用浮点数。

全部线性表达式使用 16 位有符号位向量，单个二元变量使用 1 位，其他变量按上界定宽并零扩展；支持 2–128 轮。该范围内总计数项远小于有符号范围。读取求解结果后，程序还以 Python 整数重新核验所有变量范围和约束。

## 验证与适用范围

本机未安装 STP，尚未直接用 STP 执行。用 Z3 的 SMT-LIB2 前端作兼容性检查时，2 轮 `(ini,mat)=(0,1)` 的最优目标为 16，与原 Gurobi 模型相同；把生成的全部变量赋值固定到原 Gurobi 模型后仍可行。原仓库的局部规则测试通过 19 项中的 17 项（2 项因依赖跳过）。这不是 23 轮的验证，也不能证明两种求解器或两个模型在所有实例上的完全等价。

本模型的局部消耗与匹配秩不保证跨轮全局独立性；灰色匹配方程也未必构成有效过滤。LIN 的提示集合未被嵌入真实 S 盒求值，正向和逆向候选须结合 `lin_rules.py` 中的分片分别检查。最终仍需对输出方程、自由度和具体密钥恢复过程独立验证，不能仅凭 SAT 宣称得到真实攻击。
