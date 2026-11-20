#!/usr/bin/env python3
"""STP feasibility search for the accompanying SKINNY MITM model."""
import argparse
import json
import math
import re
import subprocess
import sys
import time
from decimal import Decimal, ROUND_CEILING, localcontext
from pathlib import Path
from types import SimpleNamespace

from model_stp import build_model, extract_solution, format_report
from lin_rules import ALPHA

with localcontext() as ctx:
    ctx.prec = 60
    A = Decimal(3).ln() / Decimal(2).ln() / Decimal(4)


def candidates():
    # t = min(dB-alpha*n, dR-alpha*n, dM), with n in {0,1,2}.
    # Each candidate is the exact symbolic number k - n0*alpha.
    return sorted(((k, n0) for k in range(1, 17) for n0 in range(3)),
                  key=lambda item: Decimal(item[0]) - A*item[1])


def ceil_decimal(x):
    return int(x.to_integral_value(rounding=ROUND_CEILING))


def conditions(index, n, threshold):
    k, n0 = threshold
    # alpha is in (0, 1/2), so ceil((n-n0)*alpha) is 1 iff n > n0.
    key_min = k + int(n > n0)
    match_min = k
    return [index.nlin == n, index.GDeg1 >= key_min,
            index.GDeg2 >= key_min, index.GMat >= match_min]


def run_solver(executable, text, timeout):
    try:
        proc = subprocess.run([executable, '--SMTLIB2'], input=text, text=True,
                              capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 'timeout', ''
    except FileNotFoundError as exc:
        raise RuntimeError(f'找不到 STP 可执行文件：{executable}') from exc
    statuses = re.findall(r'(?m)^\s*(sat|unsat|unknown)\s*$', proc.stdout)
    if proc.returncode != 0 or not statuses or '(error' in proc.stdout or 'unsupported' in proc.stdout:
        raise RuntimeError(f'STP 调用失败（退出码 {proc.returncode}）：\n{proc.stderr}\n{proc.stdout[:2000]}')
    return statuses[0], proc.stdout


def load_values(model, output):
    pattern = r'\(\s*\|?([A-Za-z_][A-Za-z0-9_]*)\|?\s+(#b[01]+|#x[0-9a-fA-F]+|\(_\s+bv\d+\s+\d+\))\s*\)'
    found = {}
    for name, val in re.findall(pattern, output):
        found[name] = int(val[2:], 2 if val.startswith('#b') else 16) if val.startswith('#') else int(re.search(r'bv(\d+)', val).group(1))
    missing = model.vars.keys() - found.keys()
    if missing: raise RuntimeError(f'STP 未返回全部变量：缺少 {len(missing)} 个，例如 {next(iter(missing))}')
    for name, var in model.vars.items(): var.X = found[name]


def solve(args, ini, mat):
    model, index = build_model(args.tr, ini, mat)
    stem = Path(args.outdir) / f'TR{args.tr}_ini{ini}_mat{mat}'
    stem.parent.mkdir(parents=True, exist_ok=True)
    if args.write_only:
        # Base feasibility query; add constraints and get-value manually if desired.
        stem.with_suffix('.smt2').write_text(model.smt(), encoding='utf-8')
        print(f'已写入 {stem}.smt2（基础可行性模型）', flush=True)
        return {'status': 'write_only'}
    deadline = time.monotonic() + args.time if args.time else None
    vals = candidates(); lo = 0; hi = len(vals)-1; best = None; interrupted = False
    while lo <= hi:
        mid = (lo + hi)//2; threshold = vals[mid]
        any_sat = False
        for n in range(3):
            remaining = None if deadline is None else max(0, deadline-time.monotonic())
            if remaining is not None and remaining <= 0:
                interrupted = True; break
            extra = conditions(index, n, threshold)
            print(f'TR={args.tr} ini={ini} mat={mat} t≥{float(Decimal(threshold[0]) - A*threshold[1]):.6f} nLIN={n}', flush=True)
            status, _ = run_solver(args.stp, model.smt(extra), remaining)
            if status in ('timeout', 'unknown'):
                interrupted = True; break
            if status == 'sat':
                # Re-solve with value query. Save the exact query for reproducibility.
                query = model.smt(extra, values=True)
                remaining = None if deadline is None else max(0, deadline-time.monotonic())
                if remaining is not None and remaining <= 0:
                    interrupted = True; break
                status2, output = run_solver(args.stp, query, remaining)
                if status2 in ('timeout', 'unknown'):
                    interrupted = True; break
                if status2 != 'sat': raise RuntimeError('重解时结果不一致')
                load_values(model, output)
                model.validate(extra)
                index.GObj = SimpleNamespace(X=min(index.GDeg1.X-ALPHA*n, index.GDeg2.X-ALPHA*n, index.GMat.X))
                solution = extract_solution(index)
                best = (solution, query, output)
                any_sat = True; break
        if interrupted: break
        if any_sat: lo = mid + 1
        else: hi = mid - 1
    if best is None:
        status = 'unknown' if interrupted else 'unsat'
        print(f'状态：{status}', flush=True)
        return {'TR': args.tr, 'ini_r': ini, 'mat_r': mat, 'status': status}
    sol, query, output = best
    sol['status'] = 'feasible_unproven' if interrupted else 'optimal'
    stem.with_suffix('.smt2').write_text(query, encoding='utf-8')
    stem.with_suffix('.stp.out').write_text(output, encoding='utf-8')
    stem.with_suffix('.json').write_text(json.dumps(sol, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    stem.with_suffix('.txt').write_text(format_report(sol) + f"status={sol['status']}\n", encoding='utf-8')
    print(f"{sol['status']}: GObj={sol['GObj']:.6f}, dB={sol['GDeg1']}, dR={sol['GDeg2']}, dM={sol['GMat']}, nLIN={sol['nLIN']}", flush=True)
    return sol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tr', type=int, default=23); p.add_argument('--ini', type=int, default=2)
    p.add_argument('--mat', type=int, default=12); p.add_argument('--all', action='store_true')
    p.add_argument('--time', type=float, default=120, help='每个切分总时限（秒）；0 为不限时')
    p.add_argument('--stp', default='stp', help='STP 可执行文件路径')
    p.add_argument('--outdir', default='stp_solutions'); p.add_argument('--write-only', action='store_true')
    args = p.parse_args()
    if not 2 <= args.tr <= 128: p.error('轮数须在 2..128，确保 16 位有符号算术不溢出')
    if args.time < 0: p.error('--time 不能为负')
    pairs = ((i,j) for i in range(args.tr) for j in range(args.tr) if i != j) if args.all else [(args.ini,args.mat)]
    results = []
    try:
        for ini, mat in pairs:
            results.append(solve(args, ini, mat))
    except (RuntimeError, ValueError) as exc:
        print(f'错误：{exc}', file=sys.stderr); return 1
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir)/f'Result_TR{args.tr}.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return 0

if __name__ == '__main__': sys.exit(main())
