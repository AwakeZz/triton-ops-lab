"""
算子练习验收框架。

你只写 kernels/*.py 里的 kernel，这里负责：生成输入 → 比对正确性 → 测带宽 → 给 roofline 结论。

用法:
    python check.py              # 跑全部
    python check.py --only 06_softmax
    python check.py --selftest   # 验证框架自身(用内置参考实现)
    python check.py --only 03_sum --odd        # 加测非常规尺寸
    python check.py --odd hard                 # 再加"非 2 的幂"尺寸

--odd 会在常规 shape 之外再跑几组别扭的尺寸, 只验正确性和启动次数, 不测带宽。
理由: 常规 shape 是 (4096, 1024), 总元素数恰好能被 BLOCK 整除,
grid 写成 n // BLOCK 也能"通过"。非常规尺寸专门用来抓这类错。

设计上故意不提供任何 kernel 实现。--selftest 里那几个是最平凡的版本，
只用来确认框架逻辑没坏，不是给你抄的。
"""

import os

# 必须在 import triton 之前, 否则编译缓存会落到用户主目录
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))
os.environ.setdefault("TRITON_HOME", os.path.join(_HERE, ".triton-cache"))

import argparse
import importlib.util
import json
import statistics
import sys
import time
import traceback
from datetime import date
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.runtime.jit import JITFunction

ROOT = Path(__file__).resolve().parent
KDIR = ROOT / "kernels"
RESULT_DIR = ROOT / "results"

# ---- 本机实测基线 (2026-10-18, 见 bench_baseline.py) -------------------------
PEAK_GBPS = 180.5
PEAK_TFLOPS = 8.2
RIDGE = PEAK_TFLOPS * 1e12 / (PEAK_GBPS * 1e9)  # 45.4 FLOP/Byte, 用实测带宽算

M, N = 4096, 1024
DEV = "cuda"


# ---- 统计 kernel 启动次数: 直接反映你的 kernel 走了几趟 ----------------------
_LAUNCH = {"n": 0}
_orig_run = JITFunction.run


def _counting_run(self, *args, **kwargs):
    _LAUNCH["n"] += 1
    return _orig_run(self, *args, **kwargs)


JITFunction.run = _counting_run


# ---- 计时 -------------------------------------------------------------------
def bench(fn, target_total_ms=80.0, min_iters=20):
    """测前先跑 0.8s 把时钟拉满, 迭代数按耗时自适应。笔记本必须这么做。"""
    ramp_until = time.perf_counter() + 0.8
    while time.perf_counter() < ramp_until:
        fn()
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5):
        fn()
    e.record()
    torch.cuda.synchronize()
    est = max(s.elapsed_time(e) / 5, 1e-4)
    iters = min(max(min_iters, int(target_total_ms / est)), 5000)

    samples = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        samples.append(a.elapsed_time(b))
    return statistics.median(samples), iters


# ---- 题目定义 ---------------------------------------------------------------
def _mk_binop(m=M, n=N):
    x = torch.randn(m, n, device=DEV)
    y = torch.randn(m, n, device=DEV)
    return dict(x=x, y=y, out=torch.empty_like(x))


def _mk_unary(m=M, n=N):
    x = torch.randn(m, n, device=DEV)
    return dict(x=x, out=torch.empty_like(x))


def _mk_row_reduce(m=M, n=N):
    x = torch.randn(m, n, device=DEV)
    return dict(x=x, out=torch.empty(m, device=DEV))


def _mk_tp(m=M, n=N):
    x = torch.randn(m, n, device=DEV)
    return dict(x=x, out=torch.empty(n, m, device=DEV))


# ---- 非常规尺寸 (--odd) ------------------------------------------------------
# tail: N 仍是 2 的幂, 但行数凑不出 BLOCK 的整数倍 -> 专抓 "grid 用整除" 和尾部 mask
# hard: N 也不是 2 的幂 -> 归约类 kernel 必须自己做 next_power_of_2 + mask
TAIL_SHAPES = [(997, 1024), (1003, 512), (1, 4096)]
HARD_SHAPES = [(333, 1000), (77, 127)]
ODD_SHAPES = {"tail": TAIL_SHAPES, "hard": TAIL_SHAPES + HARD_SHAPES}


PROBLEMS = [
    dict(
        id="01_add", file="01_add.py", title="vector add",
        args=("x", "y", "out"), make=_mk_binop,
        ref=lambda t: t["x"] + t["y"],
        ideal_bytes=lambda t: 3 * t["x"].numel() * 4,
        ideal_launch=1, tol=(1e-6, 1e-6),
        note="读 2 写 1, 算术强度 0.25 → 铁定访存受限。目标就是贴着 180 GB/s",
    ),
    dict(
        id="02_copy", file="02_copy.py", title="copy",
        args=("x", "out"), make=_mk_unary,
        ref=lambda t: t["x"].clone(),
        ideal_bytes=lambda t: 2 * t["x"].numel() * 4,
        ideal_launch=1, tol=(0.0, 0.0),
        note="读 1 写 1。这是你的尺子, 后面所有 kernel 都拿它比",
    ),
    dict(
        id="03_sum", file="03_sum.py", title="row sum (dim=1)",
        args=("x", "out"), make=_mk_row_reduce,
        ref=lambda t: t["x"].sum(dim=1),
        ideal_bytes=lambda t: t["x"].numel() * 4 + t["x"].shape[0] * 4,
        ideal_launch=1, tol=(1e-4, 1e-3),
        note="沿 dim=1 归约。难点: 一个 program 处理一行还是多行, 决定能不能占满带宽",
    ),
    dict(
        id="04_max", file="04_max.py", title="row max (dim=1)",
        args=("x", "out"), make=_mk_row_reduce,
        ref=lambda t: t["x"].max(dim=1).values,
        ideal_bytes=lambda t: t["x"].numel() * 4 + t["x"].shape[0] * 4,
        ideal_launch=1, tol=(0.0, 0.0),
        note="和 03 结构一样, 换成 max。顺手对比两者带宽差多少",
    ),
    dict(
        id="05_cumsum", file="05_cumsum.py", title="row cumsum (dim=1)",
        args=("x", "out"), make=_mk_unary,
        ref=lambda t: torch.cumsum(t["x"], dim=1),
        ideal_bytes=lambda t: 2 * t["x"].numel() * 4,
        ideal_launch=1, tol=(1e-4, 1e-4),
        note="前缀和。一行装不进一个 block 时怎么办? 这是第一次遇到跨 block 依赖",
    ),
    dict(
        id="06_softmax", file="06_softmax.py", title="row softmax",
        args=("x", "out"), make=_mk_unary,
        ref=lambda t: torch.softmax(t["x"], dim=1),
        ideal_bytes=lambda t: 2 * t["x"].numel() * 4,
        ideal_launch=1, tol=(1e-4, 1e-5),
        note="关键题。朴素写法读 3 趟写 1 趟; online softmax 读 1 写 1。启动次数会暴露你写的是哪种",
    ),
    dict(
        id="07_layernorm", file="07_layernorm.py", title="layernorm (dim=1)",
        args=("x", "out"), make=_mk_unary,
        ref=lambda t: (t["x"] - t["x"].mean(dim=1, keepdim=True))
        / torch.sqrt(t["x"].var(dim=1, unbiased=False, keepdim=True) + 1e-5),
        ideal_bytes=lambda t: 2 * t["x"].numel() * 4,
        ideal_launch=1, tol=(1e-4, 1e-5),
        note="mean 和 var 要在一趟里同时算(Welford 或 sum/sumsq), 分两趟就是 4 次访问",
    ),
    dict(
        id="08_transpose", file="08_transpose.py", title="transpose + contiguous",
        args=("x", "out"), make=_mk_tp,
        ref=lambda t: t["x"].t().contiguous(),
        ideal_bytes=lambda t: 2 * t["x"].numel() * 4,
        ideal_launch=1, tol=(0.0, 0.0),
        note="读和写必有一边不连续。靠 tl.trans + 合适的 BLOCK 让两边都合并访存",
    ),
]

PROB_BY_ID = {p["id"]: p for p in PROBLEMS}


# ---- 执行 -------------------------------------------------------------------
def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_one(p, mod=None, shape=None, benchmark=True):
    t = p["make"](*(shape or (M, N)))
    ref = p["ref"](t)  # 先算参考值, 防止 kernel 就地改写输入
    out = t["out"]
    args = [t[a] for a in p["args"]]

    if mod is None:
        kern = _SELFTEST[p["id"]]
    else:
        kern = getattr(mod, "run", None)
        if kern is None:
            return dict(id=p["id"], status="无 run()", note="文件里要有 run(...)")
    try:
        kern(*args)
        torch.cuda.synchronize()
    except NotImplementedError:
        return dict(id=p["id"], status="未实现")
    except Exception as exc:
        return dict(id=p["id"], status="报错", note=f"{type(exc).__name__}: {exc}",
                    tb=traceback.format_exc())

    ok = out.shape == ref.shape and torch.allclose(
        out, ref, rtol=p["tol"][0], atol=p["tol"][1], equal_nan=True)
    if not ok:
        delta = (out.float() - ref.float()).abs().max().item() if out.shape == ref.shape else float("nan")
        return dict(id=p["id"], status="结果错", note=f"max|diff|={delta:.3e}")

    _LAUNCH["n"] = 0
    kern(*args)
    torch.cuda.synchronize()
    launch = _LAUNCH["n"]

    if not benchmark:
        return dict(id=p["id"], status="通过", launch=launch,
                    ideal_launch=p["ideal_launch"])

    ms, iters = bench(lambda: kern(*args))
    ideal_bytes = p["ideal_bytes"](t)
    gbps = ideal_bytes / (ms * 1e-3) / 1e9
    return dict(
        id=p["id"], status="通过", ms=round(ms, 4), iters=iters,
        gbps=round(gbps, 1), pct_peak=round(gbps / PEAK_GBPS * 100, 1),
        launch=launch, ideal_launch=p["ideal_launch"],
    )


def _selftest_kernels():
    @triton.jit
    def _add(x, y, out, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(out + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

    @triton.jit
    def _copy(x, out, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(out + i, tl.load(x + i, mask=m), mask=m)

    def run_add(x, y, out):
        n = x.numel()
        _add[(triton.cdiv(n, 4096),)](x, y, out, n, BLOCK=4096)

    def run_copy(x, out):
        n = x.numel()
        _copy[(triton.cdiv(n, 4096),)](x, out, n, BLOCK=4096)

    return {"01_add": run_add, "02_copy": run_copy}


_SELFTEST = _selftest_kernels()


def _dw(s):
    """显示宽度: 中文按 2 列算, 否则 f-string 的对齐会错位。"""
    return sum(2 if ord(c) > 0x2E7F else 1 for c in str(s))


def pad(s, width, right=False):
    s = str(s)
    fill = " " * max(width - _dw(s), 0)
    return fill + s if right else s + fill


def run_odd(plist, level):
    """非常规尺寸检测: 只验正确性 + 启动次数, 不测带宽。
    返回失败数, 0 表示全通过。
    """
    shapes = ODD_SHAPES[level]
    print(f"\n非常规尺寸检测 ({level})  只验正确性和启动次数, 不测带宽\n")
    head = (pad("题目", 14) + pad("尺寸", 15) + pad("状态", 10)
            + pad("启动", 6, True) + pad("理想", 6, True))
    print(head)
    print("-" * (_dw(head) + 2))

    fails = []
    for p in plist:
        path = KDIR / p["file"]
        if not path.exists():
            print(pad(p["id"], 14) + pad("-", 15) + "缺文件")
            continue
        try:
            mod = load_module(path)
        except Exception as exc:
            print(pad(p["id"], 14) + pad("-", 15) + pad("导入失败", 10)
                  + f"  {type(exc).__name__}: {exc}")
            fails.append(p["id"])
            continue

        skipped = False
        for sh in shapes:
            r = run_one(p, mod, shape=sh, benchmark=False)
            tag = f"({sh[0]}, {sh[1]})"
            if r["status"] == "通过":
                print(pad(r["id"], 14) + pad(tag, 15) + pad("通过", 10)
                      + pad(r["launch"], 6, True) + pad(r["ideal_launch"], 6, True))
            elif r["status"] == "未实现" and not skipped:
                print(pad(r["id"], 14) + pad(tag, 15) + pad("未实现", 10))
                skipped = True
            elif not skipped:
                print(pad(r["id"], 14) + pad(tag, 15) + pad(r["status"], 10)
                      + f"  {r.get('note', '')}")
                if r.get("tb"):
                    print(r["tb"])
                fails.append(f"{r['id']} {tag}")

    print("-" * (_dw(head) + 2))
    if fails:
        print(f"非常规尺寸: {len(fails)} 处失败")
        print("  · 结果错 -> 大概率 grid 用了整除(要 cdiv), 或尾部 mask 没盖住")
        print("  · 报错   -> 大概率 BLOCK 假设了 N 是 2 的幂, 越界读写了")
        return len(fails)
    print("非常规尺寸全部通过")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="只跑某一题, 例如 06_softmax")
    ap.add_argument("--selftest", action="store_true", help="用内置参考实现验证框架")
    ap.add_argument("--odd", nargs="?", const="tail", choices=["tail", "hard"],
                    help="额外跑非常规尺寸: --odd 只跑尾数, --odd hard 再加非 2 的幂")
    a = ap.parse_args()

    if a.selftest:
        targets = [PROB_BY_ID[k] for k in _SELFTEST]
        for p in targets:
            r = run_one(p, mod=None)
            print(f"[self] {r['id']:<14} {r['status']:<6} "
                  f"{r.get('gbps', 0):>7.1f} GB/s  {r.get('pct_peak', 0):>5.1f}%  "
                  f"launch={r.get('launch', '-')}")
        return

    if not KDIR.exists():
        KDIR.mkdir(parents=True)
    plist = [PROB_BY_ID[a.only]] if a.only else PROBLEMS

    print(f"基线: {PEAK_GBPS} GB/s, 平衡点 {RIDGE:.1f} FLOP/Byte\n")
    head = (pad("题目", 14) + pad("状态", 10) + pad("ms", 10, True)
            + pad("GB/s", 10, True) + pad("%峰值", 9, True)
            + pad("启动", 8, True) + pad("理想", 8, True))
    print(head)
    print("-" * (_dw(head) + 2))

    rows = []
    for p in plist:
        path = KDIR / p["file"]
        if not path.exists():
            print(pad(p["id"], 14) + pad("缺文件", 10))
            continue
        try:
            mod = load_module(path)
        except Exception as exc:
            print(pad(p["id"], 14) + pad("导入失败", 10) + f"  {type(exc).__name__}: {exc}")
            continue
        r = run_one(p, mod)
        rows.append(r)
        if r["status"] == "通过":
            print(pad(r["id"], 14) + pad("通过", 10)
                  + pad(f"{r['ms']:.4f}", 10, True) + pad(f"{r['gbps']:.1f}", 10, True)
                  + pad(f"{r['pct_peak']:.1f}%", 9, True)
                  + pad(r["launch"], 8, True) + pad(r["ideal_launch"], 8, True))
        else:
            print(pad(r["id"], 14) + pad(r["status"], 10) + f"  {r.get('note', '')}")
            if r.get("tb"):
                print(r["tb"])

    print("-" * (_dw(head) + 2))
    passed = [r for r in rows if r["status"] == "通过"]
    print(f"通过 {len(passed)}/{len(plist)}")
    for r in passed:
        if r["launch"] > r["ideal_launch"]:
            p = PROB_BY_ID[r["id"]]
            print(f"  · {r['id']}: 启动 {r['launch']} 次, 理想 {r['ideal_launch']} 次 → "
                  f"多走了 {r['launch'] - r['ideal_launch']} 趟访存, 这就是优化空间")
        if r["pct_peak"] < 85:
            p = PROB_BY_ID[r["id"]]
            print(f"  · {r['id']}: 只有 {r['pct_peak']}% 峰值 —— {p['note']}")

    if passed:
        RESULT_DIR.mkdir(exist_ok=True)
        out = RESULT_DIR / f"check_{date.today()}.json"
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n结果已存 {out}")

    if a.odd:
        nfail = run_odd(plist, a.odd)
        if nfail:
            sys.exit(1)


if __name__ == "__main__":
    main()
