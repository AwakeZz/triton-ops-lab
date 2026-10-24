"""
算子练习验收框架。

你只写 kernels/*.py 里的 kernel，这里负责：生成输入 → 比对正确性 → 测性能 → 给 roofline 结论。

用法:
    python check.py              # 跑全部
    python check.py --only 06_softmax
    python check.py --selftest   # 验证框架自身(用内置参考实现)
    python check.py --only 03_sum --odd        # 加测非常规尺寸
    python check.py --odd hard                 # 再加"非 2 的幂"尺寸

--odd 会在常规 shape 之外再跑几组别扭的尺寸, 只验正确性和启动次数, 不测带宽。
理由: 常规 shape 是 (4096, 1024), 总元素数恰好能被 BLOCK 整除,
grid 写成 n // BLOCK 也能"通过"。非常规尺寸专门用来抓这类错。
GEMM 题的尺寸是三元组 (M, N, K), 见 GEMM_ODD。

两类题分开看指标:
    01~08  elementwise / reduction / scan / transpose -> 访存受限, 看 GB/s
    09~12  GEMM 四版                                  -> 计算受限, 看 TFLOPS
    同一个 kernel 该盯哪个指标, 用 roofline 平衡点判断, 不是凭感觉。

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

# ---- fp16 tensor core 实测峰值 (2026-10-24, 见 _probe_peak.py) ---------------
# 按尺寸扫 1024/2048/3072/4096 最好成绩 18.2 TFLOPS (cuBLAS, 持续 1s 后取中位数)。
# 官方规格是 40.7 TFLOPS, 只跑到 45% —— 笔记本功耗墙, SM 时钟撑不住。
# 和 PEAK_GBPS 一样, 分母用本机实测值, 不用规格值。
PEAK_TFLOPS_TC = 18.2

M, N = 4096, 1024
GM = GN = GK = 1024     # GEMM 的标准尺寸, 三方都是 1024
DEV = "cuda"

# 参考值必须是 fp32 精确算, 不能被 TF32 截断, 否则 GEMM 的正确性比对没意义
torch.backends.cuda.matmul.allow_tf32 = False


# ---- 统计 kernel 启动次数: 直接反映你的 kernel 走了几趟 ----------------------
_LAUNCH = {"n": 0}
_orig_run = JITFunction.run


def _counting_run(self, *args, **kwargs):
    _LAUNCH["n"] += 1
    return _orig_run(self, *args, **kwargs)


JITFunction.run = _counting_run


# ---- 计时 -------------------------------------------------------------------
def bench(fn, target_total_ms=80.0, min_iters=20, ramp_ms=800):
    """先预热把时钟拉满, 再按实测耗时自适应迭代数。笔记本 GPU 必须这么做。

    预热次数按单次耗时反推, 不能写成"跑满 0.8 秒" —— launch 是异步的,
    慢 kernel 在 0.8 秒墙钟里能往队列里塞进上千次 launch, 接下来那次
    synchronize 就要等上几十分钟。09_gemm_naive 单次 0.475 秒, 这么卡死过一次。
    """
    fn()                      # 第一次调用含 JIT 编译, 排除在计时和预热之外
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    single = max((time.perf_counter() - t0) * 1e3, 1e-3)   # 单次耗时, ms

    warm = max(1, min(int(ramp_ms / single), 4000))
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()

    # 单次就超过 50 ms 的 kernel 不必强跑 20 次, 3 次取中位数已经够稳
    floor = 3 if single > 50 else min_iters
    iters = min(max(floor, int(target_total_ms / single)), 5000)

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


def _mk_gemm(m=GM, n=GN, k=GK):
    """A(m,k) fp16  x  B(k,n) fp16  ->  out(m,n) fp32

    fp16 是因为 tl.dot 只有喂 fp16/bf16 才走 tensor core 的 mma 指令;
    out 用 fp32 是因为累加器必须是 fp32, 顺便让正确性比对可以卡得紧一点。
    """
    a = torch.randn(m, k, device=DEV, dtype=torch.float16)
    b = torch.randn(k, n, device=DEV, dtype=torch.float16)
    return dict(a=a, b=b, out=torch.empty(m, n, device=DEV, dtype=torch.float32))


# ---- 非常规尺寸 (--odd) ------------------------------------------------------
# tail: N 仍是 2 的幂, 但行数凑不出 BLOCK 的整数倍 -> 专抓 "grid 用整除" 和尾部 mask
# hard: N 也不是 2 的幂 -> 归约类 kernel 必须自己做 next_power_of_2 + mask
TAIL_SHAPES = [(997, 1024), (1003, 512), (1, 4096)]
HARD_SHAPES = [(333, 1000), (77, 127)]
ODD_SHAPES = {"tail": TAIL_SHAPES, "hard": TAIL_SHAPES + HARD_SHAPES}

# GEMM 的尺寸是三元组 (M, N, K)
# tail: M/N/K 不是 BLOCK 的整数倍 -> 抓 tile 边界的 mask
# hard: 再加大到小到 1、K 不是 2 的幂 -> 抓 K 方向尾块和退化尺寸
GEMM_ODD = {
    "tail": [(1000, 1000, 1024), (1024, 1024, 1000), (2048, 512, 768)],
    "hard": [(1000, 1000, 1000), (17, 33, 65), (1, 1, 1024), (128, 128, 129)],
}


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
    # ---- 以下四题是同一个 GEMM 的四次迭代, 形状相同, 看 TFLOPS 怎么爬 ----------
    dict(
        id="09_gemm_naive", file="09_gemm_naive.py", title="gemm (1 elem / prog)",
        args=("a", "b", "out"), make=_mk_gemm,
        ref=lambda t: t["a"].float() @ t["b"].float(),
        ideal_flops=lambda t: 2 * t["a"].shape[0] * t["b"].shape[1] * t["a"].shape[1],
        metric="tflops", shape=(GM, GN, GK), ideal_launch=1, tol=(1e-4, 1e-3),
        odd_shapes=GEMM_ODD, target_pct=1,
        note="这一版是故意的下限: 一个 program 一个元素, K 串行循环, 0 复用。"
             "它的存在只是给后面三版当分母",
    ),
    dict(
        id="10_gemm_tiled", file="10_gemm_tiled.py", title="gemm (tile, no dot)",
        args=("a", "b", "out"), make=_mk_gemm,
        ref=lambda t: t["a"].float() @ t["b"].float(),
        ideal_flops=lambda t: 2 * t["a"].shape[0] * t["b"].shape[1] * t["a"].shape[1],
        metric="tflops", shape=(GM, GN, GK), ideal_launch=1, tol=(1e-4, 1e-3),
        odd_shapes=GEMM_ODD, target_pct=12,
        note="tile 化 + K 方向分块累加, 但累加用广播外层积, 还是 FMA。"
             "比 09 快 500 倍, 但离 tensor core 差 5 倍 —— 这就是第 11 题的理由",
    ),
    dict(
        id="11_gemm_dot", file="11_gemm_dot.py", title="gemm (tl.dot)",
        args=("a", "b", "out"), make=_mk_gemm,
        ref=lambda t: t["a"].float() @ t["b"].float(),
        ideal_flops=lambda t: 2 * t["a"].shape[0] * t["b"].shape[1] * t["a"].shape[1],
        metric="tflops", shape=(GM, GN, GK), ideal_launch=1, tol=(1e-4, 1e-3),
        odd_shapes=GEMM_ODD, target_pct=70,
        note="把累加换成 tl.dot, 一下子进 tensor core。这一行就是 GEMM 和前面八题的分水岭",
    ),
    dict(
        id="12_gemm_pipe", file="12_gemm_pipe.py", title="gemm (dot + pipeline)",
        args=("a", "b", "out"), make=_mk_gemm,
        ref=lambda t: t["a"].float() @ t["b"].float(),
        ideal_flops=lambda t: 2 * t["a"].shape[0] * t["b"].shape[1] * t["a"].shape[1],
        metric="tflops", shape=(GM, GN, GK), ideal_launch=1, tol=(1e-4, 1e-3),
        odd_shapes=GEMM_ODD, target_pct=80,
        note="同样的 tl.dot, 调 tile 尺寸 / num_warps / num_stages。"
             "注意 tile 不是越大越好, 要看 grid 能不能填满 20 个 SM",
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
    t = p["make"](*(shape or p.get("shape", (M, N))))
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

    if p.get("metric") == "tflops":
        tflops = p["ideal_flops"](t) / (ms * 1e-3) / 1e12
        return dict(
            id=p["id"], status="通过", ms=round(ms, 4), iters=iters,
            tflops=round(tflops, 4),
            pct_peak=round(tflops / PEAK_TFLOPS_TC * 100, 1),
            launch=launch, ideal_launch=p["ideal_launch"],
        )

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
    print(f"\n非常规尺寸检测 ({level})  只验正确性和启动次数, 不测带宽\n")
    head = (pad("题目", 14) + pad("尺寸", 19) + pad("状态", 10)
            + pad("启动", 6, True) + pad("理想", 6, True))
    print(head)
    print("-" * (_dw(head) + 2))

    fails = []
    for p in plist:
        shapes = p.get("odd_shapes", ODD_SHAPES)[level]
        path = KDIR / p["file"]
        if not path.exists():
            print(pad(p["id"], 14) + pad("-", 19) + "缺文件")
            continue
        try:
            mod = load_module(path)
        except Exception as exc:
            print(pad(p["id"], 14) + pad("-", 19) + pad("导入失败", 10)
                  + f"  {type(exc).__name__}: {exc}")
            fails.append(p["id"])
            continue

        skipped = False
        for sh in shapes:
            r = run_one(p, mod, shape=sh, benchmark=False)
            tag = "(" + ", ".join(str(v) for v in sh) + ")"
            if r["status"] == "通过":
                print(pad(r["id"], 14) + pad(tag, 19) + pad("通过", 10)
                      + pad(r["launch"], 6, True) + pad(r["ideal_launch"], 6, True))
            elif r["status"] == "未实现" and not skipped:
                print(pad(r["id"], 14) + pad(tag, 19) + pad("未实现", 10))
                skipped = True
            elif not skipped:
                print(pad(r["id"], 14) + pad(tag, 19) + pad(r["status"], 10)
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


def _fmt_tf(v):
    """0.0045 这种小值用两位小数会被抹成 0.00, 09_gemm_naive 正好踩这个。"""
    return f"{v:.4f}" if v < 1 else f"{v:.2f}"


def _row_line(r, metric):
    val = _fmt_tf(r["tflops"]) if metric == "tflops" else f"{r['gbps']:.1f}"
    return (pad(r["id"], 14) + pad("通过", 10)
            + pad(f"{r['ms']:.4f}", 10, True) + pad(val, 10, True)
            + pad(f"{r['pct_peak']:.1f}%", 9, True)
            + pad(r["launch"], 8, True) + pad(r["ideal_launch"], 8, True))


def _print_group(plist, metric, rows):
    """按指标分表打印。访存类看 GB/s, 计算类看 TFLOPS, 混在一张表里没法读。"""
    sub = [p for p in plist if p.get("metric", "gbps") == metric]
    if not sub:
        return
    title = ("计算类 (compute-bound) · 分母 = fp16 tensor core 实测 "
             f"{PEAK_TFLOPS_TC} TFLOPS") if metric == "tflops" else \
            ("访存类 (memory-bound) · 分母 = 实测带宽 "
             f"{PEAK_GBPS} GB/s")
    print(f"\n【{title}】")
    col = "TFLOPS" if metric == "tflops" else "GB/s"
    head = (pad("题目", 14) + pad("状态", 10) + pad("ms", 10, True)
            + pad(col, 10, True) + pad("%峰值", 9, True)
            + pad("启动", 8, True) + pad("理想", 8, True))
    print(head)
    print("-" * (_dw(head) + 2))

    for p in sub:
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
            print(_row_line(r, metric))
        else:
            print(pad(r["id"], 14) + pad(r["status"], 10) + f"  {r.get('note', '')}")
            if r.get("tb"):
                print(r["tb"])
    print("-" * (_dw(head) + 2))


def _cublas_ms(t):
    """cuBLAS 在同样输入下的耗时, 输出缓冲预分配, 避免把 malloc 算进计时。"""
    key = (t["a"].shape, t["b"].shape)
    buf = _CBUF.get(key)
    if buf is None:
        buf = _CBUF[key] = torch.empty(t["a"].shape[0], t["b"].shape[1],
                                       device=DEV, dtype=torch.float16)
    fn = lambda: torch.matmul(t["a"], t["b"], out=buf)
    fn()
    torch.cuda.synchronize()
    return bench(fn)[0]


_CBUF = {}


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

    print(f"基线: 带宽 {PEAK_GBPS} GB/s, FP32 {PEAK_TFLOPS} TFLOPS, "
          f"fp16 tensor core {PEAK_TFLOPS_TC} TFLOPS, 平衡点 {RIDGE:.1f} FLOP/Byte")

    rows = []
    _print_group(plist, "gbps", rows)
    _print_group(plist, "tflops", rows)

    passed = [r for r in rows if r["status"] == "通过"]
    print(f"\n通过 {len(passed)}/{len(plist)}")
    for r in passed:
        p = PROB_BY_ID[r["id"]]
        if r["launch"] > r["ideal_launch"]:
            print(f"  · {r['id']}: 启动 {r['launch']} 次, 理想 {r['ideal_launch']} 次 → "
                  f"多走了 {r['launch'] - r['ideal_launch']} 趟访存, 这就是优化空间")
        if r["pct_peak"] < p.get("target_pct", 85):
            print(f"  · {r['id']}: 只有 {r['pct_peak']}% 峰值 —— {p['note']}")

    # GEMM 单独给一行 cuBLAS 对照, 这是唯一能和工业实现直接比的题
    gemm_rows = [r for r in passed if PROB_BY_ID[r["id"]].get("metric") == "tflops"]
    if gemm_rows:
        print("\n与 cuBLAS 对照 (同尺寸、同 dtype, 输出缓冲预分配):")
        for r in gemm_rows:
            p = PROB_BY_ID[r["id"]]
            t = p["make"](*p.get("shape", (M, N)))
            tms = _cublas_ms(t)
            ttf = p["ideal_flops"](t) / (tms * 1e-3) / 1e12
            rel = tms / r["ms"]
            verdict = "超过" if rel > 1 else "落后"
            print(f"  · {r['id']}: cuBLAS {tms:.4f} ms ({_fmt_tf(ttf)} TFLOPS) | "
                  f"你 {r['ms']:.4f} ms ({_fmt_tf(r['tflops'])} TFLOPS) → "
                  f"{verdict} {abs(1 - rel) * 100:.0f}%")
            del t
            torch.cuda.empty_cache()

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
