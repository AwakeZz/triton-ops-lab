"""
Zero baseline: 设备画像 + 带宽实测。

第一周之前先跑这个，把"这张卡的真实上限"钉下来。
后面每一个 kernel 的验收都拿这里的数字做基准。

用法:
    python bench_baseline.py
"""

import os

# 必须在 import triton 之前设置, 否则编译缓存会落到用户主目录
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))
os.environ.setdefault("TRITON_HOME", os.path.join(_HERE, ".triton-cache"))

import json
import statistics
import time
from datetime import date
from pathlib import Path

import torch
import triton
import triton.language as tl

SPEC_PEAK_GBPS = 192.0  # RTX 3050 Ti Laptop 理论带宽
RESULT_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------- kernels
@triton.jit
def copy_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)


# ---------------------------------------------------------------- timing
def bench(fn, target_total_ms=80.0, min_iters=20):
    """
    返回 (中位数ms, 最小值ms, 迭代数)。

    两个必须处理的问题:
    1) 笔记本 GPU 有频率阶梯, 测太短时钟还没升上去 → 必须先跑够时长把 clock 拉满
    2) 迭代数要按耗时自适应, 否则小 kernel 只测到启动开销
    """
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
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), min(samples), iters


def run_copy(n_bytes):
    n = n_bytes // 4
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    grid = (triton.cdiv(n, 4096),)
    fn = lambda: copy_kernel[grid](x, y, n, BLOCK=4096)
    med, best, iters = bench(fn)
    moved = 2 * n_bytes  # 读一份 + 写一份
    return {
        "size_mb": round(n_bytes / 2**20, 2),
        "median_ms": round(med, 4),
        "best_ms": round(best, 4),
        "iters": iters,
        "gbps": round(moved / (med * 1e-3) / 1e9, 1),
    }


def run_add(n_bytes):
    n = n_bytes // 4
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, 4096),)
    fn = lambda: add_kernel[grid](x, y, out, n, BLOCK=4096)
    med, best, iters = bench(fn)
    moved = 3 * n_bytes
    return {
        "size_mb": round(n_bytes / 2**20, 2),
        "median_ms": round(med, 4),
        "iters": iters,
        "gbps": round(moved / (med * 1e-3) / 1e9, 1),
        "gflops": round(n / (med * 1e-3) / 1e9, 2),
    }


# ---------------------------------------------------------------- report
def main():
    p = torch.cuda.get_device_properties(0)
    sm = p.multi_processor_count
    l2 = getattr(p, "L2_cache_size", 0)

    print("=" * 62)
    print("设备画像")
    print("=" * 62)
    print(f"设备          {p.name}")
    print(f"算力等级      sm_{p.major}{p.minor}")
    print(f"显存          {p.total_memory / 2**30:.2f} GiB")
    print(f"SM 数量       {sm}")
    print(f"L2 缓存       {l2 / 2**20:.1f} MiB" if l2 else "L2 缓存       n/a")
    print(f"torch         {torch.__version__}")
    print(f"triton        {triton.__version__}")
    print(f"bf16 支持     {torch.cuda.is_bf16_supported()}")
    print(f"Triton 缓存   {os.environ['TRITON_CACHE_DIR']}")

    # 平衡点: 每字节需要多少 FLOP 才不浪费带宽
    # 用 FP32 CUDA core 峰值估, 20 SM * 128 core * 2 * ~1.6GHz ~ 8.2 TFLOPS
    fp32_tflops = sm * 128 * 2 * 1.6 / 1000
    ridge = fp32_tflops * 1e12 / (SPEC_PEAK_GBPS * 1e9)
    print(f"\nFP32 峰值(估) {fp32_tflops:.1f} TFLOPS")
    print(f"平衡点        {ridge:.1f} FLOP/Byte  (A100 约 12.5)")
    print("              → 算术强度低于这个值的 kernel, 一律访存受限")

    print("\n" + "=" * 62)
    print("实测带宽 (copy: 读 1 写 1)")
    print("=" * 62)
    print(f"{'尺寸(MB)':>10} {'中位数(ms)':>12} {'带宽(GB/s)':>12} {'占理论%':>9}  迭代")
    rows = []
    for mb in (1, 2, 4, 16, 64, 256):
        n_bytes = mb * 2**20
        r = run_copy(n_bytes)
        r["l2_resident"] = 2 * n_bytes <= l2
        r["peak_pct"] = round(r["gbps"] / SPEC_PEAK_GBPS * 100, 1)
        rows.append(r)
        tag = "   ← L2 命中, 不作数" if r["l2_resident"] else ""
        print(
            f"{r['size_mb']:>10.2f} {r['median_ms']:>12.4f}"
            f" {r['gbps']:>12.1f} {r['peak_pct']:>8.1f}%  {r['iters']:<5d}{tag}"
        )

    print("\n" + "=" * 62)
    print("vector add (读 2 写 1)")
    print("=" * 62)
    add_rows = []
    for mb in (16, 64, 256):
        r = run_add(mb * 2**20)
        add_rows.append(r)
        print(
            f"{r['size_mb']:>8.0f} MB  {r['median_ms']:.4f} ms"
            f"  {r['gbps']:>7.1f} GB/s  {r['gflops']:>5.1f} GFLOP/s"
        )

    big = max([r for r in rows if not r["l2_resident"]] or rows,
              key=lambda r: r["size_mb"])["gbps"]
    print("\n" + "-" * 62)
    print(f"大尺寸持续带宽   {big:.1f} GB/s  ({big / SPEC_PEAK_GBPS * 100:.0f}% of 192)")
    print(f"第一周验收线     165 GB/s")
    print(f"当前是否达标     {'是' if big >= 165 else '否 —— 先调 block size / num_warps'}")
    print("-" * 62)

    RESULT_DIR.mkdir(exist_ok=True)
    out = RESULT_DIR / f"{date.today()}.json"
    out.write_text(json.dumps({
        "device": {
            "name": p.name,
            "cc": f"{p.major}.{p.minor}",
            "vram_gib": round(p.total_memory / 2**30, 2),
            "sm": sm,
            "l2_mib": round(l2 / 2**20, 1) if l2 else None,
        },
        "versions": {"torch": torch.__version__, "triton": triton.__version__},
        "copy": rows,
        "add": add_rows,
        "ridge_flop_per_byte": round(ridge, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已存 {out}")


if __name__ == "__main__":
    main()
