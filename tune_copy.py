"""
copy kernel 调参实验。

这个文件是给你"照着改着玩"的: 改 BLOCK、改 num_warps, 看数字怎么动。
它自带一份 copy 实现, 因为要调参总得有个能跑的版本。

    python tune_copy.py

两件事会让你意外:
1. BLOCK 不是越大越好, 有个拐点
2. 去掉 mask 能白捡一点性能(前提是 N 能被 BLOCK 整除)
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))
os.environ.setdefault("TRITON_HOME", os.path.join(_HERE, ".triton-cache"))

import statistics
import time

import torch
import triton
import triton.language as tl

PEAK_GBPS = 180.5
N = 1 << 25  # 3355 万个数 = 128 MB, 远超 L2, 保证测的是显存带宽


@triton.jit
def copy_masked(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


@triton.jit
def copy_nomask(x_ptr, o_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs))


def bench(fn, target_total_ms=60.0, min_iters=10):
    # 第一次调用包含 JIT 编译(几百毫秒到几秒), 必须挡在预热外面。
    # 否则编译会把预热时间吃光, GPU 还停在待机频率上, 测出来直接假低 10 倍。
    fn()
    torch.cuda.synchronize()

    ramp = time.perf_counter() + 0.7
    while time.perf_counter() < ramp:
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
    iters = min(max(min_iters, int(target_total_ms / est)), 2000)

    samples = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        samples.append(a.elapsed_time(b))
    return statistics.median(samples)


def main():
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    o = torch.empty_like(x)
    moved = 2 * N * 4

    print(f"测试规模: {N} 个 float = {N * 4 / 2**20:.0f} MB, 搬运 {moved / 2**20:.0f} MB")
    print(f"理论峰值: {PEAK_GBPS} GB/s (本机实测值)\n")

    print("一、BLOCK 与 num_warps 扫参（带 mask）")
    head = f"{'BLOCK':>7}{'warps':>7}{'每线程元素':>11}{'ms':>10}{'GB/s':>9}{'%峰值':>8}"
    print(head)
    print("-" * len(head.encode('gbk', errors='ignore')))
    best = (0, 0, 0)
    for blk in (256, 512, 1024, 2048, 4096, 8192, 16384):
        for w in (1, 2, 4, 8):
            if blk // (w * 32) < 1:
                continue
            grid = (N // blk,)
            ms = bench(lambda: copy_masked[grid](x, o, N, BLOCK=blk, num_warps=w))
            gbps = moved / (ms * 1e-3) / 1e9
            per_thread = blk // (w * 32)
            print(f"{blk:>7}{w:>7}{per_thread:>11}{ms:>10.4f}{gbps:>9.1f}"
                  f"{gbps / PEAK_GBPS * 100:>7.1f}%")
            if gbps > best[2]:
                best = (blk, w, gbps)

    print(f"\n最优: BLOCK={best[0]}, num_warps={best[1]}, {best[2]:.1f} GB/s"
          f" ({best[2] / PEAK_GBPS * 100:.1f}%)\n")

    print("二、去掉 mask（N 能被 BLOCK 整除时才合法）")
    for blk in (1024, 2048, 4096, 8192):
        grid = (N // blk,)
        ms_m = bench(lambda: copy_masked[grid](x, o, N, BLOCK=blk, num_warps=4))
        ms_n = bench(lambda: copy_nomask[grid](x, o, BLOCK=blk, num_warps=4))
        gm = moved / (ms_m * 1e-3) / 1e9
        gn = moved / (ms_n * 1e-3) / 1e9
        print(f"BLOCK={blk:>6}  mask {gm:>7.1f} GB/s   无 mask {gn:>7.1f} GB/s"
              f"   差 {gn - gm:+.1f}")

    print("\n三、抄 PyTorch 的作业")
    ms = bench(lambda: o.copy_(x))
    gbps = moved / (ms * 1e-3) / 1e9
    print(f"o.copy_(x)  {gbps:.1f} GB/s ({gbps / PEAK_GBPS * 100:.1f}%)  "
          f"← 你至少要追到这个数的 95%")


if __name__ == "__main__":
    main()
