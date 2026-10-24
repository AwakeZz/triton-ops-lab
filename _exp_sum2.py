"""
实验二: BLOCK_N == N 什么时候会失效。

上一步发现 (4096,1024) 上 16 种配置全是 180.2 GB/s, 与 BLOCK_N 无关。
那 BLOCK_N 到底在什么情况下才重要? 测两个边界:
  A. N 很小 (N < 32): 一行的字节数不够一次访存事务, 必须多行合并
  B. N 很大 (N >= 8192): 一行装不进寄存器, 必须分段

固定总数据量 16 MB, 只改 N。
"""
import os

os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")

import statistics
import time

import torch
import triton
import triton.language as tl

PEAK_GBPS = 180.5
TOTAL_MB = 16


def bench(fn, target_total_ms=60.0, min_iters=20):
    fn()
    ramp_until = time.perf_counter() + 0.7
    while time.perf_counter() < ramp_until:
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5):
        fn()
    e.record()
    torch.cuda.synchronize()
    est = max(s.elapsed_time(e) / 5, 1e-4)
    iters = min(max(min_iters, int(target_total_ms / est)), 5000)
    samples = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        samples.append(a.elapsed_time(b))
    return statistics.median(samples)


@triton.jit
def row_full(x_ptr, out_ptr, n, BLOCK_N: tl.constexpr):
    """一行一 program, BLOCK_N >= n"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n
    v = tl.load(x_ptr + row * n + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row, tl.sum(v, axis=0))


@triton.jit
def row_seg(x_ptr, out_ptr, n, BLOCK_N: tl.constexpr):
    """一行一 program, 分 ceil(n/BLOCK_N) 段累加"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    base = row * n
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for s in range(0, n, BLOCK_N):
        m = (s + cols) < n
        acc += tl.load(x_ptr + base + s + cols, mask=m, other=0.0)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))


@triton.jit
def rows_block(x_ptr, out_ptr, m, n, ROWS: tl.constexpr, BLOCK_N: tl.constexpr):
    """一个 program 处理 ROWS 行"""
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_N)
    rmask = rows < m
    cmask = cols < n
    ptrs = x_ptr + rows[:, None] * n + cols[None, :]
    v = tl.load(ptrs, mask=rmask[:, None] & cmask[None, :], other=0.0)
    tl.store(out_ptr + rows, tl.sum(v, axis=1), mask=rmask)


def run_case(N):
    M = (TOTAL_MB * 2**20 // 4) // N
    x = torch.randn(M, N, device="cuda")
    out = torch.empty(M, device="cuda")
    ref = x.sum(dim=1)
    nbytes = M * N * 4 + M * 4

    def row(label, launch):
        try:
            launch()
        except Exception as exc:
            print(f"  {label:<30}{'编译失败':>10}  {type(exc).__name__}")
            return
        if not torch.allclose(out, ref, rtol=1e-3, atol=1e-3):
            print(f"  {label:<30}{'结果错':>10}")
            return
        ms = bench(launch)
        g = nbytes / (ms * 1e-3) / 1e9
        flag = "" if g / PEAK_GBPS > 0.9 else "  <<<"
        print(f"  {label:<30}{ms * 1000:>7.1f} us{g:>8.1f}{g / PEAK_GBPS * 100:>7.1f}%{flag}")

    print(f"\nN={N}, M={M}, 每行 {N * 4} 字节, 总 {nbytes / 2**20:.1f} MB")
    print(f"  {'配置':<30}{'耗时':>10}{'GB/s':>8}{'占峰值':>8}")

    BN = triton.next_power_of_2(N)
    row(f"一行/prog, BLOCK_N={BN}", lambda: row_full[(M,)](x, out, N, BLOCK_N=BN))

    if BN > 32:
        half = BN // 4
        row(f"一行/prog, 分4段 BLOCK_N={half}",
            lambda: row_seg[(M,)](x, out, N, BLOCK_N=half))

    # 多行合并: 让每个 program 搬约 4KB
    target_bytes = 4096
    for r in (2, 4, 8, 16, 32, 64):
        if r * BN * 4 > 64 * 1024 or r * BN * 4 < 256:
            continue
        if r * N > M:
            continue
        grid = (triton.cdiv(M, r),)
        row(f"{r} 行/prog, 共 {r * BN * 4 // 1024}KB",
            lambda r=r, grid=grid: rows_block[grid](
                x, out, M, N, ROWS=r, BLOCK_N=BN, num_warps=4))


def main():
    torch.manual_seed(0)
    print("=" * 66)
    print(f"固定 16 MB 数据, 改 N: BLOCK_N == N 何时失效")
    print("=" * 66)
    for N in (8, 16, 32, 128, 1024, 4096, 8192, 32768, 131072):
        run_case(N)
    print(f"\n<<< 标记 = 低于峰值 90%")


if __name__ == "__main__":
    main()
