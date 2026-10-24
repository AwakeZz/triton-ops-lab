"""
实验: 行求和里 BLOCK_N 到底该取多大, 以及"一 program 一行"够不够。

问题: BLOCK_N == N 是不是最优?
答案分两个维度:
  (1) 不浪费 -> BLOCK_N 取到 >= N 的 2 的幂时, 没有无效线程
  (2) 够不够喂饱显存 -> 取决于每个 program 搬多少字节, 与 BLOCK_N 无关

用法:
    G:/venvs/triton/Scripts/python.exe _exp_sum.py
"""
import os

os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")

import statistics
import time

import torch
import triton
import triton.language as tl

PEAK_GBPS = 180.5
M, N = 4096, 1024


def bench(fn, target_total_ms=80.0, min_iters=20):
    """先跑 0.8s 拉满时钟; 第一次调用(含 JIT 编译)必须排除在外。"""
    fn()                     # <- 单独一次, 吃掉编译时间
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


# --- 变体 1: 一行一 program, 一次性装下整行 (BLOCK_N >= N) -------------------
@triton.jit
def row_full(x_ptr, out_ptr, n, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n
    v = tl.load(x_ptr + row * n + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row, tl.sum(v, axis=0))


# --- 变体 2: 一行一 program, 分段累加 (BLOCK_N < N) -------------------------
@triton.jit
def row_seg(x_ptr, out_ptr, n, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    base = row * n
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for s in range(0, n, BLOCK_N):
        m = (s + cols) < n
        acc += tl.load(x_ptr + base + s + cols, mask=m, other=0.0)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))


# --- 变体 3: 一个 program 处理 ROWS 行 (二维 block) -------------------------
@triton.jit
def rows_block(x_ptr, out_ptr, m, n, ROWS: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_N)
    rmask = rows < m
    cmask = cols < n
    ptrs = x_ptr + rows[:, None] * n + cols[None, :]
    v = tl.load(ptrs, mask=rmask[:, None] & cmask[None, :], other=0.0)
    s = tl.sum(v, axis=1)
    tl.store(out_ptr + rows, s, mask=rmask)


def main():
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda")
    out = torch.empty(M, device="cuda")
    ref = x.sum(dim=1)
    nbytes = M * N * 4 + M * 4

    def gbps(ms):
        return nbytes / (ms * 1e-3) / 1e9

    print("=" * 76)
    print(f"行求和  x=({M},{N})  共 {nbytes / 2**20:.2f} MB, 峰值 {PEAK_GBPS} GB/s")
    print("=" * 76)

    ms, _ = bench(lambda: torch.sum(x, dim=1, out=out))
    print(f"{'torch.sum(dim=1)':<34}{ms * 1000:>9.1f} us{gbps(ms):>9.1f}{gbps(ms) / PEAK_GBPS * 100:>7.1f}%")

    # 变体 1: BLOCK_N 从 N 一路加到 4N, 看"浪费线程"的代价
    print("\n[变体 1] 一行一 program, 一次性装下整行")
    print(f"{'配置':<34}{'耗时':>9}{'GB/s':>9}{'占峰值':>8}")
    print("-" * 76)
    for bn in (N, 2 * N, 4 * N):
        for w in (4, 8):
            def launch(bn=bn, w=w):
                row_full[(M,)](x, out, N, BLOCK_N=bn, num_warps=w)
            launch()
            if not torch.allclose(out, ref, rtol=1e-4, atol=1e-4):
                print(f"BLOCK_N={bn} warps={w} 结果错")
                continue
            ms, _ = bench(launch)
            tag = f"BLOCK_N={bn:<5} warps={w}"
            print(f"{tag:<34}{ms * 1000:>9.1f} us{gbps(ms):>9.1f}{gbps(ms) / PEAK_GBPS * 100:>7.1f}%")

    # 变体 2: BLOCK_N < N, 分段累加
    print("\n[变体 2] 一行一 program, 分成多段累加")
    print(f"{'配置':<34}{'耗时':>9}{'GB/s':>9}{'占峰值':>8}")
    print("-" * 76)
    for bn in (512, 256, 128):
        for w in (4, 8):
            def launch(bn=bn, w=w):
                row_seg[(M,)](x, out, N, BLOCK_N=bn, num_warps=w)
            launch()
            if not torch.allclose(out, ref, rtol=1e-4, atol=1e-4):
                print(f"BLOCK_N={bn} warps={w} 结果错")
                continue
            ms, _ = bench(launch)
            tag = f"BLOCK_N={bn:<5} warps={w} seg={N // bn}"
            print(f"{tag:<34}{ms * 1000:>9.1f} us{gbps(ms):>9.1f}{gbps(ms) / PEAK_GBPS * 100:>7.1f}%")

    # 变体 3: 一个 program 处理多行
    print("\n[变体 3] 一个 program 处理 ROWS 行 (BLOCK_N=N)")
    print(f"{'配置':<34}{'耗时':>9}{'GB/s':>9}{'占峰值':>8}")
    print("-" * 76)
    for rows in (1, 2, 4, 8, 16):
        for w in (4, 8):
            grid = (triton.cdiv(M, rows),)

            def launch(rows=rows, w=w, grid=grid):
                rows_block[grid](x, out, M, N, ROWS=rows, BLOCK_N=N, num_warps=w)
            try:
                launch()
            except Exception as exc:
                print(f"ROWS={rows} warps={w} 编译失败: {type(exc).__name__}")
                continue
            if not torch.allclose(out, ref, rtol=1e-4, atol=1e-4):
                print(f"ROWS={rows} warps={w} 结果错")
                continue
            ms, _ = bench(launch)
            tag = f"ROWS={rows:<3} warps={w} grid={grid[0]}"
            print(f"{tag:<34}{ms * 1000:>9.1f} us{gbps(ms):>9.1f}{gbps(ms) / PEAK_GBPS * 100:>7.1f}%")

    # --- 关键对照: 每个 program 搬的字节数 ---------------
    print("\n" + "=" * 76)
    print("每个 program 搬的字节数 (决定能不能喂饱显存的真正变量)")
    print("=" * 76)
    for rows in (1, 2, 4, 8, 16):
        kb = rows * N * 4 / 1024
        print(f"  ROWS={rows:<3} -> 每 program {kb:>6.1f} KB,  "
              f"grid {triton.cdiv(M, rows):>5} 个")
    print(f"\n   参考: copy kernel 每 program 16 KB, 只需 5-10 个并发就吃满带宽")


if __name__ == "__main__":
    main()
