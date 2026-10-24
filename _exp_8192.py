"""
N=8192 到底要不要单独写一个版本?
   A: 一个 program 一行, BLOCK_N = 8192(整行一次装下)
   B: 一个 program 一行, BLOCK_N = 1024, 块内串行 8 段

跑: G:/venvs/triton/Scripts/python.exe _exp_8192.py
"""
import os

os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")
import statistics
import time

import torch
import triton
import triton.language as tl

PEAK = 180.5          # 本机实测 copy 峰值 GB/s


# ---------- A: 整行一个 block ----------
@triton.jit
def cumsum_whole(x_ptr, o_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK_N)
    m = tl.arange(0, BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + offs, tl.cumsum(v, 0), mask=m)


# ---------- B: 分块 + 块间串行 carry ----------
@triton.jit
def cumsum_chunked(x_ptr, o_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * N
    nchunk = tl.cdiv(N, BLOCK_N)
    carry = 0.0
    for c in range(0, nchunk):
        cols = c * BLOCK_N + tl.arange(0, BLOCK_N)
        m = cols < N
        v = tl.load(x_ptr + base + cols, mask=m, other=0.0)
        tl.store(o_ptr + base + cols, tl.cumsum(v, 0) + carry, mask=m)
        carry += tl.sum(v, 0)


def bench(fn, warm=0.7, target=0.08, minimum=10):
    fn()
    torch.cuda.synchronize()
    ramp = time.perf_counter() + warm
    while time.perf_counter() < ramp:
        fn()
    torch.cuda.synchronize()
    samples, spent = [], 0.0
    while len(samples) < minimum or spent < target:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
        spent += samples[-1] / 1000
    return statistics.median(samples)


def main():
    M, N = 512, 8192
    x = torch.randn(M, N, device="cuda")
    ref = x.cumsum(dim=1)
    nbytes = x.numel() * 4 * 2

    print(f"x = ({M}, {N})  {x.numel() * 4 / 2**20:.0f} MB  ×2(读+写) = {nbytes / 2**20:.0f} MB\n")
    print(f"{'方案':<34}{'正确':>6}{'ms':>9}{'GB/s':>9}{'占峰值':>8}")
    print("-" * 66)

    # A 组: 不同 num_warps
    for w in (4, 8, 16):
        out = torch.empty_like(x)
        try:
            ms = bench(lambda: cumsum_whole[(M,)](x, out, N, BLOCK_N=8192, num_warps=w))
            ok = torch.allclose(out, ref, rtol=1e-4, atol=1e-3)
            gbps = nbytes / (ms * 1e-3) / 1e9
            print(f"{f'A 整行一块 BLOCK=8192 warps={w}':<34}{str(ok):>6}"
                  f"{ms:>9.4f}{gbps:>9.1f}{gbps / PEAK * 100:>7.1f}%")
        except Exception as exc:
            print(f"{f'A warps={w}':<34}  失败 {type(exc).__name__}: {str(exc).splitlines()[0][:40]}")

    # B 组: 分块
    for w in (2, 4):
        out = torch.empty_like(x)
        ms = bench(lambda: cumsum_chunked[(M,)](x, out, N, BLOCK_N=1024, num_warps=w))
        ok = torch.allclose(out, ref, rtol=1e-4, atol=1e-3)
        gbps = nbytes / (ms * 1e-3) / 1e9
        print(f"{f'B 分块 BLOCK=1024(串行8段) warps={w}':<34}{str(ok):>6}"
              f"{ms:>9.4f}{gbps:>9.1f}{gbps / PEAK * 100:>7.1f}%")

    # 参考
    ms = bench(lambda: torch.cumsum(x, dim=1))
    gbps = nbytes / (ms * 1e-3) / 1e9
    print(f"{'torch.cumsum(dim=1)':<34}{str(torch.allclose(torch.cumsum(x, 1), ref)):>6}"
          f"{ms:>9.4f}{gbps:>9.1f}{gbps / PEAK * 100:>7.1f}%")

    # 寄存器
    print("\n编译产物:")
    for name, k, cfg in [
        ("A BLOCK=8192 warps=8 ", cumsum_whole, dict(BLOCK_N=8192)),
        ("B BLOCK=1024 warps=4 ", cumsum_chunked, dict(BLOCK_N=1024)),
    ]:
        kk = k.warmup(x, torch.empty_like(x), N, **cfg, grid=(1,))
        kk._init_handles()
        print(f"  {name}: 每线程 {kk.n_regs} 寄存器, spill {kk.n_spills} 个, "
              f"shared {kk.metadata.shared} 字节")


if __name__ == "__main__":
    main()
