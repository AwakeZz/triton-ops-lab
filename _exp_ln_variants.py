"""比较 layernorm 求 var 的几种写法：正确性 + 带宽"""
import torch, triton, triton.language as tl


# A: 用户现在（无 other，N 必须是 2 的幂）
@triton.jit
def ln_A(x_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask)
    m = tl.sum(v, 0) / N
    c = tl.sum((v - m) * (v - m), 0) / N
    tl.store(out_ptr + offs, (v - m) / tl.sqrt(c + 1e-5), mask=mask)


# B: sum + sumsq
@triton.jit
def ln_B(x_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(v, 0)
    ss = tl.sum(v * v, 0)
    m = s / N
    var = ss / N - m * m
    tl.store(out_ptr + offs, (v - m) / tl.sqrt(var + 1e-5), mask=mask)


# C: 先求 mean，再用 where 把越界位置的中心化差值抹成 0
@triton.jit
def ln_C(x_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    m = tl.sum(v, 0) / N
    d = tl.where(mask, v - m, 0.0)
    var = tl.sum(d * d, 0) / N
    tl.store(out_ptr + offs, (v - m) / tl.sqrt(var + 1e-5), mask=mask)


# D: 同 C，用乘 mask 代替 where
@triton.jit
def ln_D(x_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    m = tl.sum(v, 0) / N
    d = (v - m) * mask
    var = tl.sum(d * d, 0) / N
    tl.store(out_ptr + offs, (v - m) / tl.sqrt(var + 1e-5), mask=mask)


IMPLS = [("A 无other", ln_A), ("B sumsq", ln_B), ("C where", ln_C), ("D 乘mask", ln_D)]


def ref(x):
    return (x - x.mean(1, keepdim=True)) / torch.sqrt(
        x.var(1, unbiased=False, keepdim=True) + 1e-5)


def timeit(call, n=200):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(n):
        call()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


for R, N in [(4096, 1024), (4096, 1000), (4096, 333)]:
    x = torch.randn(R, N, device="cuda")
    r = ref(x)
    BN = triton.next_power_of_2(N)
    print("=== shape", (R, N), "BLOCK_N", BN)
    for name, fn in IMPLS:
        for nw in (4, 8):
            out = torch.empty_like(x)
            call = lambda: fn[(R,)](x, out, N, BN, num_warps=nw)
            call()
            err = (out - r).abs().max().item()
            ms = timeit(call)
            gbs = 2 * x.numel() * 4 / ms / 1e6
            k = fn.warmup(x, out, N, BN, grid=(1,), num_warps=nw)
            k._init_handles()
            print("  %-10s w%d regs=%3d spill=%2d %7.1f GB/s  err=%.2e %s"
                  % (name, nw, k.n_regs, k.n_spills, gbs, err,
                     "PASS" if err < 1e-4 else "FAIL"))
    print()
