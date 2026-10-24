"""08_transpose 对照实验：朴素散写 vs 2D tile + tl.trans vs 2D tile 不 trans"""
import torch
import triton
import triton.language as tl

M, N = 4096, 1024


# ---------- V0 朴素：1D 线性读，用 div/mod 算转置后的地址（写完全不合并） ----------
@triton.jit
def tp_scatter(x_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    v = tl.load(x_ptr + offs, mask=mask)
    i = offs // N
    j = offs % N
    tl.store(out_ptr + j * M + i, v, mask=mask)


# ---------- V1 正确版：2D tile，读合并，tl.trans 后写也合并 ----------
@triton.jit
def tp_trans(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m2 = offs_m[:, None] < M
    n2 = offs_n[None, :] < N
    v = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=m2 & n2)
    vt = tl.trans(v)
    tl.store(out_ptr + offs_n[:, None] * M + offs_m[None, :], vt,
             mask=(offs_n[:, None] < N) & (offs_m[None, :] < M))


# ---------- V2 错误示范：tile 但漏了 trans，值与地址 shape 对齐但写不合并 ----------
@triton.jit
def tp_notrans(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m2 = offs_m[:, None] < M
    n2 = offs_n[None, :] < N
    v = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=m2 & n2)
    tl.store(out_ptr + offs_m[:, None] + offs_n[None, :] * M, v, mask=m2 & n2)


def bench(call, n=100):
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        call()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


x = torch.randn(M, N, device="cuda")
ref = x.t().contiguous()
nbytes = 2 * x.numel() * 4

print("M,N = %d,%d  理想访存 %.1f MB  (峰值 180.5 GB/s)" % (M, N, nbytes / 1e6))
print("%-26s | %8s | %10s | %8s" % ("实现", "GB/s", "ms", "正确"))
print("-" * 62)

out = torch.empty(N, M, device="cuda")
for BLK in (1024, 4096):
    call = lambda: tp_scatter[(triton.cdiv(M * N, BLK),)](x, out, M, N, BLK)
    call()
    ok = torch.equal(out, ref)
    ms = bench(call)
    print("%-26s | %8.1f | %10.4f | %8s" % ("V0 朴素散写 BLOCK=%d" % BLK,
                                           nbytes / ms / 1e6, ms, ok))

for bm, bn in [(16, 16), (32, 32), (64, 64), (32, 64), (64, 32), (128, 64)]:
    call = lambda: tp_trans[(triton.cdiv(M, bm), triton.cdiv(N, bn))](x, out, M, N, bm, bn)
    out.zero_()
    call()
    ok = torch.equal(out, ref)
    ms = bench(call)
    print("%-26s | %8.1f | %10.4f | %8s" % ("V1 trans %dx%d" % (bm, bn),
                                           nbytes / ms / 1e6, ms, ok))

for bm, bn in [(32, 32), (64, 64)]:
    call = lambda: tp_notrans[(triton.cdiv(M, bm), triton.cdiv(N, bn))](x, out, M, N, bm, bn)
    out.zero_()
    call()
    ok = torch.equal(out, ref)
    ms = bench(call)
    print("%-26s | %8.1f | %10.4f | %8s" % ("V2 无trans %dx%d" % (bm, bn),
                                           nbytes / ms / 1e6, ms, ok))

call = lambda: ref.copy_(x.t())
torch.cuda.synchronize()
out2 = torch.empty(N, M, device="cuda")
ms = bench(lambda: out2.copy_(x.t()))
print("%-26s | %8.1f | %10.4f | %8s" % ("torch x.t().copy_", nbytes / ms / 1e6, ms, True))
