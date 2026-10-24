"""v 的维数：编译期 static_print + TTIR 类型双重确认"""
import torch
import triton
import triton.language as tl


@triton.jit
def probe_kernel(x_ptr, out_ptr, M, N,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    tl.static_print("---- 编译期形状 ----")
    tl.static_print("offs_m.shape          ", offs_m.shape)
    tl.static_print("offs_n.shape          ", offs_n.shape)

    row_part = offs_m[:, None] * N          # (BM,1)
    col_part = offs_n[None, :]              # (1,BN)
    tl.static_print("offs_m[:,None]*N      ", row_part.shape)
    tl.static_print("offs_n[None,:]        ", col_part.shape)

    addr = row_part + col_part
    tl.static_print("addr = row + col      ", addr.shape)
    tl.static_print("len(addr.shape)       ", len(addr.shape))

    v = tl.load(x_ptr + addr)
    tl.static_print("v = load(x_ptr+addr)  ", v.shape)
    tl.static_print("len(v.shape)          ", len(v.shape))

    s0 = tl.sum(v, 0)
    s1 = tl.sum(v, 1)
    tl.static_print("tl.sum(v, 0)          ", s0.shape)
    tl.static_print("tl.sum(v, 1)          ", s1.shape)

    if BLOCK_M == BLOCK_N:
        vt = tl.trans(v)
        tl.static_print("tl.trans(v)           ", vt.shape)

    tl.store(out_ptr + addr, v)


M, N = 64, 128
x = torch.randn(M, N, device="cuda")
o = torch.zeros_like(x)

print("\n========== 第一次编译（BLOCK_M=BLOCK_N=32）==========")
probe_kernel[(M // 32,)](x, o, M, N, 32, 32)

print("\n========== 换成非方形 tile（BLOCK_M=16, BLOCK_N=64）==========")
probe_kernel[(M // 16,)](x, o, M, N, 16, 64)

print("\n数值正确性:", torch.allclose(o, x))

# 1D 反例：同样写法但去掉两个 None
@triton.jit
def flat_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.static_print("1D 版 v.shape        ", v.shape)
    tl.static_print("1D 版 len(v.shape)   ", len(v.shape))
    tl.store(out_ptr + offs, v)

print("\n========== 对照组：不加 None 时 ==========")
flat_kernel[(1024,)](x.reshape(-1), o.reshape(-1), 1024)
