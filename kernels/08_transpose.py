"""
08  转置          out = x.T  (contiguous)

x  : (4096, 1024) float32
out: (1024, 4096) float32
理想访存: 读 1 份 + 写 1 份 = 2 * numel * 4 字节

读和写必有一边是非连续的。直接按行读按行写, 写完的那一边 stride 是 4096,
每次 store 只落 4 字节到一个 32 字节的 sector → 实际带宽掉到 1/8。

思路: 用 BLOCK_M x BLOCK_N 的二维 tile。读按 m*N+n 算地址, 写按 n*M+m 算地址,
      行距从 N 换成 M —— 同一批值落到不同位置, 这就是转置。
      性能来自 tile 化本身: 每个 program 落 128 字节一段, 段内连续才吃得满 sector。

      注: 实测 tl.trans 对带宽没有增益(32x32 加不加都是 176~179 GB/s),
          它的作用是把值的 shape 转成和写地址矩阵一致, 属语义需要。
          对照: 1D + div/mod 的朴素散写只有 65.8 GB/s。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # (BLOCK_N,)

    # (BLOCK_M, 1) + (1, BLOCK_N) → (BLOCK_M, BLOCK_N) 的地址矩阵
    v = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :],
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    # 写：(BLOCK_N, 1) * M + (1, BLOCK_M) → (BLOCK_N, BLOCK_M)
    tl.store(out_ptr + offs_n[:, None] * M + offs_m[None, :],
             tl.trans(v),
             mask=(offs_n[:, None] < N) & (offs_m[None, :] < M))


def run(x, out):
    M, N = x.shape
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](x, out, M, N, BLOCK_M, BLOCK_N)
