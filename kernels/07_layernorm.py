"""
07  layernorm     out = (x - mean) / sqrt(var + 1e-5)      沿 dim=1

x  : (4096, 1024) float32
out: 同上
理想访存: 读 1 份 + 写 1 份 = 2 * numel * 4 字节

关键: mean 和 var 必须一趟同时算出来。
      分两趟算(先 mean 再 var) = 读 3 趟, 带宽直接掉 3 倍。
      一趟的方法: 同时累计 sum 和 sum of squares, 或者 Welford。

注意: 用 biased var (除以 N, 不是 N-1), 和 torch.nn.LayerNorm 一致。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(x_ptr, out_ptr, N, eps, BLOCK_N: tl.constexpr):
    cows=tl.program_id(0)
    offs = cows * N + tl.arange(0,BLOCK_N)
    mask = tl.arange(0,BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask=mask)
    m = tl.sum(v, 0) / N
    d = tl.where(mask, v - m, 0.0)      # 越界槽的 (0-m)² 在这里被抹成 0
    var = tl.sum(d * d, 0) / N
    tl.store(out_ptr + offs, (v-m) / (tl.sqrt(var+eps)),mask=mask)


def run(x, out):
    grid = (x.shape[0],)
    eps = 1e-5
    layernorm_kernel[grid](x, out, x.shape[1], eps, triton.next_power_of_2(x.shape[1]))
