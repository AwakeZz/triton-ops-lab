"""
05  行前缀和      out[i, j] = sum(x[i, :j+1])

x  : (4096, 1024) float32
out: 同上
理想访存: 读 1 份 + 写 1 份 = 2 * numel * 4 字节

这是第一次遇到跨 block 依赖。N=1024 一行装得进一个 block,
所以可以先在一行内做 tl.cumsum, 不需要 block 间扫描。
先做这个简单版, 再想把 N 放大到 8192 该怎么办。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def row_cumsum_kernel(x_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    cows = tl.program_id(0)
    offs = cows * N + tl.arange(0,BLOCK_N)
    mask = tl.arange(0,BLOCK_N) < N
    v = tl.load(x_ptr+offs,mask)
    tl.store(out_ptr+offs,tl.cumsum(v,0),mask=mask)

def run(x, out):
    n= x.numel()
    grid = (triton.cdiv(n,x.shape[1]),)
    row_cumsum_kernel[grid](x,out,x.shape[1],triton.next_power_of_2(x.shape[1]))
