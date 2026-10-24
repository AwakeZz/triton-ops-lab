"""
04  行最大值      out[i] = max(x[i, :])

x  : (4096, 1024) float32
out: (4096,)      float32
理想访存: numel * 4 + 4096 * 4 字节

和 03 结构完全一样, 只换成 tl.max。
写完后对比两者的 GB/s —— 如果 max 明显慢, 想清楚为什么(值得想)。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def row_max_kernel(x_ptr, out_ptr, M, N, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0)
    offs = rows * N + tl.arange(0,BLOCK_N)
    mask = tl.arange(0,BLOCK_N) < N
    v = tl.load(x_ptr + offs,mask,other=float("-inf"))
    tl.store( out_ptr + rows,tl.max(v,0))



def run(x, out):
    n =x.numel()
    grid = (triton.cdiv(n,x.shape[1]),)
    row_max_kernel[grid](x,out,x.shape[0],x.shape[1],triton.next_power_of_2(x.shape[1]))
