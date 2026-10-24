"""
06  行 softmax     out[i, :] = softmax(x[i, :])

x  : (4096, 1024) float32
out: 同上
理想访存: 读 1 份 + 写 1 份 = 2 * numel * 4 字节

这是本组最重要的一题。

朴素写法: 先读一遍求 max, 再读一遍求 sum(exp), 再读一遍写结果 + 写
          → 读 3 趟写 1 趟
online 写法: 一遍扫过同时维护 running max 和 running sum, 最后再读一遍写
          → 读 2 趟写 1 趟
一行装进一个 block 时: 只读一遍到寄存器, 算完写回
          → 读 1 趟写 1 趟 (理想)

启动次数会被框架统计, 直接暴露你写的是哪一种。
"""


import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0)
    offs = rows * N + tl.arange(0,BLOCK_N)
    mask = tl.arange(0,BLOCK_N) < N
    v = tl.load(x_ptr + offs ,mask,other=float("-inf"))
    m = tl.max(v,0)
    s = tl.sum(tl.exp(v-m),0)
    s2 = tl.exp(v-m)
    tl.store(out_ptr+offs,s2/s,mask=mask)



def run(x, out):
    n= x.numel()
    grid = (triton.cdiv(n,x.shape[1]),)
    softmax_kernel[grid](x,out,x.shape[1],triton.next_power_of_2(x.shape[1]))
