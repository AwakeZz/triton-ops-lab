"""
01  vector add     out = x + y

x, y:  (4096, 1024) float32  contiguous
out :  同上
理想访存: 读 2 份 + 写 1 份 = 3 * numel * 4 字节
目标   : 贴着 180 GB/s

提示: 一维 grid, BLOCK 试 1024 / 2048 / 4096, 注意尾块 mask
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0,BLOCK)
    mask = offs < n
    v1 = tl.load(x_ptr+offs,mask=mask)
    v2 = tl.load(y_ptr+offs,mask=mask)
    tl.store(out_ptr + offs, v1+v2,mask=mask)


def run(x, y, out):
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n,BLOCK),)
    add_kernel[grid](x,y,out,n,BLOCK)
