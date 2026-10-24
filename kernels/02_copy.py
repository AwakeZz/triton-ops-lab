"""
02  copy          out = x

x  : (4096, 1024) float32
out: 同上
理想访存: 读 1 份 + 写 1 份 = 2 * numel * 4 字节

这题是你的尺子。后面每个 kernel 都问一句: 它比 copy 慢几倍?
慢 2 倍就是多走了一趟访存, 慢 4 倍就是三趟。

这一份实现是给你的(尺子), 03~08 自己写。
跑: check.py --only 02_copy        扫参: tune_copy.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # 1) 我是第几个 program
    pid = tl.program_id(0)
    # 2) 我负责哪一段: pid*BLOCK 是起点, arange 生成段内偏移向量
    offs = pid * BLOCK + tl.arange(0,BLOCK)

    # 3) 越界的 lane 屏蔽掉(最后一段通常不满)
    mask = offs < n

    # 4) 一次 load 拿整个 BLOCK, 一次 store 写回去
    v = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, v, mask=mask)


def run(x, out):
    n = x.numel()
    # grid 必须盖满整个数组。用 cdiv 向上取整, 少一个 program 尾巴就漏了
    BLOCK=4096
    grid = (triton.cdiv(n, BLOCK),)
    copy_kernel[grid](x, out, n, BLOCK)
