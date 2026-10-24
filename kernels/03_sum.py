"""
03  行求和        out[i] = sum(x[i, :])

x  : (4096, 1024) float32
out: (4096,)      float32
理想访存: numel * 4 + 4096 * 4 字节

提示: 先按最简写法做对 —— 一行一 program, BLOCK_N = next_power_of_2(N)。
      写完再问自己"它真的慢吗", 用 _exp_sum.py 的 bench 量一遍, 不要凭直觉假定要优化。

      实测(3050 Ti, 见 ops-lab/_exp_sum.py): 这个尺寸下 16 种配置(BLOCK_N 从
      N/8 到 4N、一 program 1~16 行)全部是 180.2 GB/s = 99.8% 峰值。
      也就是说这题在 (4096,1024) 上没有优化空间, 已经贴着 roofline。

      边界(见 _exp_sum2.py): 只有当 N < 32 时, 一行填不满一个 warp,
      必须一个 program 处理多行才能吃满带宽。N >= 128 之后与分块方式无关。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(x_ptr, out_ptr, M, N, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0)
    offs = rows * N + tl.arange(0,BLOCK_N)
    mask = tl.arange(0,BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + rows, tl.sum(v,0))


def run(x, out):
    grid = (x.shape[0],)
    row_sum_kernel[grid](x,out,x.shape[0],x.shape[1],triton.next_power_of_2(x.shape[1]))