"""
12  GEMM 调优版        out = A @ B

A  : (1024, 1024) float16
B  : (1024, 1024) float16
out: (1024, 1024) float32
契约: run(a, b, out)

kernel 主体和第 11 题一样, 这一题调的是**外面三个旋钮**:

1. tile 尺寸 (BLOCK_M / BLOCK_N / BLOCK_K)
   tile 越大, 同一份 A/B 数据被复用的次数越多, 算术强度越高。
   但 tile 一大, grid 就小, 填不满 20 个 SM 的话反而掉。
   1024³ 下这是个真实的两难 —— 见下面实测。

2. num_warps
   64x64 用 4 warps 刚好; 128x128 必须 8 warps, 否则每个 warp 分到的
   mma 份数太多, 排队等 tensor core。

3. num_stages (软件流水深度)
   让第 k+1 块的 load 和第 k 块的 mma 重叠起来。默认 3, 试 2/3/4。
   它不是越大越好: 每一级 stage 都要占一块 shared memory,
   4 级 128x128x32 就是 4*(128*32*2 + 32*128*2) = 64 KB, 而一张 SM 只有 100 KB
   左右, 再大就直接掉占用率。

实测 (这张卡, cuBLAS 在 1024³ 是 0.1501 ms / 14.31 TFLOPS):

    BLOCK_M/N/K     warps  stages   1024³ TFLOPS   %峰
    32/32/32        4      1          10.92        60.0
    64/64/32        4      1          14.19        78.0
    64/64/32        4      3          14.98        82.3   <- 1024³ 最好
    128/128/32      8      3          13.52        74.3
    128/128/64      8      3          12.87        70.7
    128/64/64       4      4          12.42        68.2
    64/128/64       4      4          13.32        73.2

    (同样的配置, 换到 2048³ 之后 128/128/32 反而成了最好: 16.05 TFLOPS / 88.2%,
     而 64/64/32 只有 11.89 / 65.3%。原因还是 grid 大小 —— 2048³ 下
     128x128 有 256 个 block, 正好铺满; 1024³ 下只有 64 个, 填不满。
     **所以 tile 尺寸没有全局最优, 它和 M/N 绑在一起。**)

一个小实验就够看清 num_stages 的作用:
    把 s1 和 s3 的 PTX 各 dump 一份, 搜 cp.async —— s1 里没有或很少, s3 里成排出现。
    delta 就在这儿: 访存提前发出去, 和 mma 并行。

验收标准:
    通过  -> 正确性 + 启动 1 次
    目标  -> >= 80% 峰值 (约 14.6 TFLOPS)。达到 82% 就是超过 cuBLAS。
    注: 1024³ 的 100% 是 18.2 TFLOPS, 但那是 4096³ 才能贴到的渐近值;
        1024³ 只有 0.14 ms, kernel 启动和尾部空转占掉了将近两成, 谁都到不了 100%。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gemm_pipe_kernel(a_ptr, b_ptr, out_ptr, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr):
    raise NotImplementedError("kernel 主体照抄 11 题, 这题调的是启动参数")


def run(a, b, out):
    raise NotImplementedError("写完 gemm_pipe_kernel 再补 run, 记得传 num_warps / num_stages")
