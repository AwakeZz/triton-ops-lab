"""
11  GEMM + tl.dot        out = A @ B

A  : (1024, 1024) float16
B  : (1024, 1024) float16
out: (1024, 1024) float32
契约: run(a, b, out)

和第 10 题的差别只有一行:

    10:  acc += tl.sum(a[:, :, None] * b[None, :, :], axis=1)
    11:  acc = tl.dot(a, b, acc)

整个项目到这一行才第一次离开"标量/向量运算", 进入矩阵单元。

为什么快 5 倍:
    tl.sum 那版编译出来是一堆 ffma (每个线程一条一条地乘加)。
    tl.dot 编译成 mma.sync.aligned.m16n8k16 —— 一个 warp 的 32 个线程协作,
    一条指令完成 16x8x16 = 2048 次乘加, 而且操作数直接从寄存器矩阵喂进去。
    print 出来的 PTX 里能直接搜到 mma.sync。

三个硬约束 (不满足会在编译期报错, 报错信息指向性很强):
    1. 两个输入都必须是 fp16 或 bf16 —— 别写 .to(tl.float32), 一转就退回 FMA
    2. BLOCK_M / BLOCK_N / BLOCK_K 都要 >= 16
    3. 累加写成 acc = tl.dot(a, b, acc) 而不是 acc += tl.dot(a, b) ——
       把 acc 作为参数传进去, 编译器才让累加器一直留在 tensor core 里

建议 BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4。
为什么 64x64 比 32x32 好, 而且比 128x128 好 (在 1024³ 这个尺寸下):
    64x64 -> grid (16,16) = 256 个 block。20 个 SM, 每个 SM 能挂 2~4 个块,
             正好几个波次铺满, 尾部空转少。
    32x32 -> 1024 个 block, mma 的 16x8 粒度下每个 warp 分到的活太少, 调度开销占主导。
    128x128 -> 只有 64 个 block, 20 个 SM 分不到 3 个块, 尾部空一大截。

验收标准:
    通过  -> 正确性 + 启动 1 次
    目标  -> >= 70% 峰值 (约 12.7 TFLOPS)。实测参考:
             32/32/32 w4 s1 = 10.92 TFLOPS (60%)
             64/64/32 w4 s1 = 14.19 TFLOPS (78%)  <- 和 cuBLAS 的 14.31 打平
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gemm_dot_kernel(a_ptr, b_ptr, out_ptr, M, N, K,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
    raise NotImplementedError("把 10 的累加换成 tl.dot")


def run(a, b, out):
    raise NotImplementedError("写完 gemm_dot_kernel 再补 run")
