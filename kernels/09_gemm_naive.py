"""
09  GEMM 朴素版         out = A @ B

A  : (1024, 1024) float16
B  : (1024, 1024) float16
out: (1024, 1024) float32

第一次出现 fp16。原因只有一个: tl.dot 只有喂 fp16/bf16 才走 tensor core 的
mma 指令, fp32 进去只能走 FMA。累加器则必须是 float32, K=1024 用 fp16 累加
误差会到 1e-1 量级。

这一版故意不优化, 一个 program 算一个输出元素:

    out[row, col] = sum_k A[row, k] * B[k, col]
                  = A 的第 row 行  ·  B 的第 col 列

grid = (M, N) = 1048576 个 program。每个 program 沿 k 串行循环 1024 次,
每轮两个标量 load + 一次 fma。A 的行是连续的(K 方向步长 1), B 的列不是
(步长 N) —— 但这一版连合并访存都没管, 就按定义老老实实地算。

三个下标各自的作用:
    a_ptr + row * K + k     A 是 (M, K), 行长 K, 所以行步长是 K
    b_ptr + k * N + col     B 是 (K, N), 行长 N, 所以行步长是 N
    out_ptr + row * N + col out 是 (M, N), 行长 N

实测 (RTX 3050 Ti, 1024³):
    单次 470 ms, 0.005 TFLOPS —— 比第 10 题的 tile 版慢 560 倍, 比 cuBLAS 慢 3000 倍。

它存在的意义只有一个: 给后面三版当分母。跑一次就知道为什么 GEMM 必须分块:
    总乘加 = 1024³ = 10.7 亿次, 但每个 program 只算其中 1024 次乘加,
    拿到的数据一点都复用不了 —— A 的一行被读了 N 次, B 的一列被读了 M 次。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gemm_naive_kernel(a_ptr, b_ptr, out_ptr, N, K):
    row = tl.program_id(0)
    col = tl.program_id(1)

    acc = 0.0
    for k in range(0, K):
        av = tl.load(a_ptr + row * K + k).to(tl.float32)
        bv = tl.load(b_ptr + k * N + col).to(tl.float32)
        acc += av * bv

    tl.store(out_ptr + row * N + col, acc)


def run(a, b, out):
    M, K = a.shape
    N = b.shape[1]
    gemm_naive_kernel[(M, N)](a, b, out, N, K)
