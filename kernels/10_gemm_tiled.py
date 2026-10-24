"""
10  GEMM 分块版 (不用 tl.dot)        out = A @ B

A  : (1024, 1024) float16
B  : (1024, 1024) float16
out: (1024, 1024) float32
契约: run(a, b, out)

比 09 只多一件事: **一个 program 算一块, 不是算一个元素**。

为什么这样能快 500 倍 —— 算一下访存账:
    09:  每个输出元素读 K 个 A + K 个 B  ->  总共 M*N*K*2 次元素读
    10:  每个 BMxBN 的块, 读 K/ BK 次 A 块和 B 块, 每块被 BM(或 BN) 个元素共用
         同一份数据被复用 BM 次, 访存降到 1/BM 量级
    GEMM 的算术强度 = 2*BM*BN*BK / (4*(BM+BN)*BK) ≈ BM*BN/(2*(BM+BN)),
    BM=BN=64 时是 16 FLOP/Byte —— 还不够 45.4 的平衡点, 所以这题仍然受带宽约束,
    这也是为什么第 11 题换成 tensor core 之后还能再快 5 倍。

结构 (三个下标 + 一个循环):

    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)      # 这一块覆盖哪些行
    offs_n = pid_n * BN + tl.arange(0, BN)      # 这一块覆盖哪些列

    acc = tl.zeros((BM, BN), tl.float32)        # 累加器, 必须是 float32
    for k0 in range(0, tl.cdiv(K, BK)):         # K 方向分块
        offs_k = k0 * BK + tl.arange(0, BK)
        a = load  A[offs_m, offs_k]             # (BM, BK)
        b = load  B[offs_k, offs_n]             # (BK, BN)
        acc += 外层积求和(a, b)                  # (BM, BK) x (BK, BN) -> (BM, BN)

    最后 store out[offs_m, offs_n] = acc        # 只写一次, 不是每轮都写

"外层积求和"怎么写:
    a[:, :, None]  -> (BM, BK, 1)
    b[None, :, :]  -> (1, BK, BN)
    两者广播成 (BM, BK, BN), 再 tl.sum(..., axis=1) 消掉 k 那一维 -> (BM, BN)
    注意 axis=1 消的是 BK —— 中间那一维, 不是最后那维。

四个 mask 都不能省 (--odd 会抓):
    load A: (offs_m < M) & (offs_k < K)
    load B: (offs_k < K) & (offs_n < N)
    store : (offs_m < M) & (offs_n < N)

建议 BM=BN=32, BK=32, grid=(triton.cdiv(M,BM), triton.cdiv(N,BN))。
先把 32/32/32 跑通, 再试 64/64/32 看能到多少。

验收标准:
    通过  -> 正确性 + 启动 1 次
    目标  -> >= 12% 峰值 (约 2.2 TFLOPS)。实测参考: 32/32/32 是 2.80 TFLOPS (15.4%),
             64/64/32 是 4.66 TFLOPS (25.6%)。
    注: 那三个维度 (BM,BK,BN) 会一起进寄存器, 64/64/32 就是 131072 个 float,
        寄存器根本装不下, 所以这一版无论怎么调都到不了 tensor core 的水平。
        —— 这就是第 11 题的理由。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gemm_tiled_kernel(a_ptr, b_ptr, out_ptr, M, N, K,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr):
    raise NotImplementedError("写 10_gemm_tiled 的 kernel")


def run(a, b, out):
    raise NotImplementedError("写完 gemm_tiled_kernel 再补 run")
