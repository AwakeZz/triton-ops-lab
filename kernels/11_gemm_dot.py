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

三个硬约束 (前两个的坑法不一样, 只有第 2 个会报错):

    1. 两个输入都必须是 fp16 或 bf16 —— 别写 .to(tl.float32)。
       这一条**不报错**, 是静默降级, 所以最坑:
           fp16       -> mma.sync.aligned.m16n8k16.row   一次 K=16
           .to(f32)   -> mma.sync.aligned.m16n8k8.row    TF32, 一次 K=8, 吞吐腰斩
       实测同一个 kernel 只差这一处: 14.36 -> 7.85 TFLOPS。
       (注意它并没有"退回 FMA", 而是换成了 TF32 的 tensor core, 更难发现。)

    2. BLOCK_M / BLOCK_N / BLOCK_K 都要 >= 16, 不满足编译期直接报错。

    3. 累加写成 acc = tl.dot(a, b, acc) 而不是 acc += tl.dot(a, b) ——
       把 acc 作为参数传进去, 编译器才让累加器一直留在 tensor core 里,
       否则每轮都要把 acc 落回寄存器和内存。

精度这件事和第 10 题正好相反, 别把教训带过来:
    第 10 题手写 FMA 必须在乘法前 .to(f32), 因为 fp16 乘积会舍入 (误差 3.5e-2)。
    这里 mma 内部 fp16*fp16 的乘积是精确的、累加在 fp32, 不需要转。
    实测误差 4.7e-4, 比第 10 题 FMA 版的 2.1e-4 略大一点点 (累加顺序不同),
    但已经远小于 tol, 而且比"转 f32 走 TF32"的 6.7e-4 更准。

建议 BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4。
为什么 64x64 比 32x32 好, 而且比 128x128 好 (在 1024³ 这个尺寸下):
    64x64 -> grid (16,16) = 256 个 block。20 个 SM, 每个 SM 能挂 2~4 个块,
             正好几个波次铺满, 尾部空转少。
    32x32 -> 1024 个 block, mma 的 16x8 粒度下每个 warp 分到的活太少, 调度开销占主导。
    128x128 -> 只有 64 个 block, 20 个 SM 分不到 3 个块, 尾部空一大截。

验收标准:
    通过  -> 正确性 + 启动 1 次
    目标  -> >= 70% 峰值 (约 12.7 TFLOPS)。实测参考 (1024^3, w4 s1):
             32/32/32 带 .to(f32) =  7.85 TFLOPS (43.1%)  <- 老板第一版踩的就是这个
             32/32/32 去掉 .to(f32) = 11.54 TFLOPS (63.4%)
             64/64/32 去掉 .to(f32) = 14.36 TFLOPS (78.9%)
             64/64/32 用本框架 check.py 测 = 15.66 TFLOPS (86.1%), 超 cuBLAS 9%
    注: 上面前三个数是用逐次 synchronize 的探针量的, 比 check.py 的口径低 8~9%
        (探针把每次 launch 开销都算进去了)。横向比配置用探针, 报成绩用 check.py。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gemm_dot_kernel(a_ptr, b_ptr, out_ptr, M, N, K,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0,BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0,BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):         # K 方向分块
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        # 直接喂 fp16: mma 内部 fp16*fp16 的乘积是精确的, 累加在 fp32 里做,
        # 不存在第 10 题那种"乘积已经在小精度里舍入"的问题 —— 那段教训不要带过来。
        # 一旦 .to(tl.float32), 指令会从 m16n8k16 掉到 m16n8k8 (TF32), 吞吐直接腰斩。
        acc = tl.dot(a, b, acc)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(a, b, out):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"shape mismatch: a{K} vs b{K2}"
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_dot_kernel[grid](a, b, out, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                          num_warps=4, num_stages=1)
