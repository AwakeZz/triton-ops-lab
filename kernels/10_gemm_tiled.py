"""
10  GEMM 分块版 (不用 tl.dot)        out = A @ B

A  : (1024, 1024) float16
B  : (1024, 1024) float16
out: (1024, 1024) float32
契约: run(a, b, out)

比 09 只多一件事: **一个 program 算一块, 不是算一个元素**。

为什么这样能快 500 倍 —— 算一下访存账:
    09:  每个输出元素读 K 个 A + K 个 B  ->  总共 M*N*K*2 次元素读
    10:  同一份数据被 BM(或 BN) 个输出共用, 访存降到 1/BM 量级

    一个 BMxBN 的块, 每个 k-step 只需读 (BM + BN) 个元素, 共 K/BK 步:
        总量 = (BM + BN) * K              (09 是 2 * BM * BN * K)
        A 的每个元素被 BN 个输出共用, B 的每个元素被 BM 个共用。
        BM=BN=32 时复用 32 倍, 64 时复用 64 倍。

    GEMM 的算术强度 (注意 fp16 是 2 字节, 不是 4):
        I = 2*BM*BN*BK / (2*(BM+BN)*BK) = BM*BN/(BM+BN)   [FLOP/Byte]
        BM=BN=32 -> I=16;  BM=BN=64 -> I=32;  BM=BN=128 -> I=64
        平衡点是 45.4, 所以这几档**全都还在带宽墙左边**。

    实测印证 (180.5 GB/s 带宽线):
        32/32/32: 16 * 180.5 = 2.89 TFLOPS 上界, 实测 2.76 -> 贴着 96%
        64/64/32: 32 * 180.5 = 5.78 TFLOPS 上界, 实测 4.72 -> 贴着 82%
    两档都顶在带宽线上, 说明这一版的天花板是访存, 不是 FMA 算力。
    想再往上只能推大 tile 提高 I, 但 128x128 的 3D 中间张量寄存器装不下
    —— 这就是第 11 题必须换 tensor core 的理由 (mma 在硬件内部复用操作数,
    不吃寄存器带宽, 也不吃这份 I 的账)。

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
    目标  -> >= 12% 峰值 (约 2.2 TFLOPS)。实测参考: 32/32/32 是 2.8-3.0 TFLOPS (~16%),
             64/64/32 是 4.72 TFLOPS (25.9%)。
    注: 别信"三个维度一起进寄存器所以装不下"这个说法 —— 实测 64/64/32 只有
        96 个寄存器、0 spill, 编译器把中间结果走 shared memory 中转 (smem 32KB),
        并没有真把 131072 个数摊在寄存器里。真正的软肋是 smem 占用抬高后
        occupancy 塌掉, 以及 tile 越大 layout 转换越贵。
        第 11 题换 tl.dot 的理由也不是"寄存器装不下", 而是 mma 在硬件内部
        复用操作数, 同样的 tile 不用付这份 layout 转换的账。

两个实测出来的坑 (踩过, 记下来):

  1. 精度: a / b 是 fp16, `a * b` 在 fp16 里就舍入了, 1024 项累起来
     max|diff| 能到 3.5e-2, 直接被 tol 卡死。必须先 .to(tl.float32) 再乘。
     tl.sum(x, axis, dtype=tl.float32) 也没用 —— 那只是把"已经舍入过的
     fp16 乘积"转成 fp32, 舍入已经发生了。

  2. tile 怎么选: 决定性能的只有两条, 都算得出来。
     (a) 算术强度 I = BM*BN/(BM+BN), 决定带宽上界 I*180.5 GB/s。
     (b) smem 占用 = (BM+BN)*BK*2 字节, 决定 occupancy。
     实测 (M=N=K=1024, 每档独立进程跑 3 轮, min/max 差 <0.5%):

        tile        ms       TFLOPS   I      带宽线利用率
        32/32/32    0.7793   2.76     16.0   95.6%
        32/32/64    0.7803   2.75     16.0   95.2%   BK 不影响 I 也不影响速度
        64/32/32    0.6766   3.17     21.3   82.3%
        32/64/32    0.5888   3.65     21.3   94.8%   同为 21.3, 但更快
        64/64/32    0.4548   4.72     32.0   81.7%   <- 这题最优
        64/64/64    0.6185   3.47     32.0   60.1%   I 相同却慢 36%: smem 64KB 压低 occupancy
        128/64/32   0.4656   4.61     42.7   59.9%   寄存器 168, 只有 1 个 CTA/SM
        128/128/32  不稳     0.4~3.5  64.0   —       三次跑出 3 个数量级, 不要用

     所以 "tile 越大越好" 和 "必须方形" 都是错的。真正决定的是 I 和 occupancy,
     而 I 一旦推到 42 以上, occupancy 塌下来吃掉的比涨上去的多。

  3. 顺带验证过: 生成的 PTX 里 0 条 mma.sync, 256 条 fma.rn.f32 —— 编译器
     没有偷偷把 mul+reduce 融合成 tl.dot, 这题确实是纯 FMA 基线。
     (不带 .to(f32) 的版本编译器会在 TritonCombineOps 崩:
      'tt.splat requires the same element type', 歪打正着暴露了它想融合。)

  4. 测量陷阱 (调这题必须知道): 在同一个 Python 进程里连着测多个 tile 配置,
     某些档会跑出慢 8-9 倍的假数据 (实测 64/32/32 在进程内连测得 0.39
     TFLOPS, 独立进程稳定得 3.17)。原因没查清, 但结论明确:
     **一个配置一个进程, 跑 3 轮看 min/max 是否收敛, 再信这个数。**
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gemm_tiled_kernel(a_ptr, b_ptr, out_ptr, M, N, K,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):         # K 方向分块
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        # f16 x f16 的乘积在 f16 里就舍入了, 1024 项累起来误差能到 1e-2 量级,
        # 所以先升到 f32 再乘 —— 这是纯 FMA 版本必须付的代价。
        acc += tl.sum(a.to(tl.float32)[:, :, None] * b.to(tl.float32)[None, :, :], 1)
    # store 在循环外, 只写一次
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(a, b, out):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"shape mismatch: a{K} vs b{K2}"
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_tiled_kernel[grid](a, b, out, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
