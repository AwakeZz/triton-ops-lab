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
   它不是越大越好: 每一级 stage 都要占一块 shared memory (fp16):
       smem = (BM*BK + BK*BN) * 2 * max(1, stages-1)     <- 是 stages-1, 不是 stages
   而一张 SM 只有 100 KB, 超了直接掉 occupancy。

实测 (2026-10-29 重测, 探针口径, cuBLAS 在 1024³ 是 0.1488 ms / 14.43 TFLOPS):

    BLOCK_M/N/K     warps  stages   1024³ TFLOPS   %峰     备注
    32/32/32        4      1          11.57        63.6
    64/64/32        4      1          14.31        78.6    <- 等于第 11 题, 没做这题
    64/64/32        4      2          14.27        78.4    cp.async 有了, 但 smem 没翻倍, 白给
    64/64/32        4      3          15.20        83.5    <- 1024³ 最好
    64/64/32        4      4          14.24        78.2    smem 24KB -> 每 SM 只放得下 4 个 CTA (s3 是 5)
    128/128/32      8      3          12.19        67.0
    128/128/64      8      3          11.65        64.0
    128/64/64       4      4          11.32        62.2
    64/128/64       4      4          12.47        68.5

    (同样的配置换到 2048³, 排序整个反转, 已复验:
     128/128/32 w8 s3 = 15.97 TFLOPS (87.7%), 64/64/32 w4 s3 只有 11.94 (65.6%)。
     原因还是 grid 大小 —— 2048³ 下 128x128 有 256 个 block, 正好铺满;
     1024³ 下只有 64 个, 20 个 SM 分不到 4 个, 尾部空转占掉两成。
     **所以 tile 尺寸没有全局最优, 它和 M/N 绑在一起。**)

num_stages 到底改了什么 —— 看 PTX, 一眼就明白 (64/64/32 w4):

    stages  smem    cp.async  commit  wait  ld.global  bar.sync
    1        8192         0       0     0          4         6     <- 同步加载, load 完才 mma
    2        8192         8       4     5          0         8     <- 双缓冲, 但 smem 没翻倍
    3       16384        12       6     5          0         9     <- 真流水
    4       24576        16       8     5          0        10     <- 3 个 buffer, occupancy 5 -> 4

    s1 里 cp.async 是 0, 走的是同步 ld.global: 搬第 k 块的时候 tensor core 在闲着。
    s2 有了 cp.async 但 smem 还是 8192 —— 只给了 1 个 buffer, 重叠不起来, 所以没提升。
    s3 才真的把 smem 翻到 16KB (2 个 buffer), 第 k+1 块的搬运和第 k 块的 mma 并行。
    s4 给了 3 个 buffer = 24KB, 但 floor(102400/24576) = 4, 每 SM 的 CTA 从 5 掉到 4。
    多出来的那一级流水, 不够付 occupancy 掉 20% 的账。
    (以前这里写的是"barrier 变多", 那是错的 —— barrier 多一条是结果, 不是原因。)

shared memory 的三本账 (fp16, 这张卡: 每 SM 100 KB / 65536 寄存器 / 128 B per clk):

  容量:  smem = (BM*BK + BK*BN) * 2 * max(1, stages-1)
         占每 SM 的比例 = smem / 102400
         CTA_per_SM    = min( 102400/smem, 65536/regs_per_cta, 1536/threads )

  带宽:  每 FLOP 要读 (BM+BN)/(BM*BN) 字节的 smem
         (一轮 K 从 smem 读 (BM+BN)*BK*2 字节, 做 BM*BN*BK 次乘加)
         32/32 = 0.0625   64/64 = 0.03125   128/128 = 0.015625  B/FLOP
         跑到 16 TFLOPS (每 SM 0.8 TFLOPS) 时 128/128 只要 12.5 GB/s,
         而 smem 带宽 = 128 B/clk * 1.695 GHz = 217 GB/s/SM —— 只用掉 5.8%。
         **所以 smem 带宽从来不是瓶颈**, 它是 HBM 的 24 倍。tile 的全部意义就在这:
         把 180 GB/s 的慢带宽换成 4.34 TB/s 的快带宽。

  占用:  64/64/32 s3 = 16 KB (16%) -> 寄存器先卡, 5 个
         64/64/32 s4 = 24 KB (24%) -> smem 先卡, 4 个     <- 唯一翻转的一档
         64/64/64 s3 = 32 KB (32%) -> smem 先卡, 3 个
         128/128/32 s3 = 32 KB (32%) -> 寄存器先卡, 1 个
         128/128/64 s3 = 64 KB (64%) -> 两个都卡, 1 个

  反过来用: 想让 CTA_per_SM 保持 N 个, 就要求 smem <= 102400/N。
  64/64/32 想保 5 个 -> 每块 <= 20480 -> stages-1 <= 2.5 -> stages <= 3。
  **s3 是算出来的, 不是试出来的。**

验收标准:
    通过  -> 正确性 + 启动 1 次
    目标  -> >= 80% 峰值 (约 14.6 TFLOPS)。达到 82% 就是超过 cuBLAS。
    实测: 64/64/32 w4 s3 在本框架 check.py 口径是 15.92 TFLOPS (87.5%), 超 cuBLAS 10%。
    注: 1024³ 的 100% 是 18.2 TFLOPS, 但那是 4096³ 才能贴到的渐近值;
        1024³ 只有 0.14 ms, kernel 启动和尾部空转占掉了将近两成, 谁都到不了 100%。
    注 2: 这题唯一要动的旋钮就是 num_stages。
        沿用第 11 题的 num_stages=1 等于没做这道题 —— 78.6%, 连 80% 的门槛都够不到。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gemm_pipe_kernel(a_ptr, b_ptr, out_ptr, M, N, K,
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
        acc = tl.dot(a, b, acc)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(a, b, out):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"shape mismatch: a{K} vs b{K2}"
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_pipe_kernel[grid](a, b, out, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                          num_warps=4, num_stages=3)
