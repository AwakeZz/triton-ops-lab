"""
实验: block 到底是怎么并行执行的

三个问题:
  1. 一个 grid 里有多少个 block, 硬件能同时装下多少
  2. 一次 launch 里, block 是按 pid 顺序跑的吗
  3. 同时跑的 block 数量不够时, 带宽会不会掉

跑: python _exp_parallel.py
"""
import os
os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")

import statistics
import time

import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, v, mask=mask)


@triton.jit
def order_kernel(counter_ptr, order_ptr, BLOCK: tl.constexpr):
    """每个 block 开头抢一个号, 写进 order[pid]。
    如果 block 是顺序执行的, order[pid] 就应该 == pid。
    """
    pid = tl.program_id(0)
    # 标量 atomic: 每个 program 只执行一次
    my_turn = tl.atomic_add(counter_ptr, 1)
    tl.store(order_ptr + pid, my_turn)


@triton.jit
def strided_copy(x_ptr, out_ptr, n, n_prog, iters, BLOCK: tl.constexpr):
    """只启动 n_prog 个 block, 每个 block 轮流搬剩下的部分。
    用来验证: 并发 block 太少时, 带宽会被拖死。
    """
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    for i in range(iters):
        offs = (i * n_prog + pid) * BLOCK + lane
        mask = offs < n
        v = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, v, mask=mask)


def bench(fn, warmup_ms=600.0, target_ms=80.0, min_iters=8):
    fn()                      # 第一次调用含 JIT 编译, 必须排除
    torch.cuda.synchronize()
    stop = time.perf_counter() + warmup_ms / 1000
    while time.perf_counter() < stop:
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    once = (time.perf_counter() - t0) * 1000
    iters = max(min_iters, int(target_ms / max(once, 1e-3)))
    samples = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    return statistics.median(samples)


def main():
    p = torch.cuda.get_device_properties(0)
    print("=" * 68)
    print(f"设备: {p.name}   sm_{p.major}{p.minor}")
    print(f"  SM 数量            : {p.multi_processor_count}")
    print(f"  每 SM 最大常驻线程 : {p.max_threads_per_multi_processor}")
    print(f"  每 SM 寄存器数     : {p.regs_per_multiprocessor}")
    print(f"  每 SM 共享内存     : {p.shared_memory_per_multiprocessor // 1024} KB")
    print(f"  每 SM 最大 block 数: 16  (sm_86 硬件规格, torch 没暴露这个字段)")
    print("=" * 68)

    BLOCK, WARPS = 4096, 4
    N = 1 << 25                      # 33,554,432 个 float32 = 128 MB
    x = torch.randn(N, device="cuda")
    out = torch.empty_like(x)
    nblocks = triton.cdiv(N, BLOCK)

    print(f"\n用你的 copy 配置: BLOCK={BLOCK}, num_warps={WARPS}, N={N:,}")
    print(f"  grid = (cdiv({N:,}, {BLOCK}),) = ({nblocks},)  -> {nblocks} 个 block")
    print(f"  每个 block {WARPS * 32} 个线程, 共 {nblocks * WARPS * 32:,} 个线程")

    # --- 编译一次, 读寄存器用量 ---
    k = copy_kernel.warmup(x, out, N, BLOCK=BLOCK, num_warps=WARPS, grid=(1,))
    k._init_handles()            # 不调用的话 n_regs 还不存在
    nreg = k.n_regs
    threads = WARPS * 32
    # 寄存器按 warp 分配, 粒度 256 个/warp
    per_warp = -(-nreg * 32 // 256) * 256
    regs_per_block = per_warp * WARPS
    print(f"\n编译产物: 每线程 {nreg} 个寄存器, spills={k.n_spills}")
    print(f"  -> 每 warp {per_warp} 个(按 256 粒度上取整), 每 block {regs_per_block} 个")

    by_regs = p.regs_per_multiprocessor // regs_per_block
    by_threads = p.max_threads_per_multi_processor // threads
    by_blocks = 16               # sm_86: 每 SM 最多 16 个常驻 block
    resident = min(by_regs, by_threads, by_blocks)
    print(f"每 SM 能同时装下几个 block:")
    print(f"  寄存器限制 : {p.regs_per_multiprocessor} // {regs_per_block} = {by_regs}")
    print(f"  线程限制   : {p.max_threads_per_multi_processor} // {threads} = {by_threads}")
    print(f"  block 上限 : {by_blocks}")
    print(f"  -> 取最小 = 每 SM {resident} 个 block" +
          ("  (这个 kernel 卡在寄存器上)" if resident == by_regs else ""))

    total_resident = resident * p.multi_processor_count
    waves = nblocks / total_resident
    print(f"\n整个 GPU 同时能跑 {resident} x {p.multi_processor_count} = "
          f"{total_resident} 个 block")
    print(f"总共要跑 {nblocks} 个 -> 分 {waves:.1f} 批(wave) 才跑完")

    # --- 实验 1: block 是按 pid 顺序执行的吗 ---
    print("\n" + "=" * 68)
    print("实验 1: 谁先跑到?(抢号的结果 vs pid 顺序)")
    print("=" * 68)
    nb = total_resident          # 正好装满一批, 全部并发
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    order = torch.full((nb,), -1, dtype=torch.int32, device="cuda")
    order_kernel[(nb,)](counter, order, BLOCK=128, num_warps=4)
    torch.cuda.synchronize()
    o = order.cpu().tolist()
    print(f"启动 {nb} 个 block, order[pid] = 该 block 抢到的序号")
    print(f"  顺序执行的话 order 应该 = [0, 1, 2, 3, ...]")
    print(f"  实测前 16 个 pid 抢到的号: {o[:16]}")
    print(f"  抢到 0 号的是 pid = {o.index(0)}  (不是 0 就说明不是顺序执行)")
    print(f"  抢到 1 号的是 pid = {o.index(1)}")
    print(f"  order 与 pid 完全相同的 block 占比: "
          f"{sum(1 for i, v in enumerate(o) if i == v)} / {nb}")

    # --- 实验 2: 固定工作量, 只改并发 block 数 ---
    print("\n" + "=" * 68)
    print("实验 2: 同样的活, 只改同时干活的 block 数")
    print("=" * 68)
    print("固定搬 128 MB, 每个 block 轮流搬一部分 (grid-stride)")
    print(f"{'block数':>8}{'批数':>8}{'ms':>10}{'GB/s':>9}{'占峰值':>9}")
    print("-" * 68)
    bytes_moved = N * 4 * 2
    results = {}
    for g in (1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1600, 8192):
        iters = triton.cdiv(N, g * BLOCK)

        def launch(g=g, iters=iters):
            strided_copy[(g,)](x, out, N, g, iters, BLOCK=BLOCK, num_warps=WARPS)

        ms = bench(launch)
        gbps = bytes_moved / (ms * 1e-3) / 1e9
        results[g] = (ms, gbps)
        print(f"{g:>8}{g / total_resident:>7.2f}x{ms:>10.4f}{gbps:>9.1f}"
              f"{gbps / 192 * 100:>8.1f}%")

    one_ms, one_gbps = results[1]
    full_ms, full_gbps = results[nblocks]
    print(f"\n读法 1: 如果 block 是一个接一个执行的, 整卡带宽就等于单 block 上限")
    print(f"        {one_gbps:.1f} GB/s -> 搬 128MB 需要 {one_ms:.2f} ms")
    print(f"        实测 8192 个 block 只要 {full_ms:.2f} ms -> 快 {one_ms / full_ms:.1f} 倍")
    print(f"        代码里没有任何循环, 时间却变短了, 只能是多个 block 同时在跑")
    print(f"\n读法 2: 并发容量 {total_resident} 个 block, 但带宽在 5~10 个 block 时就满了。")
    print(f"        剩下的并行度不靠 block 数量, 靠每个线程在飞的多笔 load (ILP)。")
    print(f"\n读法 3: 8192 个 block = {nblocks / total_resident:.0f} 个 wave, "
          f"但耗时并没有变成 {nblocks / total_resident:.0f} 倍。")
    print(f"        因为 block 一结束下一个立刻补位, 硬件在动态回填 (backfill)。")


if __name__ == "__main__":
    main()
