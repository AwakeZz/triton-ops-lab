"""
实验: tl.sum 的 axis 到底指的是哪一维, 以及"标量 store"是什么。

回答 tl.store(out_ptr + row, tl.sum(v, axis=0)) 里的几个问题:
  1. axis=0 在 1D 上是"全压成一个数", 在 2D 上却是"每列求和" —— 同一个 0 不同含义
  2. tl.sum 返回标量, 不是长度 1 的向量
  3. 标量 store 在硬件上怎么摊到 128 个线程

用法:
    G:/venvs/triton/Scripts/python.exe _exp_sum_axis.py
"""
import os

os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")

import re

import torch
import triton
import triton.language as tl


@triton.jit
def sum_1d(x_ptr, o_ptr, n, N: tl.constexpr):
    """v 是 1D -> axis=0 把整条压成一个标量"""
    offs = tl.arange(0, N)
    v = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    s = tl.sum(v, axis=0)
    tl.store(o_ptr, s)            # 标量指针 + 标量值, 没有 + offs


@triton.jit
def sum_2d_ax0(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
    """v 是 2D -> axis=0 压掉行维, 得到每列之和 (C,)"""
    r = tl.arange(0, R)
    c = tl.arange(0, C)
    v = tl.load(x_ptr + r[:, None] * C + c[None, :])
    s = tl.sum(v, axis=0)
    tl.store(o_ptr + c, s)


@triton.jit
def sum_2d_ax1(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
    """v 是 2D -> axis=1 压掉列维, 得到每行之和 (R,)"""
    r = tl.arange(0, R)
    c = tl.arange(0, C)
    v = tl.load(x_ptr + r[:, None] * C + c[None, :])
    s = tl.sum(v, axis=1)
    tl.store(o_ptr + r, s)


def main():
    torch.manual_seed(0)
    print("=" * 72)
    print("实验 1: 1D 上的 tl.sum(v, axis=0)")
    print("=" * 72)
    x = torch.arange(1, 9, dtype=torch.float32, device="cuda")
    o = torch.zeros(1, device="cuda")
    sum_1d[(1,)](x, o, 8, N=8)
    torch.cuda.synchronize()
    print(f"  v = {x.cpu().tolist()}")
    print(f"  tl.sum(v, axis=0) = {o.item():.0f}    (torch: {x.sum().item():.0f})")
    print(f"  o 的形状 {tuple(o.shape)} <- 存进 1 个位置, 是整个 v 的和")

    print("\n" + "=" * 72)
    print("实验 2: 2D 上 axis=0 和 axis=1 的区别")
    print("=" * 72)
    v = torch.arange(1, 17, dtype=torch.float32, device="cuda").reshape(4, 4)
    print("  v =")
    for row in v.cpu().tolist():
        print("     ", [int(t) for t in row])

    o0 = torch.zeros(4, device="cuda")
    sum_2d_ax0[(1,)](v, o0, R=4, C=4)
    torch.cuda.synchronize()
    print(f"\n  tl.sum(v, axis=0) -> {[int(t) for t in o0.cpu().tolist()]}")
    print(f"  torch v.sum(dim=0) -> {[int(t) for t in v.sum(dim=0).cpu().tolist()]}")
    print(f"  形状 {tuple(o0.shape)}  <- 压掉行, 每列一个数(列和)")

    o1 = torch.zeros(4, device="cuda")
    sum_2d_ax1[(1,)](v, o1, R=4, C=4)
    torch.cuda.synchronize()
    print(f"\n  tl.sum(v, axis=1) -> {[int(t) for t in o1.cpu().tolist()]}")
    print(f"  torch v.sum(dim=1) -> {[int(t) for t in v.sum(dim=1).cpu().tolist()]}")
    print(f"  形状 {tuple(o1.shape)}  <- 压掉列, 每行一个数(行和)")

    print("\n  " + "=" * 68)
    print("  关键: axis 指的是【被消掉的那一维】, 不是被保留的那一维")
    print("        (4,4) axis=0 -> (4,)  消掉第 0 维")
    print("        (4,4) axis=1 -> (4,)  消掉第 1 维")
    print("  " + "=" * 68)

    print("\n" + "=" * 72)
    print("实验 3: 标量 store 在 PTX 里长什么样")
    print("=" * 72)
    k = sum_1d.warmup(x, o, 8, N=8, grid=(1,))
    ptx = k.asm["ptx"]
    stores = [ln.strip() for ln in ptx.splitlines() if "st.global" in ln]
    print("  编译出的全局写指令:")
    for s in stores[:3]:
        print(f"    {s}")
    pred = [s for s in stores if s.startswith("@")]
    print(f"\n  带谓词的 store: {len(pred)} 条")
    print(f"  含义: 128 个线程里只有满足谓词的那个真正写内存,")
    print(f"        不会出现 128 个线程抢写同一地址。")

    print("\n" + "=" * 72)
    print("实验 4: 和 copy 的 store 对比")
    print("=" * 72)
    print(f"  copy : tl.store(out_ptr + offs, v, mask=mask)     指针向量 + 值向量")
    print(f"         一次写 {4096} 个位置(整个 BLOCK)")
    print(f"  sum  : tl.store(out_ptr + row, tl.sum(v, axis=0)) 标量指针 + 标量值")
    print(f"         一次写 1 个位置")
    print(f"\n  所以 sum 的 grid 是 M 个 program, 每个只贡献 1 个结果。")
    print(f"  这就是为什么 run() 里 grid 要用 x.shape[0], 不是 numel()/BLOCK。")


if __name__ == "__main__":
    main()
