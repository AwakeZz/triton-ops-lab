"""tl.max(v, 0) 到底取的是谁的最大值"""
import os

os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")
import torch
import triton
import triton.language as tl


@triton.jit
def max1d(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n, other=float("-inf"))
    tl.store(o_ptr, tl.max(v, 0))                 # 1D -> 标量


@triton.jit
def max1d_default(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n, other=float("-inf"))
    tl.store(o_ptr, tl.max(v))                    # 不写 axis


@triton.jit
def max2d_ax0(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
    v = tl.load(x_ptr + tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :])
    r = tl.max(v, 0)                              # 消掉第 0 维(行)
    tl.store(o_ptr + tl.arange(0, C), r)


@triton.jit
def max2d_ax1(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
    v = tl.load(x_ptr + tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :])
    r = tl.max(v, 1)                              # 消掉第 1 维(列)
    tl.store(o_ptr + tl.arange(0, R), r)


@triton.jit
def max_other(x_ptr, o_ptr, n, BLOCK: tl.constexpr, OTHER: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n, other=OTHER)
    tl.store(o_ptr, tl.max(v, 0))


def main():
    print("=" * 66)
    print("[1] 1D: tl.max(v, 0) 是整条向量的最大值吗")
    x = torch.tensor([3.0, -1.0, 7.5, 2.0, -9.0], device="cuda")
    o = torch.zeros(1, device="cuda")
    max1d[(1,)](x, o, 5, BLOCK=8)
    torch.cuda.synchronize()
    print(f"  v        = {x.tolist()}   (BLOCK=8, 后 3 个被 mask, 填 -inf)")
    print(f"  tl.max(v,0) = {o.item()}     torch.max = {x.max().item()}")

    o2 = torch.zeros(1, device="cuda")
    max1d_default[(1,)](x, o2, 5, BLOCK=8)
    torch.cuda.synchronize()
    print(f"  tl.max(v)   = {o2.item()}     <- 不写 axis 默认就是 0, 结果一样")

    print("\n" + "=" * 66)
    print("[2] 它取的是【这个 program 装进 v 的那些元素】的最大值")
    N = 4096
    big = torch.zeros(N, device="cuda")
    big[2000] = 999.0                       # 全局最大值藏在下标 2000
    big[:2000] = torch.linspace(0, 10, 2000, device="cuda")
    print(f"  整条数组长度 {N}, 最大值 {big.max().item():.0f} 在下标 2000")
    for BLOCK in (256, 1024, 4096):
        o = torch.zeros(1, device="cuda")
        max1d[(1,)](big, o, BLOCK, BLOCK=BLOCK)   # n=BLOCK, 只看前 BLOCK 个
        seen = "看到了" if o.item() == 999.0 else "没看到 -> 返回的是这段里的局部最大值"
        print(f"  BLOCK={BLOCK:<5} -> {o.item():>7.1f}   {seen}")
    print("  结论: tl.max 不知道 v 之外还有数据, 它只压平手上这条向量")

    print("\n" + "=" * 66)
    print("[3] 2D 时 axis 的含义变不变")
    v = torch.tensor([[1.0, 9.0, 3.0, 0.0],
                      [8.0, 2.0, 7.0, 1.0],
                      [4.0, 6.0, 5.0, 2.0],
                      [0.0, 3.0, 1.0, 4.0]], device="cuda")
    print("  v =")
    for r in v.cpu().tolist():
        print("     ", r)
    o0 = torch.zeros(4, device="cuda")
    max2d_ax0[(1,)](v, o0, R=4, C=4)
    torch.cuda.synchronize()
    print(f"  tl.max(v, 0) -> {o0.tolist()}   torch v.max(dim=0) -> {v.max(dim=0).values.tolist()}  形状 {tuple(o0.shape)}")
    o1 = torch.zeros(4, device="cuda")
    max2d_ax1[(1,)](v, o1, R=4, C=4)
    torch.cuda.synchronize()
    print(f"  tl.max(v, 1) -> {o1.tolist()}   torch v.max(dim=1) -> {v.max(dim=1).values.tolist()}  形状 {tuple(o1.shape)}")

    print("\n" + "=" * 66)
    print("[4] mask 掉的位置填什么, 直接改变结果")
    neg = torch.tensor([-3.0, -1.5, -8.0, -2.0], device="cuda")
    print(f"  v = {neg.tolist()}  真实最大值 = {neg.max().item()}")
    for other_name, other in [("0.0(默认)", 0.0), ("-inf", float("-inf"))]:
        o = torch.zeros(1, device="cuda")
        max_other[(1,)](neg, o, 4, BLOCK=8, OTHER=other)
        torch.cuda.synchronize()
        print(f"  other={other_name:<12} -> {o.item():.4f}"
              f"   {'正确' if abs(o.item() - neg.max().item()) < 1e-6 else '错! mask 区的 0 冒成了最大值'}")


if __name__ == "__main__":
    main()
