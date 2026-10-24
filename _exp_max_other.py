"""
实验: 归约类 kernel 里 tl.load 的 other 参数, sum 和 max 要求不一样。

04_max 里如果这样写:
    v = tl.load(x_ptr + offs, mask=mask)      # other 默认 0
    tl.max(v, 0)

mask 掉的位置被填成 0。如果这一行全是负数, max 就会得 0 而不是真实最大值。

用法:
    G:/venvs/triton/Scripts/python.exe _exp_max_other.py
"""
import os

os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")

import torch
import triton
import triton.language as tl


@triton.jit
def max_default_other(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask)          # other 默认 0
    tl.store(o_ptr, tl.max(v, 0))


@triton.jit
def max_neg_inf(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask, other=float("-inf"))
    tl.store(o_ptr, tl.max(v, 0))


@triton.jit
def sum_default_other(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask)          # other 默认 0
    tl.store(o_ptr, tl.sum(v, 0))


def main():
    torch.manual_seed(0)
    n, BLOCK = 127, 1024          # N 不是 2 的幂, 897 个位置被 mask 掉

    for tag, x in [
        ("全负数", -torch.rand(n, device="cuda") - 1.0),
        ("正负混合", torch.randn(n, device="cuda")),
    ]:
        o = torch.zeros(1, device="cuda")
        print(f"\n{tag}:  n={n}, BLOCK={BLOCK}, 被 mask 掉 {BLOCK - n} 个位置")
        print(f"  torch 的 max  = {x.max().item():.4f}")

        max_default_other[(1,)](x, o, n, BLOCK=BLOCK)
        torch.cuda.synchronize()
        a = o.item()
        print(f"  other 默认 0  = {a:.4f}   {'<- 错' if abs(a - x.max().item()) > 1e-5 else '正确'}")

        max_neg_inf[(1,)](x, o, n, BLOCK=BLOCK)
        torch.cuda.synchronize()
        b = o.item()
        print(f"  other=-inf    = {b:.4f}   {'正确' if abs(b - x.max().item()) < 1e-5 else '<- 错'}")

    print("\n" + "=" * 68)
    print("为什么 sum 不用管: 0 是加法的单位元, 填 0 不影响结果")
    x = -torch.rand(127, device="cuda") - 1.0
    o = torch.zeros(1, device="cuda")
    sum_default_other[(1,)](x, o, 127, BLOCK=1024)
    torch.cuda.synchronize()
    print(f"  全负数求和: kernel {o.item():.4f}  vs  torch {x.sum().item():.4f}")
    print("=" * 68)
    print("所以: sum 可以用默认 other; max 必须显式 other=-inf")
    print("      这是'归约的单位元'问题 —— min 对应 +inf")


if __name__ == "__main__":
    main()
