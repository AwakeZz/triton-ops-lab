"""
tl.max(v, 0) 是"这个偏置之前"的最大值吗?  -> 不是, 是 v 里那 BLOCK_N 个位置的最大值

跑: G:/venvs/triton/Scripts/python.exe _exp_max_scope.py
"""
import os

os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")
import torch
import triton
import triton.language as tl


@triton.jit
def max_with_offs(x_ptr, o_ptr, oi_ptr, BLOCK: tl.constexpr):
    """offs 由外部传进来, 便于换各种取法"""
    offs = tl.load(oi_ptr + tl.arange(0, BLOCK))       # 读一串下标
    v = tl.load(x_ptr + offs)                          # 按这串下标取值
    tl.store(o_ptr, tl.max(v, 0))


@triton.jit
def _max2(a, b):
    return tl.maximum(a, b)


@triton.jit
def cmpmax(x_ptr, o_ptr, BLOCK: tl.constexpr):
    """前缀最大(cummax), 用于对照 tl.max"""
    v = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.store(o_ptr + tl.arange(0, BLOCK), tl.associative_scan(v, 0, _max2))


def run_offs(x, idx):
    BLOCK = len(idx)
    o = torch.zeros(1, device="cuda")
    d_idx = torch.tensor(idx, dtype=torch.int32, device="cuda")
    max_with_offs[(1,)](x, o, d_idx, BLOCK=BLOCK)
    torch.cuda.synchronize()
    return o.item()


def main():
    x = torch.tensor([3.0, 9.0, 1.0, 7.0, 5.0, 2.0, 8.0, 4.0], device="cuda")
    print("x =", x.tolist())
    print("整条最大值 =", x.max().item())
    print()

    cases = [
        ("连续前 4 个     offs=0..3", [0, 1, 2, 3]),
        ("连续后 4 个     offs=4..7", [4, 5, 6, 7]),
        ("倒序取全部      offs=7..0", [7, 6, 5, 4, 3, 2, 1, 0]),
        ("跨步取偶数位    offs=0,2,4,6", [0, 2, 4, 6]),
        ("只取绝对值最大那个位置 offs=1", [1]),
    ]
    print(f"{'取法':<32}{'拿到的 max':>12}")
    print("-" * 46)
    for name, idx in cases:
        print(f"{name:<32}{run_offs(x, idx):>12.1f}")

    print("\n读法:")
    print("  - 倒序取全部 = 9.0, 和正序一样 -> max 与顺序无关")
    print("  - 换一组 offs, 结果就跟着变 -> 结果只取决于【装了哪些位置】")
    print("  - 没有'之前'这个概念: 取前 4 个得 9.0 是因为 9 恰好在前 4 个里")

    print("\n" + "=" * 56)
    print("对照: 前缀最大(cummax)才和位置有关")
    o = torch.zeros(8, device="cuda")
    cmpmax[(1,)](x, o, BLOCK=8)
    torch.cuda.synchronize()
    print(f"  cummax(v) = {o.tolist()}")
    print(f"  tl.max(v,0) = {o[-1].item()}  <- 等于 cummax 的最后一个, 但只是个标量")


if __name__ == "__main__":
    main()
