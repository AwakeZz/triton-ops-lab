"""
实验: tl.cumsum 是"扫描", 不是"归约" —— 输出形状和输入一样。

05 里写的是:
    tl.store(out_ptr + cows, tl.cumsum(v, 0))     # 标量指针 + 向量值
这会形状不匹配。为什么? 因为 sum/max 压成一个数, cumsum 不压。

用法:
    G:/venvs/triton/Scripts/python.exe _exp_cumsum.py
"""
import os

os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")

import torch
import triton
import triton.language as tl


@triton.jit
def probe_shapes(x_ptr, sum_o, cum_o, n, BLOCK: tl.constexpr):
    """同一个 v, 分别做 sum 和 cumsum, 看输出形状"""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask)
    tl.store(sum_o + 0, tl.sum(v, 0))                  # 标量 -> 存 1 个位置
    c = tl.cumsum(v, 0)
    tl.store(cum_o + offs, c, mask)                    # 向量 -> 存 BLOCK 个位置


@triton.jit
def cum_bad(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """故意用标量指针接 cumsum 的向量结果"""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask)
    c = tl.cumsum(v, 0)
    tl.store(o_ptr, c)          # <- 标量指针, 向量值


def main():
    torch.manual_seed(0)
    n = 8
    x = torch.arange(1, n + 1, dtype=torch.float32, device="cuda")
    s = torch.zeros(1, device="cuda")
    c = torch.zeros(n, device="cuda")

    probe_shapes[(1,)](x, s, c, n, BLOCK=8)
    torch.cuda.synchronize()

    print("=" * 70)
    print("同一个向量 v, 两个操作的结果形状对比")
    print("=" * 70)
    print(f"  v        = {[int(t) for t in x.cpu().tolist()]}")
    print(f"  tl.sum    = {s.item():.0f}                       -> 标量, 存 1 个位置")
    print(f"  tl.cumsum = {[int(t) for t in c.cpu().tolist()]}  -> 向量, 存 {n} 个位置")
    print(f"  torch.cumsum = {[int(t) for t in torch.cumsum(x, 0).cpu().tolist()]}")

    print("\n  形状:")
    print(f"    tl.sum(v).shape    = ()      归约: 消掉一维")
    print(f"    tl.cumsum(v).shape = ({n},)   扫描: 形状不变")

    print("\n" + "=" * 70)
    print("用标量指针接 cumsum 会报什么错")
    print("=" * 70)
    try:
        cum_bad[(1,)](x, c, n, BLOCK=8)
        torch.cuda.synchronize()
        print("  没报错")
    except Exception as exc:
        text = str(exc)
        lines = [ln.rstrip() for ln in text.splitlines()]
        tail = [ln for ln in lines if ln.strip()][-3:]
        print(f"  {type(exc).__name__}")
        print(f"  {'-' * 60}")
        for ln in lines:
            print(f"  {ln}")
        print(f"  {'-' * 60}")
        print(f"  末尾关键行: {tail}")

    print("\n" + "=" * 70)
    print("05 的 store 该怎么写")
    print("=" * 70)
    print("  cumsum 输出和输入同形状, 所以:'值向量 -> 地址向量'")
    print("     tl.store(out_ptr + offs, 结果, mask=mask)")
    print("  而 sum/max 是:'值标量 -> 地址标量'")
    print("     tl.store(out_ptr + row,  结果)")
    print("\n  一句话: 看左边是标量还是向量, 右边必须跟它同形状。")


if __name__ == "__main__":
    main()
