"""
实验: tl.sum 的 axis 到底消掉哪一维, 以及 store 的标量/向量两种形态。

用法:
    G:/venvs/triton/Scripts/python.exe _exp_axis.py
"""
import os

os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")

import torch
import triton
import triton.language as tl


@triton.jit
def sum_axis1(x_ptr, o_ptr, ROWS: tl.constexpr, COLS: tl.constexpr):
    """v: (ROWS, COLS)  ->  sum(axis=1): (ROWS,)  —— 每行求和"""
    r = tl.arange(0, ROWS)
    c = tl.arange(0, COLS)
    v = tl.load(x_ptr + r[:, None] * COLS + c[None, :])
    s = tl.sum(v, axis=1)
    tl.store(o_ptr + r, s)


@triton.jit
def sum_axis0(x_ptr, o_ptr, ROWS: tl.constexpr, COLS: tl.constexpr):
    """v: (ROWS, COLS)  ->  sum(axis=0): (COLS,)  —— 每列求和"""
    r = tl.arange(0, ROWS)
    c = tl.arange(0, COLS)
    v = tl.load(x_ptr + r[:, None] * COLS + c[None, :])
    s = tl.sum(v, axis=0)
    tl.store(o_ptr + c, s)


@triton.jit
def sum_to_one(x_ptr, o_ptr, N: tl.constexpr):
    """v: (N,)  ->  sum(axis=0): 一个标量。注意 o_ptr 后面没有 + offsets"""
    c = tl.arange(0, N)
    v = tl.load(x_ptr + c)
    s = tl.sum(v, axis=0)
    tl.store(o_ptr, s)          # 标量指针, 标量值


@triton.jit
def bad_axis(x_ptr, o_ptr, N: tl.constexpr):
    """1D 张量上传 axis=1, 应该报错"""
    c = tl.arange(0, N)
    v = tl.load(x_ptr + c)
    s = tl.sum(v, axis=1)
    tl.store(o_ptr, s)


def main():
    torch.manual_seed(0)
    ROWS, COLS = 4, 8
    x = torch.arange(ROWS * COLS, dtype=torch.float32, device="cuda").reshape(ROWS, COLS)

    print("=" * 70)
    print(f"x = arange({ROWS * COLS}).reshape({ROWS}, {COLS})")
    print("=" * 70)
    for row in x.cpu().tolist():
        print("   ", [int(v) for v in row])

    print("\n" + "=" * 70)
    print("axis=1 -> 消掉第 1 维, 得到 (ROWS,) = 每行之和")
    print("=" * 70)
    o1 = torch.zeros(ROWS, device="cuda")
    sum_axis1[(1,)](x, o1, ROWS=ROWS, COLS=COLS)
    torch.cuda.synchronize()
    print(f"  kernel 结果 : {[int(v) for v in o1.cpu().tolist()]}")
    print(f"  x.sum(dim=1): {[int(v) for v in x.sum(dim=1).cpu().tolist()]}")
    print(f"  shape       : {tuple(o1.shape)}  <- 与 torch dim=1 完全一致")

    print("\n" + "=" * 70)
    print("axis=0 -> 消掉第 0 维, 得到 (COLS,) = 每列之和")
    print("=" * 70)
    o0 = torch.zeros(COLS, device="cuda")
    sum_axis0[(1,)](x, o0, ROWS=ROWS, COLS=COLS)
    torch.cuda.synchronize()
    print(f"  kernel 结果 : {[int(v) for v in o0.cpu().tolist()]}")
    print(f"  x.sum(dim=0): {[int(v) for v in x.sum(dim=0).cpu().tolist()]}")
    print(f"  shape       : {tuple(o0.shape)}")

    print("\n" + "=" * 70)
    print("1D 张量: tl.sum(v, axis=0) 返回标量 -> store 用标量指针")
    print("=" * 70)
    v1 = torch.arange(8, dtype=torch.float32, device="cuda")
    o = torch.zeros(1, device="cuda")
    sum_to_one[(1,)](v1, o, N=8)
    torch.cuda.synchronize()
    print(f"  v           = {[int(t) for t in v1.cpu().tolist()]}")
    print(f"  tl.store(o_ptr, tl.sum(v, axis=0))  ->  o[0] = {o.item():.0f}")
    print(f"  tl.sum 返回的是一个标量, 不是长度为 1 的向量")

    print("\n" + "=" * 70)
    print("如果 1D 张量上写 axis=1")
    print("=" * 70)
    try:
        bad_axis[(1,)](v1, o, N=8)
        torch.cuda.synchronize()
        print("  没报错?")
    except Exception as exc:
        msg = str(exc).splitlines()
        head = [ln for ln in msg if "axis" in ln.lower() or "rank" in ln.lower()][:3]
        print(f"  {type(exc).__name__}")
        for ln in (head or msg[:3]):
            print(f"    {ln.strip()[:90]}")

    print("\n  " + "=" * 66)
    print("  记法: axis 指向的是'被消掉'的那一维, 不是'被保留'的")
    print("        v(4,8) axis=1 -> (4,)   每行压成一个数")
    print("        v(4,8) axis=0 -> (8,)   每列压成一个数")
    print("        v(8,)  axis=0 -> 标量   整个压成一个数")
    print("  " + "=" * 66)


if __name__ == "__main__":
    main()
