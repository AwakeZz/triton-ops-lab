"""other 取值对归约结果的影响：语义演示 + GPU 实测"""
import math
import torch
import triton
import triton.language as tl

# ---------- 1. 语义演示：一行 6 个真实数据，BLOCK_N=8，后 2 槽越界 ----------
base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
N, BN = 6, 8
OTHERS = [0.0, 1.0, -1.0, 100.0, -math.inf, math.inf, float("nan")]


def fmt(x):
    if isinstance(x, str):
        return x
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "-inf" if x < 0 else "inf"
    return "%g" % x


print("真实数据 [1,2,3,4,5,6]，N=6，补到 8 槽")
print("真值      sum=21  max=6  min=1  sumsq=91  prod=720  mean=3.5")
print()
print("%-8s | %10s | %8s | %8s | %12s | %10s" % ("other", "sum", "max", "min", "sumsq", "prod"))
print("-" * 72)
for o in OTHERS:
    v = base + [o] * (BN - N)
    s = sum(v)
    mx = max(v)
    mn = min(v)
    ss = sum(t * t for t in v)
    p = 1.0
    for t in v:
        p *= t
    print("%-8s | %10s | %8s | %8s | %12s | %10s" % (
        fmt(o) if not math.isnan(o) else "nan",
        fmt(s), fmt(mx), fmt(mn), fmt(ss), fmt(p)))
print()


# ---------- 2. GPU 实测：确认 Triton 行为与手算一致 ----------
@triton.jit
def k_reduce(x_ptr, out_ptr, N, other, OP: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask=mask, other=other)
    if OP == 0:
        r = tl.sum(v, 0)
    elif OP == 1:
        r = tl.max(v, 0)
    elif OP == 2:
        r = tl.min(v, 0)
    else:
        r = tl.sum(v * v, 0)
    tl.store(out_ptr + row, r)


@triton.jit
def k_ln(x_ptr, out_ptr, N, other, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask=mask, other=other)
    m = tl.sum(v, 0) / N
    c = tl.sum((v - m) * (v - m), 0) / N
    tl.store(out_ptr + offs, (v - m) / tl.sqrt(c + 1e-5), mask=mask)


x1 = torch.tensor([base], device="cuda")
o1 = torch.empty(1, device="cuda")
print("GPU 实测（同一行 [1..6]，BLOCK_N=8）")
print("%-8s | %8s | %6s | %6s | %8s" % ("other", "sum", "max", "min", "sumsq"))
print("-" * 50)
for o in [0.0, 1.0, 100.0, -math.inf]:
    vals = []
    for op in range(4):
        k_reduce[(1,)](x1, o1, N, o, op, BN)
        vals.append(o1.item())
    print("%-8s | %8s | %6s | %6s | %8s" % (
        fmt(o), fmt(vals[0]), fmt(vals[1]), fmt(vals[2]), fmt(vals[3])))
print()

# ---------- 3. layernorm 里 other 填错的误差放大 ----------
print("layernorm 实测（N=1000, BLOCK_N=1024, 越界 24 槽）")
x = torch.randn(4096, 1000, device="cuda")
ref = (x - x.mean(1, keepdim=True)) / torch.sqrt(
    x.var(1, unbiased=False, keepdim=True) + 1e-5)
print("%-8s | %10s | %12s | %10s" % ("other", "mean误差", "var相对误差", "输出 max_err"))
print("-" * 54)
for o in [0.0, 0.001, -0.001, 0.5, 1.0, -math.inf]:
    out = torch.empty_like(x)
    k_ln[(4096,)](x, out, 1000, o, 1024)
    v = torch.full((4096, 1024), o, device="cuda")
    v[:, :1000] = x
    m = v.sum(1) / 1000
    c = ((v - m[:, None]) ** 2).sum(1) / 1000
    mr = (m - x.mean(1)).abs().max().item()
    vr = ((c - x.var(1, unbiased=False)) / x.var(1, unbiased=False)).abs().max().item()
    err = (out - ref).abs().max().item()
    print("%-8s | %10.2e | %12.2e | %10.2e" % (
        fmt(o) if not math.isinf(o) else "-inf", mr, vr, err))
