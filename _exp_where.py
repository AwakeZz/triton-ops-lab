"""tl.where 语义实验：选择而非过滤 / 不短路 / 广播 / 与 other 的取舍"""
import torch
import triton
import triton.language as tl

# ---------- 1. 语义：选择，元素个数不变 ----------
@triton.jit
def k_select(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    r = tl.where(v % 2 == 0, v, -1)
    tl.store(out_ptr + offs, r)


x = torch.arange(8, device="cuda", dtype=torch.int32)
o = torch.empty(8, device="cuda", dtype=torch.int32)
k_select[(1,)](x, o, 8)
print("输入      ", x.tolist())
print("where 结果", o.tolist(), "  <- 仍是 8 个元素")
print("numpy 布尔索引", x[x % 2 == 0].tolist(), "  <- 只有 4 个，这是过滤")
print()


# ---------- 2. 不短路：两个分支都会算 ----------
@triton.jit
def k_noshort(x_ptr, out_ptr, N, MODE: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    q = 1.0 / v
    if MODE == 0:
        r = tl.sum(q, 0)
    elif MODE == 1:
        r = tl.sum(tl.where(mask, q, 0.0), 0)
    else:
        r = tl.sum(1.0 / tl.where(mask, v, 1.0), 0)
    tl.store(out_ptr + row, r)


xn = torch.tensor([[2.0, 4.0, 5.0, 8.0, 10.0, 2.0]], device="cuda")  # N=6, 补到 8
on = torch.empty(1, device="cuda")
print("输入 [2,4,5,8,10,2]，越界 2 槽 v 被填 0，然后算 1/v")
for mode, name in [(0, "先归约(错)"), (1, "先 where 再归约"), (2, "在除数上 where")]:
    k_noshort[(1,)](xn, on, 6, mode, 8)
    got = on.item()
    print("  MODE%d %-16s = %s" % (mode, name, got))
print("  真值 1/2+1/4+1/5+1/8+1/10+1/2 = %.4f" % (0.5 + 0.25 + 0.2 + 0.125 + 0.1 + 0.5))
print()


# ---------- 3. 广播：标量条件 = 整块统一选择 ----------
@triton.jit
def k_bcast(x_ptr, out_ptr, flag, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    r = tl.where(flag, v, 0.0)
    tl.store(out_ptr + offs, r)


xb = torch.arange(6, device="cuda", dtype=torch.float32)
ob = torch.zeros(6, device="cuda")
k_bcast[(1,)](xb, ob, True, 8)
print("标量条件 True  ->", ob.tolist())
k_bcast[(1,)](xb, ob, False, 8)
print("标量条件 False ->", ob.tolist())
print()


# ---------- 4. 性能：other / where / 乘 mask 三种填法 ----------
@triton.jit
def ln_other(x_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    m = tl.sum(v, 0) / N
    var = tl.sum(tl.where(mask, v - m, 0.0) * tl.where(mask, v - m, 0.0), 0) / N
    tl.store(out_ptr + offs, (v - m) / tl.sqrt(var + 1e-5), mask=mask)


@triton.jit
def ln_where(x_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    m = tl.sum(v, 0) / N
    d = tl.where(mask, v - m, 0.0)
    var = tl.sum(d * d, 0) / N
    tl.store(out_ptr + offs, (v - m) / tl.sqrt(var + 1e-5), mask=mask)


@triton.jit
def ln_mul(x_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    m = tl.sum(v, 0) / N
    d = (v - m) * mask
    var = tl.sum(d * d, 0) / N
    tl.store(out_ptr + offs, (v - m) / tl.sqrt(var + 1e-5), mask=mask)


def ref(x):
    return (x - x.mean(1, keepdim=True)) / torch.sqrt(
        x.var(1, unbiased=False, keepdim=True) + 1e-5)


print("性能：N=1000 时三种写法（同时验证正确性）")
R, N = 4096, 1000
x = torch.randn(R, N, device="cuda")
rr = ref(x)
print("%-12s | %6s | %6s | %10s | %8s" % ("写法", "regs", "spill", "max_err", "GB/s"))
print("-" * 58)
for name, fn in [("other+where", ln_other), ("where", ln_where), ("乘 mask", ln_mul)]:
    out = torch.empty_like(x)
    call = lambda: fn[(R,)](x, out, N, 1024)
    call()
    err = (out - rr).abs().max().item()
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(200):
        call()
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / 200
    gbs = 2 * x.numel() * 4 / ms / 1e6
    k = fn.warmup(x, out, N, 1024, grid=(1,))
    k._init_handles()
    print("%-12s | %6d | %6d | %10.2e | %8.1f" % (name, k.n_regs, k.n_spills, err, gbs))
