"""01_add：越界写入检测 + 覆盖检测 + 带宽"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_now(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < n
    v1 = tl.load(x_ptr + offs, mask=mask)
    v2 = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, v1 + v2, mask=mask)


@triton.jit
def add_fix(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v1 = tl.load(x_ptr + offs, mask=mask, other=0.0)
    v2 = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, v1 + v2, mask=mask)


PAD = 1 << 18


def probe(kern, shape, block, grid_mode, label):
    n = shape[0] * shape[1]
    xb = torch.randn(n + PAD, device="cuda")
    yb = torch.randn(n + PAD, device="cuda")
    ob = torch.zeros(n + PAD, device="cuda")
    x, y, out = xb[:n], yb[:n], ob[:n]
    ref = x + y
    grid = (shape[0],) if grid_mode == "rows" else (triton.cdiv(n, block),)
    try:
        kern[grid](x, y, out, n, block)
        torch.cuda.synchronize()
    except Exception as e:
        return label, "CUDA err", "-", "-"
    d = (out - ref).abs().max().item()
    tail = ob[n:].abs().max().item()
    cov = grid[0] * block / n * 100
    return (label,
            "OK" if d < 1e-5 else f"错 {d:.1e}",
            "干净" if tail == 0 else f"越界写 {tail:.1e}",
            f"{cov:.0f}%")


print("=" * 78)
print("grid=(行数,), BLOCK=next_pow2(N)  ← 你现在的写法")
print("=" * 78)
print(f"{'shape':>14} {'结果':>10} {'尾部缓冲':>14} {'覆盖率':>7}")
for s in [(4096, 1024), (997, 1024), (1003, 512), (1, 4096), (333, 1000), (77, 127)]:
    B = triton.next_power_of_2(s[1])
    r = probe(add_now, s, B, "rows", "now")
    print(f"{str(s):>14} {r[1]:>10} {r[2]:>14} {r[3]:>7}")

print()
print("=" * 78)
print("同样 grid=(行数,) 但 BLOCK 改小（提示让你试 1024/2048/4096，试 512 会怎样）")
print("=" * 78)
s = (4096, 1024)
print(f"{'BLOCK':>7} {'结果':>12} {'尾部缓冲':>14} {'覆盖率':>7}")
for B in (512, 1024, 2048, 4096):
    r = probe(add_fix, s, B, "rows", "fix")
    print(f"{B:>7} {r[1]:>12} {r[2]:>14} {r[3]:>7}")

print()
print("=" * 78)
print("修正版：grid=cdiv(numel, BLOCK)")
print("=" * 78)
print(f"{'shape':>14} {'结果':>10} {'尾部缓冲':>14} {'覆盖率':>7}")
for s in [(4096, 1024), (997, 1024), (1003, 512), (1, 4096), (333, 1000), (77, 127)]:
    for B in (1024,):
        r = probe(add_fix, s, B, "cdiv", "fix")
        print(f"{str(s):>14} {r[1]:>10} {r[2]:>14} {r[3]:>7}")

print()
print("=" * 78)
print("带宽（M=4096 N=1024，理想 3*numel*4 = 50.3 MB）")
print("=" * 78)
n = 4096 * 1024
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")
o = torch.zeros(n, device="cuda")
ideal = 3 * n * 4


def bench(fn, label):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    it = 200
    for _ in range(it):
        fn()
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / it
    print(f"  {label:<24} {ms:.4f} ms   {ideal / ms / 1e6:6.1f} GB/s")


bench(lambda: torch.add(x, y, out=o), "torch x+y")
for B in (1024, 2048, 4096):
    bench(lambda B=B: add_fix[(triton.cdiv(n, B),)](x, y, o, n, B), f"triton BLOCK={B}")
