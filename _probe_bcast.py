import torch, triton, triton.language as tl

@triton.jit
def a_both(x_ptr, N, BM: tl.constexpr, BN: tl.constexpr):
    m = tl.arange(0, BM); n = tl.arange(0, BN)
    addr = m[:, None] * N + n[None, :]
    tl.static_print("both None       ", addr.shape)

@triton.jit
def b_only_m(x_ptr, N, BM: tl.constexpr, BN: tl.constexpr):
    m = tl.arange(0, BM); n = tl.arange(0, BN)
    addr = m[:, None] * N + n
    tl.static_print("only m[:,None]  ", addr.shape)

@triton.jit
def c_only_n(x_ptr, N, BM: tl.constexpr, BN: tl.constexpr):
    m = tl.arange(0, BM); n = tl.arange(0, BN)
    addr = m * N + n[None, :]
    tl.static_print("only n[None,:]  ", addr.shape)

x = torch.randn(64, 128, device="cuda")
for f, name in [(a_both,"A"), (b_only_m,"B"), (c_only_n,"C")]:
    print("=== " + name)
    try:
        f[(1,)](x, 128, 16, 64)
    except Exception as e:
        print("   编译失败:", str(e).split("\n")[-1][:110])
