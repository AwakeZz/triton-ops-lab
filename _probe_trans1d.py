import torch
import triton
import triton.language as tl


@triton.jit
def t_1d(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, tl.trans(v))


x = torch.arange(8, device="cuda", dtype=torch.float32)
o = torch.empty(8, device="cuda")
try:
    t_1d[(1,)](x, o, 8)
    print("1D trans 竟然通过了:", o.tolist())
except Exception as e:
    print(type(e).__name__)
    print(str(e)[:600])
