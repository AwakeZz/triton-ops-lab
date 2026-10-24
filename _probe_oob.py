"""临时验证: 归一化 PTX 是否相同 + 不 mask 的越界读到底读到什么"""
import importlib.util
import os
import re

os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "_exp_mask.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def norm(p):
    p = re.sub(r"\bk_(pos|kw)\b", "KERN", p)
    p = re.sub(r"_exp_mask\.py:\d+:\d+", "LOC", p)
    p = re.sub(r"\$L__\w+", "LBL", p)
    return p


n1, n2 = norm(m.ptx_of(m.k_pos)), norm(m.ptx_of(m.k_kw))
print("[1] 归一化(抹掉函数名/行号/标号)后 PTX 相同?", n1 == n2)


@triton.jit
def dump(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)                # 完全不 mask
    tl.store(o_ptr + row * BLOCK + tl.arange(0, BLOCK), v)


N, BLOCK, M = 1000, 1024, 4
x = torch.ones(M * N, device="cuda")
o = torch.zeros(M * BLOCK, device="cuda")
dump[(M,)](x, o, N, BLOCK=BLOCK)
torch.cuda.synchronize()
o = o.view(M, BLOCK)
print(f"\n[4] 不 mask 读, 每行第 {N}..{N+3} 个位置读到:")
for r in range(M):
    print(f"    row {r}: {o[r, N:N + 4].tolist()}")
print("    最后一行确实读到了数组末尾之外的内存, Triton 不检查也不报错")
