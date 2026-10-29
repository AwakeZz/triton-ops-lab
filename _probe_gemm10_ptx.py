import os, re, torch, triton, importlib.util
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))
spec = importlib.util.spec_from_file_location("k10", os.path.join(_HERE, "kernels", "10_gemm_tiled.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

M=N=K=1024
a = torch.randn(M,K,device="cuda",dtype=torch.float16)
b = torch.randn(K,N,device="cuda",dtype=torch.float16)
out = torch.empty(M,N,device="cuda",dtype=torch.float32)
m.run(a,b,out)
ref = a.float() @ b.float()
print("max|diff| =", (out-ref).abs().max().item())

k = m.gemm_tiled_kernel
for key, ker in k.device_caches[0][0].items():
    pass
# 从 cache 里捞编译产物
import triton.compiler
found=False
for kk, ker in list(k.device_caches[0][0].items()):
    asm = ker.asm
    ptx = asm["ptx"]
    print("=== kernel", kk[1] if isinstance(kk,tuple) else kk)
    print("  n_regs:", ker.n_regs, " n_spills:", ker.n_spills, " shared:", ker.metadata.shared)
    for pat in ["mma.sync", "wgmma", "fma.rn.f32", "hmma", "ld.shared"]:
        print(f"  {pat}: {len(re.findall(re.escape(pat), ptx))}")
    found=True
if not found:
    print("no cache hit")
