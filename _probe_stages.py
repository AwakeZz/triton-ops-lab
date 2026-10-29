import os, re, sys, torch, triton, triton.language as tl
from collections import Counter
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))
sys.path.insert(0, _HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("k12", os.path.join(_HERE,"kernels","12_gemm_pipe.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

M=N=K=1024
a = torch.randn(M,K,device="cuda",dtype=torch.float16)
b = torch.randn(K,N,device="cuda",dtype=torch.float16)
out = torch.empty(M,N,device="cuda",dtype=torch.float32)

print(f"{'stages':>7s} {'smem':>7s} {'regs':>5s} {'cp.async':>9s} {'commit':>7s} {'wait':>5s} {'ld.glob':>8s} {'bar.sync':>9s} {'ld.shared':>10s}")
for s in [1,2,3,4]:
    k = m.gemm_pipe_kernel[(16,16)](a,b,out,M,N,K,64,64,32,num_warps=4,num_stages=s)
    ptx = k.asm["ptx"]
    c = lambda p: len(re.findall(p, ptx))
    print(f"{s:7d} {k.metadata.shared:7d} {k.n_regs:5d} {c(r'cp.async.cg'):9d} {c(r'cp.async.commit_group'):7d} "
          f"{c(r'cp.async.wait_group'):5d} {c(r'ld.global'):8d} {c(r'bar.sync'):9d} {c(r'ld.shared'):10d}")
