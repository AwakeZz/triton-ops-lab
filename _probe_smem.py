import os, re, torch, importlib.util
from collections import Counter
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))
spec = importlib.util.spec_from_file_location("k10", os.path.join(_HERE,"kernels","10_gemm_tiled.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

M=N=K=1024
a = torch.randn(M,K,device="cuda",dtype=torch.float16)
b = torch.randn(K,N,device="cuda",dtype=torch.float16)
out = torch.empty(M,N,device="cuda",dtype=torch.float32)

print(f"{'tile':12s} {'regs':>5s} {'spill':>6s} {'smem':>7s} {'ld.sh':>6s} {'st.sh':>6s} {'bar.sync':>8s} {'cp.async':>8s}  TFLOPS")
for BM,BN,BK in [(32,32,32),(64,64,32),(64,64,64),(64,32,32)]:
    grid=(triton.cdiv(M,BM) if False else (M+BM-1)//BM, (N+BN-1)//BN)
    k = m.gemm_tiled_kernel[grid](a,b,out,M,N,K,BM,BN,BK)
    ptx = k.asm["ptx"]
    cnt = lambda p: len(re.findall(p, ptx))
    import time
    for _ in range(3): m.gemm_tiled_kernel[grid](a,b,out,M,N,K,BM,BN,BK)
    torch.cuda.synchronize()
    ts=[]
    for _ in range(30):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); m.gemm_tiled_kernel[grid](a,b,out,M,N,K,BM,BN,BK); e.record()
        torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); ms = ts[len(ts)//2]
    tf = 2*M*N*K/ms/1e9
    print(f"{BM}/{BN}/{BK:<6d} {k.n_regs:5d} {k.n_spills:6d} {k.metadata.shared:7d} "
          f"{cnt(r'ld.shared'):6d} {cnt(r'st.shared'):6d} {cnt(r'bar.sync'):8d} {cnt(r'cp.async.cg'):8d}  {tf:6.2f}")
