import os, sys, torch, triton, importlib.util, statistics
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))
spec = importlib.util.spec_from_file_location("k10", os.path.join(_HERE,"kernels","10_gemm_tiled.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

M=N=K=1024
a = torch.randn(M,K,device="cuda",dtype=torch.float16)
b = torch.randn(K,N,device="cuda",dtype=torch.float16)
out = torch.empty(M,N,device="cuda",dtype=torch.float32)
flops = 2*M*N*K

BM,BN,BK = [int(x) for x in sys.argv[1:4]]
for trial in range(3):
    grid = (triton.cdiv(M,BM), triton.cdiv(N,BN))
    for _ in range(5): m.gemm_tiled_kernel[grid](a,b,out,M,N,K,BM,BN,BK)
    torch.cuda.synchronize()
    ts=[]
    for _ in range(30):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); m.gemm_tiled_kernel[grid](a,b,out,M,N,K,BM,BN,BK); e.record()
        torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort()
    print(f"trial{trial}: med={statistics.median(ts):7.4f} min={ts[0]:7.4f} max={ts[-1]:7.4f} ms  "
          f"-> {flops/statistics.median(ts)/1e9:6.2f} TFLOPS")
