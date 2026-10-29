import os, torch, triton, importlib.util, statistics
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))
spec = importlib.util.spec_from_file_location("k10", os.path.join(_HERE, "kernels", "10_gemm_tiled.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

M=N=K=1024
a = torch.randn(M,K,device="cuda",dtype=torch.float16)
b = torch.randn(K,N,device="cuda",dtype=torch.float16)
out = torch.empty(M,N,device="cuda",dtype=torch.float32)
ref = a.float() @ b.float()
flops = 2*M*N*K

for BM,BN,BK in [(32,32,32),(64,64,32),(64,32,32),(32,64,32),(64,64,64),(128,64,32),(32,32,64)]:
    try:
        grid = (triton.cdiv(M,BM), triton.cdiv(N,BN))
        kern = m.gemm_tiled_kernel[grid](a,b,out,M,N,K,BM,BN,BK)
        for _ in range(3): m.gemm_tiled_kernel[grid](a,b,out,M,N,K,BM,BN,BK)
        torch.cuda.synchronize()
        ts=[]
        for _ in range(20):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True)
            s.record(); m.gemm_tiled_kernel[grid](a,b,out,M,N,K,BM,BN,BK); e.record()
            torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        ms = statistics.median(ts)
        err = (out-ref).abs().max().item()
        print(f"{BM}/{BN}/{BK}  {ms:7.4f} ms  {flops/ms/1e9:6.2f} TFLOPS ({flops/ms/1e9/18.2*100:5.1f}%)  regs={kern.n_regs} spill={kern.n_spills} smem={kern.metadata.shared}  err={err:.2e}")
    except Exception as ex:
        print(f"{BM}/{BN}/{BK}  FAIL: {str(ex)[:110]}")
