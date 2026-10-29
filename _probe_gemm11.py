import os, re, sys, torch, triton, triton.language as tl, statistics
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))

@triton.jit
def kern(a_ptr, b_ptr, out_ptr, M, N, K,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
         USE_F32: tl.constexpr, USE_DOT: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, tl.cdiv(K, BK)):
        offs_k = k0 * BK + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        if USE_DOT:
            if USE_F32:
                acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc)
            else:
                acc = tl.dot(a, b, acc)
        else:
            acc += tl.sum(a.to(tl.float32)[:, :, None] * b.to(tl.float32)[None, :, :], 1)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

SZ = int(os.environ.get('SZ', '1024'))
M=N=K=SZ
a = torch.randn(M,K,device="cuda",dtype=torch.float16)
b = torch.randn(K,N,device="cuda",dtype=torch.float16)
out = torch.empty(M,N,device="cuda",dtype=torch.float32)
ref = a.float() @ b.float()
flops = 2*M*N*K

BM,BN,BK,use_f32,use_dot = int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
warps = int(sys.argv[6]) if len(sys.argv)>6 else 4
stages = int(sys.argv[7]) if len(sys.argv)>7 else 1

grid = (triton.cdiv(M,BM), triton.cdiv(N,BN))
k = kern[grid](a,b,out,M,N,K,BM,BN,BK,use_f32,use_dot,num_warps=warps,num_stages=stages)
ptx = k.asm["ptx"]
mma = re.findall(r"mma\.sync\.aligned\.(m\d+n\d+k\d+)\.([a-z0-9]+)", ptx)
mma_desc = ",".join(sorted({f"{x[0]}.{x[1]}" for x in mma})) or "none"

for _ in range(5): kern[grid](a,b,out,M,N,K,BM,BN,BK,use_f32,use_dot,num_warps=warps,num_stages=stages)
torch.cuda.synchronize()
best=None; worst=None
for t in range(3):
    ts=[]
    for _ in range(30):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); kern[grid](a,b,out,M,N,K,BM,BN,BK,use_f32,use_dot,num_warps=warps,num_stages=stages); e.record()
        torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    med = statistics.median(ts)
    best = med if best is None else min(best,med)
    worst = med if worst is None else max(worst,med)

err = (out-ref).abs().max().item()
tag = ("dot" if use_dot else "sum") + ("+f32cast" if use_f32 else "")
print(f"{BM}/{BN}/{BK} {tag:12s} w{warps} s{stages} | {best:7.4f} ms {flops/best/1e9:6.2f} TF ({(flops/best/1e9/18.2)*100:5.1f}%) | "
      f"mma={mma_desc:22s} n={len(mma):3d} fma={len(re.findall(r'fma.rn.f32', ptx)):4d} regs={k.n_regs} spill={k.n_spills} | err={err:.2e} | jitter={(worst-best)/best*100:.1f}%")
