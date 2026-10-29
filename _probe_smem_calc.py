import os, math, torch, triton, importlib.util
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))
spec = importlib.util.spec_from_file_location("k12", os.path.join(_HERE,"kernels","12_gemm_pipe.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

SM, SMEM_SM, REGS_SM, THR_SM = 20, 102400, 65536, 1536
a = torch.randn(256,256,device="cuda",dtype=torch.float16)
b = torch.randn(256,256,device="cuda",dtype=torch.float16)
o = torch.empty(256,256,device="cuda",dtype=torch.float32)

print(f"{'BM/BN/BK':12s} {'w':>2s} {'s':>2s} | {'期望buf':>7s} {'smem实':>7s} {'buf数':>5s} {'占100KB':>8s} | {'regs':>4s} | {'限reg':>5s} {'限smem':>6s} {'限thr':>5s} | {'CTA':>3s} {'瓶颈':>6s}")
for BM,BN,BK,w,s in [(32,32,32,4,3),(64,64,32,4,1),(64,64,32,4,2),(64,64,32,4,3),(64,64,32,4,4),
                     (64,64,64,4,3),(96,96,32,4,3),(128,64,32,4,3),(64,128,32,4,3),
                     (128,128,32,8,3),(128,128,64,8,3),(128,128,32,8,4)]:
    try:
        k = m.gemm_pipe_kernel[(1,)](a,b,o,256,256,256,BM,BN,BK,num_warps=w,num_stages=s)
        regs, smem, thr = k.n_regs, k.metadata.shared, w*32
        per_buf = (BM*BK + BK*BN)*2
        nb = smem // per_buf
        lim_r = REGS_SM // (math.ceil(regs*thr/256)*256)
        lim_s = SMEM_SM // smem
        lim_t = THR_SM // thr
        cta = min(lim_r, lim_s, lim_t)
        who = "regs" if cta==lim_r else ("smem" if cta==lim_s else "thr")
        print(f"{BM}/{BN}/{BK:<7d} {w:2d} {s:2d} | {per_buf:7d} {smem:7d} {nb:5d} {smem/SMEM_SM*100:7.1f}% | {regs:4d} | {lim_r:5d} {lim_s:6d} {lim_t:5d} | {cta:3d} {who:>6s}")
    except Exception as e:
        print(f"{BM}/{BN}/{BK:<7d} {w:2d} {s:2d} | FAIL {str(e)[:60]}")
