import os, torch, triton, importlib.util, math
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))
spec = importlib.util.spec_from_file_location("k12", os.path.join(_HERE,"kernels","12_gemm_pipe.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

SM, SMEM_SM, REGS_SM, THREADS_SM = 20, 102400, 65536, 1536
a = torch.randn(128,128,device="cuda",dtype=torch.float16)
b = torch.randn(128,128,device="cuda",dtype=torch.float16)
o = torch.empty(128,128,device="cuda",dtype=torch.float32)

print(f"{'tile':18s} {'regs':>5s} {'smem':>7s} {'thr':>4s} | {'限regs':>6s} {'限smem':>6s} {'限thr':>6s} -> {'CTA/SM':>6s} {'slot/wave':>9s}")
info={}
for BM,BN,BK,w in [(64,64,32,4),(128,128,32,8)]:
    k = m.gemm_pipe_kernel[(1,)](a,b,o,128,128,128,BM,BN,BK,num_warps=w,num_stages=3)
    regs, smem, thr = k.n_regs, k.metadata.shared, w*32
    # 寄存器按 warp 分配粒度 256 round up
    r_per_cta = math.ceil(regs*thr/256)*256
    lim_r = REGS_SM // r_per_cta
    lim_s = SMEM_SM // smem
    lim_t = THREADS_SM // thr
    cta = min(lim_r, lim_s, lim_t)
    info[(BM,BN)] = (cta, cta*SM)
    print(f"{BM}/{BN}/{BK} w{w:<10d} {regs:5d} {smem:7d} {thr:4d} | {lim_r:6d} {lim_s:6d} {lim_t:6d} -> {cta:6d} {cta*SM:9d}")

print()
print(f"{'size':>6s} | {'64/64 grid':>11s} {'wave':>5s} {'fill':>6s} | {'128/128 grid':>13s} {'wave':>5s} {'fill':>6s}")
for SZ in [1024,2048,4096]:
    row=f"{SZ:6d} |"
    for (BM,BN) in [(64,64),(128,128)]:
        g = (SZ//BM)*(SZ//BN)
        slot = info[(BM,BN)][1]
        waves = math.ceil(g/slot)
        fill = g/(waves*slot)*100
        row += f" {g:6d} {waves:8d} {fill:5.1f}% |"
    print(row)

print()
print("算术强度 I = BM*BN/(BM+BN):  64/64 ->", 64*64//128, "  128/128 ->", 128*128//256)
print("矩阵体量 (fp16, A+B):  1024 ->", 2*1024*1024*2/1e6, "MB   2048 ->", 2*2048*2048*2/1e6, "MB   4096 ->", 2*4096*4096*2/1e6, "MB   (L2 = 2 MB)")
