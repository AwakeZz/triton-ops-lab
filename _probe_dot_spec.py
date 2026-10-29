import os, re, torch, triton, triton.language as tl
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))

@triton.jit
def d2(a_ptr, b_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
       PREC: tl.constexpr, OUT_DT: tl.constexpr):
    om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
    a = tl.load(a_ptr + om[:, None] * K + ok[None, :])
    b = tl.load(b_ptr + ok[:, None] * N + on[None, :])
    if OUT_DT == 0:
        c = tl.dot(a, b, input_precision=PREC)
    else:
        c = tl.dot(a, b, input_precision=PREC, out_dtype=tl.float16)
    tl.store(o_ptr + om[:, None] * N + on[None, :], c)

@triton.jit
def d3(a_ptr, b_ptr, o_ptr, B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    ob = tl.arange(0, B); om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
    a = tl.load(a_ptr + ob[:, None, None] * (M * K) + om[None, :, None] * K + ok[None, None, :])
    b = tl.load(b_ptr + ob[:, None, None] * (K * N) + ok[None, :, None] * N + on[None, None, :])
    c = tl.dot(a, b)
    tl.store(o_ptr + ob[:, None, None] * (M * N) + om[None, :, None] * N + on[None, None, :], c)

print("=== 1. 最小尺寸 (fp16 输入) ===")
for M, N, K in [(8,8,8), (8,8,16), (16,16,8), (16,16,16), (16,16,32), (32,32,16), (64,64,16)]:
    try:
        a = torch.randn(M,K,device="cuda",dtype=torch.float16)
        b = torch.randn(K,N,device="cuda",dtype=torch.float16)
        o = torch.empty(M,N,device="cuda",dtype=torch.float32)
        d2[(1,)](a,b,o,M,N,K,"tf32",0)
        err = (o - a.float()@b.float()).abs().max().item()
        print(f"  M={M:3d} N={N:3d} K={K:3d}  OK     err={err:.2e}")
    except Exception as e:
        msg = str(e).replace("\n"," ")[:90]
        print(f"  M={M:3d} N={N:3d} K={K:3d}  FAIL   {msg}")

print("\n=== 2. fp32 输入 x 三种 input_precision (K=512 累加, 看速度与精度) ===")
M=N=512; K=1024
a32 = torch.randn(M,K,device="cuda",dtype=torch.float32)
b32 = torch.randn(K,N,device="cuda",dtype=torch.float32)
ref64 = (a32.double() @ b32.double()).float()
import statistics
for prec in ["tf32", "tf32x3", "ieee"]:
    try:
        o = torch.empty(M,N,device="cuda",dtype=torch.float32)
        k = d2[(1,)](a32,b32,o,M,N,K,prec,0)
        ptx = k.asm["ptx"]
        mma = sorted({x+y for x,y in re.findall(r"mma\.sync\.aligned\.(m\d+n\d+k\d+)\.([a-z0-9]+)", ptx)})
        for _ in range(5): d2[(1,)](a32,b32,o,M,N,K,prec,0)
        torch.cuda.synchronize()
        ts=[]
        for _ in range(20):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True)
            s.record(); d2[(1,)](a32,b32,o,M,N,K,prec,0); e.record()
            torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        ms = statistics.median(ts)
        err = (o-ref64).abs().max().item()
        print(f"  {prec:7s} {ms:8.4f} ms  {2*M*N*K/ms/1e9:7.2f} TFLOPS  err={err:.3e}  n_mma={len(re.findall('mma', ptx)):4d}  mma={','.join(mma) or 'none'}")
    except Exception as e:
        print(f"  {prec:7s} FAIL {str(e)[:80]}")

print("\n=== 3. fp16 输入时 input_precision 是否被忽略 ===")
a16 = a32.half(); b16 = b32.half()
for prec in ["tf32", "ieee"]:
    o = torch.empty(M,N,device="cuda",dtype=torch.float32)
    k = d2[(1,)](a16,b16,o,M,N,K,prec,0)
    mma = sorted({x+y for x,y in re.findall(r"mma\.sync\.aligned\.(m\d+n\d+k\d+)\.([a-z0-9]+)", k.asm["ptx"])})
    print(f"  fp16 + prec={prec:5s} -> mma={','.join(mma) or 'none'}   (fp16 的 mma 形状没变即说明被忽略)")

print("\n=== 4. out_dtype=tl.float16 ===")
o16 = torch.empty(M,N,device="cuda",dtype=torch.float16)
k = d2[(1,)](a16,b16,o16,M,N,K,"tf32",1)
print(f"  out_dtype=fp16 -> 输出 dtype={o16.dtype}, 误差={(o16.float()-a16.float()@b16.float()).abs().max().item():.3e}")

print("\n=== 5. 3D batched dot ===")
B,M2,N2,K2 = 4,64,64,32
ab = torch.randn(B,M2,K2,device="cuda",dtype=torch.float16)
bb = torch.randn(B,K2,N2,device="cuda",dtype=torch.float16)
ob = torch.empty(B,M2,N2,device="cuda",dtype=torch.float32)
d3[(1,)](ab,bb,ob,B,M2,N2,K2)
refb = torch.bmm(ab.float(), bb.float())
print(f"  batch={B} {M2}x{N2}x{K2}  误差={(ob-refb).abs().max().item():.3e}  (等价于 torch.bmm)")
