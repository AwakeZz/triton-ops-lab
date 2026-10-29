import os, re, torch, importlib.util
from collections import Counter
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))

def stat(ptx, tag):
    print(f"\n### {tag}")
    c = Counter()
    for line in ptx.splitlines():
        s = line.strip()
        m = re.match(r"(ld\.global\S*|st\.global\S*|cp\.async\S*|ld\.shared\S*|st\.shared\S*|ldmatrix\S*|mbarrier\S*)", s)
        if m:
            op = m.group(1)
            # 归一化: 保留 op 名 + 向量宽度
            v = re.search(r"(\.v[248])", op)
            c[op.split(".")[0] + "." + op.split(".")[1] + (v.group(1) if v else ".v1") + "." + op.split(".")[-1]] += 1
    for k, v in sorted(c.items()):
        print(f"  {k:34s} x{v}")

def go(name, path, launch, kernname):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    launch(m)
    k = getattr(m, kernname)
    for _, ker in list(k.device_caches[0][0].items()):
        print(f"\n=== {name}  regs={ker.n_regs} spill={ker.n_spills} smem={ker.metadata.shared} warps={ker.metadata.num_warps}")
        stat(ker.asm["ptx"], "memory ops")
        return

M=N=K=1024
a = torch.randn(M,K,device="cuda",dtype=torch.float16)
b = torch.randn(K,N,device="cuda",dtype=torch.float16)
out = torch.empty(M,N,device="cuda",dtype=torch.float32)
go("10_gemm_tiled 32/32/32", os.path.join(_HERE,"kernels","10_gemm_tiled.py"),
   lambda m: m.run(a,b,out), "gemm_tiled_kernel")

x = torch.randn(1024,1024,device="cuda",dtype=torch.float16)
o8 = torch.empty(1024,1024,device="cuda",dtype=torch.float16)
go("08_transpose 32x32", os.path.join(_HERE,"kernels","08_transpose.py"),
   lambda m: m.run(x,o8), "transpose_kernel")

o1 = torch.empty(1024*1024, device="cuda", dtype=torch.float32)
x1 = torch.randn(1024*1024, device="cuda", dtype=torch.float32)
y1 = torch.randn(1024*1024, device="cuda", dtype=torch.float32)
go("01_add 1024", os.path.join(_HERE,"kernels","01_add.py"),
   lambda m: m.run(x1,y1,o1), "add_kernel")
