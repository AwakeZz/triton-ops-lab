import os, re, torch, importlib.util
from collections import Counter
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))

def go(name, path, launch, kernname):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    launch(m)
    k = getattr(m, kernname)
    for _, ker in list(k.device_caches[0][0].items()):
        ptx = ker.asm["ptx"]
        print(f"\n=== {name}  regs={ker.n_regs} spill={ker.n_spills} smem={ker.metadata.shared} warps={ker.metadata.num_warps}")
        c = Counter()
        for line in ptx.splitlines():
            mm = re.search(r"\b(ld\.global|st\.global|cp\.async\.bulk|cp\.async)\S*", line)
            if mm:
                c[mm.group(0).rstrip(",;").split(" ")[0]] += 1
        for kk, v in sorted(c.items()):
            print(f"  {kk:48s} x{v}")
        return

M=N=K=1024
a = torch.randn(M,K,device="cuda",dtype=torch.float16)
b = torch.randn(K,N,device="cuda",dtype=torch.float16)
out = torch.empty(M,N,device="cuda",dtype=torch.float32)
go("10_gemm_tiled 32/32/32", "kernels/10_gemm_tiled.py", lambda m: m.run(a,b,out), "gemm_tiled_kernel")

x = torch.randn(1024,1024,device="cuda",dtype=torch.float16)
o8 = torch.empty(1024,1024,device="cuda",dtype=torch.float16)
go("08_transpose 32x32", "kernels/08_transpose.py", lambda m: m.run(x,o8), "transpose_kernel")

x1 = torch.randn(1024*1024, device="cuda", dtype=torch.float32)
y1 = torch.randn(1024*1024, device="cuda", dtype=torch.float32)
o1 = torch.empty(1024*1024, device="cuda", dtype=torch.float32)
go("01_add 1024", "kernels/01_add.py", lambda m: m.run(x1,y1,o1), "add_kernel")
