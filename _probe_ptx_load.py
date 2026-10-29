import os, re, torch, triton, importlib.util
from collections import Counter
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, ".triton-cache"))

def shape_of_ld(ptx):
    """统计 ld.global / st.global 的向量宽度分布"""
    c = Counter()
    for m in re.finditer(r"(ld|st)\.global(?:\.\w+)*?(\.v2|\.v4)?\.(b16|b32|b64|u16|u32|f16|f32)", ptx):
        w = m.group(2) or ".v1"
        c[f"{m.group(1)}.global{w}.{m.group(3)}"] += 1
    return c

def dump(name, mod_path, launch):
    spec = importlib.util.spec_from_file_location(name, mod_path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    launch(m)
    for _, ker in list(getattr(m, [n for n in dir(m) if n.endswith("kernel")][0]).device_caches[0][0].items()):
        print(f"\n### {name}   regs={ker.n_regs} spill={ker.n_spills} smem={ker.metadata.shared} warps={ker.metadata.num_warps}")
        for k, v in sorted(shape_of_ld(ker.asm["ptx"]).items()):
            print(f"    {k:28s} x{v}")
        return

M=N=K=1024
a = torch.randn(M,K,device="cuda",dtype=torch.float16)
b = torch.randn(K,N,device="cuda",dtype=torch.float16)
out = torch.empty(M,N,device="cuda",dtype=torch.float32)

# --- 10: a 行内连续 (offs_m[:,None]*K + offs_k[None,:]) ---
dump("10_gemm_tiled 32/32/32", os.path.join(_HERE,"kernels","10_gemm_tiled.py"),
     lambda m: m.run(a,b,out))

# --- 08 转置: load 行内连续, store 跨行 (offs_n[:,None]*M + offs_m[None,:]) ---
x = torch.randn(1024,1024,device="cuda",dtype=torch.float16)
o8 = torch.empty(1024,1024,device="cuda",dtype=torch.float16)
dump("08_transpose 32x32", os.path.join(_HERE,"kernels","08_transpose.py"),
     lambda m: m.run(x,o8))
