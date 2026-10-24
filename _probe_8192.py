"""检查 05_cumsum 当前版本(N=8192 硬编码 + 8*arange 的 mask)到底做了什么"""
import importlib.util
import os

os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("c05", os.path.join(HERE, "kernels", "05_cumsum.py"))
c05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c05)


def probe(M, N, tag):
    print(f"\n--- {tag}: 输入 x = ({M}, {N}) ---")
    x = torch.ones(M, N, device="cuda")
    out = torch.full((M, N), -1.0, device="cuda")

    # 复现 run() 里的 grid 计算
    n = x.numel()
    grid = (triton.cdiv(n, x.shape[1]),)
    print(f"  run() 里: numel={n}  grid={grid[0]} 个 program")
    print(f"  但 kernel 收到的 N 是写死的 8192, 而真实 N = {N}")
    print(f"  每个 program 最多写到的下标 = ({grid[0]}-1)*8192 + 1023 = "
          f"{(grid[0] - 1) * 8192 + 1023}    数组长度 = {n}")

    try:
        c05.run(x, out)
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"  报错: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        return

    ref = x.cumsum(dim=1)
    written = int((out != -1.0).sum())
    ok = int((out == ref).sum())
    print(f"  被写过的元素: {written}/{n}  其中正确: {ok}")
    print(f"  第 0 行前 6 个: {out[0, :6].tolist()}")
    print(f"  第 0 行第 1018..1023 个: {out[0, 1018:1024].tolist()}")


def mask_truth_table():
    print("\n--- kernel 里那行 mask 到底算出什么 ---")
    for N in (1024, 8192):
        ar = torch.arange(0, 1024, device="cuda")
        m = (8 * ar) < N
        true_cnt = int(m.sum())
        first_false = int((~m).nonzero()[0]) if not m.all() else -1
        print(f"  N={N:>4}: mask 为真 {true_cnt}/1024 个"
              f"{f'，从第 {first_false} 个开始为假' if first_false >= 0 else '（全真, 等于没写 mask）'}")


if __name__ == "__main__":
    mask_truth_table()
    probe(2, 8192, "真实 N=8192")
    probe(2, 1024, "check.py 实际用的 N=1024")
