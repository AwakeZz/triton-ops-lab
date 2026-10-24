"""
为什么 05_cumsum 在 N 不是 2 的幂时结果错?
    store 少了 mask -> 每个 program 写了 BLOCK_N 个数, 但只有 N 个属于自己,
    多出来的 BLOCK_N - N 个"溢出"到了下一行的开头。

跑: G:/venvs/triton/Scripts/python.exe _exp_cumsum_store.py
"""
import os

os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")
import torch
import triton
import triton.language as tl


@triton.jit
def cumsum_nomask(x_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.cumsum(v, 0))          # ← 缺 mask


@triton.jit
def cumsum_masked(x_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    v = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.cumsum(v, 0), mask=mask)   # ← 有 mask


def main():
    M, N = 4, 1000
    BLOCK_N = triton.next_power_of_2(N)
    spill = BLOCK_N - N
    x = torch.ones(M, N, device="cuda")            # 全 1, 前缀和 = 1,2,3,...,N

    print(f"M={M}  N={N}  BLOCK_N={BLOCK_N}  每行溢出 {spill} 个位置")
    print(f"输入全 1, 所以第 i 行正确的输出应该是 [1, 2, 3, ..., {N}]\n")

    for name, kern in [("缺 mask", cumsum_nomask), ("有 mask", cumsum_masked)]:
        out = torch.full((M * N,), -1.0, device="cuda")
        kern[(M,)](x, out, N, BLOCK_N=BLOCK_N)
        torch.cuda.synchronize()
        o = out.view(M, N)

        ok = torch.allclose(o, x.cumsum(dim=1))
        print(f"[{name}] 整块正确? {ok}")

        # 看第 1 行开头那几个位置
        print(f"  第 1 行前 {spill + 2} 个值 : "
              f"{[round(t, 1) for t in o[1, :spill + 2].tolist()]}")
        print(f"  (正确应为 1.0, 2.0, ... ; 若出现 {float(N)}.0 说明被上一行的行和覆盖)")

        # 找第 1 行从哪个位置开始回到正确值
        ref1 = x[1].cumsum(0)
        bad = (o[1] != ref1).nonzero().flatten()
        if len(bad):
            print(f"  第 1 行错的位置: 前 {len(bad)} 个 (下标 {bad[0].item()}..{bad[-1].item()})")
        else:
            print("  第 1 行完全正确")
        print()

    # 越界写: 最后一个 program 会写到 out 之外
    last_start = (M - 1) * N
    print(f"越界检查: 最后一个 program 写 [{last_start}, {last_start + BLOCK_N - 1}]")
    print(f"          out 长度 {M * N}, 越界 {last_start + BLOCK_N - M * N} 个 float")
    print("          -> PyTorch 分配器一般有富余, 所以不会崩, 只是静默改写别人内存")


if __name__ == "__main__":
    main()
