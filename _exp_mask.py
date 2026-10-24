"""
mask=None / mask=mask / 位置参数传 mask 三者到底差在哪

跑: G:/venvs/triton/Scripts/python.exe _exp_mask.py
"""
import os

os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")
import re

import torch
import triton
import triton.language as tl

N = 1000          # 一行 1000 个数
BLOCK = 1024      # 但一次申请 1024 个位置 -> 尾部 24 个越界


# ---------------- 三种写法, 逻辑完全相同 ----------------
@triton.jit
def k_nomask(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, cols < N)          # 读有 mask
    tl.store(o_ptr + offs, v * 2.0)              # 写没 mask


@triton.jit
def k_pos(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK)
    m = tl.arange(0, BLOCK) < N
    v = tl.load(x_ptr + offs, m)                 # 位置参数
    tl.store(o_ptr + offs, v * 2.0, m)           # 位置参数


@triton.jit
def k_kw(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK)
    m = tl.arange(0, BLOCK) < N
    v = tl.load(x_ptr + offs, mask=m)            # 关键字
    tl.store(o_ptr + offs, v * 2.0, mask=m)      # 关键字


@triton.jit
def k_read_nomask(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    """读也不带 mask, 看会读到什么"""
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)                    # 读没 mask
    keep = tl.where(cols < N, v, -0.0)           # 只为了输出观察, 再 mask 掉
    tl.store(o_ptr + offs, keep, mask=cols < N)


def ptx_of(kern, dtype=torch.float32):
    x = torch.ones(N, device="cuda", dtype=dtype)
    o = torch.ones(N, device="cuda", dtype=dtype)
    k = kern.warmup(x, o, N, BLOCK=BLOCK, num_warps=4, grid=(1,))
    return k.asm["ptx"]


def count(ptx):
    ld = re.findall(r"\n\s*(?:@%\w+\s+)?ld\.global[\w.]*", ptx)
    st = re.findall(r"\n\s*(?:@%\w+\s+)?st\.global[\w.]*", ptx)
    return [s.strip() for s in ld], [s.strip() for s in st]


def main():
    print("=" * 72)
    print(f"N={N}, BLOCK={BLOCK}, 尾部 {BLOCK - N} 个位置越界")
    print("=" * 72)

    # ---------- 1. 位置参数 vs 关键字: 编译产物一样吗 ----------
    print("\n[1] 位置参数 mask  vs  关键字 mask=mask")
    p1 = ptx_of(k_pos)
    p2 = ptx_of(k_kw)
    print(f"  PTX 完全相同? {p1 == p2}")
    l1, s1 = count(p1)
    print(f"  load : {l1}")
    print(f"  store: {s1}")
    print("  -> 第二种参数之后全都能用位置传, 两者是同一件事, 只差可读性")

    # ---------- 2. store 带不带 mask: PTX 差在哪 ----------
    print("\n[2] tl.store 带不带 mask, 生成的 PTX 差在哪")
    for name, k in [("不带 mask", k_nomask), ("带 mask", k_kw)]:
        p = ptx_of(k)
        _, st = count(p)
        predicated = all(s.startswith("@") for s in st)
        print(f"  {name:<10}: {st}   谓词化={predicated}")
    print("  不带 mask 的 st.global 前面没有 @%p -> 1024 个 lane 全部真的写")
    print("  带 mask 的 st.global 前面有 @%p   -> 只有谓词为真的 lane 才写")

    # ---------- 3. 不写 mask = 真的写出去 ----------
    print("\n[3] 实际结果: out 前后各垫一块哨兵, 看有没有被踩")
    M = 4
    for name, k in [("不带 mask", k_nomask), ("带 mask", k_kw)]:
        pad = 32
        buf = torch.full((M * N + 2 * pad,), 9.0, device="cuda")
        x = torch.ones(M, N, device="cuda")
        out = buf[pad:pad + M * N]
        k[(M,)](x, out, N, BLOCK=BLOCK)
        torch.cuda.synchronize()

        head = buf[:pad]
        tail = buf[pad + M * N:]
        dirty = int((head != 9.0).sum()) + int((tail != 9.0).sum())
        wrong = int((out.view(M, N) != 2.0).sum())
        print(f"  {name:<10}: 哨兵被改 {dirty} 个   out 里错 {wrong} 个")

    # ---------- 4. 读不写 mask, 会读到什么 ----------
    print("\n[4] tl.load 不带 mask 会读到什么(超出数组的部分)")
    buf2 = torch.full((M * N + 64,), 7.0, device="cuda")
    x2 = torch.ones(M, N, device="cuda")
    buf2[:M * N] = x2.flatten()
    buf2[M * N:] = 7.0                                  # 哨兵: 位于 x 之后的内存
    o = torch.zeros(M * N, device="cuda")
    k_read_nomask[(M,)](buf2[:M * N], o, N, BLOCK=BLOCK)
    torch.cuda.synchronize()
    print(f"  越界读到的值出现频率: {torch.unique(o).tolist()}")
    print("  1.0 = 真实数据, 7.0 = 读到了数组外面的哨兵 -> 不报错, 直接算进了结果")

    # ---------- 5. other 只在有 mask 时才有意义 ----------
    print("\n[5] mask 为假的位置, 值由 other 决定(默认 0)")
    x3 = torch.arange(1, BLOCK + 1, dtype=torch.float32, device="cuda")[:N]
    o3 = torch.zeros(BLOCK, device="cuda")
    k_kw[(1,)](x3, o3, N, BLOCK=BLOCK)
    torch.cuda.synchronize()
    print(f"  输出尾部 8 个(不该被写): {o3[N:N + 8].tolist()}")


if __name__ == "__main__":
    main()
