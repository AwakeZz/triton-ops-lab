"""[:, None] / [None, :] 的形状与广播语义实验"""
import torch
import triton
import triton.language as tl

print("=" * 62)
print("第一部分：[None] 是一个省略号的别名，等价于 torch.unsqueeze")
print("=" * 62)

a = torch.arange(6)
print("a            shape =", tuple(a.shape), " a =", a.tolist())
print("a[:, None]   shape =", tuple(a[:, None].shape))
print("a[None, :]   shape =", tuple(a[None, :].shape))
print("a[None]      shape =", tuple(a[None].shape), "  ← 无冒号时 None 只是前面补轴")
print("a[None,None,:] shape =", tuple(a[None, None, :].shape))

# 三种写法的等价性
b1 = a[:, None]
b2 = a.unsqueeze(1)
b3 = a.view(6, 1)
print("\n[:, None] == unsqueeze(1):", torch.equal(b1, b2))
print("[:, None] == view(6,1)   :", torch.equal(b1, b3))

print("\n" + "=" * 62)
print("第二部分：元素个数不变，只是重排成矩阵")
print("=" * 62)
print("a[:, None] 实际内容:")
for i in range(6):
    print(f"  [{i}] ->", a[:, None][i].tolist())
print("numel: a =", a.numel(), " a[:, None] =", a[:, None].numel())

print("\n" + "=" * 62)
print("第三部分：外积 —— 两个 1D 向量拼出二维地址矩阵")
print("=" * 62)
m = torch.tensor([0, 1, 2])   # 代表 offs_m
n = torch.tensor([10, 11, 12, 13])  # 代表 offs_n
outer = m[:, None] * 100 + n[None, :]
print("offs_m =", m.tolist(), "  offs_n =", n.tolist())
print("offs_m[:,None].shape =", tuple(m[:, None].shape), "  offs_n[None,:].shape =", tuple(n[None, :].shape))
print("\noffs_m[:,None]*100 + offs_n[None,:]:")
print(outer)
print("shape =", tuple(outer.shape))

print("\n两个向量直接相加会怎样:")
try:
    r = m + n
    print(r)
except Exception as e:
    print("  报错:", type(e).__name__, e)
print("  注意：m 和 n 长度不同 → 报错；就算长度相同也是逐元素相加，不是外积")

print("\n" + "=" * 62)
print("第四部分：真实二维矩阵的取址（转置 kernel 里那一行）")
print("=" * 62)
M, N = 3, 4
x = torch.arange(M * N).reshape(M, N)
print("x (3x4):")
print(x)

offs_m = torch.arange(M)
offs_n = torch.arange(N)
addr = offs_m[:, None] * N + offs_n[None, :]
print("\noffs_m[:,None]*N + offs_n[None,:]  (生成线性偏移矩阵):")
print(addr)
print("\n这种取址方式能否还原 x：", torch.equal(addr, x))

print("\n换成转置的地址 n*M + m:")
addr_t = offs_n[:, None] * M + offs_m[None, :]
print(addr_t)
print("形状:", tuple(addr_t.shape), " ← (N, M)，和读的时候 (M, N) 不一样")

print("\n" + "=" * 62)
print("第五部分：广播失败的典型案例（mask 少加一个轴）")
print("=" * 62)
offs_m = torch.arange(32)
offs_n = torch.arange(64)
mask_bad = offs_m < 20            # (32,)
mask_good = (offs_m[:, None] < 20) & (offs_n[None, :] < 50)  # (32,64)
print("(32,) 的 mask 直接配 (32,64) 的地址:")
try:
    v = x.new_zeros(32, 64)[:, :] + 0
    _ = v[mask_bad]
    print("  torch 索引不报错但语义完全错位")
except Exception as e:
    print("  ", type(e).__name__, e)
print("正确 mask shape =", tuple(mask_good.shape))
print("mask_good 为 True 的元素个数 =", int(mask_good.sum()), " 期望 = 20*50 =", 20 * 50)

print("\n" + "=" * 62)
print("第六部分：Triton 里的报错长什么样")
print("=" * 62)

@triton.jit
def bad_ptr_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    # 少了 [:, None]：一维地址 + 意图二维
    v = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, v)

@triton.jit
def ok_2d_kernel(x_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    addr = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    v = tl.load(x_ptr + addr)
    tl.store(out_ptr + addr, v + 1.0)

@triton.jit
def trans_kernel(x_ptr, out_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    v = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    tl.store(out_ptr + offs_n[:, None] * M + offs_m[None, :], tl.trans(v))

M, N = 1024, 512
x = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
o = torch.empty(N, M, device="cuda", dtype=torch.float32)
BF = {"BM": 32, "BN": 32}
try:
    trans_kernel[(M // 32, N // 32)](x, o, M, N, 32, 32)
    ok = torch.equal(o, x.t().contiguous())
    print("2D 转置 kernel 正确:", ok)
except Exception as e:
    print("2D 报错:", type(e).__name__, str(e)[:200])

print("\n1D 张量调 tl.trans 的报错:")
try:
    trans_kernel[(1, 1)](x.reshape(-1), o.reshape(-1), 32, 32, 32, 32)
    print("  没报错？")
except Exception as e:
    print(" ", type(e).__name__, str(e)[:160])

print("\n" + "=" * 62)
print("第七部分：容易写错的两个变体")
print("=" * 62)
a = torch.arange(6)
print("a[:, None]  shape =", tuple(a[:, None].shape), " ← 插入到第 1 轴（末尾）")
print("a[None, :]  shape =", tuple(a[None, :].shape), " ← 插入到第 0 轴（开头）")
a3 = torch.arange(24).reshape(2, 3, 4)
print("3D a.shape =", tuple(a3.shape))
print("a3[:, None]     shape =", tuple(a3[:, None].shape), " ← 在位置 1 插入")
print("a3[..., None]   shape =", tuple(a3[..., None].shape), " ← 在最后插入")
print("a3[:, None, :]  shape =", tuple(a3[:, None, :].shape))
try:
    a3[:, None, None]
    print("a3[:, None, None] shape =", tuple(a3[:, None, None].shape))
except Exception as e:
    print(e)
