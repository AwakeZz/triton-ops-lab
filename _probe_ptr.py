"""
实验: x_ptr 和 x、output_ptr 和 output 到底是什么关系。

三个问题:
  1. x_ptr + offsets 里的 "+1" 是加一个元素还是加一个字节?
  2. 值向量 x 读出来之后, 还记不记得自己来自哪个地址?
  3. tl.store 写到哪里, 由第一个参数还是第二个参数决定?

用法:
    G:/venvs/triton/Scripts/python.exe _probe_ptr.py
"""
import os

os.environ.setdefault("TRITON_CACHE_DIR", "G:/triton-cache")

import torch
import triton
import triton.language as tl


# ---- 实验 1: 把指针当整数打印出来 ------------------------------------------
@triton.jit
def probe_addr(x_ptr, addr_out, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    p = x_ptr + offs                  # 指针向量
    tl.store(addr_out + offs, p.to(tl.int64), mask=offs < n)


# ---- 实验 2/3: 指针决定去哪, 值只是数据 -----------------------------------
@triton.jit
def add_normal(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def add_reverse_y(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """x 正向读, y 反向读, 输出正向写 —— 完全合法"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + (n - 1 - offs), mask=mask)   # 反向
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def store_x_only(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """算对了 x+y 但只存 x —— 写进去的是第一个值参数, 不是算式的"结果" """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    acc = x + y
    tl.store(out_ptr + offs, x, mask=mask)   # 故意存 x


@triton.jit
def store_shifted(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """值算对, 但写到错开的地址 —— 目的地只由第一个参数决定"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + (offs + 1) % n, x + y, mask=mask)   # 右移一格


def main():
    torch.manual_seed(0)
    print("=" * 70)
    print("实验 1: x_ptr + 1 到底加了什么")
    print("=" * 70)

    x = torch.arange(8, dtype=torch.float32, device="cuda")
    addr = torch.zeros(8, dtype=torch.int64, device="cuda")
    probe_addr[(1,)](x, addr, 8, BLOCK=8)
    torch.cuda.synchronize()

    base = x.data_ptr()
    addrs = addr.cpu().tolist()
    print(f"  torch 侧 x.data_ptr() = {base}  (0x{base:x})")
    print(f"  kernel 里 (x_ptr + offs).to(int64) 的返回值:")
    for i, a in enumerate(addrs):
        print(f"    offs={i}: {a}  (0x{a:x})   减去 base = {a - base} 字节")
    print(f"\n  结论: 相邻差 {addrs[1] - addrs[0]} 字节 = float32 的 4 字节")
    print(f"        所以 x_ptr + 1 是 '下一个元素', 不是 '下一个字节'")
    print(f"        指针的类型 x_ptr 是 *fp32, 编译器按 4 字节缩放")

    print("\n" + "=" * 70)
    print("实验 2: 值向量还记不记得自己的地址")
    print("=" * 70)
    n = 8
    a = torch.arange(n, dtype=torch.float32, device="cuda")
    b = torch.arange(100, 100 + n, dtype=torch.float32, device="cuda")
    out = torch.zeros(n, device="cuda")

    add_normal[(1,)](a, b, out, n, BLOCK=8)
    torch.cuda.synchronize()
    print(f"  a          = {a.cpu().tolist()}")
    print(f"  b          = {b.cpu().tolist()}")
    print(f"  a + b      = {out.cpu().tolist()}   <- add_normal")

    out.zero_()
    add_reverse_y[(1,)](a, b, out, n, BLOCK=8)
    torch.cuda.synchronize()
    expect = (a + b.flip(0))
    print(f"\n  b 反向读   = {b.flip(0).cpu().tolist()}")
    print(f"  a + b[::-1]= {out.cpu().tolist()}   <- add_reverse_y")
    print(f"  正确?        {torch.allclose(out, expect)}")
    print(f"\n  结论: x 和 y 从不同方向取数据, 相加后照样能写回正向位置。")
    print(f"        值向量里没有地址信息, 位置是靠指针那边的 offsets 维持的。")

    print("\n" + "=" * 70)
    print("实验 3: store 写到哪里, 由谁决定")
    print("=" * 70)
    out.zero_()
    store_x_only[(1,)](a, b, out, n, BLOCK=8)
    torch.cuda.synchronize()
    print(f"  算了 x+y 但 store 传 x : {out.cpu().tolist()}")
    print(f"  a                      : {a.cpu().tolist()}")

    out.zero_()
    store_shifted[(1,)](a, b, out, n, BLOCK=8)
    torch.cuda.synchronize()
    print(f"\n  算对但写到 offs+1      : {out.cpu().tolist()}")
    print(f"  正确结果 a+b           : {(a + b).cpu().tolist()}   <- 右移了一格")

    print("\n  " + "=" * 66)
    print("  结论:")
    print("    tl.load(地址向量)  -> 值向量, 第 i 个值来自第 i 个地址")
    print("    tl.store(地址向量, 值向量)")
    print("      写到哪里 = 第一个参数(地址向量) 决定")
    print("      写入什么 = 第二个参数(值向量)   决定")
    print("      两者只靠'形状相同 + 位置对应'联系, 没有名字绑定")
    print("  " + "=" * 66)


if __name__ == "__main__":
    main()
