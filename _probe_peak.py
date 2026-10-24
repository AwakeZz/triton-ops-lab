"""fp16 tensor core 峰值: 按尺寸扫一遍, 找出这张卡真实可达到的算力。

上一版只测 4096 就下结论, 结果 4096 反而比 2048 慢 —— 说明 cuBLAS 选算法
对尺寸敏感, 不能拿单点当"峰值"。这里 1024/2048/3072/4096 各测两轮。
"""
import os
import statistics
import time

import torch

os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".triton-cache"))


def bench(fn, ramp_s=1.0, iters=40, rounds=2):
    best = []
    for _ in range(rounds):
        until = time.perf_counter() + ramp_s
        while time.perf_counter() < until:
            fn()
        torch.cuda.synchronize()
        xs = []
        for _ in range(iters):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            xs.append(a.elapsed_time(b))
        best.append(statistics.median(xs))
    return min(best), best


def main():
    torch.manual_seed(0)
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"{'尺寸':>6}{'ms(最好)':>11}{'TFLOPS':>10}{'各轮 ms':>24}")
    print("-" * 52)
    peak = 0.0
    for S in (1024, 2048, 3072, 4096):
        a = torch.randn(S, S, device="cuda", dtype=torch.float16)
        b = torch.randn(S, S, device="cuda", dtype=torch.float16)
        out = torch.empty(S, S, device="cuda", dtype=torch.float16)
        torch.matmul(a, b, out=out)
        torch.cuda.synchronize()
        ms, rounds = bench(lambda: torch.matmul(a, b, out=out))
        flops = 2 * S ** 3
        tf = flops / (ms * 1e-3) / 1e12
        peak = max(peak, tf)
        print(f"{S:>6}{ms:>11.3f}{tf:>10.2f}{'  '.join(f'{r:.3f}' for r in rounds):>24}")
        del a, b, out
        torch.cuda.empty_cache()

    print(f"\n最好成绩 {peak:.2f} TFLOPS")
    print("规格参考: RTX 3050 Ti Laptop 官方 FP16 tensor 40.7 TFLOPS (dense)")


if __name__ == "__main__":
    main()
