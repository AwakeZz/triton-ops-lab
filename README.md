# triton-ops-lab

从零手写的 Triton GPU kernel 练习集。8 道题覆盖 elementwise、reduction、scan、
归一化和转置，每一题都用 roofline 模型验收 —— 不是"能跑就算完"，而是问一句
**它离这张卡的内存带宽上限还有多远**。

## 硬件基线

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 3050 Ti Laptop (sm_86, 20 SM, 4 GB, L2 2 MB) |
| 理论带宽 | 192 GB/s |
| 实测带宽 | 180.5 GB/s（94%） |
| FP32 峰值 | 8.2 TFLOPS |
| roofline 平衡点 | 45.4 FLOP/Byte |

平衡点 = FP32 峰值 ÷ 实测带宽。算术强度低于 45.4 的 kernel 一律访存受限，
本仓库全部 kernel 都在这个范围内，所以优化目标只有一个：把带宽压到 180 GB/s 附近。

## 验收结果

2026-10-24 完整跑一遍，8/8 通过：

| # | kernel | ms | GB/s | % 峰值 | 启动次数 |
|---|---|---|---|---|---|
| 01 | vector add | 0.2804 | 179.5 | 99.4% | 1 |
| 02 | copy | 0.1886 | 177.9 | 98.5% | 1 |
| 03 | row sum | 0.0932 | 180.2 | 99.8% | 1 |
| 04 | row max | 0.0932 | 180.2 | 99.8% | 1 |
| 05 | row cumsum | 0.1874 | 179.1 | 99.2% | 1 |
| 06 | row softmax | 0.1874 | 179.1 | 99.2% | 1 |
| 07 | layernorm | 0.1874 | 179.1 | 99.2% | 1 |
| 08 | transpose | 0.1902 | 176.4 | 97.7% | 1 |

"启动次数"由 monkey-patch `JITFunction.run` 统计，等价于**访存趟数**。8 个 kernel
全是 1 趟，说明没有一个是"读一遍求 max、再读一遍求 sum"的朴素实现。

## 与 PyTorch 对照

| 算子 | 本仓库 | torch | 差距 |
|---|---|---|---|
| copy (256 MB) | 180.5 | 179.4 `copy_` | 打平 |
| add | 179.4 | 179.7 `x + y` | 打平 |
| transpose 4096×1024 | 176.4 | 102.0 `x.t().contiguous()` | **1.73×** |
| cumsum (N=8192) | 178.8 | 132.7 `torch.cumsum` | **1.35×** |

elementwise 已经贴着 roofline，没有优化空间；能明显超过 torch 的是
**一边访存不连续**的算子 —— 转置和长行扫描。

## 8 道题

| # | 算子 | 难点 |
|---|---|---|
| 01 | `out = x + y` | 一维 grid、尾块 mask，算术强度 0.25 |
| 02 | `out = x` | 尺子：后面每题都拿它比 |
| 03 | `out[i] = sum(x[i, :])` | 一 program 一行还是多行 |
| 04 | `out[i] = max(x[i, :])` | 和 03 同构，换 `tl.max`，`other` 必须是 `-inf` |
| 05 | `out[i, :] = cumsum(x[i, :])` | 前缀和跨 block 有依赖 |
| 06 | `softmax(x[i, :])` | online 递推，能否一趟做完 |
| 07 | `layernorm` | mean 和 var 必须同一趟算出 |
| 08 | `x.T` contiguous | 读和写必有一边不连续 |

## 几个反直觉的实测结论

这些结论都是在同一张卡上扫参跑出来的，有几条推翻了我最初的猜测：

1. **elementwise 没有优化空间。** copy 扫了 28 种 `BLOCK × num_warps` 组合，
   正常组合全部落在 180 GB/s，只有"每线程元素 > 128"开始掉
   （8192 元素 / 1 warp 只有 145.1）。调参是在做无用功。

2. **多行合并对带宽零收益。** row sum 从 `BLOCK_N = N/8` 到 `4N`、一 program
   处理 1~16 行，12 种配置全部 180.2 GB/s。原本以为多行能提升 ILP，实测否定。

3. **但 N < 32 时必须合并。** N=16 时一行一个 program 只有 30.1 GB/s
   （一个 warp 的 32 个 lane 填不满），合并 16 行才回到 178.3。
   分块方式要不要改，取决于 N 而不是数据总量。

4. **`mask` 的开销可以忽略。** 带 mask 与不带 mask 的 copy 都是 177.4 GB/s ——
   PTX 里只是给 `st.global` 加了个 `@%p` 谓词。不要为省 mask 冒越界的风险。

5. **`tl.trans` 不影响性能。** 转置 32×32 加不加 `tl.trans` 都是 176~179 GB/s，
   Triton 3.8 的 layout 优化自己会处理。它的作用是让值的 shape 与写地址矩阵对齐，
   属语义需要。性能来自 tile 化本身：1D + `div/mod` 的朴素散写只有 65.8 GB/s。

6. **`other` 填错不会报错。** `tl.load(ptr, mask)` 省略 `other` 就是默认填 0。
   对 sum 恰好正确（0 是单位元），对 max 就致命（全负行返回 0）。统一显式写。

## 目录

```
ops-lab/
├── check.py            # 验收框架: 生成输入 → 比对 → 测带宽 → roofline 结论
├── bench_baseline.py   # 设备画像 + 带宽基线实测
├── tune_copy.py        # copy 的 BLOCK / num_warps 扫参
├── kernels/            # 8 道题的实现，每题一个文件
└── results/            # 每次验收的 JSON 记录
```

## 快速开始

```bash
pip install torch triton
python bench_baseline.py         # 先把自己的卡量一遍，基线因卡而异
python check.py                  # 跑全部 8 题
python check.py --only 06_softmax
python check.py --odd            # 加测非常规尺寸
python check.py --odd hard       # 再加"非 2 的幂"尺寸
```

`PEAK_GBPS = 180.5` 写在 `check.py` 里，换卡需要先跑 `bench_baseline.py` 改这个数。

`--odd` 是这套框架的关键设计。常规尺寸 (4096, 1024) 恰好都是 2 的幂，
`grid` 写成整除也能通过；非常规尺寸专门用来抓"只在特定 shape 正确"的错：

| 级别 | 尺寸 | 抓什么 |
|---|---|---|
| `--odd` | (997,1024) (1003,512) (1,4096) | grid 用整除、尾部 mask 没盖住 |
| `--odd hard` | 追加 (333,1000) (77,127) | BLOCK 假设了 N 是 2 的幂 |

## 验收框架做的三件普通 benchmark 不做的事

1. **统计 kernel 启动次数**：`launch` 与 `ideal_launch` 并排打印。多一次启动 =
   多一趟访存 = 可直接量化的优化空间。

2. **测前跑 0.8 秒把时钟拉满**：笔记本 GPU 有频率阶梯，测太短会低估一个数量级。
   更重要的是把第一次调用（含 JIT 编译）挡在预热之外 —— 否则编译时间吃掉预热窗口，
   GPU 还停在待机频率上。这个坑实测踩出来过四次，每次看到的都是假数据
   （22 GB/s 的 layernorm、13.4 GB/s 的转置），重跑即恢复。

3. **自适应迭代数**：按耗时反推迭代次数，避免小 kernel 只测到启动开销。

## 环境

Python 3.13 / torch 2.11.0+cu128 / triton 3.8.0，Windows。
Triton 编译缓存默认写到仓库目录下的 `.triton-cache/`（已在 `.gitignore` 里），
也可以用 `TRITON_CACHE_DIR` 环境变量覆盖。
