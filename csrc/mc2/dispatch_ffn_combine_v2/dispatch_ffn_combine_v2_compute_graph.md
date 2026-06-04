# dispatch_ffn_combine_v2 Compute Graph

本文记录当前对 v2 计算链路的理解，目的是后续排查 golden 和 kernel 正确性时不丢上下文。

## 入口和类型

Host 入口：

- `run.sh` 生成数据、编译、设置 `DISPATCH_FFN_COMBINE_V2_CASE_DIR` 后用 `mpirun` 拉起。
- `scripts/gen_data.py` 生成 `x / expert_idx / probs / weight1 / weight2 / scale1 / scale2 / expected_out`。
- `main.cpp` 读入数据，构造 device buffer 和 tensor-list wrapper，然后调用 `launchDispatchFFNCombine()`。
- `tiling_builder.cpp` 填 `DispatchFFNCombineTilingData` 和 `MoeInitRoutingQuantV2TilingData`。

Kernel 入口：

- `op_kernel/dispatch_ffn_combine.cpp`
- 当前编译定义为 `DTYPE_W1=int8_t`、`DTYPE_OUT=half`。
- kernel 实例为 `DispatchFFNCombine<int8_t, int8_t, half, false, true>`。
- 因此 GMM 输入 A/B 是 int8，GMM 累加落到 half workspace，最终输出 `out` 是 half。
- `Nz_=true`，weight 使用 Catlass `layout::zN`，host 侧需要按 ZN 方式 pack。

重要维度：

- 输入 token: `x[M, K]`
- topK expert id: `expert_idx[M, topK]`
- GMM1 weight: `W1[expert_per_rank, K, N]`
- SwiGLU 后 hidden: `N / 2`
- GMM2 weight: `W2[expert_per_rank, N / 2, K]`
- 输出: `out[M, K]`

## 总体阶段图

```text
Host gen_data.py
  x, expert_idx, probs, W1, W2, scale1, scale2
        |
        v
main.cpp H2D + launch
        |
        v
dispatch_ffn_combine kernel
        |
        +-- AIV path ------------------------------------------------------+
        |                                                                  |
        |  ApplyXActiveMask                                                |
        |        |                                                         |
        |        v                                                         |
        |  InitRoutingDynamicQuant                                         |
        |        |                                                         |
        |        +--> peer offsetA: routed int8 x + per-token scale1       |
        |        +--> workspace.expandedRowIdx                             |
        |        +--> peer tokenPerExpert count vector                     |
        |                                                                  |
        |  CrossRank token count all-gather                                |
        |        |                                                         |
        |        +--> cumsumMM / preSumBeforeRank / expert_token_nums      |
        |        |                                                         |
        |        v                                                         |
        |  Copy remote routed A into workspace gmA/gmPerTokenScale1        |
        |        |                                                         |
        |        v                                                         |
        |  wait GMM1 -> SwiGLU + dynamic quant -> gmPermutedToken          |
        |        |                                                         |
        |        v                                                         |
        |  wait GMM2 -> CombineV1/V2 -> peer offsetD                       |
        |        |                                                         |
        |        v                                                         |
        |  MoeTokenUnpermute(expandedRowIdx, probs, offsetD) -> out        |
        |                                                                  |
        +------------------------------------------------------------------+
        |
        +-- AIC path ------------------------------------------------------+
                                                                           |
           wait routing/cumsum                                             |
                |                                                          |
                v                                                          |
           GMM1: routed int8 x @ W1 -> gmC                                 |
                |                                                          |
                v                                                          |
           wait SwiGLU quant                                               |
                |                                                          |
                v                                                          |
           GMM2: quantized SwiGLU @ W2 -> gmC2                             |
```

## Host Golden 计算链路

对应代码：`scripts/gen_data.py::compute_outputs_and_workload()`。

对每个目标 rank 和本地 expert：

```text
for dst_rank:
  for local_expert:
    global_expert = dst_rank * expert_per_rank + local_expert
    for src_rank, token, topk:
      if expert_idx[src_rank][token, topk] != global_expert:
        continue

      x_token = x[src_rank][token]

      qx, per_token_scale1 = dynamic_quant(x_token)
      product1 = qx @ W1[dst_rank][local_expert]
      gmm1_half = fp16(product1 * scale1[dst_rank][local_expert])
      h1 = fp32(gmm1_half) * per_token_scale1

      swiglu = silu(h1[:N/2]) * h1[N/2:]

      qswiglu, per_token_scale2 = dynamic_quant(swiglu)
      product2 = qswiglu @ W2[dst_rank][local_expert]
      gmm2_half = fp16(product2 * scale2[dst_rank][local_expert])
      y_half = fp16(fp32(gmm2_half) * per_token_scale2)

      out[src_rank][token] += probs[src_rank][token, topk] * fp32(y_half)
```

这个 host 语义和 kernel 的最终 unpermute 是一致的，前提是：

- `expandedRowIdx[token * topK + topk]` 指向该 routed token 在 `offsetD` 里的最终行号。
- combine 阶段按同一 sorted row 顺序把 GMM2 输出写回源 rank 的 `offsetD`。
- `probs` 的布局为 `[M, topK]`，和 unpermute 读取顺序一致。

## Kernel 阶段拆解

### 1. ApplyXActiveMask

对应代码：`DispatchFFNCombineKernel::ApplyXActiveMask()`。

```text
expert_idx[M, topK] + x_active_mask[M]
        |
        v
inactive token 的 expert id 改成 sentinel expert
```

当前 sentinel 是 `expertPerRank * EP`。`MoeInitRoutingQuantV2` tiling 里 `expertNum = expertPerRank * EP + 1`，多出的 1 个 expert 用来承接 inactive token。

### 2. InitRoutingDynamicQuant

对应代码：

- `moe_init_routing_quant_v2<ElementD2>()`
- `MoeV2FullLoadDynamicQuant<T>`

输入：

```text
x[M, K]
expert_idx[M, topK]
```

输出：

```text
workspace.expandedRowIdx[M * topK]
peer offsetA: sorted/routed int8 rows, 每行 stride 是 K + UB_ALIGN
peer offsetPeerPerTokenScale: 每个 routed row 的 dynamic quant scale
peer offsetPeerTokenPerExpert: 当前 rank 的 expert token count vector
```

核心语义：

```text
flat_row = token * topK + topk
expert = expert_idx[flat_row]

按 expert 排序，得到 sorted_row
expandedRowIdx[flat_row] = sorted_row

对 x[token] 做 per-token dynamic quant:
  scale = max(abs(x[token])) / 127
  qx = round(x[token] / scale)

写到 offsetA[sorted_row]
```

注意：kernel 内部 dynamic quant 不是简单的 numpy round。当前代码里会先把 normalized fp32 cast 到 half，再 round 到 int8：

```text
temp = x / scale
temp_half = Cast(temp, CAST_TRUNC)
qx = Cast(temp_half, CAST_ROUND)
```

所以 host golden 如果要 bit-level 贴近 kernel，dynamic quant 也要模拟这一步。

### 3. Token Count All-Gather

对应代码：

- `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2()`
- `GetCumsumForMMAIV()`

阶段图：

```text
local tokenPerExpert count vector
        |
        v
写入本 rank peer window
        |
        v
跨 rank all-gather 到各 rank peer window
        |
        +--> tokenPerExpert[src_rank, global_expert]
        +--> preSumBeforeRank[src_rank, local_expert]
        +--> cumsumMM[source_rank prefix, local_expert]
        +--> expert_token_nums[local_expert]
```

`expertTokensCountOrCumsumFlag` 当前 tiling 设置为 `2`，在 `moe_v2_common.h` 中对应 `EXERPT_TOKENS_COUNT`，也就是 count，不是 cumsum。

### 4. Copy Routed A Into Workspace

对应代码：`DispatchAndCombine()` 中 copy remote A 的循环。

当前 rank 作为目标 expert rank，按本地 expert group 收集所有源 rank 发来的 token：

```text
for groupIdx in local experts:
  for src_rank:
    rows = tokenPerExpert[src_rank, current_rank * expert_per_rank + groupIdx]
    copy remote peer offsetA rows -> local workspace gmA
    copy per-token scale -> gmPerTokenScale1
```

这里会按 `maxOutputSize` 截断。host golden 里 `kept_tokens` 也是按 `dst_rank` 维度做同样的容量限制。

### 5. GMM1

对应代码：`DispatchFFNCombineKernel::GMM1()`。

阶段图：

```text
gmA[int8, routed rows, K]
W1[int8, local_expert, K, N]
scale1[local_expert, N]
        |
        v
int8 GEMM + per-channel dequant
        |
        v
gmC[half, routed rows, N]
```

weight 和 scale 通过 tensor-list 取地址：

```text
GetTensorAddr(index, ptrB1 / ptrScale1)
```

当前 host 用 `listLen=1`，tensor-list 只有一个 entry，entry 指向所有 local experts 拼起来的连续 buffer。kernel 内部再通过 `groupIdx` 做 expert offset。

### 6. SwiGLU + Dynamic Quant

对应代码：`block_epilogue_pertoken_swiglu.hpp`。

阶段图：

```text
gmC[half, rows, N]
per_token_scale1[rows]
        |
        v
fp32 dequant: gmC * per_token_scale1
        |
        v
SwiGLU:
  x0 = first N/2 channels
  gate = second N/2 channels
  y = silu(x0) * gate
        |
        v
per-token dynamic quant:
  scale2 = max(abs(y)) / 127
  qy = round(y / scale2)
        |
        +--> gmPermutedToken[int8, rows, N/2]
        +--> gmPerTokenScale2[rows]
```

kernel 的 `silu(x0)` 写法是：

```text
silu(x0) = x0 / (1 + exp(-x0))
```

### 7. GMM2

对应代码：`DispatchFFNCombineKernel::GMM2()`。

阶段图：

```text
gmPermutedToken[int8, rows, N/2]
W2[int8, local_expert, N/2, K]
scale2[local_expert, K]
        |
        v
int8 GEMM + per-channel dequant
        |
        v
gmC2[half, rows, K]
```

`gmPerTokenScale2` 不在 GMM2 内直接乘，而是在 combine epilogue 里乘。

### 8. CombineV1 / CombineV2

对应代码：

- `CombineV1()` for larger `M * topK`
- `CombineV2()` for `M * topK <= 4096`

阶段图：

```text
gmC2[half, rows, K]
gmPerTokenScale2[rows]
tokenPerExpert / preSumBeforeRank
        |
        v
fp32 dequant: gmC2 * per_token_scale2
        |
        v
按源 rank 和 expert 内 offset 写回 peer offsetD
```

`CombineV2` 的关键写回逻辑：

```text
for src_rank:
  lenRankInExpert = tokenPerExpert[src_rank, current_rank * expert_per_rank + groupIdx]
  dstExpertOffset = preSumBeforeRank[src_rank, groupIdx]
  write remote peer offsetD[dstExpertOffset + local_row]
```

### 9. MoeTokenUnpermute

对应代码：`unpermute/moe_token_unpermute.h`。

阶段图：

```text
peer offsetD[sorted/routed rows, K]
workspace.expandedRowIdx[M * topK]
probs[M, topK]
        |
        v
for token in M:
  acc = 0
  for topk in topK:
    row = expandedRowIdx[token * topK + topk]
    acc += probs[token, topk] * offsetD[row]
  out[token] = cast_half(acc)
```

这就是 host golden 中：

```text
outputs[src_rank][token] += probs[src_rank][token, topk] * result
```

的 kernel 对应阶段。

## 当前已发现的错位风险

### 1. x dtype 当前不一致

当前 CMake 编译为：

```text
DTYPE_OUT=half
```

kernel 里 init routing 使用：

```text
moe_init_routing_quant_v2<ElementD2>(...)
```

所以 `x` 会按 `half` 读。但当前 `scripts/gen_data.py` 写 `x` 时使用的是 bf16 bit：

```text
write(rank_x.bin, fp32_to_bf16_bits(xs[rank]))
```

这意味着同一个 16-bit pattern：

- host golden 按 bf16 值计算。
- kernel 按 fp16 值解释。

这是确定存在的 host/kernel 输入错位。要么生成 fp16 输入并按 fp16 rounded value 做 golden，要么把 kernel 输入类型改成 bf16。

### 2. Dynamic quant rounding 不完全一致

host 当前是：

```text
round(x * 127 / max_abs)
```

kernel 是：

```text
scale = max_abs / 127
temp = x / scale
temp_half = Cast(temp, CAST_TRUNC)
q = Cast(temp_half, CAST_ROUND)
```

这个通常是小误差来源，不应该导致 actual 全 0，但 golden 要严谨需要补齐。

### 3. GMM dequant 和 per-token dequant 的 cast 顺序不一致

device 侧 GMM1 的真实顺序是：

```text
product1 = int8_gemm(qx, W1)
gmC_half = FixpipeVDEQF16(product1, scale1)
swiglu_input = fp32(gmC_half) * per_token_scale1
```

当前 host 曾使用：

```text
fp16(product1 * scale1 * per_token_scale1)
```

这和 device 不一致。per-token scale 不能提前并入 GMM1 的 half cast 前面。

device 侧 GMM2 + combine 的真实顺序是：

```text
product2 = int8_gemm(qswiglu, W2)
gmC2_half = FixpipeVDEQF16(product2, scale2)
offsetD_half = fp16(fp32(gmC2_half) * per_token_scale2)
unpermute_acc += probs * fp32(offsetD_half)
out_half = fp16(unpermute_acc)
```

当前 host 曾使用：

```text
product2 * scale2 * per_token_scale2
```

并且只在最终 `expected_out` 写文件时 cast half，这也和 device 不一致。host golden 需要补齐 GMM2 后的 half cast，以及 combine 写 `offsetD` 前的 half cast。

### 4. compare 打印口径仍需调整

当前 compare 的 PASS 逻辑是：

```text
mismatch = abs(actual - expected) > atol + rtol * abs(expected)
pass = mismatch <= total_count * rtol
```

这个判断逻辑可以暂时保留，但打印成 `err=mismatch/err_threshold` 会让输出类似 `err=1930/2`，不直观。更接近 `gemm_ar` 的口径应打印：

```text
err=mismatch/total_count
```

如果需要保留允许错误个数，可以单独打印 `allow=err_threshold`，不要把它放到 `err` 分母。

### 5. actual 全 0 不是阈值问题

当前观察到的失败形态是 expected 非零、actual 全 0：

```text
rank=0 max_diff=0.00850677 max_ratio=1 err=1930/2 -> FAIL
rank=1 max_diff=0.00940704 max_ratio=1 err=1969/2 -> FAIL
```

这不属于 atol/rtol 设置问题。下一步应该先回读 `expert_token_nums`：

- 如果 `expert_token_nums` 全 0，问题在 routing、token count all-gather、peer window 或 launch/runtime 链路。
- 如果 `expert_token_nums` 非 0，但 `out` 全 0，问题在 GMM1/SwiGLU/GMM2/combine/unpermute 其中一段。

## 推荐排查顺序

1. 回读并打印 `expert_token_nums`，确认 init routing 后是否真的有 token 进入本地 experts。
2. 修正 host golden 的数值链路：`x` dtype、GMM1/GMM2 half cast 顺序、combine half cast、dynamic quant rounding。
3. 生成 fp16 `x`，同时用 fp16 rounded input 重算 golden，排除 dtype 错位。
4. 若 token count 非零，按阶段加最小回读点：`gmA`/`gmC`/`gmPermutedToken`/`offsetD`，二分确认 first zero stage。
5. 再收敛 compare 阈值和打印口径。
