# dispatch_ffn_combine_v2 Standalone 改造设计

## 目标

`dispatch_ffn_combine_v2` 的目标是做原版 `csrc/mc2/dispatch_ffn_combine` 的 standalone kernel benchmark，而不是实现一条新的通信路径。

改造后应满足：

- 在当前 910B1 机器上可构建、可运行。
- standalone host 负责生成数据、初始化 HCCL/MC2 资源、准备 tiling/workspace、launch kernel、验证精度和性能。
- kernel 侧行为尽量保持原版一致，包括 tiling key、混合核类型、HCCL context/window 获取方式、workspace 布局和核心计算逻辑。
- v2 的测试结论能代表原版 kernel 路径，避免因为 standalone shim 改变 kernel 行为。

## 当前问题

当前 v2 为了能 standalone launch，引入了几处和原版不一致的改造：

- `op_kernel/dispatch_ffn_combine.cpp` 将原版 `TILING_KEY_IS(1000010)` 改成了 `TILING_KEY_IS(0)`。
- 生成的 direct-launch stub 当前调用 `LaunchAscendKernel(..., func_key=0, ...)`，只会匹配 key 0。
- v2 通过 tiling 传入 `runtimeInfo.symmetricPtr/rank/rankSize/segmentSize`，kernel 侧再用 `window_table_dev` 寻址通信 window；原版则通过 `AscendC::GetHcclContext<HCCL_GROUP_ID_0>()` 读取 HCCL context。
- standalone host 手写的通信资源配置使用 `opType=18` 和 `"BatchWrite=level0:fullmesh"`，原版 host tiling 使用 `opType=8` 和 `"AlltoAll=level0:fullmesh;level1:pairwise"`。
- v2 workspace 只计算了 `SYSTEM_NEED_WORKSPACE + cocWorkspace`，原版使用 `SYSTEM_NEED_WORKSPACE + max(cocWorkspace, initRoutingWorkspace)`。

这些差异会让 v2 测试对象偏离原版 kernel。尤其是通信 window 改为 tiling 传参后，即使跑通，也不能直接说明原版 HCCL context 路径正确。

后续实测补充：

- `TILING_KEY_IS(1000010)` 是 device 编译/tiling 分支语义，direct-launch runtime function key 仍是生成物注册的 key `0`。把 `LaunchAscendKernel` 的 key 改成 `1000010` 会返回 `507000`。
- 生成的 mix AIC/AIV direct-launch wrapper 已包含混合核任务配置。standalone launch 传 framework TSCH blockDim `60` 会让 AIV 侧 `get_block_num()` 变成 `60`，导致 init-routing 的 40 AIV tiling 和 `SyncAll()` 参与者不匹配。standalone 应传 `aivNum=20`，AIV 侧再通过 `get_subblockdim()==2` 得到 40 个逻辑 AIV。
- 当前 910B compact HCCL context 布局为 `workSpace/workSpaceSize/rankId/rankNum/winSize/windowsIn[]/windowsOut[]`，device 侧需要按 `HcclA2CombineOpParam` 读取 `windowsIn[rank]`。

## 推荐方案

采用“kernel 尽量回归原版，standalone host 对齐原版上下文”的方案。

### 1. 恢复 kernel 侧原版语义

`op_kernel` 中和原版一致的部分应保持一致：

- 恢复 `TILING_KEY_IS(1000010)` 和 `KERNEL_TASK_TYPE(1000010, KERNEL_TYPE_MIX_AIC_1_2)`。
- 恢复 `DispatchFFNCombine::Init(..., GM_ADDR tilingGM)` 的接口形式。
- 恢复通过 `GET_TILING_DATA` 读取 tiling 数据。
- 恢复通过 `AscendC::GetHcclContext<HCCL_GROUP_ID_0>()` 获取 rank/rankSize/window。
- 移除 v2 tiling 中仅为 standalone window table 服务的 `runtimeInfo` 字段，或至少不让 kernel 依赖它。
- `HcclShmem` 优先恢复为原版 context 寻址逻辑。

保留 v2 host harness 文件，例如 `main.cpp`、`runtime_context.cpp`、`tiling_builder.cpp`、`data_utils.cpp`。

### 2. 区分 device tiling key 和 direct-launch function key

实测确认这两个 key 不能混用：

- device 侧主算子 tiling key 保持原版 `1000010`，用于 `TILING_KEY_IS(1000010)` 和 `KERNEL_TASK_TYPE(1000010, KERNEL_TYPE_MIX_AIC_1_2)`。
- standalone direct-launch 的 runtime function key 保持生成物注册 key `0`。该 key 选择 `dispatch_ffn_combine_0_mix_aic/aiv` 入口；改成 `1000010` 会 launch 失败。
- `tiling_builder` 仍可在 `launchConfig.tilingKey` 中记录主算子 key，作为文档/校验信息，但 host wrapper 实际传给 `LaunchAscendKernel` 的 `func_key` 应使用 `DISPATCH_FFN_COMBINE_STANDALONE_FUNC_KEY=0`。

因此 v2 需要保留源码层可传 key 的 host launch wrapper，但默认传 `0`，不要修改 build 目录下的生成产物。

### 3. host 侧准备原版 MC2/HCCL 通信资源

standalone host 可以启动 HCCL，但必须让 kernel 看到和原版一致的 device HCCL context。

host 侧设计：

- `HcclCommInitRootInfo(world_size, root_info, rank_id, &comm)` 初始化 communicator。
- `HcclGetCommName(comm, group)` 获取 group 名。
- 使用和原版 tiling 一致的通信配置：
  - `opType = 8`
  - `algConfig = "AlltoAll=level0:fullmesh;level1:pairwise"`
- 通过 MC2/HCCL resource API 分配通信资源，使 `AscendC::GetHcclContext<HCCL_GROUP_ID_0>()` 能在 kernel 内读取到正确 context。
- 启动 kernel 前确认 host 侧 context 中 rank、rankNum、winSize、local/remote windows 合法。

如果 bare direct launch 无法注入 `GetHcclContext<HCCL_GROUP_ID_0>()` 所需上下文，则应优先走更接近原版的 aclnn/op executor 路径；不要继续扩大 kernel 侧 window table shim。

### 4. tiling 和 workspace 对齐原版

`tiling_builder.cpp` 应复刻原版 `op_host/dispatch_ffn_combine_tiling.cpp` 的关键逻辑：

- `M/K/N/topK/expertPerRank/listLen/maxOutputSize/worldSize` 的来源保持一致。
- `isTransposeB = false`，`isWeightNz = true` 时主算子 key 为 `1000010`。
- `CoCTiling` 常量保持一致：
  - `m0 = 128`
  - `k0 = 256`
  - `n0 = 256`
  - `swizzleDirect = 1`
  - `swizzleOffset = 7`
  - `ubMoveNum = 16 * 1024`
  - `pValue = 1`
  - `commNpuSplit = worldSize`
  - `commDataSplit = 1`
  - `lenPerLoop = m0 * n0 / 2`
- `MoeInitRoutingQuantV2TilingBase::DoTiling(...)` 参数保持原版一致。
- workspace 使用：

```cpp
workspaceBytes = SYSTEM_NEED_WORKSPACE + std::max(cocWorkspace, initRoutingWorkspace);
```

- 保留原版 HCCL window size 检查，避免 window 不足导致通信越界。
- standalone direct launch 的 `blockDim` 使用 `aivNum=20`。原版 op executor 的 `context->SetBlockDim(60)` 是 framework TSCH blockDim；直接传给 `LaunchAscendKernel` 会改变 device 侧 `get_block_num()` 语义。

### 5. 当前 910B1 机器适配

当前机器 `npu-smi info` 显示 8 张 `910B1`，设备健康。v2 应支持这台机器直接运行：

- `run.sh` 如果 `ASCEND_HOME_PATH` 未设置，应探测：
  - `/usr/local/Ascend/cann-8.5.0`
  - `/usr/local/Ascend/ascend-toolkit/latest`
- 仍允许外部通过 `ASCEND_HOME_PATH` 覆盖。
- `SOC_VERSION` 默认使用当前 CANN 可编译的 910B 目标，例如 `ascend910_9391`，并保留 `-DSOC_VERSION=...` 覆盖。
- 先支持 `mpirun -n 2`，再验证 `mpirun -n 8`。
- rank 和 device 默认一一对应：rank 0 使用 device 0，rank 1 使用 device 1。

## 实施步骤

1. 回滚 v2 `op_kernel` 中偏离原版 kernel 行为的改造，仅保留 standalone 编译所需的最小包装。
2. 保持 launch wrapper 参数化，但 standalone runtime launch key 使用 `0`。
3. 修改 `tiling_builder.cpp`，补齐 `initRoutingWorkspace` 并恢复 workspace `max(...)` 计算。
4. 修改 `runtime_context.cpp`，将通信资源配置对齐原版 AlltoAll 配置。
5. 修改 `run.sh` 和 CMake 默认值，支持当前 910B1/CANN 环境自动探测。
6. 构建并运行 2 卡最小 case。
7. 扩展到 8 卡 case，记录精度和性能输出。

## 验证计划

### 构建验证

```bash
cd csrc/mc2/dispatch_ffn_combine_v2
bash run.sh
```

如果环境变量未设置，`run.sh` 应能自动找到 CANN 路径。构建日志需要确认 SoC 目标为 910B 可用配置。

### 2 卡正确性

使用默认小 case：

```bash
cd csrc/mc2/dispatch_ffn_combine_v2
DISPATCH_FFN_COMBINE_V2_WORLD_SIZE=2 \
DISPATCH_FFN_COMBINE_V2_WARMUP_ITERS=1 \
DISPATCH_FFN_COMBINE_V2_MEASURE_ITERS=1 \
bash run.sh
```

预期：

- rank 0/1 都输出 `PASS`。
- `expert_token_nums` 不应长期全 0。
- kernel 能正常同步，不出现 stream sync failure。

### 8 卡验证

```bash
cd csrc/mc2/dispatch_ffn_combine_v2
DISPATCH_FFN_COMBINE_V2_WORLD_SIZE=8 \
DISPATCH_FFN_COMBINE_V2_WARMUP_ITERS=1 \
DISPATCH_FFN_COMBINE_V2_MEASURE_ITERS=3 \
bash run.sh
```

预期：

- 8 rank 都输出 `PASS`。
- profile 输出 kernel/e2e 耗时；kernel 计时使用 device `get_sys_cnt()` 的入口/出口 envelope，e2e 仍为 host 侧参考。
- HCCL window size 检查通过。

## 风险和判断标准

最大风险是 direct kernel launch 无法像原版 op 执行路径一样注入 `GetHcclContext<HCCL_GROUP_ID_0>()`。如果确认该 context 无法通过 standalone resource API 注入，则应停止扩大 kernel 改造，改为走 aclnn/op executor 路径，以保证测试对象仍是原版 kernel。

判断 v2 改造成功的标准不是“能跑出非零输出”，而是：

- kernel 侧通信路径和原版一致。
- device tiling key 与原版 `1000010` 一致，standalone runtime function key 与生成物注册 key `0` 一致。
- host 侧通信资源配置与原版 AlltoAll tiling 一致。
- workspace 和 tiling 字段与原版 host tiling 一致。
- 当前 910B1 机器上 2 卡和 8 卡验证通过。
