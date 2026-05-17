# dispatch_ffn_combine_v4 设计：PTO-first 的 standalone MegaMoE fused kernel

## 1. 目标与定位

`dispatch_ffn_combine_v4` 不是 `dispatch_ffn_combine_v3` 的继续翻译，也不是把 legacy helper 再清洗一轮。

它的目标产物是一个：

- **standalone** 的项目
- **AICORE-only / Ascend NPU-only** 的项目
- **以 PTO 为主编程模型** 的项目
- **面向 MegaMoE fused operator** 的项目
- **不依赖 `dispatch_ffn_combine_v4/` 目录外业务代码** 的项目

一句话：

> **v4 要把 MegaMoE 的 fused dispatch / compute / combine 语义，直接重写成 PTO-native 的 standalone kernel project。**

这份设计文档不是只给实现机器看，也要给人看。因此它的组织顺序必须是：
1. 先解释 **MegaMoE 算法到底在做什么**；
2. 再解释 **v4 要保留哪些算法真值、放弃哪些历史工程包袱**；
3. 最后才落到 **protocol、layout、模块、脚手架、验证**。

---

## 2. 文档阅读顺序与章节逻辑

### 2.1 先读算法，再读模块

本设计按下面的逻辑阅读：

1. **第 3 章：MegaMoE 算法说明**
   - 回答“为什么 dispatch / combine 不能对称”“routing 真值到底是什么”“overlap 是怎么成立的”。
2. **第 4-5 章：设计输入、约束、PTO 承接**
   - 回答“哪些是需求真值”“哪些接口必须 PTO-first”“哪些内容明确不做”。
3. **第 6-13 章：把算法链逐段落到模块**
   - system boundary → dataflow → workspace/layout → signal → routing → exchange → compute → output。
4. **第 14-18 章：把工程脚手架、实施顺序、验证闭环补齐**
   - standalone harness、run.sh、CMake、阶段落地、golden、perf、风险管理。
5. **第 19 章：最终收束**
   - 用一页话重申最终边界，避免实现时又退回 legacy 习惯。

### 2.2 章节之间的前后依赖

章节之间不是并列堆砌，而是明确依赖：

- **算法说明** 决定后面所有模块 contract；
- **routing 真值** 决定 dispatch/combine 偏移和 output restore 的正确性；
- **dispatch/combine 非对称** 决定 `TGET/TPUT` 的方向选择；
- **overlap 模型** 决定 signal/ready-queue/summary-counter 的设计；
- **工程脚手架** 只是承载算法，不反过来定义算法。

如果后面某个细节章节与第 3 章算法总述冲突，应该以算法总述为准，并回头修正文档细节，而不是让实现去猜哪一处才算真值。

---

## 3. MegaMoE 算法说明

### 3.1 要解决的根问题

MegaMoE 要解决的不是“把 dispatch、FFN、combine 三个算子简单串起来”，而是：

- 在 MoE 场景下，token 会被路由到不同 expert，天然形成 **不规则跨 rank 数据交换**；
- 如果仍按传统 BSP 方式做“先全量通信，再重排，再 GMM，再全量回传”，会出现长时间同步空泡；
- 尤其是 dispatch 之后，若接收端拿到的仍是按 source-rank 拼接的数据，就还要再做一轮“通信后重排”才能形成 expert-major 连续矩阵，这会阻塞 GMM 提前启动。

MegaMoE 的核心改进是：

> **先把 routing metadata 真值算清楚，再让通信直接把 payload 落到 GMM 可消费的最终 expert-major 位置；随后让 dispatch、compute、combine 尽量分段重叠，而不是整段 BSP 串行。**

### 3.2 端到端算法主链

可以把 v4 要实现的算法链看成下面这张图：

```text
输入 X / topkIds / topkProbs / x_active_mask / maxOutputSize
                |
                v
+----------------------------------------------+
|  routing prelude (control plane)             |
|  1) expand top-k                             |
|  2) mask inactive token -> sentinel expert   |
|  3) local reorder / local count              |
|  4) gathered count / exact prefix / offsets  |
|  5) materialize DispatchPullTask/Combine...  |
+----------------------------------------------+
                |
                v
+----------------------------------------------+
|  dispatch (data plane, receiver-driven pull) |
|  source: publish packed payload + ready      |
|  dst   : TWAIT/TTEST -> TGET                 |
|  dst   : direct landing to expert-major      |
+----------------------------------------------+
                |
                v
+----------------------------------------------+
|  compute                                      |
|  expert-major input -> GMM1 -> SwiGLU -> GMM2|
|  first MVP keeps legacy two-stage overlap     |
+----------------------------------------------+
                |
                v
+----------------------------------------------+
|  combine (data plane, producer-driven push)  |
|  producer: TPUT partial outputs to row owner |
|  owner   : wait ready -> restore -> reduce   |
+----------------------------------------------+
                |
                v
最终输出 Y
```

这条主链里有一个非常重要的分层：

- **routing prelude** 是 control plane；
- **dispatch / compute / combine** 是 hot-path data plane。

前者负责得出真值，后者只负责消费真值执行。

### 3.3 dispatch 前的 routing 真值到底是什么

这里要先把几个概念分开：

- **gating**：决定一个 token 去哪些 expert；
- **routing**：在 gating 结果已知后，继续完成 expand、分组、count、prefix、offset、task materialization；
- **dispatch**：按 routing 真值把 payload 送到目标 rank / 目标 expert-major 输入区。

因此，routing 不是“再做一次 gating”，而是把 gating 的结果变成后续通信和计算可以直接消费的执行真值。

这组真值至少包括：

- `expandedRowIdx[]`
- `expandedExpertIdx[]`
- `expandedProb[]`
- `expandedRankIdx[]`
- `expandedLocalExpertSlot[]`
- `tokenPerExpert[srcRank][localExpert]`
- `cumsumMM[localExpert][srcRank]`
- `srcOffset[]`
- `dstOffset[]`
- `rowToExpandedRange[row]`

它们的作用不是“记录一些调试信息”，而是直接回答 3 个关键问题：

1. **某个 expanded token 从哪里来？**
   - 由 `expandedRowIdx[]`、`srcOffset[]` 回答；
2. **它应该落到哪个 rank / 哪个 local expert / 哪个连续片段？**
   - 由 `expandedRankIdx[]`、`expandedLocalExpertSlot[]`、`dstOffset[]` 回答；
3. **后面 combine 恢复时，它属于原始哪一行、权重是多少？**
   - 由 `rowToExpandedRange[]`、`expandedProb[]` 回答。

### 3.4 “通信后重排”的真正含义

MegaMoE 优化的重点不是“通信前完全不做重排”，而是：

- 本地在 dispatch 前，仍然要做 expand、local count、local pack；
- 被消掉的是那种“远端收完 source-rank-major 数据后，再做一次大规模 sparse-to-expert-major 重排”的阶段。

换句话说，MegaMoE 要求 dispatch 后的数据天然就是：

> **GMM 可直接消费的 expert-major 连续矩阵，而不是一堆还要再挪一次的数据块。**

这也是为什么 `dstOffset[]`、`cumsumMM[]`、`DispatchPullTask[]` 在设计里是核心真值：
它们直接决定 `TGET` 的目标落点，而不是在通信完成后再临时推导。

### 3.5 为什么 dispatch 与 combine 必须不对称

MegaMoE 不是一个对称 all-to-allv 算法。

#### dispatch
- 本质是 **receiver-driven pull**；
- receiver 只有在自己掌握完整 routing 真值后，才知道应该从每个 source rank 拉哪些连续片段；
- 因此更适合：source publish ready，destination `TGET` 直接拉到 expert-major 终态。

#### combine
- 本质是 **producer-driven push**；
- 当某个 expert-group 的 `GMM2` 结果完成时，producer 已经掌握“这批 partial output 应该回哪个 row owner”的信息；
- 因此更适合：producer `TPUT` 到 owner 的 combine region，owner 再做 restore / weighted reduce。

所以 v4 不能为了“接口统一看起来整齐”把两边都写成同一种通信方向。算法上它们就是不对称的。

### 3.6 compute 主链与 combine 恢复

compute 部分的业务链是：

```text
expert-major input -> GMM1 -> SwiGLU -> GMM2 -> partial outputs
```

这里要保留两个事实：

1. **第一版 compute 不追求最激进 subtile overlap**；
2. **但也不能退回完全串行的“全体 GMM1 完成后再统一激活、再统一 GMM2”**。

因此 MVP 先保留 legacy 已验证的两段式基线：
- 前段 expert 先完成一波 `GMM1 -> SwiGLU -> GMM2` 错拍；
- 后段再推进第二波；
- 这保证 v4 一开始就具备基本 compute overlap，而不是纯串行 reference。

combine 恢复部分则依赖：
- `CombinePushTask[]`
- `rowToExpandedRange[]`
- `expandedProb[]`

其中一个关键点是：

> **combine 的目标落点不能从 `cumsumMM` 反推。**

`cumsumMM` 是 dispatch / GMM 输入侧的 expert-major 前缀真值；
combine 回写时需要的是 row owner 视角的 prefix 真值，它更接近 legacy 里的 `preSumBeforeRank` 一类语义，或者直接固化在 `dstPayloadOffsetBytes` 中。

### 3.7 sentinel expert 与 capacity 语义

MegaMoE 的真实执行链里还有两个很容易被忽略、但必须固定下来的算法约束。

#### A. sentinel expert
`x_active_mask` 不是简单“把这个 token 整体删掉”，而是：
- host 侧保留 `expertNum = worldSize * expertPerRank + 1`
- device 侧把 inactive token 的 `expertIdx` 写成最后那个 sentinel expert

因此 routing prelude 必须能解释这个哨兵 expert；否则 `expandedRowIdx[]`、count、offset 都会错位。

#### B. capacity-bounded execution
`maxOutputSize` 不是一个无关紧要的 tiling 参数，而是：
- 每个目的 rank 的 routed-token capacity；
- 它会把“理论上 expand 后的 token 真值”截断成“真正会进入 GMM / combine 的 executable truth”。

因此设计里必须区分：
- **before-capacity truth**：原始 expand / count / cumsum；
- **after-capacity truth**：真正 materialize 成 task 并实际执行的部分。

### 3.8 overlap 模型

MegaMoE 的 overlap 不是一句“通信计算重叠”就能带过，它至少有三层意思：

#### A. dispatch → GMM 的 many-to-many 依赖
- 一个 rank 的 compute 开始，依赖多个 source rank 的 dispatch arrival；
- 所以这里的重点是 `ready queue + summary counter + TWAIT/TTEST`；
- 它不是简单的一对一生产者消费者。

#### B. GMM → combine 的 producer-to-consumer 依赖
- 某个 expert-group 的 `GMM2` 一旦完成，就可以把 partial outputs 往 owner 侧推；
- combine 的消费粒度更接近 group / tile 级任务完成。

#### C. 三段式流水掩盖
MegaMoE 文档强调的深度融合，可以抽象成：

1. **前一段**：下一个 chunk / group 的 dispatch-side quant / pack / ready publish
2. **中间段**：当前 chunk / group 的 `GMM1 -> SwiGLU -> GMM2`
3. **后一段**：上一个 chunk / group 的 combine push / restore / output-ready publish

v4 的 Phase 0-5 先保留更保守的两段式 compute overlap；
到 Phase 6 再进一步引入：
- ready queue
- summary counter
- probe-first `TTEST`
- coarse/fine expert grouping（如 `{8, 4, 2, 1, 1}`）

### 3.9 v4 重写时哪些东西要保留，哪些不要保留

v4 不是“把 legacy 工程抄一遍”，但也不是“只保留一个高层想法然后随便重写”。

必须保留的是：
- MegaMoE 的算法主链
- routing metadata 真值
- dispatch/combine 非对称
- direct landing 到 expert-major 输入布局
- combine 回 owner 后的 restore / weighted reduce
- sentinel expert / capacity / overlap 这些执行约束

明确不保留的是：
- legacy 的 helper 命名
- 大块 AscendC 业务接口面
- 把 host/runtime 包袱直接带进 kernel
- 为了迁就旧工程而牺牲 PTO-first 表达

---

## 4. 输入优先级与设计真值来源

### 4.1 第一优先级：MegaMoE 需求真相

主输入：
- `/home/ntlab/zy/megamoe/昇腾高效支持MegaMoE融合算子，实现MoE计算通信深度融合.html`
- 用户给出的 GitHub 阅读笔记与设计纠偏说明
- 原始 `dispatch_ffn_combine` 代码链路

这里决定：
- 为什么要做 fused MegaMoE
- dispatch / compute / combine 的目标数据流
- 为什么 dispatch 与 combine 的通信方向不应对称
- 为什么必须追求通信计算重叠，而不是 BSP 串行
- routing / offset / restore 的真值应如何定义

### 4.2 第二优先级：PTO 公开能力与示例

主输入：
- `/home/ntlab/zy/code/zhangyuan/pto-isa-zy/docs/pto-knowledge.md`
- `/home/ntlab/zy/code/zhangyuan/pto-isa-zy/docs/isa/comm/TGET.md`
- `/home/ntlab/zy/code/zhangyuan/pto-isa-zy/docs/isa/comm/TPUT.md`
- `/home/ntlab/zy/code/zhangyuan/pto-isa-zy/docs/isa/comm/TWAIT.md`
- `/home/ntlab/zy/code/zhangyuan/pto-isa-zy/docs/isa/comm/TTEST.md`
- `/home/ntlab/zy/code/zhangyuan/pto-isa-zy/kernels/manual/a2a3/dispatch_combine_moe/`
- `/home/ntlab/zy/code/zhangyuan/pto-isa-zy/kernels/manual/a2a3/dispatch_combine_moe_v2/DESIGN.md`
- `/home/ntlab/zy/code/zhangyuan/pto-isa-zy/kernels/manual/a2a3/allgather_gemm/README.md`
- `/home/ntlab/zy/code/zhangyuan/pto-isa-zy/kernels/manual/a2a3/gemm_ar/README.md`
- `/home/ntlab/zy/code/zhangyuan/pto-isa-zy/tests/npu/a2a3/`

这里决定：
- PTO 已经有哪些可直接承接 MegaMoE 的 building blocks
- 哪些地方必须直接用 PTO 原语
- 哪些地方仍只需要很薄的 protocol / substrate shell

### 4.3 第三优先级：legacy 项目经验

辅助输入：
- `csrc/mc2/dispatch_ffn_combine/`
- `csrc/mc2/dispatch_ffn_combine_v2/`
- `csrc/mc2/dispatch_ffn_combine_v3/`

这里只用于：
- 校对计算链是否完整
- 校对 overlap 策略是否合理
- 校对 standalone host harness 的最小职责

不用于：
- 直接沿用目录拆分
- 沿用历史 helper 命名
- 把 legacy 工程约束当作 v4 的需求来源

---

## 5. 设计约束：原则、范围、非目标与 PTO 承接

### 5.1 设计原则

1. **MegaMoE 语义优先**
   - 先保证 fused dispatch / compute / combine 数据流正确，再谈历史工程兼容性。
2. **PTO-first**
   - 只要 PTO 已有等价接口、类型、布局、协议语义，就直接用 PTO 风格表达。
3. **薄壳化**
   - AscendC / Gemm / Fixpipe 只能留在 substrate 层，不允许反向渗透到业务层。
4. **metadata first**
   - 先得到 routing metadata 真值，再执行 payload 交换和计算。
5. **protocol explicit**
   - payload / signal / epoch / completion 全部显式建模，不能靠隐式调用顺序猜。
6. **correctness before performance**
   - 先完成功能闭环，再记录性能，再单独做收敛。

### 5.2 本次范围

v4 必须覆盖完整链路：
- route expand
- local reorder / local count
- gathered count / exact offsets
- dispatch pull
- GMM1
- SwiGLU
- GMM2
- combine push
- unpermute + weighted reduce
- standalone host launch / golden / perf harness

### 5.3 明确非目标

本设计不做：
- 兼容 CPU / 非 AICORE 路径
- 兼容 v1 / v2 / v3 的 helper 命名和内部 ABI
- 把规则 collective 硬套成 MegaMoE 的不规则 all-to-allv
- 在第一版中就实现最激进的 subtile 级 overlap
- 为了短期性能回退到大块 AscendC 历史壳结构

### 5.4 PTO 边界必须拆成三层

#### A. 业务层 public PTO surface

v4 的业务主链模块（routing / dispatch / compute / combine / output）允许直接依赖的，只能是 PTO 已公开的通用数据与指令面。

可直接使用的数据描述：
- `Shape`
- `Stride`
- `Layout`
- `GlobalTensor`
- `Tile`

可直接使用的通信原语：
- `pto::comm::Signal`
- `pto::comm::Signal2D`
- `pto::comm::TGET`
- `pto::comm::TPUT`
- `pto::comm::TWAIT`
- `pto::comm::TTEST`
- `pto::comm::TNOTIFY`

可直接使用的搬运 / 计算原语：
- `TLOAD`
- `TSTORE`
- `TMATMUL`
- `TMATMUL_ACC`
- `TSTORE_FP`
- `TQUANT`
- PTO 已公开的常见向量 / 规约原语

这些 public PTO surface 对应到 MegaMoE 主链时，语义必须保持直接映射：
- Dispatch = receiver-driven pull = `TWAIT/TGET/TNOTIFY`
- Combine = producer-driven push = `TPUT/TNOTIFY`
- GMM1 / SwiGLU / GMM2 = PTO public compute surface 上的计算链
- task 的输入输出位置 = `GlobalTensor + Tile + offset` 的组合，而不是历史 helper 的隐式 side effect

因此 v4 的业务合同必须直接建立在 PTO public types / ops 上，而不是先写一层 AscendC tensor helper，再局部翻译成 PTO。

#### B. protocol-shell：借鉴 PTO 示例的 sequencing，但不把示例 helper 当 public API

`dispatch_combine_moe`、`allgather_gemm`、`gemm_ar` 这些 PTO 示例，v4 要复用的是协议模式，不是把示例 helper 的命名直接抬升成业务层接口。

v4 允许从示例中继承的模式主要是：
- payload transfer 与 ready publication 分离
- monotonic epoch / summary-counter 风格的完成判定
- local-first compute
- `TTEST` probe-first，必要时 `TWAIT` 阻塞
- publish / consume fence discipline
- ready queue / coarse batch / progress record 这类调度骨架

因此 v4 可以在项目内形成自己的最小 protocol shell，例如：
- `RemoteWindowContext`
- `PublishPayloadFence()`
- `ConsumePayloadFence()`
- `DispatchGroupProgress`
- `CombineGroupProgress`
- `ReadyQueue`

这些 shell 的职责，是把业务 task 映射到 PTO public 指令序列上；它们是项目私有协议壳，不是 PTO 官方的 business-layer public surface。

特别说明：
- `HcclRemotePtr`
- `CrossRankSync`
- `PtoTGetContiguous`
- `PtoTPutContiguous`
- `PtoNotifySet`
- `PtoWaitGe`

这类名字即使在示例、旧工程或局部包装里出现，也只能被当作 sequencing 参考；不能被抬升成 v4 对外算法合同的一部分。

#### C. substrate / internal shell territory

如果 PTO 当前还没有把某些 AICORE 细节完整公开成 business-facing surface，v4 允许保留最薄的一层内部壳，但只能收敛在 `substrate/`，不能反向污染业务层。

允许暂留的范围只包括：
- PTO 当前仍需依附的 buffer / queue / lifecycle substrate
- PTO public matmul 面还未覆盖完整的 L1/L0/fixpipe/Gemm 外围壳
- 极少数必须与 AscendC runtime 对接的内部 glue

边界要求固定为：
1. `routing/dispatch/combine/output` 这些业务层头文件，不直接暴露 AscendC / Gemm 业务接口；
2. 业务 task / plan / progress 合同里，不出现 AscendC / Gemm 类型；
3. 任何残留壳都必须能解释为“PTO 当前缺口”，而不是回退到 legacy API；
4. HCCL/HCOM 只保留在 host runtime 的 bootstrap / remote GM window 获取层，kernel 侧只看 `RemoteWindowContext + Signal + GM window`。

### 5.5 PTO 示例对 v4 的正确作用

#### A. `dispatch_combine_moe`
可借鉴的不是 helper 名字本身，而是：
- PTO comm 原语已经足够组成跨 rank protocol shell；
- payload region、signal region、epoch table 可以被显式建模；
- 需要补的是 MegaMoE workload 级协议，不是另造通信原语。

#### B. `allgather_gemm`
最值得直接承接的是：
- payload transfer 与 ready publication 分离；
- monotonic summary counter；
- local-first compute；
- `TWAIT` 驱动 chunk arrival。

#### C. `gemm_ar`
最值得直接承接的是：
- ready queue；
- `TTEST` probe-first；
- 必要时 `TWAIT` 阻塞；
- publish / consume fence discipline。

#### D. `tests/npu/a2a3/*`
这些测试的作用，是给每条 PTO 原语提供最小语义基线：
- `tget`
- `tput`
- `twait`
- `ttest`
- `tmatmul`
- `tquant`

它们证明的是 PTO public instruction 的语义边界，而不是“MegaMoE 业务层已经存在现成 helper 可以直接照搬”。

---

## 6. 最终系统边界

### 6.1 host/runtime 层
职责：
- ACL 生命周期
- HCCL bootstrap
- remote GM window 导出
- workspace 分配
- tiling blob 组装
- case 读写
- golden / perf 记录

### 6.2 kernel business 层
只允许表达 PTO 语义：
- tensor / view / layout / tile
- `TGET/TPUT/TWAIT/TTEST/TNOTIFY`
- `TMATMUL/TMATMUL_ACC/TSTORE_FP`
- routing / task / schedule / output restore 业务语义

### 6.3 substrate 层
只保留 PTO 目前还没完全吞掉的底层壳：
- `kernel_operator.h`
- `lib/matmul_intf.h`
- Fixpipe / Gemm helper
- cache shell
- cross-core shell

### 6.4 禁止项
- 不依赖 `dispatch_ffn_combine_v4/` 目录外业务代码
- 不把 HCCL host 语义泄漏进 kernel 业务层
- 不在 business 层散落 AscendC / Gemm / Fixpipe 细节

---

## 7. 端到端数据流

单次执行固定成 8 步：

1. **route expand**
   - 输入 `topk expert ids + probs + row ids`
   - 输出 expanded 条目 `(row, expert, prob)`
2. **local route reorder**
   - 按目标 expert / rank / group 形成本地重排顺序
3. **count exchange**
   - 只交换 metadata，不交换 payload
   - 生成 `gatheredExpertCount`、`localExpertPrefix/globalExpertPrefix/cumsumMM`、`srcOffset`、`dstOffset`
4. **dispatch pull**
   - receiver 根据 `DispatchPullTask[]` 主动 `TGET` 远端 payload
5. **compute chain**
   - AIC 对 ready 的 expert-group 执行 `GMM1 -> SwiGLU -> GMM2`
6. **combine push**
   - producer 根据 `CombinePushTask[]` 将结果 `TPUT` 回目标 rank
7. **unpermute + weighted reduce**
   - 按 `rowToExpandedRange[] + expandedProb[] + CombinePushTask[]` 恢复 token 顺序并做 top-k 加权求和
8. **final verify**
   - 写回最终输出，host 侧做 golden 与 perf 记录

这里固定三条主结论：
- **dispatch 是 receiver-driven**
- **combine 是 producer-driven**
- **routing metadata 是唯一真值源**

### 7.1 control-plane prelude 与 hot-path pipeline 必须分开理解

为了避免把 MegaMoE 错理解成“所有步骤都在同一条热流水里”，v4 需要明确区分两层：

#### A. control-plane prelude
这部分先完成，主要负责让后续通信地址可解：
- `MoeInitRoutingQuant` 风格的本地 routing / expand / quant
- `expandedRowIdx`
- `localTokenPerExpert`
- token count 同步 / allgather
- `tokenPerExpert[srcRank][localExpert]`
- `cumsumMM[localExpert][srcRank]`
- exact `srcOffset/dstOffset`

这里还要特别强调一条实现约束：
- **count/control exchange 是独立的 metadata 协议**；
- 它可以采用更接近 allgather / push-notify 的实现，不必强行复用 dispatch payload 的 receiver-pull 方向；
- 也就是说，`dispatch payload` 与 `count/control metadata` 的通信方向可以故意不同，只要最终真值表一致。

这部分的目标不是直接产出 GMM 结果，而是：
- 让 dispatch 后的数据天然成为 expert-major 的 GMM-ready 连续块；
- 把“通信后的稀疏到密排重排”提前吸收到通信地址计算里。

#### B. hot-path pipeline
真正做细粒度重叠的是：
- `dispatch expert i+1`
- `GMM1 -> SwiGLU -> GMM2 expert i`
- `combine expert i-1`

这是一条按 expert 语义推进的错拍流水，而不是“count 同步也在热流水里并行推进”。

### 7.2 expert 与 expert-group 的关系

- `expert` 是算法语义上的最小消费单位：某个 expert 的 token 连续块 ready 后，理论上就具备启动对应 GMM 的条件。
- `expert-group` 是工程实现上的调度打包单位：为了控制 signal 数量、buffer 预算和 AIC/AIV 调度复杂度，第一版允许把多个 expert 合成一组推进。
- 因此：
  - **语义主轴是 expert**；
  - **MVP 调度主轴可以是 expert-group**；
  - 但为了和 legacy / v2 / v3 的真实主链保持一致，**Phase 0-5 默认取 `expertGroupId == expertId`，也就是一组只含一个 local expert**；
  - Phase 6 再按需要把多个 expert 打包成 group，或把 group 内粒度缩小到 expert-block / sub-group，而不是一开始就强行做到 subtile 级。

### 7.3 为什么 dispatch 与 combine 必须刻意不对称

MegaMoE 的关键不是“通信和计算都融合了”，而是**dispatch 侧与 combine 侧面对的是两种完全不同的数据依赖图**。

#### A. dispatch = receiver-driven pull
- dispatch 的消费方是目标 rank 的本地 compute；
- 目标 rank 在 count exchange 结束后，已经知道：
  - `tokenPerExpert[srcRank][localExpert]`
  - `cumsumMM[localExpert][srcRank]`
  - 每个来源 rank 对本地 expert-major buffer 的精确落点；
- 因此最合理的做法是：**由 receiver 直接 `TGET` 到最终 `expert-major` 地址**。

这样做的直接收益：
- source rank 只负责把本地 token 按 dispatch source 视角 packed 到 `dispatch_region`；
- receiver 负责把不同来源 rank 的 payload 直接落到 GMM-ready 的最终连续区；
- “通信后再做一遍 source-major -> expert-major 重排”不再是独立热路径步骤，而是被吸收进 `dstOffset` 的地址规划里。

所以，dispatch 的核心语义不是“先传过去再重排”，而是：

> **receiver 用 metadata 真值把远端数据直接拉到最终 expert-major 布局。**

#### B. combine = producer-driven push
- combine 的生产方是本地 compute；
- `GMM2` 完成时，producer 已经掌握待回传 partial output 的连续块；
- 这些结果的目标去向已经由 `CombinePushTask[]` 固定，因此**producer 最适合算完即 `TPUT`**。

这样做的直接收益：
- compute 完成即发布，不需要等待目标 rank 逐个轮询远端 expert 结果；
- output engine 只要按 `combineDstReadyEpoch` 消费 `combine_region` 并做 row restore / weighted reduce；
- 不会把 combine 重新变回“row owner 从多个 remote experts 主动拉结果”的高耦合路径。

所以，combine 的关键不是对称地模仿 dispatch，而是：

> **producer 把已经形成的连续结果块立即推回 row owner，让消费侧只做 restore/reduce。**

#### C. 结论
- **dispatch 的优化重心是：让通信直接落到最终 GMM 输入布局；**
- **combine 的优化重心是：让 compute 完成结果立刻可发布、可还原；**
- 两侧都用 PTO comm 原语，但**协议方向故意不对称**。

### 7.4 Dispatch→GMM 与 GMM→Combine 的同步模型不应等价处理

#### A. Dispatch→GMM 是 many-to-many 依赖
某个 local expert 的输入块，通常来自多个 `srcRank`：
- 只有其中一部分来源 ready，并不代表整个 expert-major 输入都 ready；
- 若强制对整个 group 做 `SyncAll`，会把已经 ready 的专家也拖住，形成明显空泡；
- 因此 MegaMoE 需要的是**软件计分板 / min-status / summary-counter** 语义，而不是“一刀切大栅栏”。

在 v4 中，这体现在：
- `dispatchSrcReadyEpoch[peer][group]` 提供每个来源 peer 的可见进度；
- `summaryCounter[group]` 提供 group 级聚合进度；
- Phase 0-5 允许先在 `expert-group` 级等待；
- Phase 6 再把它细化成 `expert-block` / `sub-group` 级 ready queue，让 compute 尽量在“该 expert 所需来源已满足”时提前启动，而不是等整个大组彻底齐。

#### B. GMM→Combine 更接近 producer-to-consumer 切块
一旦某个 expert block 的 `GMM2` 结果完成：
- 它已经在本地形成连续输出块；
- 目标 row/rank 也已由 `CombinePushTask[]` 固定；
- 这更像 producer 把完成块按 row/tile 切开后发布给 downstream consumer。

因此 combine 侧通常不需要复制 dispatch 那种 many-to-many 计分板复杂度：
- 生产方以本地完成为主时钟；
- 消费方按 `combineDstReadyEpoch` 或 task completion 消费；
- 真正需要的更多是 row-block / tile-block 的切分策略，而不是重新做一层“大量来源 -> 一个 expert 输入”的汇聚判定。

#### C. 设计落点
- **Dispatch→GMM**：重点设计 `ready queue + summary + scoreboard`；
- **GMM→Combine**：重点设计 `task completion + row/tile restore`；
- 两段都可以重叠，但它们的“为什么能重叠”不同，不能抽象成同一种模板。

---

## 8. workspace / remote window 设计

### 8.1 顶层 region 划分

每个 rank 的 remote window 固定成 5 个一级 region：

```text
workspace
├── R0 control_region
├── R1 dispatch_region
├── R2 compute_region
├── R3 combine_region
└── R4 signal_region
```

### 8.2 `control_region`
只放 metadata，不放大 payload。

内容：
- launch config snapshot
- routing headers
- `localTokenPerExpert[]`
- `tokenPerExpert[srcRank][localExpert]`
- `gatheredExpertCount`（仅作为 host/debug 视角的归档投影）
- `cumsumMM[localExpert][srcRank]`
- prefix / offset tables
- `DispatchPullTask[]`
- `CombinePushTask[]`
- per-group progress summary

二级布局：

```text
control_region
├── control_header
├── routing_tables
├── count_tables
├── offset_tables
├── dispatch_task_table
└── combine_task_table
```

### 8.3 `dispatch_region`
只放待被 pull 的输入 payload。

内容：
- packed activation blocks
- optional scale / quant sideband
- per-expert contiguous slices
- dispatch source headers

二级布局：

```text
dispatch_region
├── dispatch_header
├── expert_payload_blocks
└── scale_sideband_blocks
```

规则：
- source 先写这里
- receiver 只从这里 `TGET`
- compute 不直接读原始输入区
- **MVP 的物理合同优先采用 legacy/v2/v3 一致的“per-row packed layout”**：一行 dispatch 记录默认是 `payload + per-token scale sidecar (+ 对齐填充)` 的单地址连续块；
- `scale_sideband_blocks` 在设计文档里可以作为逻辑命名存在，但第一版不要把它误实现成“必须二次寻址才能拿到 scale 的完全分离平面”，否则会偏离当前 MegaMoE 主链里 `TGET -> 本地拆包 -> GMM-ready` 的简单通信合同。

### 8.4 `compute_region`
只放本 rank 本地 compute scratch。

内容：
- local expert-major input buffer
- GMM1 output
- SwiGLU temp
- GMM2 output
- optional dequant / scale temp
- per-group local state

二级布局：

```text
compute_region
├── compute_header
├── expert_input_blocks
├── gmm1_out_blocks
├── swiglu_tmp_blocks
└── gmm2_out_blocks
```

规则：
- 这是纯本地 region
- 不作为远端 payload 协议区

### 8.5 `combine_region`
只放待被 push 回来的结果 payload。

内容：
- remote experts 回传的 partial output
- per-destination row slices
- optional reduce staging blocks

二级布局：

```text
combine_region
├── combine_header
├── destination_payload_blocks
└── reduce_staging_blocks
```

规则：
- producer `TPUT` 到目标 rank 的这里
- consumer 从这里做 unpermute / reduce
- combine region 不反向复用为 dispatch source

### 8.6 `signal_region`
只放 signal / epoch / summary counter。

内容：
- dispatch-ready epochs
- dispatch-done epochs
- compute-stage epochs
- combine-ready epochs
- combine-done epochs
- group summary counters

二级布局：

```text
signal_region
├── dispatch_signal_table
├── compute_signal_table
├── combine_signal_table
└── summary_counter_table
```

规则：
- 不放大结构体
- 只放单调推进的小控制量
- payload readiness 只由这里发布

### 8.7 `RemoteWindowContext`

职责：kernel 唯一可见的 remote window 描述。

字段固定为：
- `workspaceBase`
- `workspaceBytes`
- `rank`
- `rankSize`
- `segmentBytes`
- `controlRegionOffset`
- `dispatchRegionOffset`
- `computeRegionOffset`
- `combineRegionOffset`
- `signalRegionOffset`
- `windowIn[]`
- `windowOut[]`

要求：
- 只描述地址和布局
- 不携带 host HCCL 语义
- 不携带业务态状态

---

## 9. signal / epoch 协议

### 9.1 统一 signal 模型

v4 不用一次性 bool flag，统一用 **monotonic epoch / counter**。

最小信号族：
- `dispatchSrcReadyEpoch[peer][group]`
- `dispatchDstDoneEpoch[peer][group]`
- `computeGmm1DoneEpoch[group]`
- `computeSwigluDoneEpoch[group]`
- `computeGmm2DoneEpoch[group]`
- `combineDstReadyEpoch[peer][group]`
- `combineSrcDoneEpoch[peer][group]`
- `summaryCounter[group]`

### 9.2 为什么用 epoch

1. bool flag 容易复用出错
2. epoch 更适合 chunk / group 多轮推进
3. `TWAIT/TTEST` 天然适合等“达到某 epoch”
4. 后续做 overlap 不需要重写协议，只需细化推进粒度

### 9.3 publish / consume 纪律

#### dispatch source publish
source rank：
1. 写 `dispatch_region` payload
2. 本地 fence / barrier
3. 更新 `dispatchSrcReadyEpoch`
4. remote receiver 才可见

receiver rank：
1. `TWAIT/TTEST` 等 `dispatchSrcReadyEpoch`
2. 执行 `TGET`
3. 本地 fence
4. 更新 `dispatchDstDoneEpoch`

#### combine publish
producer rank：
1. 写本地 `gmm2_out`
2. 生成 `CombinePushTask`
3. 执行 `TPUT` 到 remote `combine_region`
4. fence
5. 更新 `combineDstReadyEpoch`

consumer rank：
1. `TWAIT/TTEST` 等 `combineDstReadyEpoch`
2. 从 `combine_region` 消费 payload
3. 执行 unpermute / reduce
4. 更新 `combineSrcDoneEpoch`

### 9.4 `TWAIT` 与 `TTEST` 分工

- `TWAIT`：
  - Phase 0-5 的明确阶段边界
  - 必须阻塞等待的 critical dependency
- `TTEST`：
  - Phase 6 的 probe-first overlap
  - ready queue / summary counter 驱动推进

---

## 10. routing planner 详细 contract

### 10.1 职责边界

routing planner 只做 4 件事：
1. expand
2. group / sort
3. count / prefix / exact offset
4. task materialization

不负责：
- 执行 `TGET/TPUT`
- GMM 主算
- unpermute / reduce
- 底层 signal wait / notify 实现

### 10.2 `RoutingMetadataView`

这是 routing planner 的唯一真值视图。

字段固定为：
- `expandedRowIdx[]`
- `expandedExpertIdx[]`
- `expandedProb[]`
- `expandedRankIdx[]`
- `expandedLocalExpertSlot[]`
- `localTokenPerExpert[localExpert]`
- `tokenPerExpert[srcRank][localExpert]`
- `gatheredExpertCount[rank][globalExpert]`（归档 / debug 投影，不是地址真值主表）
- `localExpertPrefix[]`
- `globalExpertPrefix[]`
- `cumsumMM[localExpert][srcRank]`
- `srcOffset[]`
- `dstOffset[]`
- `packedRowCount[rank][group]`
- `groupRowCount[group]`
- `rowToExpandedRange[row]`

字段语义：
- `expandedRowIdx[]`：第 i 个 expanded 条目来自哪个原始 row
- `expandedExpertIdx[]`：第 i 个 expanded 条目路由到哪个 global expert
- `expandedProb[]`：第 i 个 expanded 条目的路由权重
- `expandedRankIdx[]`：该 expert 所在目标 rank
- `expandedLocalExpertSlot[]`：该 expert 在目标 rank 内的 local slot
- `localTokenPerExpert[]`：本 rank 发往每个 local expert 的 token 数
- `tokenPerExpert[srcRank][localExpert]`：目标 rank 视角下，各来源 rank 给各本地 expert 贡献了多少 token
- `gatheredExpertCount`：全局专家视角的 host/debug 汇总矩阵
- `cumsumMM[localExpert][srcRank]`：**按 source-rank 顺序做的累积和表**；单元格存的是“截至该 `srcRank` 为止”的累计 token 数，因此第一个来源的起点是 0，后续来源的落点取前一个 `srcRank` 的累计值
- `srcOffset[]`：expanded 条目在源 payload 中的 packed 偏移
- `dstOffset[]`：expanded 条目在目标 expert-major input 中的 packed 偏移
- `rowToExpandedRange[row]`：原始 row 对应的 expanded 条目范围

### 10.3 mask / sentinel expert 语义

legacy / v2 / v3 的真实链路里，`x_active_mask` 不是“跳过整个 token”，而是把 inactive token 的 `expertIdx` 改写到一个 **sentinel expert**：

- host 侧 `expertNum = worldSize * expertPerRank + 1`
- device 侧对 inactive token 写入 `expertId == worldSize * expertPerRank`

因此 v4 必须保留这条语义：
- routing prelude 允许存在一个 **仅用于屏蔽 inactive token 的逻辑哨兵 expert**；
- 这个 expert 不进入真实 `localExpertSlot` 计算，不参与 GMM，也不进入 `DispatchPullTask[]` / `CombinePushTask[]` 的正常 local expert 空间；
- 但 planner 必须能解释它，否则 `x_active_mask` 场景下的 `expandedRowIdx` / count 真值会错位。

### 10.4 capacity / truncation 语义

legacy / v2 / v3 的主链不是“所有路由到本 rank 的 token 都必进 GMM”，而是**受 `maxOutputSize` 限制的 capacity-bounded 执行**。

更准确地说，前置阶段要区分两层真值：
1. **before-capacity truth**
   - 原始 expand、`expandedRowIdx`、`tokenPerExpert`、`cumsumMM`
2. **after-capacity executable truth**
   - 真正进入本地 `expert-major` 输入区、真正进入 `GMM1/GMM2`、真正进入 combine 的那部分 rows

因此 v4 需要明确：
- `maxOutputSize` 是每个目的 rank 的 routed-token capacity，而不是单个 expert 的 capacity；
- `DispatchPullTask[]` 与 `CombinePushTask[]` 必须在 materialize 时就完成截断，不要把“先全量建 task，运行时再丢弃”留给 exchange engine；
- output restore 只消费 after-capacity 的已执行结果，不能假设所有 expanded 条目都会产出有效 expert output。

### 10.5 `DispatchPullTask`

routing planner 生成，exchange engine 消费。

字段固定为：
- `srcRank`
- `dstRank`
- `expertGroupId`
- `expertId`
- `localExpertSlot`
- `srcRowBegin`
- `dstRowBegin`
- `rowCount`
- `hiddenBytes`
- `scaleBytes`
- `srcPayloadOffsetBytes`
- `dstPayloadOffsetBytes`
- `readyEpoch`
- `taskFlags`

约束：
- 一个 task 对应一个连续源片段和一个连续目标片段
- exchange engine 不允许在运行时随意重拆 task
- task 粒度就是后续 `TGET` / chunk scheduling 的最小单位

### 10.6 `CombinePushTask`

字段固定为：
- `srcRank`
- `dstRank`
- `expertGroupId`
- `expertId`
- `srcRowBegin`
- `dstRowBegin`
- `rowCount`
- `outputBytes`
- `srcPayloadOffsetBytes`
- `dstPayloadOffsetBytes`
- `completionEpoch`
- `taskFlags`

约束：
- `CombinePushTask` 必须能由 compute 完成后直接 materialize
- output engine 只依赖它和 `rowToExpandedRange[]`
- **combine 的目标落点不能从 `cumsumMM` 反推**；它必须来自 owner-rank 视角的前缀真值（legacy 中接近 `preSumBeforeRank` 的角色），或者直接固化进 `dstPayloadOffsetBytes`

### 10.7 planner 生成顺序

顺序固定为：
1. expand：`topkIds/topkProbs -> expandedRowIdx/expandedExpertIdx/expandedProb`
2. annotate：根据 expert->rank 映射补 `expandedRankIdx/expandedLocalExpertSlot`
3. local count：生成 `localTokenPerExpert/groupRowCount`
4. gathered count：count exchange 后形成 `gatheredExpertCount`
5. prefix / exact offsets：生成 `srcOffset/dstOffset/packedRowCount`
6. combine-owner prefix：生成 owner-rank 视角的目的落点前缀，供 `CombinePushTask.dstPayloadOffsetBytes` 使用
7. task materialization：生成 `DispatchPullTask[]/CombinePushTask[]`

结论：
- **count exchange 是 routing planner 的最后一个同步点**
- 一旦 offsets 固定，后面的 exchange / compute / output 都没有布局解释权

---

## 11. exchange engine 详细设计

### 11.1 角色划分

exchange engine 运行在 AIV 侧，负责：
- 执行 dispatch pull
- 执行 combine push
- 维护 group / chunk 级 progress
- 驱动 payload 与 signal 的正确时序

AIV 不负责：
- 重新推导 routing 布局
- 执行 GMM 主算
- 做最终 weighted reduce

### 11.2 dispatch engine

职责：
- 根据 `DispatchPullTask[]` 执行 `TGET`
- 先搬 payload，再搬 optional sideband
- 写入 `compute_region.expert_input_blocks`
- 更新本地 dispatch progress

最小执行流：
1. 读取当前 group 的 `DispatchPullTask[]`
2. 对每个源 peer 用 `TWAIT/TTEST` 等 `dispatchSrcReadyEpoch`
3. 按 task 连续执行 `TGET`
4. 本地 fence
5. 更新 `dispatchDstDoneEpoch`
6. 若该 group 全部完成，增加 `summaryCounter[group]`

### 11.3 combine engine

职责：
- 根据 `CombinePushTask[]` 执行 `TPUT`
- 把 `gmm2_out_blocks` 发回目标 rank 的 `combine_region`
- 正确发布 `combineDstReadyEpoch`

最小执行流：
1. 等本地 `computeGmm2DoneEpoch[group]`
2. materialize 当前 group 的 `CombinePushTask[]`
3. 对每个目标 peer 执行 `TPUT`
4. 本地 fence
5. 发布 `combineDstReadyEpoch`
6. 若该 group 全部完成，更新 `combineSrcDoneEpoch`

### 11.4 chunk / group 推进规则

第一版固定：
- 先按 `expertGroupId` 推进
- group 内 task 保持连续块粒度
- 不做更细的 subtile scheduling

这样做的原因：
- 先建立稳定协议边界
- 避免第一版就把调度粒度复杂化
- 为 Phase 6 预留向 queue / summary 的细化空间

### 11.5 exchange 对外 contract

输入：
- `RemoteWindowContext`
- `RoutingMetadataView`
- `DispatchPullTask[]`
- `CombinePushTask[]`
- signal / epoch 表

输出：
- `compute_region.expert_input_blocks`
- `combine_region.destination_payload_blocks`
- dispatch/combine progress epochs

---

## 12. compute engine 详细设计

### 12.1 职责边界

compute engine 只负责：
- 读取已 ready 的 expert-major 输入
- 执行 `GMM1 -> SwiGLU -> GMM2`
- 产出可直接被 combine engine 消费的连续输出块

不负责：
- 推导 routing offset
- 管理 remote window
- 解释 dispatch / combine 协议

### 12.2 计算数据面

compute engine 输入固定来自：
- `compute_region.expert_input_blocks`

中间结果固定落到：
- `compute_region.gmm1_out_blocks`
- `compute_region.swiglu_tmp_blocks`
- `compute_region.gmm2_out_blocks`

这样做的目的：
- 每个阶段 buffer 都有固定归属
- combine engine 只需要消费 `gmm2_out_blocks`
- debug / golden 更容易定位

### 12.3 计算子模块

#### `gmm1.hpp`
职责：
- 对 ready 的 expert-major block 执行第一层 matmul
- 通过 `substrate/pto_mmad_shell.hpp` 进入底层 mmad

#### `swiglu.hpp`
职责：
- 对 `gmm1_out_blocks` 做门控与激活
- 使用 PTO 向量 / 规约原语表达业务逻辑

#### `gmm2.hpp`
职责：
- 执行第二层 matmul
- 输出最终 expert partial result

#### `epilogue.hpp`
职责：
- 做 dequant / requant / store-side formatting
- 保证输出块布局和 `CombinePushTask` 一致

### 12.4 compute 与 substrate 的边界

业务层只允许：
- 描述 block / group / layout / tile 关系
- 调用 `pto_mmad_shell` / `pto_fixpipe_shell` 暴露的窄接口

业务层禁止：
- 直接 include `kernel_operator.h`
- 直接 include `lib/matmul_intf.h`
- 直接调用 `Gemm::Tile::*` / `Gemm::helper::*`

### 12.5 compute 完成信号

每个 `expertGroupId` 至少发布：
- `computeGmm1DoneEpoch[group]`
- `computeSwigluDoneEpoch[group]`
- `computeGmm2DoneEpoch[group]`

Phase 0-5 中：
- `combine engine` 只等待 `computeGmm2DoneEpoch[group]`

Phase 6 中：
- 允许更细粒度地 probe `computeGmm1DoneEpoch/group`

### 12.6 MVP 先保留 legacy 的两段式 SwiGLU overlap 基线

虽然 MegaMoE 文档最终追求的是更细的 coarse/fine 激活流水，但 legacy / v2 / v3 当前已经被验证的稳定主链，其实是一个更简单的 **两段式** overlap：

- `GMM1` 先推进到一个前段边界；
- AIV 对前段执行第一波 `dequant + SwiGLU + quant/reformat`；
- `GMM2` 在收到这一波 ready 后先消费前段；
- 后段再重复一次同样流程。

当前代码里这条基线体现在 `epilogueGranularity` 这类启发式切分上；因此 v4 在 Phase 0-5 应保留同类策略作为 MVP：
- 先允许一个简单的前/后两段分界；
- 默认仍以 expert 顺序推进，不追求一开始就做 `{8,4,2,1,1}` 这种多段 coarse/fine 调度；
- 只要 `GMM1 -> SwiGLU -> GMM2` 的两段错拍成立，就满足第一版 compute overlap 基线。

一个可接受的初始策略是复用 legacy 风格的经验规则：
- `expertPerRank > 4` 时，前段边界可先取接近 `expertPerRank - 3`；
- `expertPerRank <= 4` 时，前段边界退化到接近 `expertPerRank - 1`；
- 这只是 Phase 0-5 的保守基线，不是最终的 Phase 6 调度最优解。

---

## 13. output restore 详细设计

### 13.1 职责边界

output engine 只负责：
- 消费 `combine_region.destination_payload_blocks`
- 依据 `rowToExpandedRange[]` 恢复原 token 顺序
- 依据 `expandedProb[]` 做 top-k weighted reduce
- 形成最终输出 tensor

不负责：
- 重做通信偏移推导
- 重做 expert regroup
- 管理 remote signal

### 13.2 输入真值

output engine 只依赖：
- `expandedRowIdx[]`
- `expandedProb[]`
- `rowToExpandedRange[]`
- `CombinePushTask[]` 的目标布局信息

### 13.3 执行流

1. 等目标 group 的 `combineDstReadyEpoch`
2. 从 `combine_region` 读取目标 row 的 partial outputs
3. 依据 `rowToExpandedRange[]` 聚合该 row 的 expanded entries
4. 用 `expandedProb[]` 做 weighted sum
5. 写到最终输出 tensor

### 13.4 为什么单独拆 output engine

因为它的业务语义与 compute 不同：
- compute 关注 expert-major block
- output 关注 row-major restore

把它单独拆出，能防止 `gmm2` 后又把还原逻辑揉回 compute 大文件。

---

## 14. host/runtime/harness 详细设计

### 14.1 host 侧职责

host/runtime 只负责：
- ACL init / finalize
- device / stream 生命周期
- HCCL bootstrap
- remote window resource 申请与地址导出
- workspace 分配与 region base 计算
- kernel launch 参数与 tiling blob 组装
- case 输入生成 / 装载
- CPU golden 生成
- 输出校验
- perf 记录

### 14.2 kernel 不可见的 host 语义

这些语义只允许停留在 host：
- `HcclComm`
- `Hcom*`
- remote resource bootstrap 细节
- MPI 加载细节

kernel 只看见：
- `RemoteWindowContext`
- input/output global pointers
- tiling/config blob

### 14.3 standalone harness 产物

host 层最终要能提供：
- small case
- large case
- metadata dump 开关
- golden 对比开关
- perf 记录开关

### 14.4 tiling builder 职责

tiling builder 负责：
- 根据 `m/k/n/topk/expert/world_size` 生成 block / group / chunk 配置
- 确定每个 region 的逻辑大小
- 确定每组 `expertGroupId` 的调度分区
- 生成 kernel 可直接消费的 config blob

tiling builder 不负责：
- 在 host 侧执行 routing 数据推导
- 生成运行时临时 metadata 真值

### 14.5 `main.cpp` 的模式合同

`main.cpp` 不是随意堆逻辑的调试入口，而是 v4 standalone harness 的**唯一模式分发器**。

第一版应固定支持这些 mode：
- `signal-roundtrip`
- `metadata-only`
- `dispatch-only`
- `compute-only`
- `combine-only`
- `full-chain`

职责边界：
- 解析 CLI / case 配置
- 初始化 ACL / device / stream / rank runtime
- 调用 `tiling_builder`、`kernel_launch`、host reference helpers
- 根据 mode 选择最小可验证链路
- 统一输出 `PASS/FAIL`
- 统一写 `out/` 目录下的 case、dump 产物，并在 stdout 输出 perf 日志

禁止项：
- 不在 `main.cpp` 里内联 routing 真值推导实现
- 不在 `main.cpp` 里直接解析 remote window 物理布局
- 不把 kernel 业务逻辑分支塞回 host CLI 分发层

### 14.6 `run.sh` 的脚本合同

`run.sh` 是**工程脚手架**，不是算法实现层的一部分。

它的职责固定为：
1. 加载约定环境（Ascend / MPI / PTO runtime）
2. 解析最小参数集：
   - `--soc`
   - `--world-size`
   - `--mode`
   - `--m --k --n`
   - `--topk`
   - `--experts`
   - `--max-output-size`
   - 可选 `--warmup --measure --dump-metadata --check-golden --record-perf`
3. 配置并构建工程
4. 通过 `mpirun` 启动 standalone 可执行文件
5. 把 stdout/stderr 和 `out/` 产物保留给后续分析

它**不负责**：
- 在 shell 中推导 routing truth
- 在 shell 中做 golden compare
- 在 shell 中拼装 kernel 内部协议字段

推荐行为：
- 默认把 build 输出放到 `csrc/mc2/dispatch_ffn_combine_v4/build/`
- 默认把运行产物放到 `csrc/mc2/dispatch_ffn_combine_v4/out/`
- 默认允许同一脚本直接切换上述 6 个 mode，而不是为每个 mode 再写一套独立脚本

### 14.7 perf 日志与产物合同

v4 的 perf 设计必须和 correctness harness 一起定义，但**第一版只要求 stdout/stderr 的稳定日志格式**，不额外生成 perf JSON 文件。

第一版固定产物：
- `out/case.json`
- `out/output_rank{rank}.bin`
- `out/rank{rank}_expected_out.bin`
- 可选 `out/metadata_rank{rank}.json`

perf 交付方式固定为：
- 由 standalone 可执行文件直接打印到 stdout
- `run.sh` 负责保留 stdout/stderr 原始日志
- 不要求 `out/perf_rank*.json`、`out/perf_summary.json` 这类单独 perf 文件

stdout 格式要求：
- 采用稳定的 `key=value` 风格，便于 grep/README 记录
- 至少分成“计时指标”和“逻辑工作负载派生指标”两行
- 若阶段拆分计时已经可用，可额外打印一行 stage breakdown

推荐日志样例：
- `perf kernel_us_avg=123.4 kernel_us_min=121.0 kernel_us_max=126.8 e2e_us_avg=140.2 e2e_us_min=138.7 e2e_us_max=143.9`
- `perf input_tokens_per_sec=8.1e6 routed_tokens_per_sec=1.6e7 equivalent_tflops=42.7 equivalent_gbps=318.5`
- 可选：`perf_stage dispatch_us_avg=11.2 gmm1_us_avg=37.9 swiglu_us_avg=9.3 gmm2_us_avg=34.8 combine_us_avg=14.0 output_us_avg=6.2`

### 14.8 perf 指标口径

第一版 stdout perf 日志分三层：

#### A. run config
- `warmup_iterations`
- `measure_iterations`
- `mode`
- `world_size`
- `m/k/n/topk/experts/max_output_size`

#### B. core timing
- `kernel_us_avg`
- `kernel_us_min`
- `kernel_us_max`
- `e2e_us_avg`
- `e2e_us_min`
- `e2e_us_max`
- 可选 `dispatch_us_avg / gmm1_us_avg / swiglu_us_avg / gmm2_us_avg / combine_us_avg / output_us_avg`

#### C. workload-derived metrics
延续 v2/v3 的思路，但只保留对 review 和后续优化有用的稳定口径：
- `input_tokens_all_ranks`
- `routed_tokens_all_ranks`
- `remote_routed_tokens_all_ranks`
- `compute_flops_all_ranks`
- `comm_bytes_all_ranks`
- `input_tokens_per_sec`
- `routed_tokens_per_sec`
- `equivalent_tflops`
- `equivalent_gbps`

关键约束：
- **这些是 logical workload 派生指标，不是硬件 counter**；
- README 与 stdout 日志都必须把这一点写明，避免把“等效带宽/算力”误解成芯片真实 PMU 数据；
- 第一版先保证口径稳定和跨 case 可比较，不强求上来就接硬件计数器。

### 14.9 perf 记录开关与阶段关系

- Phase 0-5：perf 开关存在，但主要用于记录，不作为准出门槛
- Phase 6：perf 开始成为显式交付物，需要能在 stdout 输出 per-stage breakdown 和 overlap 结果
- 任意阶段都不允许为了短期 perf 漂亮，回退已明确固定的 PTO-first 模块边界

---

## 15. 文件级 ownership 规则

推荐目录结构：

```text
csrc/mc2/dispatch_ffn_combine_v4/
├── CMakeLists.txt
├── DESIGN.md
├── IMPLEMENTATION_PLAN.md
├── main.cpp
├── runtime_context.hpp
├── runtime_context.cpp
├── tiling_builder.hpp
├── tiling_builder.cpp
├── case_io.hpp
├── case_io.cpp
├── data_utils.cpp
├── kernel_launch.hpp
├── run.sh
├── tests/
├── out/
└── op_kernel/
    ├── dispatch_ffn_combine.cpp
    ├── dispatch_ffn_combine.h
    ├── dispatch_ffn_combine_tiling.h
    ├── protocol/
    ├── routing/
    ├── dispatch/
    ├── compute/
    ├── combine/
    ├── output/
    └── substrate/
```

ownership 规则：
- `CMakeLists.txt`
  - 只负责 target graph、include/link 边界、test target、AscendC kernel target 与 standalone host target 的装配
  - 不承载算法常量和 case 逻辑
- `main.cpp` / `runtime_context.*` / `tiling_builder.*`
  - 只负责 host 生命周期、workspace、tiling、launch
- `run.sh`
  - 只负责环境、configure/build/run 包装和 mode 透传
- `tests/*`
  - 只负责 host-only/unit-level correctness seam，不承载运行时脚本逻辑
- `out/*`
  - 只作为运行产物目录，不作为源码或手写配置目录
- `op_kernel/protocol/*`
  - 只负责 remote window、signal、fence、task、queue contract
- `op_kernel/routing/*`
  - 只负责 routing metadata 真值，不接触 Gemm/fixpipe 壳
- `op_kernel/dispatch/*`
  - 只负责 dispatch pull 协议执行
- `op_kernel/compute/*`
  - 只描述 GMM1/SwiGLU/GMM2 业务流程
  - 若必须进入底层，只能经由 substrate 窄接口
- `op_kernel/combine/*`
  - 只负责 combine push 协议执行
- `op_kernel/output/*`
  - 只负责 restore/reduce
- `op_kernel/substrate/*`
  - 是唯一允许直接碰 `kernel_operator.h` / `lib/matmul_intf.h` / `CrossCore*` / `Fixpipe` 的目录

### 15.1 `CMakeLists.txt` 的目标图设计

v4 的 `CMakeLists.txt` 应该显式表达 4 类 target：

1. **kernel target**
   - 例如：`dispatch_ffn_combine_v4_kernel`
   - 只编译 `op_kernel/*`
   - 承载 AscendC kernel entry、kernel include、SoC 相关编译定义

2. **standalone host executable**
   - 例如：`dispatch_ffn_combine_v4`
   - 编译 `main.cpp`、`runtime_context.*`、`tiling_builder.*`、`case_io.*`、`data_utils.cpp`
   - 链接 ACL/HCCL/MPI 以及 kernel stub 所需产物

3. **host-only test targets**
   - 例如：`test_workspace_layout`
   - `test_signal_protocol`
   - `test_routing_metadata`
   - `test_task_materialization`
   - `test_compute_reference`

4. **可选聚合 target**
   - 例如 `check` 或等价聚合目标
   - 用于一次性构建/执行 host-only 测试集合

关键约束：
- host-only test target 不依赖真实多 rank NPU 运行；
- kernel target 与 host test target 的 include 面要分开，防止业务逻辑反向依赖 host harness；
- `CMakeLists.txt` 必须让“先只做 host correctness，再接 device kernel”成为自然工作流，而不是所有 target 从第一天就强耦合在一起。

### 15.2 `run.sh` 与 `CMakeLists.txt` 的关系

- `CMakeLists.txt` 定义“能构建什么”；
- `run.sh` 定义“按什么方式运行这些 target”；
- 两者之间的 mode 命名必须一一对应：
  - `signal-roundtrip`
  - `metadata-only`
  - `dispatch-only`
  - `compute-only`
  - `combine-only`
  - `full-chain`

禁止出现：
- `run.sh` 有 mode，但 `main.cpp` 不认
- `main.cpp` 有 mode，但 `run.sh` 不透传
- 文档中的模式名与 CMake/可执行实际行为不一致

---

## 16. 分阶段落地计划

### Phase 0：standalone harness + protocol skeleton
目标：
- 建最小 standalone 工程
- 跑通 ACL/HCCL/MPI 生命周期
- 建立 remote window
- 跑通最小 signal roundtrip

交付：
- `RemoteWindowContext`
- workspace region base 计算
- signal region 最小发布 / 等待样例

准出：
- 可 build
- 可启动多 rank
- 可完成最小 signal roundtrip
- 没有对 `dispatch_ffn_combine_v4/` 目录外业务代码的 include

### Phase 1：routing metadata planner only
目标：
- 只实现 routing metadata planner
- 先不进真实 dispatch / compute

交付：
- `RoutingMetadataView`
- `DispatchPullTask[]`
- `CombinePushTask[]`
- CPU golden for count / cumsum / offsets / expandedRowIdx

准出：
- 双 rank metadata 与 golden 一致
- task 表可打印 / 校验
- offsets 与 packedRowCount 稳定

### Phase 2：dispatch pull MVP
目标：
- 只实现 `Dispatch = TGET`
- payload 到本地 expert-major 输入区
- 先不接完整 compute

交付：
- `dispatch_pull.hpp`
- `dispatch_progress.hpp`
- dispatch source / ready epoch 协议

准出：
- 本地 expert-major buffer 正确
- ready / epoch protocol 正确
- `payload -> fence -> notify -> wait -> consume` 纪律成立

### Phase 3：compute MVP
目标：
- 单 rank 跑通 `GMM1 -> SwiGLU -> GMM2`
- 输入直接喂本地 expert-major buffer

交付：
- `gmm1.hpp`
- `swiglu.hpp`
- `gmm2.hpp`
- `epilogue.hpp`
- 窄接口 substrate shells

准出：
- 中间结果有 golden
- compute 业务层不直接暴露底层 Gemm/AscendC 细节
- `gmm2_out_blocks` 布局可直接供 combine 使用

### Phase 4：combine push MVP + restore
目标：
- 实现 `Combine = TPUT`
- 算完即发
- 本地恢复输出顺序并做 weighted reduce

交付：
- `combine_push.hpp`
- `combine_progress.hpp`
- `unpermute_reduce.hpp`

准出：
- final output correct
- completion epoch 正确
- output engine 只依赖 routing / combine 真值，不反推协议细节

### Phase 5：fused full chain correctness
目标：
- 把 metadata / dispatch / compute / combine / restore 串成完整链

交付：
- end-to-end standalone 可执行样例
- small / large case driver
- metadata dump / compare 工具链

准出：
- small case PASS
- large case PASS
- 最终输出与关键 metadata 都可校验

### Phase 6：overlap / scoreboard / perf
目标：
- 引入 expert-group / queue / summary counter 推进
- 用 `TTEST` + `TWAIT` 做更细粒度重叠
- 把 dispatch/compute/combine 的错拍关系从 group 级推进到更细粒度
- 记录 kernel / e2e / per-stage 性能

交付：
- queue / summary 结构
- group overlap 调度器
- README / perf 记录
- coarse-to-fine expert-group 切分策略

这一阶段要显式补上 MegaMoE 的两类重叠语义：

#### A. 三级深度融合流水掩盖
第一版 full-chain 正确性通过后，允许把一条 group 内流水细化成三段错拍：
1. **前一段**：下一个 chunk / group 的 dispatch-side quant / pack / ready publication
2. **中间段**：当前 chunk / group 的 `GMM1 -> SwiGLU -> GMM2`，其中 cube 消费按需要伴随 dequant
3. **后一段**：上一个 chunk / group 的 combine-side float store / `TPUT` / output-ready publication

这意味着 v4 的 buffer 和 signal 设计必须提前留出：
- dispatch-side quant sideband 的 staging 位；
- compute 中间块与 combine 输出块的双缓冲/轮转位；
- `dispatch ready -> compute ready -> combine ready` 的单调阶段边界。

MVP 不要求一开始就把这三段完全铺满，但文档与协议必须允许后续按这个方向细化，而不是卡死在“一个 group 一次只做一件事”的串行结构上。

#### B. SwiGLU 粗-细粒度结合调度
MegaMoE 的 expert token 数通常长尾明显，因此 Phase 6 不应把激活段永久锁死在固定大 group 上。

推荐调度原则：
- 中前段 token 较多时，允许按较粗 group 推进，减少调度和 signal 开销；
- 尾段 token 变稀时，把 group 逐渐拆细，避免少量 residual experts 被一个大 group 拖住；
- 一个可接受的参考模式是类似 `{8, 4, 2, 1, 1}` 的 coarse-to-fine 收缩，而不是“全程固定 8”或“全程都拆到 1”。

这不是在 Phase 6 硬编码具体数组，而是要求：
- `expertGroupId` 与 ready queue 设计允许 coarse/fine 切分并存；
- SwiGLU / GMM2 的激活与后处理可以跟随 group 粒度变化，而不破坏前面固定好的 metadata 真值。

准出：
- overlap 生效
- README 中开始记录 kernel/e2e/per-stage 性能
- 不为了性能回退到历史大壳结构
- queue / summary / coarse-fine group 策略能解释 dispatch/compute/combine 的实际重叠次序

---

## 17. 验证与 golden 计划

### 17.1 代码检查
- grep HCCL / HCOM：只允许停留在 host/runtime
- grep `kernel_operator.h` / `lib/matmul_intf.h`：只允许停留在 substrate 层
- grep `Gemm::Tile::*` / `Gemm::helper::*`：业务层必须为 0

### 17.2 分阶段 golden
至少要有：
- routing metadata golden
- dispatch payload golden
- GMM1 中间结果 golden
- SwiGLU 中间结果 golden
- GMM2 中间结果 golden
- combine payload golden
- final output golden

### 17.3 运行级验证
standalone build / run：
- 最小协议样例
- small case
- large case

结果要求：
- build 成功
- 双 rank PASS
- 输出与 golden 一致

### 17.4 性能记录原则
在功能链通之前：
- 允许记录性能
- 不允许为了追性能提前把架构压回历史壳层

---

## 18. 风险与选择

### 18.1 已明确选择
- dispatch 用 `TGET`
- combine 用 `TPUT`
- metadata / payload / signal 分区分离
- routing planner 是唯一 metadata 真值源
- 第一版 overlap 粒度固定到 expert-group，不直接下探 subtile

### 18.2 主要风险
- `compute_region` buffer 预算可能受 shape / group 设计影响
- first workable group size 需要 tiling builder 配合验证
- matmul/fixpipe shell 的窄接口若收得不够窄，会污染 compute 层边界

### 18.3 风险处理策略
- 先固定 contract，再让 Phase 0-3 验证容量和接口可达性
- 若容量不足，优先改 group/chunk/tiling，不破坏模块边界
- 若 substrate 壳过宽，先收窄接口，再继续功能扩展

---

## 19. 最终结论

v4 的正确设计顺序应该是：

1. **从 MegaMoE 设计拆需求**
2. **用 PTO 原语、类型系统、示例模式逐项承接这些需求**
3. **只在 PTO 没覆盖的 window / protocol / runtime 层补薄壳**
4. **最后形成新的 standalone 工程结构**

所以，v4 的本质不是“`dispatch_ffn_combine` 的新版本”，而是：

> **一个以 MegaMoE 需求为上层语义、以 PTO 为主实现语言、以 standalone harness 为运行边界、以极薄 protocol/substrate shell 为支撑的全新项目。**
