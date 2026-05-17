# dispatch_ffn_combine_v4 task status

## 当前总览

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 详细设计 | 已完成 | 已写入 [DESIGN.md](DESIGN.md)，并已重排为“算法总述 → 真值来源/约束 → 模块合同 → 工程脚手架”结构 |
| 实施计划 | 已完成 | 已写入 [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)，并已同步为同样的“算法总述优先”阅读顺序 |
| 文档 review | 已完成 | 已补齐 PTO 三层边界说明与 T3 routing truth，文档集现在可作为 T1 起点 |
| 代码实现 | 进行中 | T1-T6 已完成并通过当前阶段验证；`full-chain` 仅完成串行 MVP 贯通，真实 MegaMoE overlap 主线与最终 perf 语义仍在实现中 |

## 文档阶段任务状态

| 编号 | 任务 | 状态 | 当前产物 | 下一步 |
| --- | --- | --- | --- | --- |
| A1 | 固化 MegaMoE 算法总述、主链路图与阅读顺序 | 已完成 | `DESIGN.md` 第 2-3 章 | 已落盘 |
| A2 | 固化算法真值来源、设计原则、范围与 PTO 承接 | 已完成 | `DESIGN.md` 第 4-5 章 | 已落盘 |
| A3 | 收紧 PTO 边界为“public surface / protocol shell / substrate shell”三层 | 已完成 | `DESIGN.md` 第 5 章 | 可直接作为实现边界 |
| D1 | 固定系统边界与端到端数据流 | 已完成 | `DESIGN.md` 第 6-7 章 | 已落盘 |
| D2 | 固定 workspace / signal / routing / exchange / compute / output contract | 已完成 | `DESIGN.md` 第 8-13 章 | 已落盘 |
| D3 | 固定 host/runtime、run.sh、CMake、perf、phase、golden 与风险设计 | 已完成 | `DESIGN.md` 第 14-18 章 | 已落盘 |
| P1 | 形成文件级 implementation plan | 已完成 | `IMPLEMENTATION_PLAN.md` | 已落盘 |
| P2 | 同步 perf stdout-only 合同到实施计划 | 已完成 | `IMPLEMENTATION_PLAN.md` Task 9 | 已落盘 |
| P3 | 同步实施计划的阅读顺序为“算法总述优先” | 已完成 | `IMPLEMENTATION_PLAN.md` 头部说明与任务映射 | 已落盘 |
| P4 | 刷新 T4-T8 实现细节，使主线与 `DESIGN.md` 对齐 | 已完成 | `IMPLEMENTATION_PLAN.md` Task 4-8、准出条件 | 已落盘 |
| P5 | 强化 T3 为真实 routing truth planner，而非 toy metadata helper | 已完成 | `IMPLEMENTATION_PLAN.md` Task 3 | 可直接作为 T4/T6 control-plane 前提 |

## 实施阶段任务状态

| 编号 | 任务 | 状态 | 目标产物 | 备注 |
| --- | --- | --- | --- | --- |
| T1 | Scaffold standalone project and shared contracts | 已完成 | `CMakeLists.txt`、`main.cpp`、`runtime_context.*`、`remote_window.hpp` | 已通过 `test_workspace_layout` |
| T2 | Implement signal protocol and signal-roundtrip harness | 已完成 | `signal_protocol.hpp`、`fence.hpp`、`signal-roundtrip` 模式 | 已通过 `test_signal_protocol` 与 `signal-roundtrip` |
| T3 | Build routing metadata planner and host golden tests | 已完成 | `task_plan.hpp`、`route_plan.hpp`、metadata tests | 已通过 `test_routing_metadata`、`test_task_materialization`、`metadata-only` |
| T4 | Implement real device-backed dispatch pull MVP | 已完成 | `dispatch_pull.hpp`、`dispatch_progress.hpp`、`dispatch-only` 模式、dispatch oracle compare | 已通过 clean verification：`cmake --build build --target dispatch_ffn_combine_v4 -j16 && mpirun -np 2 build/dispatch_ffn_combine_v4 dispatch-only` 双 rank PASS；当前 remote pull 路径为 PTO `TGET -> scratch GM -> local GM copy` |
| T5 | Implement real device-backed compute MVP and narrow substrate shells | 已完成 | `gmm1.hpp`、`swiglu.hpp`、`gmm2.hpp`、substrate shells、compute oracle compare | 已通过 `build/test_compute_reference` 与 `build/dispatch_ffn_combine_v4 compute-only`；当前 compute-only 为固定 1x2 -> 1x4 -> 1x2 微链路，gmm1/gmm2 先经窄 AICORE shell 打通 device 主线 |
| T6 | Implement real device-backed combine push and output restore | 已完成 | `combine_push.hpp`、`unpermute_reduce.hpp`、`combine-only` 模式、combine oracle compare | 已通过 `build/test_task_materialization` 与 `mpirun -np 2 build/dispatch_ffn_combine_v4 combine-only`；当前为固定两卡 2/4 payload push + rank0 weighted restore 微样例 |
| T7 | Integrate real device-backed full correctness chain | 部分完成 | `full-chain` 模式、end-to-end device run、host full-chain oracle compare | 已通过 `mpirun -np 2 build/dispatch_ffn_combine_v4 full-chain` 双 rank PASS；但当前仅为串行 dispatch -> compute -> combine MVP，尚未进入 MegaMoE steady-state overlap |
| T8 | Consolidate device-backed mainline and keep host reference oracle-only | 已完成 | 统一 kernel launch surface、`HostFullChainOracle`、mode 主线收口 | host 侧仅保留 oracle/compare 入口；当前主线已收口到统一 device kernel launch surface，但仍承载串行 MVP 而非最终 overlap 主线 |
| T9 | Add overlap scaffolding and perf recording | 部分完成 | `ready_queue.hpp`、summary/progress、stdout perf 指标日志 | 已新增 `ready_queue.hpp` 与 `data_utils.*`，并输出 `perf mode=... stage=...` stdout 指标；但 queue/scoreboard 驱动的真实 overlap 与对应 perf 口径仍未完成 |

## 当前建议 review 顺序

| 顺序 | 动作 | 状态 |
| --- | --- | --- |
| 1 | review `DESIGN.md` 第 2-5 章，确认算法总述、真值来源与 PTO 约束 | 已完成 |
| 2 | review `DESIGN.md` 第 6-13 章，确认模块合同是否忠实承接算法链 | 已完成 |
| 3 | review `DESIGN.md` 第 14-18 章，确认脚手架/验证/perf 只是承载算法 | 已完成 |
| 4 | review `IMPLEMENTATION_PLAN.md`，确认任务映射顺序与算法主链一致 | 已完成 |
| 5 | 从 T1 开始进入实现 | 已完成 |
| 6 | 按 T1 -> T9 顺序推进实现 | 已完成 |

## 已落盘的关键结论

- MegaMoE 算法主链、dispatch/combine 非对称、routing 偏移真值已先在 `DESIGN.md` 中系统展开，再由模块合同承接
- `IMPLEMENTATION_PLAN.md` 已明确区分“算法核心任务（T3-T7）”和“支撑脚手架任务（T1-T2/T8-T9）”
- standalone `main.cpp` / `run.sh` / `CMakeLists.txt` 的工程边界已固定为承载算法，而不是重新定义算法
- perf 口径已固定为 stdout `key=value` 日志，不再引入额外 JSON 产物
- T3 已补成真实 routing truth planner；PTO 边界已补成 public surface / protocol shell / substrate shell 三层
- 当前 fresh verification 已覆盖 `test_compute_reference`、`test_task_materialization`、`dispatch-only`、`compute-only`、`combine-only`、`full-chain`
- 当前 `full-chain` 验证的是串行 device MVP 贯通；真实 MegaMoE overlap 的 ready-queue / scoreboard / steady-state perf 仍待补齐
