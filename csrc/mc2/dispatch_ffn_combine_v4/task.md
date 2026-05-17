# dispatch_ffn_combine_v4 task status

## 当前总览

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 详细设计 | 已完成 | 已写入 [DESIGN.md](DESIGN.md)，并已重排为“算法总述 → 真值来源/约束 → 模块合同 → 工程脚手架”结构 |
| 实施计划 | 已完成 | 已写入 [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)，并已同步为同样的“算法总述优先”阅读顺序 |
| 文档 review | 已完成 | 已补齐 PTO 三层边界说明与 T3 routing truth，文档集现在可作为 T1 起点 |
| 代码实现 | 已完成 | T1-T13 已形成可验证 standalone baseline；`full-chain` 主入口已切到 steady-state overlap，dispatch/compute/combine 分组流水、summary gate、W8A8 sideband、coarse-fine schedule 与 stdout perf 均已闭合 |

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
| P2 | 同步 perf stdout-only 合同到实施计划 | 已完成 | `IMPLEMENTATION_PLAN.md` Task 9 / Task 12 / Task 13 | 已落盘 |
| P3 | 同步实施计划的阅读顺序为“算法总述优先” | 已完成 | `IMPLEMENTATION_PLAN.md` 头部说明与任务映射 | 已落盘 |
| P4 | 刷新 T4-T8 实现细节，使主线与 `DESIGN.md` 对齐 | 已完成 | `IMPLEMENTATION_PLAN.md` Task 4-8、准出条件 | 已落盘 |
| P5 | 强化 T3 为真实 routing truth planner，而非 toy metadata helper | 已完成 | `IMPLEMENTATION_PLAN.md` Task 3 | 可直接作为 T4/T6 control-plane 前提 |
| P6 | 把剩余 MegaMoE 工作拆分为 steady-state overlap / scoreboard / task-driven combine / quant pipeline / coarse-fine SwiGLU 五条主线 | 已完成 | `IMPLEMENTATION_PLAN.md` Task 9-13 | 后续实现不再被压缩成一个弱化的 T9 |

## 实施阶段任务状态

| 编号 | 任务 | 状态 | 目标产物 | 备注 |
| --- | --- | --- | --- | --- |
| T1 | Scaffold standalone project and shared contracts | 已完成 | `CMakeLists.txt`、`main.cpp`、`runtime_context.*`、`remote_window.hpp` | 已通过 `test_workspace_layout` |
| T2 | Implement signal protocol and signal-roundtrip harness | 已完成 | `signal_protocol.hpp`、`fence.hpp`、`signal-roundtrip` 模式 | 已通过 `test_signal_protocol` 与 `signal-roundtrip` |
| T3 | Build routing metadata planner and host golden tests | 已完成 | `task_plan.hpp`、`route_plan.hpp`、metadata tests | 已通过 `test_routing_metadata`、`test_task_materialization`、`metadata-only` |
| T4 | Implement real device-backed dispatch pull MVP | 已完成 | `dispatch_pull.hpp`、`dispatch_progress.hpp`、`dispatch-only` 模式、dispatch oracle compare | 已通过 clean verification：`cmake --build build --target dispatch_ffn_combine_v4 -j16 && mpirun -np 2 build/dispatch_ffn_combine_v4 dispatch-only` 双 rank PASS；当前 remote pull 路径为 PTO `TGET -> scratch GM -> local GM copy` |
| T5 | Implement real device-backed compute MVP and narrow substrate shells | 已完成 | `gmm1.hpp`、`swiglu.hpp`、`gmm2.hpp`、substrate shells、compute oracle compare | 已通过 `build/test_compute_reference` 与 `build/dispatch_ffn_combine_v4 compute-only`；当前 compute-only 为固定 1x2 -> 1x4 -> 1x2 微链路，gmm1/gmm2 先经窄 AICORE shell 打通 device 主线 |
| T6 | Implement real device-backed combine push and output restore | 已完成 | `combine_push.hpp`、`unpermute_reduce.hpp`、`combine-only` 模式、combine oracle compare | 已通过 `build/test_task_materialization` 与 `mpirun -np 2 build/dispatch_ffn_combine_v4 combine-only`；当前 correctness path 已切到 `CombinePushTask[] + owner-prefix dst offset + weighted restore`，并使用 8-float PTO transport slot 承载 return write |
| T7 | Integrate real device-backed full correctness chain | 已完成 | `full-chain` 模式、end-to-end device run、host full-chain oracle compare | `full-chain` 已从串行 checkpoint 切到 overlap 主入口；small / large 参数路径均通过双 rank golden compare |
| T8 | Consolidate device-backed mainline and keep host reference oracle-only | 已完成 | 统一 kernel launch surface、`HostFullChainOracle`、mode 主线收口 | host 侧仅保留 oracle/compare 入口；主线已收口到统一 device kernel launch surface，并承载 overlap 主线 |
| T9 | Replace serial `full-chain` with a real steady-state overlap baseline | 已完成 | overlap `full-chain` mode、group timeline、`dispatch(g+1) / compute(g) / combine(g-1)` 执行基线 | `RunFullChainOverlap` 已作为 `full-chain` 主入口，按 `BuildSteadyStateTimeline` 驱动 dispatch/compute/combine 分组流水，并输出 per-tick/per-group perf |
| T10 | Build dispatch→compute scoreboard and probe-first readiness | 已完成 | ready queue、summary counter、software scoreboard、`TTEST`-first wait policy | 已接入 `DispatchGroupProgress`、`ReadyQueue`、summary gate 与 dispatch-side `TTEST`-first wait；当前是 standalone host-controller baseline，后续可继续下沉控制 AIV |
| T11 | Replace fixed combine micro-example with task-driven owner-prefix and tile-split return traffic | 已完成 | real `CombinePushTask[]`、owner-prefix offsets、tile/group completion publish | `combine-only` 与 overlap `full-chain` 均使用 task-driven owner-prefix offset；group combine 按全局 transport slot staging，避免 group-local task 压缩破坏 `srcPayloadOffsetBytes` |
| T12 | Restore W8A8 quant/dequant sideband pipeline and leave a clean W4A8 follow-on boundary | 已完成 | packed INT8 payload、scale sideband、compute-side dequant contract、quant-aware perf counters | compute path 已切到 `quantInput + scale1/scale2` sideband，device 侧执行 W8A8 dequant 后进入 GMM/SwiGLU/GMM2；perf 中输出 `quant_sideband_bytes`，W4A8 保留为后续扩展边界 |
| T13 | Add coarse-to-fine SwiGLU scheduling and finalize stdout-only perf characterization | 已完成 | `{8,4,2,1,1}` group policy、small/large shape perf readout、overlap interpretation | `BuildExpertGroupSchedule` 已覆盖 `{8,4,2,1,1}`，small 输出 `schedule_tag=2`，large 输出 `schedule_tag=8x4x2x1x1`；stdout perf 覆盖 dispatch/compute/combine/overlap-summary |

## MegaMoE 优化点进展

| 编号 | 优化点 | 当前状态 | 对应任务 | 当前判断 |
| --- | --- | --- | --- | --- |
| O1 | 前置 routing/quant/count/cumsum，并让 dispatch 直接落到 expert-major 终态 | 已完成 | T3 / T4 / T12 | routing truth、expert-major dispatch 落位、W8A8 payload 与 scale sideband baseline 已闭合 |
| O2 | dispatch 远端读、combine 远端写的非对称通信方向 | 已完成 | T4 / T6 / T11 | dispatch `TGET` pull 与 combine `TPUT` return 已闭合到 task-driven overlap correctness path |
| O3 | 按 expert/group 做 steady-state overlap | 已完成 | T9 | `full-chain` 主入口已按 group timeline 执行 `dispatch(g+1) / compute(g) / combine(g-1)` baseline |
| O4 | 用 software scoreboard / summary 取代粗粒度全局等齐 | 已完成 | T10 | 已有 `DispatchGroupProgress`、`ReadyQueue`、summary gate 与 probe-first wait baseline；当前仍保留 standalone host-controller 壳 |
| O5 | GMM→combine 的 tile/group 粒度切分通信 | 已完成 | T11 | combine return 已按 group task 子集推进，并保持全局 transport slot truth；更细 tile split 可作为后续性能深化项 |
| O6 | W8A8/W4A8 风格的三级量化流水 | 已完成 | T12 | W8A8 `quantInput + scale1/scale2` sideband 与 compute-side dequant 已闭合；W4A8 边界保持清晰未混入当前 baseline |
| O7 | SwiGLU 粗细粒度结合调度 | 已完成 | T13 | 已实现并测试 `{8,4,2,1,1}` schedule policy，large 运行输出 `schedule_tag=8x4x2x1x1` |
| O8 | small/large shape 的性能口径与 front-sync tradeoff 解释 | 已完成 | T13 | stdout-only perf 已覆盖 per-stage bandwidth、tick/group、summary ready counters、shape class、quant sideband bytes 与 schedule tag |

## 当前建议 review 顺序

| 顺序 | 动作 | 状态 |
| --- | --- | --- |
| 1 | review `DESIGN.md` 第 2-5 章，确认算法总述、真值来源与 PTO 约束 | 已完成 |
| 2 | review `DESIGN.md` 第 6-13 章，确认模块合同是否忠实承接算法链 | 已完成 |
| 3 | review `DESIGN.md` 第 14-18 章，确认脚手架/验证/perf 只是承载算法 | 已完成 |
| 4 | review `IMPLEMENTATION_PLAN.md`，确认任务映射顺序与算法主链一致 | 已完成 |
| 5 | 从 T1 开始进入实现 | 已完成 |
| 6 | 按 T1 -> T13 顺序推进实现 | 已完成 |

## 已落盘的关键结论

- MegaMoE 算法主链、dispatch/combine 非对称、routing 偏移真值已先在 `DESIGN.md` 中系统展开，再由模块合同承接
- `IMPLEMENTATION_PLAN.md` 现已明确区分“正确性基线任务（T3-T8）”和“剩余 MegaMoE 优化任务（T9-T13）”
- MegaMoE 工作已按五条主线闭合到 standalone baseline：steady-state overlap、dispatch→compute scoreboard、task-driven combine、W8A8 quant pipeline、coarse-fine SwiGLU + perf closeout
- MegaMoE 优化点 O1-O8 已逐项映射并更新为已完成；其中 O4/O5 仍保留为可继续下沉控制 AIV / 更细 tile split 的后续性能深化边界
- standalone `main.cpp` / `run.sh` / `CMakeLists.txt` 的工程边界已固定为承载算法，而不是重新定义算法
- perf 口径已固定为 stdout `key=value` 日志，不再引入额外 JSON 产物
- T3 已补成真实 routing truth planner；PTO 边界已补成 public surface / protocol shell / substrate shell 三层
- 当前 fresh verification 已覆盖 `test_workspace_layout`、`test_signal_protocol`、`test_routing_metadata`、`test_task_materialization`、`test_compute_reference`、`metadata-only`、`dispatch-only`、`compute-only`、`combine-only`、`full-chain` small、`full-chain` large
- 当前 `full-chain` small 输出 `schedule_tag=2 shape_class=small`，large 输出 `schedule_tag=8x4x2x1x1 shape_class=large`；两者均双 rank PASS
