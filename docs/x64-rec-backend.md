# x64 REC Backend

这是实验性的 x64 REC 后端，不属于公开 C ABI。当前在 `LW_EXPERIMENTAL_AVX2_FAST_PATH=ON` 且 CPU 支持 AVX2+FMA 时，recognizer 会为不超过配置上限的 192/320/480/640/960 宽度准备 compiled slot；不满足编译契约的宽度继续走 canonical executor。其他平台不受影响。下文较早的 Tiny/Hybrid 调优记录是历史记录；当前跨宽度共享方案和测量结果见文末。

## 当前状态

Tiny REC 已完成 192/320/480/640/960 五宽度的 standalone 与 recognizer 可执行闭环：

- `lw_x64_rec_program` 是只读图程序，保存模型签名、目标宽度、值表、物理算子表、arena/scratch 大小和 CTC 尾部描述；
- `lw_x64_rec_instance` 独占可复用的 arena、dense scratch、CTC logits、argmax 索引和 emitted probability 缓冲区；program 不持有运行时可变工作区；
- 四维激活值在可直接执行的物理算子间按 NHWC 保存；rank-3 序列/reshape 边界保留 canonical 的 channel-major 语义，并在必要处生成显式 repack；
- Conv 按形状选择 pointwise、depthwise 或 dense lowering；pointwise、depthwise、dense 已接入 AVX2+FMA kernel，形状不满足 kernel 契约时才回退到 backend 自己的 scalar 实现；
- 支持 Tiny REC 主干中的 Conv、BatchNorm、Add/Mul/Div、Relu、Erf、HardSigmoid、ReduceMean、Average/MaxPool、Transpose 和 MatMul；
- 最末 CTC 投影由独立 `ctc_projection_internal.c` 预打包权重，执行 packed MatMul + argmax + emitted softmax，并初始化 blank/repeated 行的 probability 缓冲；
- arena 使用 64 字节对齐、溢出检查与 physical lifetime 复用，instance 可重复运行；各 worker 共享只读 program，各自拥有 arena/scratch。

## 正确性门禁

`x64_rec_backend_execution` 会加载 Tiny REC960，使用确定性 BGR 输入，同时驱动 canonical backbone 与 standalone backend，逐物理算子比较 rank-4/rank-3 输出，并比较 canonical CTC greedy 与 backend 的全部索引和 probability（误差门限 `1e-6`）。`x64_rec_backend_contract` 覆盖 program/instance 创建、输入清零、执行和结构化状态输出。

## 边界与限制

该 backend 仍是实验性实现：

- 当前 x64 AVX2+FMA 的 Tiny、Small、Medium REC 已接入五个自适应宽度；非 x64 compiled 后端依各平台构建选项决定；
- 输入 API 当前要求调用方提供已经排布为 NHWC 的 `float` 输入，输出通过 program 的值表和 CTC 缓冲区读取；
- 五宽度保留各自的 physical ops 和 arena offset，但宽度无关的 packed constants 由最大宽度 Program 持有，其余宽度借用；runtime workspace 仍按现有策略共享；
- GELU fusion、Concat/Resize 等非 Tiny REC960 必需路径暂不宣称覆盖。

## 构建与测试

在 x64、AVX2+FMA 环境启用：

```text
-DLW_EXPERIMENTAL_AVX2_FAST_PATH=ON
```

构建并运行：

```text
cmake --build build --config Release --target x64-rec-backend-contract-driver x64-rec-backend-execution-driver
ctest --test-dir build -C Release -R "x64_rec_backend_(contract|execution|benchmark)" --output-on-failure
```

不满足 AVX2/FMA 或模型结构不符合当前 Tiny REC 编译契约时，调用方应继续使用 canonical recognizer。

`x64-rec-pointwise-tuning-driver` 会从编译后的 Tiny REC960 program 枚举真实 pointwise physical op，比较 `6x16`、`4x16`、`3x32` 和 `2x32`，并对每个候选做输出误差校验（`max_error <= 1e-5`）。当前本地 20 次测量显示大多数多像素 shape 仍由 `6x16` 胜出；单像素 projection shape 选择 `4x16` 或 `2x32`，compiler 只对这些 shape 启用候选，其余保持基线 kernel。`3x32` 已实现并纳入 tuning，但当前 Tiny960 没有稳定胜出，因此暂不默认 dispatch。

```text
cmake --build build --config Release --target x64-rec-pointwise-tuning-driver
build/Release/x64-rec-pointwise-tuning-driver.exe build/models/rec.lwm 20
```

## Profile 与 A/B

`x64-rec-backend-benchmark-driver` 只在测试构建且 `LW_EXPERIMENTAL_AVX2_FAST_PATH=ON` 时生成。它使用固定的 960×48 BGR 输入，先预热 3 次，再以 AB/BA 交替方式测量 canonical 与 standalone backend。正式 `canonical_ms` / `backend_ms` 只使用不带逐算子计时的完整执行路径；另外最多执行 5 次独立 profiling，结果写入 `backend_profile_ms`，因此 profiling 开销不会污染 A/B speedup。输出还包含 `backend_graph_ms`、`backend_profile_runs`、physical op 分类计数、各卷积路径的 fallback 计数、`physical_ops`、`semantic_nodes`、arena/scratch 大小和 `text_match` 门禁。

```text
cmake --build build --config Release --target x64-rec-backend-benchmark-driver
build/Release/x64-rec-backend-benchmark-driver.exe build/models/rec.lwm 30
```

当前 Windows x64 本地基线（30 次测量，结果会随 CPU 和负载变化）约为：canonical REC 19.6 ms、standalone backend 35.8 ms，`speedup=0.55x`、`text_match=true`；backend backbone 中 pointwise 约 20.7 ms、dense 约 1.2 ms，首层 Dense 尾块已从 scalar fallback 转为 AVX2 写回。正式输出的 `pointwise_fallbacks`、`dense_fallbacks`、`depthwise_fallbacks` 和 `scalar_conv_fallbacks` 均为 0。该数字是开发剖面，不是公开性能承诺。Dense 当前使用 KC=512；prepared offsets 的本地 A/B 为负优化，未接入 production path。正式 A/B 仍显示 backend 尚未超过 canonical，因此在完成剩余 pointwise/非卷积热点优化前不扩大模型或宽度覆盖。

## 2026-09：五宽度 packed constants 共享

最大可用标准宽度（通常 960）的 Program 独立打包 immutable constants；其余宽度保留自己的 op/value 表和 arena 规划，按 semantic node 与打包契约匹配后借用 Conv、可打包 MatMul、BatchNorm 派生数组和 CTC projection。借用者 retain root Program；释放顺序和 worker clone 不会使指针悬空。匹配失败时该常量独立打包，shared 编译整体失败时回退到独立编译。完整标准宽度覆盖时，Medium 不再额外建立未使用的 hybrid packed template；非标准上限或覆盖不全仍尝试原有 fallback。

内部诊断可设置 `LW_REC_MEMORY_PROFILE=1`，输出各宽度的 `owned`、`borrowed` 字节及借用次数，并报告 shared arena/scratch/CTC workspace 和 Medium hybrid 是否存在。这些字节数只统计 compiled 常量，不等于进程工作集。`x64_rec_shared_constants` 在 Tiny 五宽度比较独立/共享执行结果、借用指针、内存统计和 root 先释放后的生命周期；同一驱动也已本地通过 Small、Medium 模型。

在同一 Windows x64 主机，使用修改前 `d257646` 与本次代码、同一 500×500 样例及同一模型文件，固定 REC 960，每次新进程预热 1 次、计时 3 次，交替顺序做 3 轮配对。下表为每版本的耗时中位数和峰值工作集中位数；配对耗时变化按每轮新/旧比值取中位数。短测只用于本次回归判断，不作为跨机器性能承诺。

| 模型 | Worker | 旧版 ms | 新版 ms | 配对耗时变化 | 旧版峰值 MiB | 新版峰值 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny | 1 | 113.137 | 112.679 | -0.4% | 100.7 | 84.3 |
| Tiny | 4 | 53.531 | 54.085 | +1.0% | 118.2 | 100.6 |
| Small | 1 | 372.936 | 375.913 | +0.8% | 274.9 | 198.5 |
| Small | 4 | 219.001 | 217.741 | -0.6% | 312.0 | 233.8 |
| Medium | 1 | 1214.548 | 1217.170 | +0.6% | 1078.6 | 744.8 |
| Medium | 4 | 961.974 | 979.303 | +0.4% | 1100.2 | 763.2 |

各模型的新旧输出 checksum 完全相同（Tiny `46d99468540b5eb7`、Small `2ee4a78f9306c18b`、Medium `12aff0763cbd432b`）。五宽度 compiled 常量总量在 Tiny 为 21,660,160→4,332,032 字节，Small 为 100,255,360→20,051,072 字节，Medium 为 363,563,520→72,712,704 字节；Medium 的 192/320/480/640 每个宽度借用 72,712,704 字节、自己持有 0 字节，960 root 持有 72,712,704 字节。此处仅计算常量，不包括模型、canonical session、program metadata 和 runtime workspace。
