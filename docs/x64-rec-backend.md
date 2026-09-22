# x64 REC Backend

这是实验性的 x64 REC 后端，不属于公开 C ABI。它在 `LW_EXPERIMENTAL_AVX2_FAST_PATH=ON` 且 REC 目标宽度为 960、CPU 支持 AVX2+FMA 时由 recognizer 选择；其他平台、宽度或不满足编译契约时继续走 canonical executor。

## 当前状态

Tiny REC 已完成目标宽度 960 的 standalone 可执行闭环：

- `lw_x64_rec_program` 是只读图程序，保存模型签名、目标宽度、值表、物理算子表、arena/scratch 大小和 CTC 尾部描述；
- `lw_x64_rec_instance` 独占可复用的 arena、dense scratch、CTC logits、argmax 索引和 emitted probability 缓冲区；program 不持有运行时可变工作区；
- 四维激活值在可直接执行的物理算子间按 NHWC 保存；rank-3 序列/reshape 边界保留 canonical 的 channel-major 语义，并在必要处生成显式 repack；
- Conv 按形状选择 pointwise、depthwise 或 dense lowering；pointwise、depthwise、dense 已接入 AVX2+FMA kernel，形状不满足 kernel 契约时才回退到 backend 自己的 scalar 实现；
- 支持 Tiny REC 主干中的 Conv、BatchNorm、Add/Mul/Div、Relu、Erf、HardSigmoid、ReduceMean、Average/MaxPool、Transpose 和 MatMul；
- 最末 CTC 投影由独立 `ctc_projection_internal.c` 预打包权重，执行 packed MatMul + argmax + emitted softmax，并初始化 blank/repeated 行的 probability 缓冲；
- arena 使用 64 字节对齐和溢出检查，instance 可重复运行；当前先保证 physical lifetime 正确，尚未做跨值空间复用。

## 正确性门禁

`x64_rec_backend_execution` 会加载 Tiny REC960，使用确定性 BGR 输入，同时驱动 canonical backbone 与 standalone backend，逐物理算子比较 rank-4/rank-3 输出，并比较 canonical CTC greedy 与 backend 的全部索引和 probability（误差门限 `1e-6`）。`x64_rec_backend_contract` 覆盖 program/instance 创建、输入清零、执行和结构化状态输出。

## 边界与限制

该 backend 仍是实验性实现：

- 目前只在 x64 AVX2+FMA 和 Tiny REC960 上接入 recognizer；192/320/480/640 宽度矩阵、Small/Medium 模型和非 x64 后端尚未宣称覆盖；
- 输入 API 当前要求调用方提供已经排布为 NHWC 的 `float` 输入，输出通过 program 的值表和 CTC 缓冲区读取；
- arena 仍按所有运行时值单调分配，内存复用和 physical lifetime 压缩留在后续阶段；当前 Tiny REC960 的实例 arena 约 95.2 MiB，优先保证独立 instance 的正确性和可复用性；
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

## Profile 与 A/B

`x64-rec-backend-benchmark-driver` 只在测试构建且 `LW_EXPERIMENTAL_AVX2_FAST_PATH=ON` 时生成。它使用固定的 960×48 BGR 输入，先预热 3 次，再以 AB/BA 交替方式测量 canonical 与 standalone backend，并按 physical op 分类累计 backbone 时间。输出包含 `backend_profile_ms`、`physical_ops`、`semantic_nodes`、arena/scratch 大小和 `text_match` 门禁。

```text
cmake --build build --config Release --target x64-rec-backend-benchmark-driver
build/Release/x64-rec-backend-benchmark-driver.exe build/models/rec.lwm 30
```

当前 Windows x64 本地基线（30 次测量，结果会随 CPU 和负载变化）约为：canonical REC 19.1 ms、standalone backend 41.9 ms，`text_match=true`；backend backbone 中 pointwise 约 20.0 ms、dense 约 10.2 ms，是下一轮优化的优先热点。该数字是开发剖面，不是公开性能承诺。
