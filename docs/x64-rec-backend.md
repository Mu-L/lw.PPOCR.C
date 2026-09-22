# x64 REC Backend

这是实验性的 x64 REC 后端，不属于公开 C ABI，也不会改变现有 recognizer 的默认路径。它与历史 `x64_rec_fast_plan` 并列存在，但后端自己的编译单元不包含旧 fast-plan 头文件、布局规划器或旧执行入口。

## 当前阶段

当前已完成“大刀阔斧重构”的第一阶段：独立图程序和编译边界。

- `lw_x64_rec_program` 保存模型签名、目标宽度、值表、算子表和线性 arena；
- 编译器从模型/LWM 节点表生成值生命周期元数据，不再创建旧 `lw_x64_rec_fast_plan`；
- 所有四维激活值统一记录为 NHWC；Reshape/Squeeze/Unsqueeze/Transpose 等结构节点记录为 alias/copy 候选；
- 编译器识别 Tiny REC 的末端 `MatMul -> Add -> MatMul -> Add -> Softmax` 结构，并通过独立 `ctc_projection_internal.c` 预打包 CTC 权重，同时记录 activation、bias、rows、inner 和 classes；
- arena 使用 64 字节对齐、溢出检查和单调分配，先保证 physical lifetime 正确，暂不做跨值复用；
- `x64_rec_backend_contract` 在 Tiny REC960 上验证编译结果、159 个节点、274 个值、无 generic 节点、无布局转换并成功识别 CTC 尾；合同测试还会用零激活执行一次 packed MatMul + emitted softmax/argmax 烟测。

## 尚未宣称完成的部分

当前 `lw_x64_rec_instance_run_backbone()` 仍返回 `LW_STATUS_UNSUPPORTED`。Dense/Depthwise/NHWC 算子执行、CTC packed projection 和 recognizer 接入将在下一阶段逐个落地；因此本阶段不会改变生产 OCR 的结果或性能，也不会把未完成后端接入默认路径。

## 下一阶段边界

1. 把 Conv/BN/Pointwise/Depthwise/Pool/ReduceMean 的参数预打包到 program constants；
2. 为每个 op 实现值表寻址的 NHWC executor，先覆盖 Tiny REC960；
3. 加入独立 CTC projection（packed MatMul + emitted softmax/argmax），与 canonical executor 做逐行结果比对；
4. 通过 Tiny960 correctness driver 后，再接 recognizer，并扩展 192/320/480/640 宽度；
5. 最后再做 arena lifetime reuse 和 profile/benchmark，不在编译边界未稳定前声称性能收益。

## 构建与测试

启用 `LW_EXPERIMENTAL_AVX2_FAST_PATH=ON` 后，构建 `x64-rec-backend-contract-driver`，运行 `x64_rec_backend_contract`。不满足 AVX2/FMA 或模型结构不完整时，调用方应继续使用 canonical recognizer。
