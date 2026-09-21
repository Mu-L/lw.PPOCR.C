# x64 REC Hybrid Fast Executor prototype

This repository contains an opt-in prototype for a graph-level REC executor on x64 builds. It is intentionally private and does not change the public C ABI or the default execution path.

Enable it with:

```text
-DLW_EXPERIMENTAL_AVX2_FAST_PATH=ON
```

The current milestone provides:

- a private plan object for the REC graph;
- dual NCHW/NHWC tensor state with a private NHWC workspace;
- scalar-correct NCHW/NHWC conversion helpers;
- Pointwise 1x1, Dense (general Conv), and eligible block8 Depthwise AVX2+FMA nodes for NHWC islands;
- same-shape binary operators with scalar/channel broadcast, plus contiguous ReLU, Sigmoid, Sqrt, Erf, HardSigmoid, and Pow handling;
- NHWC ReduceMean over spatial axes, AveragePool/MaxPool, channel-axis Concat, and constant-parameter BatchNormalization;
- best-available legacy fallback, including GELU fusion and prepared packed Conv dispatch;
- authoritative per-run layout publication and NCHW/NHWC conversion accounting;
- GELU fusion safety checks for private single-consumer temporaries, plus elided intermediate tensors excluded from the NHWC workspace;
- a compiled physical-op table that executes fused semantic spans once and exposes per-kind profile timings;
- an independent physical-op table that owns packed weights, folded bias, epilogues, and semantic spans, with generic spans validated at execution time;
- plan-time Conv+BatchNorm folding plus Conv+ReLU and Conv+Add(+ReLU) epilogues for eligible Pointwise/Dense nodes;
- per-plan fusion counters (`fused_conv_bn`, `fused_conv_relu`, `fused_conv_bn_relu`, `fused_conv_add`, and `fused_conv_add_relu`) for regression and benchmark reporting;
- physical tensor lifetimes mapped from semantic spans, with workspace overlap validation before a plan is published;
- tile-first Dense scheduling that gathers border patches once per spatial tile and uses a single 6x16 accumulation tile;
- an AVX2/FMA Depthwise two-pixel interior kernel for full 32-channel blocks, with the existing border and tail fallback;
- a single-loop Depthwise dispatch that resumes the x2 interior kernel after border pixels; x1/x2 counters are collected inside the real kernel only for profiled runs, while timed runs pass stats=NULL;
- plan-time Dense K-offset metadata and a prepared AVX2/FMA entry point that removes per-call offset construction and K division/modulo from the hot loop, retaining only patch and partial-tile scratch;
- AVX2 NHWC support ops for prepared BatchNorm affine, spatial ReduceMean, and AveragePool/MaxPool kernels;
- deterministic comparison and paired AB/BA median timing against the canonical executor at REC width 960.
- v7 direct-NHWC preparation hooks: a private graph-input accessor and prepared backbone runner can consume preprocessed NHWC data without an initial NCHW-to-NHWC conversion when the model graph starts with an NHWC-capable node; graphs that require an NCHW prologue retain the existing NCHW entry path.

The fast path remains opt-in and experimental. The planner now classifies both the PP-OCRv6 Tiny and Medium REC graphs without an NHWC blocker. Fusion is conservative: it requires single-consumer, shape-compatible semantic chains and falls back to the original physical nodes when a pattern is not provably safe. The benchmark report emits `legacy_ms`, `fast_ms`, `speedup`, Pointwise/Dense/Depthwise/binary/unary/ReduceMean/Pool/Concat/BatchNorm node counts, `physical_op_count`, `elided_tensor_count`, profiled Depthwise x1/x2 dispatch counters (without a timed-run pre-walk), per-kind profile nanoseconds, conversion bytes, `unsupported_nhwc_node_count`, `argmax_mismatch`, and `max_abs <= 1e-4` correctness gates. A negative speedup is not promoted to the default path.

The prepared-NHWC hooks are internal and are not part of the public C ABI. Callers must query the graph-input accessor and treat a null pointer as an explicit fallback to lw_x64_rec_fast_run; the Tiny REC graph currently has an NCHW prologue, so it intentionally exercises that fallback.

The prototype is only built for x86/x64 when the existing experimental option is enabled. ARM64, LoongArch, WebAssembly, DET, and release builds remain on the legacy path.

The CTest entries are `nhwc_support_pool` (valid-window and channels<8 Pool regression), `layout_plan_direct_input`, and `x64_rec_fast_prototype`. It checks the layout conversion round trip, runs two warm-ups followed by nine alternating legacy/fast samples, and compares the complete REC output against `lw_execute_session_f32` with `max_abs <= 1e-4`.

Current local reference measurements (static width model, Windows x64 AVX2) are informational only. The v6 Tiny run below is a fresh post-optimization sample; it is not a release gate:

- Tiny REC960 (v6 local sample): zero mismatches (max_abs=2.72e-05, argmax_mismatch=0), four planned/runtime conversions, 40 elided GELU temporaries, about 3.7 MiB NHWC workspace, and a reported speedup ratio of about 0.69x in the latest local sample; the final profiled run reports 8330 Depthwise x2 and 19340 x1 dispatches. The physical lifetime validator passes and no fusion chain is present in this graph.
- Medium REC960: the previous `0.71x` / 21.6 MiB figures are retained only as a pre-v6 reference; rerun the Medium artifact before drawing a v6 performance conclusion.

These results confirm that planner coverage, physical lifetimes, Dense preparation, and Depthwise dispatch are improving, but the implementation is not ready for default-path integration: the current end-to-end ratio remains below 1.0x. The next stages are to connect the existing BGR-to-NHWC preprocessor to the prepared hook, then elide the terminal CTC logits/Softmax materialization and measure recognizer-level hotspots before any promotion decision.
