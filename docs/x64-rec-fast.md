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
- deterministic comparison and paired AB/BA median timing against the canonical executor at REC width 960.

The fast path remains opt-in and experimental. The planner now classifies both the PP-OCRv6 Tiny and Medium REC graphs without an NHWC blocker. Fusion is conservative: it requires single-consumer, shape-compatible semantic chains and falls back to the original physical nodes when a pattern is not provably safe. The benchmark report emits `legacy_ms`, `fast_ms`, `speedup`, Pointwise/Dense/Depthwise/binary/unary/ReduceMean/Pool/Concat/BatchNorm node counts, `physical_op_count`, `elided_tensor_count`, per-kind profile nanoseconds, conversion bytes, `unsupported_nhwc_node_count`, `argmax_mismatch`, and `max_abs <= 1e-4` correctness gates. A negative speedup is not promoted to the default path.

The prototype is only built for x86/x64 when the existing experimental option is enabled. ARM64, LoongArch, WebAssembly, DET, and release builds remain on the legacy path.

The CTest entry is `x64_rec_fast_prototype`. It checks the layout conversion round trip, runs two warm-ups followed by nine alternating legacy/fast samples, and compares the complete REC output against `lw_execute_session_f32` with `max_abs <= 1e-4`.

Current local reference measurements (static width model, Windows x64 AVX2) are informational only. The v4 Tiny run below is a fresh post-optimization sample; it is not a release gate:

- Tiny REC960 (v4 local sample): zero mismatches, four planned/runtime conversions, 40 elided GELU temporaries, about 3.7 MiB NHWC workspace, and a reported speedup ratio of `0.65x`; the physical lifetime validator passes and no fusion chain is present in this graph.
- Medium REC960: the previous `0.71x` / 21.6 MiB figures are retained only as a pre-v4 reference; rerun the Medium artifact before drawing a v4 performance conclusion.

These results confirm that planner coverage, physical lifetimes, and kernel scheduling are improving, but the implementation is not ready for default-path integration. The next optimization stages should target direct NHWC preprocessing, AVX2 support-op vectorization, terminal CTC output elision, and measured recognizer-level hotspots before any promotion decision.
