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
- deterministic comparison and paired AB/BA median timing against the canonical executor at REC width 960.

The fast path remains opt-in and experimental. The planner now classifies the PP-OCRv6 Tiny REC graph without an NHWC blocker and the Medium REC graph with only two unsupported asymmetric-pad Conv nodes. The benchmark report emits `legacy_ms`, `fast_ms`, `speedup`, Pointwise/Dense/Depthwise/binary/unary/ReduceMean/Pool/Concat/BatchNorm node counts, conversion bytes, `unsupported_nhwc_node_count`, `argmax_mismatch`, and `max_abs <= 1e-4` correctness gates. A negative speedup is not promoted to the default path.

The prototype is only built for x86/x64 when the existing experimental option is enabled. ARM64, LoongArch, WebAssembly, DET, and release builds remain on the legacy path.

The CTest entry is `x64_rec_fast_prototype`. It checks the layout conversion round trip, runs two warm-ups followed by nine alternating legacy/fast samples, and compares the complete REC output against `lw_execute_session_f32` with `max_abs <= 1e-4`.

Current local reference measurements (static width model, Windows x64 AVX2) are informational only:

- Tiny REC960: zero mismatches, four planned/runtime conversions, no NHWC blockers, and about 84.7 MiB NHWC workspace; fast execution is still slower than the canonical executor (reported speedup ratio `0.56x`, about `1.79x` the canonical time).
- Medium REC960: zero mismatches, eight planned/runtime conversions, two unsupported asymmetric-pad Conv nodes, and about 706 MiB NHWC workspace; fast execution is roughly `1.54x` the canonical executor in the current prototype (reported speedup ratio `0.65x`).

These results confirm that planner coverage and correctness are improving, but the implementation is not ready for default-path integration. The next optimization stages should target the remaining asymmetric-pad Conv nodes, direct NHWC preprocessing/output reuse, workspace lifetime/reuse, and measured kernel hotspots before any promotion decision.
