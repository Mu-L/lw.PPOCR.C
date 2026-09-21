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
- Pointwise 1x1 and Dense (general Conv) AVX2+FMA nodes for eligible NHWC islands;
- best-available legacy fallback, including GELU fusion and prepared packed Conv dispatch;
- authoritative per-run layout publication and NCHW/NHWC conversion accounting;
- deterministic comparison and paired AB/BA median timing against the canonical executor at REC width 960.

The fast path remains opt-in and experimental. The current PP-OCRv6 Tiny REC graph classifies eligible Conv nodes as Pointwise or Dense; unsupported nodes use the legacy executor. Depthwise and additional fused epilogues remain future work. The benchmark report is informational: it always emits `legacy_ms`, `fast_ms`, `speedup`, node counts, conversion bytes, and a `max_abs <= 1e-4` correctness gate. A negative speedup is not promoted to the default path.

The prototype is only built for x86/x64 when the existing experimental option is enabled. ARM64, LoongArch, WebAssembly, DET, and release builds remain on the legacy path.

The CTest entry is `x64_rec_fast_prototype`. It checks the layout conversion round trip, runs two warm-ups followed by nine alternating legacy/fast samples, and compares the complete REC output against `lw_execute_session_f32` with `max_abs <= 1e-4`.
