# x64 REC Hybrid Fast Executor prototype

This repository contains an opt-in prototype for a graph-level REC executor on x64 builds. It is intentionally private and does not change the public C ABI or the default execution path.

Enable it with:

```text
-DLW_EXPERIMENTAL_AVX2_FAST_PATH=ON
```

The first milestone provides:

- a private plan object for the REC graph;
- dual NCHW/NHWC tensor state with a private NHWC workspace;
- scalar-correct NCHW/NHWC conversion helpers;
- generic-node fallback through the existing executor dispatch;
- deterministic comparison against the canonical executor at REC width 960.

At this stage every node is classified as `generic`, so this is a correctness and integration milestone rather than a speed claim. Pointwise, Dense, and Depthwise fast nodes will be enabled only after each implementation has an independent numerical gate and a full-graph A/B measurement.

The prototype is only built for x86/x64 when the existing experimental option is enabled. ARM64, LoongArch, WebAssembly, DET, and release builds remain on the legacy path.

The CTest entry is `x64_rec_fast_prototype`. It checks the layout conversion round trip and compares the complete REC output against `lw_execute_session_f32` with `max_abs <= 1e-4`.
