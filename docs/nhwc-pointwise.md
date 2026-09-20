# Experimental NHWC 6x16 pointwise benchmark

This checkpoint implements the first A2 kernel from the NHWC fast-path design:

- `src/kernels/nhwc_pack.c` packs `[OC][IC][KH][KW]` weights as
  `[OC/16][KH][KW][IC][16]`;
- `src/simd/avx2_nhwc_pointwise.c` provides a six-pixel by sixteen-output
  AVX2/FMA microkernel;
- `tests/nhwc_pointwise_driver.c` compares it with the existing production
  NCHW packed Conv1x1 implementation in one process using alternating AB/BA
  rounds and the real Medium REC pointwise shapes.

The same driver also measures two consecutive pointwise layers while keeping each candidate layout persistent between layers. This is diagnostic only: it tests whether avoiding an intermediate layout conversion can amortize the NHWC cost.

The experiment is isolated behind `LW_EXPERIMENTAL_AVX2_FAST_PATH=ON`. It is
x86/x64-only, default-off, and is not part of `lw_ppocr_c`, the shared library,
WASM, Android, ARM64, LoongArch64, or any release package. It does not change
the C ABI, LWM format, model assets, or production dispatch.

## Build and run

```powershell
cmake -S . -B build-nhwc-a2 `
  -G "Visual Studio 16 2019" -A x64 `
  -DBUILD_TESTING=ON `
  -DLW_BUILD_HTTP_DEMO=OFF `
  -DLW_EXPERIMENTAL_AVX2_FAST_PATH=ON
cmake --build build-nhwc-a2 --config Release --target nhwc-pointwise-benchmark-driver
.\build-nhwc-a2\Release\nhwc-pointwise-benchmark-driver.exe
```

On a host without AVX2+FMA the driver prints a JSON `status` of `skipped` and
returns success. On a supported host it checks:

- `max_abs <= 1e-4`;
- `max_rel <= 1e-4`;
- zero output mismatches above `1e-4`;
- median timings from nine alternating AB/BA measurements.

The promotion rule is intentionally stricter than the correctness test:
every major hot shape must reach at least `3x` against the current NCHW packed
baseline before any graph integration is considered.

The first Windows x64 run passed all single-layer and two-layer-chain parity checks, but the NHWC candidate was slower than the existing NCHW packed path on every measured shape. Single-layer ratios were roughly `0.56x` to `0.76x` of the NCHW speed, where `1.0x` would be equal. The persistent two-layer chains also measured only about `0.70x` to `0.72x`, so layout reuse did not reverse the result.

Therefore this A2 kernel is retained as an experimental measurement target only. The next phase must change the bottleneck hypothesis before adding more NHWC kernels or enabling production execution.
