# Experimental NHWC pointwise benchmark

This checkpoint measures an AVX2/FMA NHWC pointwise Conv1x1 candidate without
changing the production runtime. It is intentionally isolated behind
`LW_EXPERIMENTAL_AVX2_FAST_PATH=ON` and remains default-off.

The experiment contains:

- `src/kernels/nhwc_pack.c`: packs `[OC][IC][KH][KW]` weights as
  `[OC/16][KH][KW][IC][16]`;
- `src/simd/avx2_nhwc_pointwise.c`: six-row by sixteen-channel AVX2/FMA tiles,
  residual/ReLU/HardSwish epilogues, and a grouped tile scheduler;
- `tests/nhwc_pointwise_driver.c`: single-layer and persistent two-layer
  comparisons against the production NCHW packed Conv1x1 path;
- `tests/test_nhwc_pointwise_benchmark.py`: JSON parity and calibration contract;
- `tests/test_source_newlines.py`: cross-compiler source hygiene contract.

The target is x86/x64-only and is not linked into `lw_ppocr_c`, the shared
library, WASM, Android, ARM64, LoongArch64, or any release package. It does not
change the C ABI, LWM format, model assets, or production dispatch.

## Build and run

```powershell
cmake -S . -B build-nhwc-a2 `
  -G "Visual Studio 16 2019" -A x64 `
  -DBUILD_TESTING=ON `
  -DLW_BUILD_HTTP_DEMO=OFF `
  -DLW_EXPERIMENTAL_AVX2_FAST_PATH=ON
cmake --build build-nhwc-a2 --config Release --target nhwc-pointwise-benchmark-driver
ctest --test-dir build-nhwc-a2 -C Release -R "source_final_newline_contract|nhwc_pointwise_benchmark" --output-on-failure
.\build-nhwc-a2\Release\nhwc-pointwise-benchmark-driver.exe
```

On a host without AVX2+FMA the driver prints a JSON `status` of `skipped` and
returns success. On a supported host it checks:

- NCHW/NHWC max absolute and relative error `<= 1e-4`;
- zero output and argmax mismatches;
- nine alternating AB/BA timing rounds;
- `group_tiles` calibration for `1`, `2`, `4`, `8`, `16`, and `all` (`0`) on every case, with representative `scheduler_cases` exported for 512→1024 and 1536→768;
- independent parity for bias-only, ReLU, HardSwish, and residual+ReLU.

## Current diagnostic result

The first grouped-scheduler run on Windows x64 passed all parity checks. The
candidate is now faster than the previous ungrouped implementation and reached
about `1.18x`–`1.62x` of the NCHW packed baseline on the five single-layer
shapes. The two persistent-layout chains reached about `1.53x`–`1.57x` in the
same run. These are diagnostic measurements, not a production claim; absolute
latency varies by CPU and build.

The scheduler calibration is deliberately informational. A future promotion
must still satisfy a reproducible multi-run gate: all major hot shapes at least
`1.0x`, at least two at `1.10x`, five-shape average at least `1.08x`, and both
persistent chains at least `1.05x` with one at least `1.10x`. Correctness and
contract checks remain mandatory. The existing `3x` promotion text is retained
only as a conservative long-term target and does not enable graph integration.

No production executor or layout planner path is changed in this checkpoint.
Keep `LW_EXPERIMENTAL_AVX2_FAST_PATH=OFF` for normal builds until calibration is
repeated on the supported CI runners and a separate runtime integration review
is completed.
