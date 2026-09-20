# Experimental NHWC Dense kernel

`lw_pack_nhwc_dense_f32` and `lw_avx2_fma_nhwc_dense_f32` are an experimental
AVX2+FMA path for evaluating the Medium REC workload. They are deliberately
kept outside the production executor and C ABI until the accuracy and
end-to-end profile gates are met.

## Packed layout

Weights are supplied in canonical `[output_channel][input_channel][kernel_y][kernel_x]`
order. The packed layout is:

```text
[output_block][input_channel][tap][lane]
```

where `output_block` is 16 output channels, `tap = kernel_y * kernel_w + kernel_x`,
and `lane` is the channel inside the 16-wide block. Pointwise (`1x1`) is the
same layout with one tap.

## Kernel contract

The experimental kernel consumes and produces NHWC tensors, supports groups=1,
dilation=1, arbitrary stride and top/left padding, 6-pixel tiles, 16-output
channel blocks, border gather, fused bias/residual and ReLU or Hardswish. K is
processed in caller-selected blocks (the default is 512); the caller owns and
reuses the scratch buffer returned by `lw_nhwc_dense_scratch_bytes`.

GELU is intentionally rejected for this first path. No production executor or
stable ABI symbol uses this code yet.

## Local correctness check

Configure the experimental x64 build and run the standalone driver:

```powershell
cmake -S . -B build-nhwc-a2 -G "Visual Studio 16 2019" -A x64 `
  -DLW_EXPERIMENTAL_AVX2_FAST_PATH=ON
cmake --build build-nhwc-a2 --config Release --target nhwc-dense-benchmark-driver
ctest --test-dir build-nhwc-a2 -C Release -R nhwc_dense_benchmark --output-on-failure
```

The driver checks packed indices, interior and border tiles, stride 1/2,
batches, and `dense_kc` values 128/256/512/1024/auto against a scalar
reference with a `1e-4` maximum absolute error gate.
Each run also emits `perf_case` records for the representative `medium-rec-3x3-h6-w240`
shape and the smaller border fixtures. `scalar_ms` is the deliberately simple
reference loop, while `dense_ms` is the experimental kernel; these numbers are
local diagnostics, not a production OCR speed claim. A promotion decision still
requires an end-to-end REC960 A/B on a fixed runner.
