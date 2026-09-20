# x64 SIMD Phase 2B — AVX2+FMA Conv1x1

Phase 2B hardens the measurement and dispatch-safety foundation for optional
AVX2+FMA kernels. It does not change the default OCR kernel selection.

## Runtime capability model

lw_simd_level remains the mutually exclusive backend selector. The internal
lw_cpu_capabilities snapshot adds:

- has_fma: hardware FMA, AVX and OSXSAVE are present and XMM/YMM state is
  enabled by the operating system;
- has_avx2_fma: the active backend is AVX2 and FMA is usable.

The SIMD level is cached by the runtime probe, and each session stores one
capability snapshot. Existing low-level wrappers therefore no longer issue
repeated CPUID/XGETBV probes during a graph run. The public C ABI, LWM format,
and model files are unchanged.

## FMA candidate

src/simd/avx2_fma_packed_conv1x1.c is an isolated Conv1x1 candidate.
It uses the same PACKED4 weight layout and geometry as the current AVX2
Conv1x1 kernel, but uses explicit _mm256_fmadd_ps instructions. The second
candidate, src/simd/avx2_fma_packed_matmul.c, covers only the Tiny terminal
CTC projection shape `[1,40,80] x [80,6906]` and keeps the existing packed
weight layout. Both are compiled with AVX2+FMA target attributes on GCC/Clang
and `/arch:AVX2` on MSVC.

The candidates are deliberately not connected to the default production dispatch.
This keeps non-FMA hosts safe and preserves the current deterministic OCR path.
A native-only experimental build may opt into a shape-aware dispatch policy
while paired A/B data is collected.

## Benchmark contract

The packed Conv1x1 benchmark invokes the regular AVX2 and FMA entry points directly; it does not use the dispatch wrapper as the AVX2 baseline. Each case records the requested batch (currently 1), input geometry including width, and seven interleaved ABBA rounds. The report exposes median, minimum, maximum, and p90 timings for both kernels, plus AVX2/FMA ratio and absolute/relative FMA error.

The packed Conv1x1 benchmark reports optional fields when the host supports
AVX2+FMA. On x64 it also measures the isolated 8-output × 8-spatial
lw_avx2_fma_packed_conv1x1_8x8_f32 candidate with a second interleaved
ABBA sequence. The packed MatMul benchmark separately measures the terminal
projection shape and checks its argmax contract:

- fma_ms;
- fma_speedup;
- fma_max_abs_error;
- fma_checksum;
- fma8_ms, fma8_vs_fma, fma8_max_abs_error, and fma8_checksum (x64 only);
- avx2_ms, avx2_min_ms, avx2_max_ms, avx2_p90_ms, and fma_vs_avx2.

The candidates must remain finite and within the current exploratory absolute
error bound of `1e-2`. The smoke tests only validate that they are measurable,
machine-readable, and numerically bounded; they do not promote them to the
default backend.

Run locally after configuring a Release build:

    cmake --build build --target packed-conv1x1-benchmark-driver packed-matmul-benchmark-driver --config Release
    build\Release\packed-conv1x1-benchmark-driver.exe 960 20
    build\Release\packed-matmul-benchmark-driver.exe 20
    ctest --test-dir build -C Release -R "packed_(conv1x1|matmul)_benchmark_smoke" --output-on-failure

## End-to-end experiment build

The candidate can be evaluated in a separate native build without changing the
normal dispatch. Configure that build with:

    cmake -S . -B build-fma -G "Visual Studio 17 2022" -A x64 -DLW_EXPERIMENTAL_AVX2_FMA_CONV1X1_DISPATCH=ON -DLW_EXPERIMENTAL_AVX2_FMA_MATMUL_DISPATCH=OFF

Then run the same `full-ocr-intra-benchmark` command against the default and
`build-fma` binaries. The experimental option is native-only, defaults to OFF,
and does not change the public ABI or model files. The performance workflow also
builds the experimental Conv1x1 and terminal MatMul drivers and runs both smoke
tests. Its end-to-end OCR candidate now enables only
`LW_EXPERIMENTAL_AVX2_FMA_CONV1X1_DISPATCH`; terminal MatMul remains separately
opt-in. It is intended for paired latency, checksum, and RSS measurements only.

The Native x64 OCR Performance workflow runs this experiment at 1 worker/1 DET
thread and 4 workers/4 DET threads. The JSON and Markdown outputs are uploaded
as the `lw-ppocr-x64-fma-ocr-results-*` artifact.
The Conv1x1 summary also lists the three slowest and three fastest shapes,
which is the input for a future shape-aware dispatch policy.

## Shape-aware experimental dispatch

`LW_EXPERIMENTAL_AVX2_FMA_CONV1X1_DISPATCH=ON` and `LW_EXPERIMENTAL_AVX2_FMA_MATMUL_DISPATCH=ON` independently enable the native-only Conv1x1 and terminal MatMul policies. The compatibility aggregate `LW_EXPERIMENTAL_AVX2_FMA_DISPATCH=ON` enables both. The Conv1x1 policy routes
only measured beneficial shapes to the FMA candidates. The current explicit
8x8 candidate allowlist is intentionally small:

- Medium 512 -> 1024 at height 6;
- late 768 -> 384 at height 3;
- late 1536 -> 768 at height 3.
The same experimental build now contains a separate FMA candidate for the REC
Node 6 stride-2 Conv3x3 shape `24 -> 48`. It is gated to batch 1, input height
24, output height 12, and widths `160 -> 80` or `480 -> 240` (REC target widths
320 and 960). It reuses the existing packed 3x3 layout and is selected only
when the cached capability snapshot reports AVX2+FMA. All other stride-2
Conv3x3 shapes continue to use the regular AVX2 path. The candidate is still
opt-in and is not part of the default build.
The 384 -> 768 late shape remains on the regular four-output FMA path; the
8x8 candidate is intentionally not used for it. Unknown shapes, 1024 -> 512,
and all other medium/late shapes remain on regular AVX2. The terminal Tiny MatMul
candidate continues to use its separate exact-shape gate. The default build keeps the
existing AVX2 dispatch and is unchanged.

The experimental benchmarks accept the small FMA rounding difference with a
`1.0e-2` maximum absolute error bound; the default benchmarks remain byte-exact.
The terminal MatMul candidate improves the isolated local benchmark by about 1.4x
over the existing AVX2 path. Full OCR remains checksum-identical in local paired
runs; promotion still requires the gates below on the full corpus.

## 100-image quality parity checkpoint

The project-owned generated corpus was replayed locally with the default AVX2
driver and the experimental FMA driver using the same Tiny DET/CLS/REC assets,
REC target width `960`, and manifest (`seed=20260907`, `614` reference lines).
The reports were compared with `tools/compare_ocr_dataset_reports.py`:

| Metric | Default AVX2 | Experimental FMA | Delta |
|---|---:|---:|---:|
| Detection F1 | 99.3517% | 99.3517% | 0.0000 pp |
| Exact reference-line rate | 58.1433% | 58.1433% | 0.0000 pp |
| CER on matched lines | 3.6458% | 3.6458% | 0.0000 pp |

Detection precision, recall, mean matched IoU, matched-line exact rate, missing
lines, and extra lines were also identical. This is a quality-parity checkpoint,
not a release gate: generated images remain local-only, and the FMA dispatch stays
opt-in until repeated runner measurements confirm the performance and working-set
gates below.

## Latest local terminal MatMul FMA checkpoint

Using the existing exact-shape Tiny terminal candidate (`rows=40`, `inner=80`, `columns=6906`) on the same Windows x64 host, the current default and `LW_EXPERIMENTAL_AVX2_FMA_DISPATCH=ON` builds were compared with five paired AB/BA rounds. The output checksum remained `0ebf8b448ab7df47` and the line count remained 16.

| Workers | Default mean (ms) | FMA mean (ms) | Mean speedup | P95 speedup | Peak RSS delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 290.167 | 275.368 | 1.054x | 1.051x | +0.004 MiB |
| 4 | 117.687 | 111.602 | 1.055x | 1.038x | +0.102 MiB |

This is a useful local checkpoint for the terminal MatMul candidate, not a promotion decision: it uses the bundled sample rather than the 100-image corpus, and the small RSS deltas still need repeated runner/corpus confirmation. The candidate therefore remains opt-in.

## Project 100-image FMA promotion checkpoint

The same candidate was then replayed against the ignored project-owned 100-image PPM set (`620` reference lines, REC width `960`, one warm-up and one measured pass). Both builds produced checksum `ea94c481ced30cbe`. The complete-corpus result does not support promotion: the single-worker run regressed materially, while the four-worker run was effectively neutral and used more RSS.

| Workers | Default mean (ms/image) | FMA mean (ms/image) | Latency change | Default peak RSS | FMA peak RSS |
|---:|---:|---:|---:|---:|---:|
| 1 | 224.176 | 253.033 | +12.87% | 108.34 MiB | 108.72 MiB |
| 4 | 174.652 | 176.181 | +0.87% | 131.99 MiB | 132.91 MiB |

This is stronger evidence than the bundled one-image sample. Keep terminal MatMul FMA opt-in; do not change the default dispatcher until repeated corpus runs on the CI runner show a stable end-to-end win with no working-set increase.

## Latest x64 CI checkpoint

The latest native x64 FMA run confirms that the shape-gated experiment is useful
but is not yet a blanket replacement for AVX2. The paired Conv3x3 Node 6
candidate measured `1.141x` at REC width 320 and `1.134x` at width 960. The
Conv1x1 direct A/B median was `1.139x` at width 320 and `1.070x` at width 960,
but the slowest measured shapes were `0.978x` and `0.981x`, respectively. This
supports keeping explicit shape allowlists rather than routing every Conv1x1
shape to FMA.

The same run measured complete OCR mean latency reductions of `3.44%` for the
1-worker profile and `4.05%` for the 4-worker profile. Both profiles retained
checksum `0ebf8b448ab7df47` and 16 lines. Peak RSS changed by only `+0.023 MiB`
and `+0.037 MiB`, but the current promotion policy requires no stable working-set
increase, so the candidate remains opt-in pending another paired run and the
Tiny/Small/Medium corpus gate.

## Medium 1024 -> 512 exclusion checkpoint

A paired local A/B was run against the Medium REC profile at target width 960
with two OCR iterations, using the same Windows x64 host and the same checksum
contract. The direct Conv1x1 benchmark had suggested that the `1024 -> 512`,
height-6 shape might benefit from FMA, so it was tested both with the regular
four-output FMA kernel and with the eight-output-by-eight-spatial candidate.
Neither variant improved the complete OCR pipeline:

| Candidate | 1 worker total | 4 workers total | checksum |
|---|---:|---:|---|
| Existing shape-gated FMA | 16182.5 ms | 11568.3 ms | `c9c15dc8d3fe01ab` |
| Enable 1024 -> 512 four-output FMA | 16219.0 ms (+0.23%) | 12000.2 ms (+3.73%) | identical |
| Enable 1024 -> 512 8x8 FMA | 16252.3 ms (+0.43%) | 11908.2 ms (+2.94%) | identical |

The result is a useful negative checkpoint: isolated-kernel speedups are not
sufficient evidence for promotion when FMA frequency effects and worker-level
scheduling are included. The shape remains on regular AVX2, and the 8x8
allowlist is unchanged.

## Small 384 -> 192 candidate checkpoint

The Small REC profile contains repeated `384 -> 192`, height-6 Conv1x1 nodes.
The isolated benchmark was added as `middle-384x192` and measured AVX2/FMA
ratios of `1.008x` at REC width 320 and `1.091x` at width 960. A paired
end-to-end Small OCR check with three iterations produced:

| REC width | 1 worker AVX2/FMA | 4 workers AVX2/FMA | checksum |
|---:|---:|---:|---|
| 320 | 1.052x | 1.075x | identical |
| 960 | 1.021x | 0.994x | identical |

Because the 960-width multi-worker result is neutral and the 320-width kernel
ratio is close to parity, this shape is not added to the experimental runtime
allowlist yet. The benchmark case remains so future CI runs can re-evaluate it
with more replicas and the same FMA contract.


## Cross-model Conv1x1-only FMA smoke

The same Conv1x1-only candidate was also exercised with the repository's Small
and Medium model assets. These are diagnostic local measurements, not release
gates: Small used the 100-image generated corpus, while Medium used a 10-image
subset because its full OCR cost is substantially higher.

| Model / corpus | Workers | Default median | Conv1x1-only FMA median | Change | Checksum |
|---|---:|---:|---:|---:|---|
| Small / 100 images, one round | 1 | 803.340 ms/image | 792.238 ms/image | -1.38% | `8705f04d72739195` |
| Medium / 10 images, three rounds | 1 | 3393.206 ms/image | 3331.481 ms/image | -1.82% | `85c4d17ff79e8f89` |
| Medium / 10 images, three rounds | 4 | 2553.116 ms/image | 2527.488 ms/image | -1.00% | `85c4d17ff79e8f89` |

Peak RSS stayed within measurement noise for both pairs (about 213 MiB for
Small and 598.65 MiB for Medium at one worker; the four-worker Medium pair was 668.65 MiB versus 668.77 MiB). Both Medium worker configurations are below the existing 2%
end-to-end promotion gate and use only a 10-image subset, so they are evidence for
further profiling rather than a default-dispatch decision. Keep the production
dispatcher on regular AVX2 until full Tiny/Small/Medium corpus runs and 4-worker
replicas satisfy the complete gate.

## AVX2 packed Conv1x1 address-arithmetic checkpoint

为 Medium 长几何 Conv1x1 做了一个仅替换地址计算的候选：把每个 input channel 的乘法/加法索引改为 `input_ptr`/`packed_ptr` 递增，保持原始输入通道累加顺序、load/store、`mul+add` 算术路径和输出布局不变。候选没有进入默认提交路径。

在同一 Windows x64/AVX2 主机、同一 100 图 PPM benchmark manifest（SHA-256 `c51474cb3761515c8c9b07c0afb1846303aba1b2304ef9a87043c9d8c282159d`）、REC 960、warm-up 1、3 次迭代下，候选结果为：

| workers | 默认 mean | 地址递增候选 mean | 变化 | 默认 peak RSS | 候选 peak RSS | checksum |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 225.517 ms/图 | 231.348 ms/图 | +2.59% | 108.24 MiB | 108.75 MiB | `d09e4c29d2d9fe6a` |
| 4 | 170.637 ms/图 | 172.206 ms/图 | +0.92% | 131.91 MiB | 133.31 MiB | `d09e4c29d2d9fe6a` |

全文 checksum 一致，但 1 worker 明显回退，4 worker 也没有收益；该候选已撤销。这个结果说明 MSVC/O3 下编译器已经能较好地消除原始索引开销，手工指针递增反而增加寄存器/依赖压力。后续 Conv1x1 优化应继续从真实 profile、缓存/线程行为和局部形状 A/B 入手，不重复启用这一写法。
## Promotion gate

Before enabling FMA in the production dispatch, collect paired measurements on
the real Tiny, Small, and Medium REC shapes at widths 320 and 960. Require:

1. at least 5% median improvement in the target kernel family;
2. no stable shape regression above 2%;
3. complete OCR paired median improvement of at least 2% on the 100-image corpus;
4. identical text, line count, and reading order;
5. detection geometry and scores within an explicitly recorded tolerance;
6. no peak working-set increase.

If the candidate does not meet these gates, remove it and retain the current
non-FMA AVX2 path. The shape-gated 8x8 kernel remains an x64 benchmark candidate only
until the same shape-aware and end-to-end gates are met; it is not part of the
production dispatcher. AVX512 remains a later, profile-driven experiment.

## Promoted narrow Conv3x3 FMA dispatch

The measured REC Node 6 Conv3x3 path is now controlled by the independent
`LW_AVX2_FMA_CONV3X3_DISPATCH` option, which defaults to `ON` for native builds
and remains disabled for WebAssembly. Runtime dispatch still requires AVX2+FMA
capability and an exact Node 6 geometry (`24 -> 48` channels, input height 24,
REC widths 320 or 960). All other Conv3x3 shapes continue to use the regular
AVX2 or scalar packed kernel.

This promotion is intentionally separate from
`LW_EXPERIMENTAL_AVX2_FMA_DISPATCH`: the broader Conv1x1 and terminal MatMul FMA
candidates remain opt-in because their shape matrix still contains regressions.
The promoted path passed the x64 full-OCR A/B checksum and 100-image quality
parity gates with no measurable RSS increase.

## Conv1x1-only FMA dispatch split

The broad `LW_EXPERIMENTAL_AVX2_FMA_DISPATCH` switch is now split into two
independent native-only switches:

- `LW_EXPERIMENTAL_AVX2_FMA_CONV1X1_DISPATCH` controls the shape-gated
  Conv1x1 FMA candidate;
- `LW_EXPERIMENTAL_AVX2_FMA_MATMUL_DISPATCH` controls the exact-shape terminal
  MatMul FMA candidate.

The original aggregate option remains supported and enables both switches for
backward-compatible experiments. Both new options default to `OFF`; the
production dispatcher is unchanged. The packed Conv1x1 and packed MatMul
benchmark drivers define only their corresponding macro, which makes it
possible to measure one candidate without contaminating the other path.

A local Windows x64 paired run used the project-owned generated 100-image PPM
corpus, Tiny assets, REC target width `960`, one warm-up, one measured pass, and
three baseline/candidate rounds. Both builds produced the same output checksum
`ea94c481ced30cbe` and 620 recognized lines:

| Workers | Default median (ms/image) | Conv1x1-only FMA median (ms/image) | Change | Default peak RSS | Candidate peak RSS |
|---:|---:|---:|---:|---:|---:|
| 1 | 224.692 | 217.902 | -3.02% | 108.30 MiB | 108.33 MiB |
| 4 | 172.307 | 169.578 | -1.58% | 132.07 MiB | 131.95 MiB |

This is encouraging but not a promotion result: it is one Windows host and one
Tiny corpus, and the four-worker improvement is below the 2% end-to-end gate.
Keep the Conv1x1-only option opt-in until Tiny, Small, and Medium paired runs
on repeated x64 runners satisfy the existing latency, checksum, and working-set
criteria. Terminal MatMul FMA remains independently opt-in.