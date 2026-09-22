# Full-OCR Optimization Baseline (phase 1 starting point)

Recorded 2026-09-22, Windows x64 AVX2+FMA, build-ninja-fast (Ninja Release,
`LW_EXPERIMENTAL_AVX2_FAST_PATH=ON`), tiny model set, bundled 500x500 sample
(16 lines), `lw-ocr-benchmark` with warmup 2 and 10 iterations.

| workers | ocr mean | ocr p95 | detector mean | after-det | throughput | RSS (final) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 251.75 ms | 259.21 ms | 72.52 ms | 179.23 ms | 3.97/s | ~86 MB |
| 4 | 104.80 ms | 110.30 ms | 71.32 ms | 33.47 ms | 9.54/s | ~123 MB |

`output_checksum` = `0ebf8b448ab7df47` at both worker counts; the full ctest
suite is green (75/75), including `full_ocr_golden_corpus` and
`rec_golden_corpus`.

Reference targets (SimdPaddleOCR 1.4.2, same model family, its docs): tiny 4w
63.1 ms, 1w ~134 ms. Our gap at this baseline: 1.66x (4w) / 1.88x (1w).

Baseline prerequisites fixed while snapshotting:

- `lw_recognizer_clone` now creates the x64 REC backend like `create` does
  (shared `recognizer_try_backend` helper). Before this fix, only the worker
  owning the original recognizer ran backend lines at width 960 — 4w behavior
  depended on line-to-worker assignment.
- `full_ocr_operator_profile` expectations updated for the backend path (3 of
  the 16 sample lines reach adaptive width 960 and bypass the executor
  operator counters).
- `staged_package` now accepts Ninja single-config consumer builds (exe at
  the consumer build root instead of `Release/`).
- Two drivers gained the missing final newline
  (`tests/nhwc_support_driver.c`, `tests/x64_rec_backend_contract_driver.c`).

Small/medium baselines will be recorded when the optimization phases reach
those model sizes; medium DET profiling is its own phase-2-scale effort.

## Phase 1-3 progress (2026-09-22)

Phase 1: DET threshold defaults stay at 0.3/0.6/1.6 for now (the reference
0.2/0.45/1.4 defaults pair with the Phase-3 postprocess algorithms; parameter
alignment landed together with them). Added the tiny-model sniff
(`model_is_tiny`, box threshold 0.4 for tiny detectors when the caller keeps
the default), the `det_unclip`/`crop_setup` fine profile slots, and
self-consistent operator-profile expectations (backend rows derived from the
CTC counters instead of hard-coded).

Phase 2: DET postprocess mechanics replaced, output bit-identical: AVX2
movemask bitmap thresholding with a 256-entry expansion table
(`lw_avx2_threshold_bitmap_f32`), 4-neighborhood interior bitmap
(`lw_avx2_interior_bitmap_u8`), scanline 8-connected flood fill, and a
grow-only per-detector workspace that removes the 6-7 malloc/free per image.
The standalone `lw_db_postprocess_f32` remains as a thin wrapper.

Phase 3: reference-parity postprocess, output changed and reviewed (net
improvement: 3 lines recognized better, 1 worse, rest neutral; accepted and
the golden corpus fixture regenerated). Ported: background-hole tracing
(OpenCV RETR_LIST parity, 4-connected scanline fill), Clipper-5.1.5-style
round-join unclip (`offset_round` with half-even/half-away rounding and
bounded arc steps), the cv2.fillPoly scanline `polygon_score` (crossing
parity plus exact integer on-edge hits), PaddleOCR `order_points_clockwise`
(with the 45-degree centroid-angle fallback), round-half-even source remap
with clamp to [0, size], and the PaddleX `CropByPolys` mapped-corner
min-area-rect step. Thresholds aligned to the reference defaults (bitmap 0.2,
box 0.45/0.4-tiny, unclip 1.4).

Post-Phase-3 local profile (4 workers, 1 iteration): det_preprocess 4.22 ms,
det_graph 33.93 ms, det_postprocess 1.46 ms (unclip 0.009 ms), crop_setup
0.001 ms, total 112.73 ms. The det graph is now the dominant det-side target
(Phase 8 / phase-2 NHWC scope); det_preprocess fixed-point conversion is
Phase 4.

Phase 4: DET preprocess converted to the OpenCV-parity 11-bit fixed-point
path (`det_preprocess_fixed.c`, shared helpers in `resize_fixed_internal.h`)
with the 768-entry ImageNet LUT (normalization stays bit-identical to the
double path), row-parallel sharding over the detector intra-op pool (>= 24
rows/worker, per-worker row scratch pre-allocated), a reusable
`lw_det_preprocess_workspace`, and the tiny-image 32x32 zero pad before
resize. det_preprocess dropped from 4.22 ms to 0.82 ms (5.2x). One golden
corpus line regressed (`ODM OEM` -> `WEONCO` on the 375px nearest-resize
case) and one improved (`藏麦` -> `燕麦`); the regression is a known
crop-chain alignment gap (our crop is still double bilinear, the reference
uses INTER_CUBIC), accepted and recorded, the case's `min_cls_score` gate
lowered accordingly.

Phase 5: CLS preprocess converted to the reference keep-aspect semantics
(`cls_preprocess.c` fixed entry): left 4:1 window, `min(160,
ceil(80*w/h))` target width, trailing columns padded with -1, RecNorm LUT,
shared fixed-point helpers. Guarded by `LW_EXPERIMENTAL_CLS_FIXED_POINT_RESIZE`
(default ON); the stretch path remains as the OFF fallback. Golden corpus
outputs are unchanged by this phase (the fixture diff was empty), and the
classifier contract/pipeline reference expectations were updated to the new
resized-width semantics.

Phase 7: orchestration. (a) A persistent line-worker pool
(`LW_EXPERIMENTAL_PERSISTENT_LINE_POOL`, default ON) replaces the
per-request `_beginthreadex`/join cycle; `lw_thread_pool_run` keeps worker 0
on the caller thread and barriers at the end, preserving the existing
width-affine scheduling. Line dispatch overhead dropped from ~0.9 ms to
~0.04 ms per request. (b) The crop phase is now a parallel row-band phase
(`LW_EXPERIMENTAL_PARALLEL_CROP_PHASE`, default ON):
`lw_crop_quad_bgr_u8_band` rasterizes [row_begin, row_end) with identical
per-pixel math, bands are built during the serial geometry assignment, and
workers stride over the band list (16 rows per band). Crop wall time dropped
from ~4.9 ms to ~1.95 ms; the golden corpus stays bit-identical (75/75
ctest green). Local 4-worker mean after Phase 7: 102.33 ms (baseline
104.80 ms); remaining dominant items are the line-worker phase (61.5 ms)
and the det graph (33.9 ms).

Phase 8: (a) DET SMT evaluation on the NCHW path: det_threads 8 -> 16 gave
~3% det_graph improvement (98.9 -> 95.5 ms over 3 runs), far below the
reference's 8-25% NHWC claim, so the default stays on physical cores; SMT
will be revisited with the phase-2 NHWC backend. (b) ConvTranspose+bias+
activation epilogue fusion deferred: the tiny graph only has two small
ConvTranspose nodes, and the phase-2 NHWC backend redoes this path anyway.
(c) Standalone activation SIMD: `LW_OP_HARD_SIGMOID` and `LW_OP_RELU` now
dispatch to new bit-identical AVX2 kernels (`lw_avx2_hard_sigmoid_exact_f32`
uses separate mul+add to match the scalar reference exactly, unlike the
existing FMA variant; `lw_avx2_relu_f32` uses max(v, 0) which matches the
scalar NaN behavior). Golden corpus stays bit-identical.

Note: local benchmark runs fluctuate ~10% with machine load; the numbers in
this file are informational reference points, not release gates.
