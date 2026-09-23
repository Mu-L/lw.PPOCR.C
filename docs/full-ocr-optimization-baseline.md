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

## Phase 2 (NHWC DET graph) progress

Phase 0 (layout coverage probe, 2026-09-22): `layout-plan-driver` now takes
`[height] [width] [direct-nhwc]` (backward-compatible `[width] [direct-nhwc]`
form kept), prints conv params (`conv nodes:` section, raw weight dims —
Conv=[OC,IC/g,kh,kw], ConvTranspose=[IC,OC,kh,kw]) and full tensor shapes
(`tensor shapes:` section). New ctest `layout_plan_det` probes det.lwm at
640x640 direct-nhwc.

Probe results (tiny det, direct-nhwc, identical at 32x64 — structure is
shape-independent): 205/242 nodes NHWC, 37 NCHW, planner conversions=2
(tensor 167 [1,16,320,320] at node 002 and tensor 169 [1,8,320,320] at node
004, both at the FPN head), 28 islands. The 8 non-aligned dense convs
(OC=4x4, 8x2, 12x1, 40x1) are 1 real conv (002, w=[8,16,2,2] at the FPN
head) plus 7 SE-gate squeezes operating on neutral [1,C,1,1] tensors —
layout-irrelevant and conversion-free. The probability-map tail
([1,1,640,640] is neutral) and SE gates cost nothing to keep NCHW.

v1 backend (island-exit nodes forced NCHW) would pay ~16 conversions /
~75 MB of copies (~4-5 ms): 007 Concat ~26 MB, 6 Resizes ~16 MB, 233
Concat ~8 MB, 236/239 ConvTranspose ~14.5 MB, planner's own ~9.8 MB. This
exceeds the 12-conversion / 2 ms thresholds, so the Phase-2 NHWC kernels are
mandatory; the probe ranks Concat as the single biggest island-exit item
(~34 MB across the two concats), so the kernel order is Resize ->
ConvTranspose -> Concat.

Phase 1 (x64_det_backend skeleton + A/B harness, 2026-09-22): new
`src/runtime/x64_det_backend_internal.h/_compile.c/_execute.c` (REC-backend
clone: two-pass lowering, greedy-by-size arena interval planner with
alternate-layout slots, per-op layout fields). New design piece: the
effective-layout walk crosses the planner's NHWC classification with the v1
lowerability set (Resize/ConvTranspose/Concat/Sigmoid stay NCHW), and tensors
crossing an effective-layout boundary get a lazily materialized
LAYOUT_CONVERT physical op writing a dedicated alternate arena slot. Both
compile strategies (NCHW / NHWC) lower det.lwm 100%: the NCHW arm covers the
det conv zoo (1x1 packed, 3x3s2/3x3s1/2x2-pad-end1 AVX2, depthwise 3x3/3x3s2x1/
5x5, scalar fallback) plus scalar ConvTranspose/Resize/Sigmoid/Concat; the
NHWC arm reuses the REC pointwise/dense/depthwise kernels. The detector gains
opt-in hooks (`lw_detector_test_enable_x64_backend`/`_disable_`, production
default stays canonical until the promotion gate) and a profiled backend
execution path that maps physical-op time onto the semantic profile.

New tests: `x64_det_backend_contract` (prob map vs canonical within
1e-3/1e-5 at 32x64 and 640x640, bit-determinism, arena-overlap checker,
per-tensor NCHW/NHWC parity replay with in-situ convert verification, profile
telemetry assertions, detector-level box smoke), `x64_det_layout_benchmark`
(rotated-order median A/B with per-round correctness), and 6 new dense-kernel
cases incl. asymmetric SAME_UPPER 2x2 pads (0,0,1,1), the ic=3 stride-2 stem,
and the 64->16 head.

Debugging note (fixed): the cloned REC `compile_conv` read the graph input's
primary slot directly, bypassing `det_slot_offset`, so the NHWC stem consumed
the NCHW input as NHWC. Localized by the snapshot-based per-tensor parity
replay (the first divergence was tensor 166, the stem output, at 13.9 vs the
FMA-noise scale of 1e-4) after end-of-run reads proved meaningless because
arena slots are legally reused. Fixed by resolving conv inputs through
`det_slot_offset`.

A/B (640x640, rotated order, median of 8): canonical 1w 110.65 ms, NCHW arm
107.59 ms (1.03x), NHWC arm 87.27 ms (1.27x) — the NHWC arm already wins
single-threaded (REC's NHWC arm was a 0.50x negative control). Canonical at 8
threads is 55.93 ms; the single-threaded NHWC arm is 0.62x of it, so Phase 3
threading is the decisive step. NHWC prob map matches canonical within
2.7e-8 at 640x640; NHWC arena 37.7 MB / NCHW 31.1 MB. Full suite 78/78.

Phase 2 (NHWC op coverage, 2026-09-23): three new kernels, all bit-exact
against the scalar references (max_abs 0, kernel-level driver
`nhwc_convtranspose_contract`): `lw_avx2_nhwc_resize_nearest_f32` (integer
scales, whole-row copies in avx2_nhwc_support.c), the 16-OC
`lw_avx2_fma_nhwc_convtranspose2x2_s2_f32` (four per-tap 1x1 GEMMs, weights
packed [tap][oc/16][ic][16] from ONNX [ic,oc,2,2] via the new
`lw_pack_nhwc_convtranspose2x2_f32` in avx2_nhwc_convtranspose.c), and the
1-OC probability-map `lw_avx2_fma_nhwc_convtranspose2x2_s2_c1_f32`
(strided-weight gather, ReLU-only epilogue, Sigmoid defused). Also fixed the
dense HARDSWISH store-path gap (avx2_nhwc_dense.c: the vectorized full-block
store silently dropped HARDSWISH; now min(max(x+3,0),6)*x/6, with 10 new
per-case hardswish parity checks in the dense driver) and the NHWC Concat
(channel-axis per-pixel copies inline in the execute switch).

Backend wiring: Resize/ConvTranspose/Concat joined the NHWC arm (Sigmoid
stays defused NCHW-scalar — canonical parity, and neutral tensors make it
conversion-free). NHWC conversions dropped 20 -> 3 (graph input + the FPN
head pair), NHWC-effective nodes 195 -> 205 (the 37 NCHW nodes are the 8
non-aligned convs, the neutral SE gates, and the Sigmoid tail), NHWC arena
37.7 -> 31.1 MB (equal to the NCHW arm). A/B (640x640, 1w, median of 8):
canonical 102.28 ms, NCHW arm 106.06 ms, NHWC arm 59.22 ms (1.73x) — the
single-threaded NHWC arm is now within 3% of the 8-thread canonical
(57.59 ms). Phase 3 threading is the remaining decisive step. Full suite
79/79.
Phase 3 (threading, 2026-09-23): generic shard wrapper in
x64_det_backend_execute.c with the reference thresholds (dense/pointwise 2M
MACs, depthwise and the 1-OC prob map 1M — depthwise MACs counted without
the channel factor), workers capped at 16, whole-tile/whole-row contiguous
ranges only (bit-identical sharded output). Partitioning per kernel:
pointwise flat pixel ranges (or OC blocks for small spatial), dense and
depthwise output-row ranges, ConvTranspose input-row ranges. Dense and
depthwise kernels gained an optional `output_row_offset` desc field
(validation became a range check: offset + height <= full height; the
serial semantics are untouched at offset 0). Per-worker dense scratch
slices come from a per-instance shard-scratch arena
(program scratch x 16 workers, ~90 KB at 640x640). The detector hookup now
passes its session pool and intra-op count via
`lw_x64_det_instance_run_profiled_ex`; thread histograms record the workers
actually used per sharded conv / prob map.

Debugging notes: the first sharded validation (`output_height + offset ==
full height`) accepted only the last row range and pushed every other worker
into a scalar fallback that itself double-offset its rows; the parity
replay (now a third sharded pass in the contract driver via the new
`lw_x64_det_instance_run_op_ex` hook) localized the first divergent op
(node 12, the 3x3 depthwise) and the standalone depthwise row-shard test
pinned the validation bug.

A/B (640x640, rotated, median of 8): canonical 1w 107.1 ms vs NHWC 60.0
(1.79x); 2w 69.3 vs 47.0 (1.48x); 4w 56.4 vs 38.5 (1.46x); 8w 53.6 vs 33.8
(1.59x); 16w 51.0 vs 31.6 (1.61x). The 8-thread default already clears the
1.15x promotion gate. Full suite 80/80 (new ctest
`x64_det_backend_contract_sharded` runs the 640x640 sharded parity,
histogram smoke, and a threaded detector smoke).

Phase 4 (promotion, 2026-09-23): the sharded NHWC DET backend is now the
production default (`x64_backend_enabled` on in `lw_detector_create`,
NHWC-first with NCHW fallback and canonical fallback; the test disable hook
remains). The promotion gate was cleared at the 8-thread default (1.59x,
median, rotated). Profile integration: every backend conv records its
worker count in the thread histograms (serial runs at index 1, matching the
canonical executor's accounting), and the full-OCR profile test now
expects `layout.selected_nodes > 0` with `fallback == candidate -
selected`, and zero prepared/kernel-path counters for the detector when the
backend is active.

SMT evaluation (negative result, recorded): det at 16 logical threads
speeds the standalone graph (~7%: 33.8 -> 31.6 ms) but slows the full OCR
pipeline by ~13% (4w 106.4 -> 92.1 ms with the physical-core cap) because
16 DET threads plus 4 line workers over-subscribe the physical cores. The
default therefore stays on physical cores (cap 8, the proven policy); the
16-thread path remains available via `lw_ocr_set_det_intra_op_thread_count`
and the backend itself is verified thread-count-independent (16-worker
sharded parity at 640x640 is bit-exact).

End-to-end (lw-ocr-benchmark, 500x500 sample): 4w OCR mean 92.2 ms
(baseline 104.8 ms, 1.14x; detector alone 40.8 ms vs baseline 71.3 ms),
throughput 10.8/s. 1w OCR mean 262.5 ms (baseline 251.8 ms): DET dropped
from 72.5 to 42.1 ms but the line phase grew ~44 ms — the Phase-3
threshold alignment (0.2/0.45/1.4) detects more/larger crops than the old
defaults, and the serial line pipeline absorbs that cost while the 4w case
parallelizes it away. Output checksum is worker-count-independent
(46d99468540b5eb7). Full suite 80/80.

Small-model regression fix (2026-09-23): the user's three-model A/B found
the small family 5.8-12.3% slower. Root cause: the small DET is the
24-channel variant, and the planner's dense rule (OC>=16 && OC%16==0)
rejected every 24-channel conv, so the whole FPN chain ran on the serial
NCHW arm (251.7 of 486 ms in the NHWC run, plus 63.9 ms of scalar
ConvTranspose) while the canonical executor sharded it. Fix: the dense rule
is now OC>=8 && OC%8==0 (the kernels already store partial 16-lane tail
blocks lane-exactly; the ConvTranspose NHWC kernel gained partial-block
bias/store handling and its pack now pads tail lanes with zeros; the REC
fast-plan prototype's own OC%16 gate was synced). Results at 960x960:
small DET 0.87x -> 1.71x (NHWC 204.7 vs canonical 349.9 ms), tiny
1.59x -> 1.80x (29.0 vs 52.2 ms, conversions 3 -> 1), medium 3.07x (584.9
vs 1795.2 ms). End-to-end tiny 4w improved to 89.3 ms (throughput 11.2/s).
The converter's tiny-only SHA guard now yields to an LW_ALLOW_ANY_DET
environment override so the small/medium models can be converted for
testing. Full suite 80/80.

Follow-up optimization pass (2026-09-23): (a) the planner's dense rule
widened again to any OC >= 8 — the 12-channel head conv of the small family
(12%8 != 0) also leaves the serial NCHW arm; the kernels store partial tail
blocks lane-exactly regardless of alignment. (b) Conv+Relu epilogue fusion
in the DET lowering: a single-consumer Relu directly after an NHWC
pointwise/dense conv folds into the epilogue (semantic span widens to 2 so
the operator profile keeps counting both nodes; the conv output tensor is
never materialized; the fused op writes the Relu output tensor's slot; the
scalar fallback now applies the fused activation). (c) The layout converts
got an AVX2 8x8-transpose path for 8-multiple channels (bit-exact, kernel-
level round-trip tests incl. 3/12-channel scalar paths); small-det convert
time dropped 22.7 -> 2.1 ms. (d) The REC fast-plan prototype's own OC gate
synced to the planner (OC >= 8, depthwise unchanged). Results at 8w:
tiny 640 1.87x (28.0 vs 52.3 ms), small 960 2.21x (164.0 vs 362.3 ms),
medium 960 2.81x (666 vs 1871 ms); tiny end-to-end 4w 89.4 ms. Full suite
80/80.
