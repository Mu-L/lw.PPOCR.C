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

Medium DET follow-up: the low-spatial, wide-output pointwise path cannot
partition output channels with the current tightly packed NHWC kernel: each
pixel still needs the full output-channel stride. Pointwise sharding now uses
spatial ranges only. The Medium 32x64 sharded contract exposed this mismatch;
the profile assertion now respects the public 256-node telemetry capacity
instead of assuming every Medium node has a per-node slot. The layout-convert
functions have explicit AVX2/FMA target attributes for GCC/Clang builds.
The shared REC converter remains Tiny-only and retains strict ONNX shape
inference: relaxing its SHA guard does not lower Small/Medium dynamic Reshape
nodes. Those variants use their dedicated conversion tools. DET's
`LW_ALLOW_ANY_DET` override does not apply to REC.


REC epilogue fusion + transpose fast path + cross-width arena sharing
(2026-09-24): (a) Conv epilogue fusion in the REC x64 backend lowering:
a single-consumer channel-bias Add, a residual Add, and the five-node
exact-GELU chain (Div/Erf/Add/Mul/Mul, two consumers of the conv output,
matched via tensor_consumed_within) fold into the preceding NHWC
pointwise/depthwise conv. The GELU epilogue uses an erff-based scalar +
AVX2 no-FMA vector helper that reproduces the standalone GELU pass bit
for bit (partial blocks: full 8-lane vectors plus scalar erff remainder,
identical to the pass layout). Dense convs are excluded from folding:
measurements showed dense+epilogue runs ~28% slower (1.81 vs 1.41 ms/line
at width 960), wiping out the gain. Depthwise folds bias only. (b) The
TRANSPOSE op got an AVX2 8x8-block 2D fast path (effective 2D
transpositions only; degenerate permutations degrade to memcpy, others
fall back to scalar): 0.89 -> 0.02 ms/line at width 960. (c) All width
slots of one recognizer instance now share a single arena/scratch/CTC
allocation sized to the max program requirement (borrowed workspace;
each worker clone owns its own shared block, thread-safe), replacing
per-slot buffers. Bit-exactness: golden corpus checksum unchanged
(46d99468540b5eb7) in every ctest and benchmark run. Full suite 88/88.

Measurements (interleaved A/B vs a clean HEAD build, quiet machine):
REC op timer total_run_ms per line, width 960: 16.02 vs 16.60 ms (-3.5%);
width 640: ~10.7 vs 10.8 (parity); width 320: 6.18 vs 6.15 (parity).
REC arena per instance: 3,317,760 -> 2,764,800 bytes (-16.6%).
End-to-end sample.ppm 4w: 101.7 vs 102.6 ms mean (-0.9%); 1w: ~229 vs
~230-250 ms (parity to slightly faster). Process private commit at
create: 4 workers 93.6 vs 118.6 MB (-25 MB, -21%); 1 worker 51.0 vs
57.2 MB (-11%); default 8 workers 150.5 vs 200.5 MB (-50 MB, -25%).


CLS compiled x64 backend (2026-09-24): the classifier reuses the REC
lowering/execution machinery. The compile entry gained an explicit input
height plus an allow-CTC-tail switch (`lw_x64_rec_backend_compile_input`;
REC wrappers keep height 48 and CTC detection on) because the CLS head is
also MatMul -> Add -> Softmax and would otherwise be misdetected as a CTC
tail. The backend learned a SOFTMAX physical op that delegates to
`lw_scalar_softmax_f32` (bit-exact with the canonical executor by
construction). `lw_avx2_nhwc_reduce_mean_hw_f32`'s vector path now divides
by H*W instead of multiplying by the reciprocal, matching its own scalar
remainder and the canonical NCHW reduce bit for bit (this cut the CLS
score drift against the canonical executor from 5.0e-6 to 2.0e-6, inside
the reference test's 5-place tolerance). classifier.c compiles the CLS
graph at create (NHWC strategy, 3x80x160, CTC off), converts the NCHW
preprocess planes into the instance input (pure data movement), runs, and
reads the two logits from the arena output value; any compile/runtime
failure falls back to the canonical session. The full-OCR profile test now
derives the expected canonical operator count from the classifier
implementation-path counters (zero when all lines take the compiled
backend), mirroring the existing compiled-REC handling.

Measurements (interleaved A/B vs clean HEAD build): CLS classify
1.66-1.70 ms vs 2.00-2.05 ms per crop (-17%, label and score bit-
identical on the sample crop). End-to-end sample.ppm: 4w 100.9 vs
103.5 ms mean (-2.5%); 8w 78.1 vs 78.4 ms (-0.4%, workers saturated).
Private commit at create: 4w 100.2 vs 118.6 MB (-15.5%); 8w 163.4 vs
200.5 MB (-18.5%) — the compiled CLS instance adds ~1.3 MB per worker
versus the REC-only shared-workspace state. Golden corpus checksum
unchanged (46d99468540b5eb7); full suite 88/88.


DET Conv+GELU epilogue fusion (2026-09-24): the DET lowering now folds the
five-node exact-GELU chain directly after an NHWC pointwise/dense conv into
the conv epilogue (mirroring the REC fold; the chain reads the conv output
twice, so a det_tensor_consumed_within range check replaces the
single-consumer test). Physical DET ops at 512x512: 176 -> 163; standalone
GELU time (4.78 ms of 20.7 ms per run at 8 workers) disappears into the
convs, DET op total 20.7 -> 15.6 ms (-25%). The scalar NHWC conv fallback
learned the exact erff-based GELU so forced-fallback contract parity holds
(bit-identical to the vector epilogue by construction). REC pointwise
kernel assessment: pure conv math already runs at 84-116 GF/s against a
~128 GF/s single-core AVX2-FMA peak, so the remaining REC graph time is
near the FP floor; CTC already uses the packed matmul + fused argmax path.
End-to-end sample.ppm vs clean HEAD: 8w 72.5 vs 75.9 ms (-4.5%), 4w 94.0
vs 97.8 ms (-3.9%). Golden corpus checksum unchanged
(46d99468540b5eb7); full suite 88/88.


## 第四轮：DET elementwise / pool 算子挂线程池分片（2026-09-24）

借鉴 SimdPaddleOCR 的 UnaryParallel 思路：此前 DET 后端只有卷积类算子分片，
elementwise 全部单线程。本轮把逐元素独立的算子挂上 DET 已有线程池，
按元素/像素/行范围分片，每元素仍由同一个逐元素核计算，逐位一致。

改动（src/runtime/x64_det_backend_execute.c，单一文件）：
- 新增 shard kind：BINARY_SAME / BINARY_SCALAR / BINARY_CHANNEL / SIGMOID /
  CONCAT_NHWC / POOL_NCHW_CHANNELS / POOL_NHWC_ROWS，统一走
  run_sharded_elementwise() 辅助函数（阈值 128K 元素，小算子保持串行）。
- add/mul/div 的 contiguous、right-scalar、channel-broadcast 三种 AVX2 路径
  全部按输出元素范围分片；sigmoid_nchw 按元素分片；concat_nhwc 按像素分片。
- max_pool/avg_pool：NCHW 按通道切（指针偏移 + dims 切片）；NHWC 按输出行切，
  begin>0 的分片把输入指针偏移 begin*stride_h-pad_top 行并置 pad_top=0
  （派发侧保证 stride_h>=pad_top），窗口裁剪与串行完全一致。
- 串行契约钩子 lw_x64_det_instance_run_op（state=NULL）行为不变。

DET 512x512 算子级（8 线程，det-op-timer，ms/次）：
| 算子 | 融合GELU后 | 分片后 | 变化 |
| --- | --- | --- | --- |
| add | 1.92 | 0.60 | -69% |
| max_pool | 1.01 | 0.17 | -83% |
| sigmoid_nchw | 0.89 | 0.15 | -83% |
| mul | 0.72 | 0.36 | -50% |
| concat_nhwc | 1.45 | 0.28 | -81% |
| DET 算子总计 | 15.6 | 10.8 | -31% |

端到端（sample.ppm 500x500，16 行，交替 A/B 各 3 轮取均值 vs HEAD）：
| 线程 | HEAD 基线 | 本轮累计 | 变化 |
| --- | --- | --- | --- |
| 8w | 90.19 ms | 75.96 ms | -15.8% |
| 4w | 113.38 ms | 106.83 ms | -5.8% |

ctest 88/88 绿；golden checksum 46d99468540b5eb7 全部轮次不变。


## 第五轮：DET conv epilogue 折叠 Add + reduce_mean 分片（2026-09-24 下午）

关键定位修正：benchmark 的 detector_ms ~56ms 远大于 det-op-timer 512x512 的
10.8ms，原因是 limit_side_length=960 把 500x500 图像放大到 960x960 跑 DET
（(960/512)^2 ≈ 3.5x）。真实负载下 DET 占端到端约 77%，仍是最大头。

改动：
1. Conv epilogue 折叠（移植 REC 的 fold_conv_epilogue 思路到 DET 编译器）：
   NHWC pointwise/dense conv 后贪婪吸收 channel-bias Add（常量）+ 残差 Add
   （非常量同尺寸张量）+ 末尾 Relu 或五节点精确 GELU 链，统一替换原来的
   Conv+Relu / Conv+GELU 两个独立块。核内顺序 accumulator -> +post_bias ->
   +residual -> activation 与原算子序完全一致，逐位不变。
   - lw_x64_det_conv_op 增加 post_bias / residual_offset / has_residual 字段。
   - 执行侧：POINTWISE/DENSE case 解析 residual 运行时指针；分片 worker 按
     片首像素/行偏移 residual（核按片内局部索引）；scalar_nhwc_conv_row_range
     增加 post_bias/residual 参数（绝对行索引），depthwise 传 NULL。
   - 限制：仅 POINTWISE/DENSE（depthwise 核只接受 bias 指针，不动）；
     要求 Add 节点 NHWC-effective、两侧 value 均 NHWC 布局、单消费者。
   - 效果：物理算子 163 -> 153，960x960 add 8.3 -> 6.0ms。剩余 add 的生产者
     是 mul（hardswish 尾部 x*h(x)+residual）和 resize_nhwc（DB head 上采样
     特征相加），不是 conv，不可折叠。
2. reduce_mean 按通道分片：新增 lw_avx2_nhwc_reduce_mean_hw_strided_f32
   （行跨步参数化，每通道像素访问顺序不变，逐位一致）；NCHW 路径在 worker
   里按通道范围跑原标量循环。960x960 reduce_mean 6.5 -> 1.7ms。

端到端（sample.ppm，交替 A/B 各 3 轮取均值 vs HEAD，checksum 全部
46d99468540b5eb7 不变，ctest 88/88 绿）：
| 线程 | HEAD 基线 | 本轮累计 | 变化 |
| --- | --- | --- | --- |
| 8w | 76.92 ms | 68.95 ms | -10.4% |
| 4w | 95.97 ms | 85.82 ms | -10.6% |

剩余热点（960x960，8 线程）：pointwise ~23ms、dense ~16ms（卷积核本身，
接近算力上限）、depthwise ~7ms、add ~6ms（mul/resize 尾部，访存受限）、
concat_nhwc ~4ms、convert ~3.7ms、relu ~3.6ms（depthwise 后，核不支持
activation）、mul ~2.8ms。


## 第六轮：pointwise 全张量展开核（REC/DET 共享）+ 三档模型占比实测（2026-09-24 傍晚）

### 阶段占比实测（当前优化构建，16 行文本样图）

DET 内算子线程数固定为物理核数（8），与 workers 无关；行级（CLS+REC）阶段
按行跨 worker 并行。ocr_ms 减去 detector_ms 即行级阶段：

| 档位 | workers | det_ms | ocr_ms | 行级阶段 | 行级占比 |
| --- | --- | --- | --- | --- | --- |
| tiny | 1 | 47.2 | 224.9 | 177.7 | 79% |
| tiny | 4 | 47.1 | 85.4 | 38.3 | 45% |
| small | 1 | 118.4 | 1309.3 | 1190.9 | 91% |
| small | 4 | 119.2 | 520.7 | 401.5 | 77% |
| medium | 1 | 975.9 | 5171.3 | 4195.4 | 81% |
| medium | 4 | 973.7 | 2054.3 | 1080.6 | 53% |

结论：低 workers 时 REC（行级阶段）确实是大头（79-91%）；8 workers 时 DET
与行级各占一半左右。两条线都要压。

### pointwise 展开核

tiny REC 单行 pointwise 占 70%（11.3/16.3ms），扩张卷积（48→96 等）只有
38 GF/s（单核峰值 ~128）。微基准定位：现有 generic 6x16 tile 的
`__m256 lo[6]/hi[6]` 数组在 MSVC 下寄存器分配不佳；手写全展开命名寄存器
版本 79 -> 111 GF/s，输出逐位一致。

但把展开 tile 塞进 grouped 循环（每 tile 一次调用）没有收益——每次调用的
栈帧序言（sub rsp,290h + 9 个 xmm 保存）吃掉了增益。最终方案：
lw_nhwc_pointwise_unrolled_all()，整个卷积一次调用，tile-outer/oc-inner
循环 + 内联 12 寄存器展开累加 + store_row16 epilogue，noinline 防止被
分组循环内联劣化。仅在 Cout%16==0 且权重 <=256KB（L2 友好）时启用，
其余形状仍走 grouped（大权重下 grouped 的 oc-outer 序更省缓存）。
微基准（2880px, 48->96）：79 -> 107 GF/s；REC 单行总耗时 -9%。

尝试过的弯路：把 REC 多像素 pointwise 切到 3x32/4x16 核——慢 2 倍
（3 行 tile 寄存器块效率低），已回退。

端到端（sample.ppm，交替 A/B 各 3 轮取均值 vs HEAD，checksum 全部
46d99468540b5eb7 不变，ctest 88/88 绿）：
| 线程 | HEAD 基线 | 本会话累计 | 变化 |
| --- | --- | --- | --- |
| 1w | 256.8 ms | 224.4 ms | -12.6% |
| 4w | 103.1 ms | 87.4 ms | -15.2% |
| 8w | 77.7 ms | 65.4 ms | -15.8% |


## Round 7: grouped pointwise oc-block tile-range worker (2026-09-24)

`lw_nhwc_tile_rows16_unrolled` (per-tile noinline call) replaced by
`lw_nhwc_ocblock16_tile_range`: one noinline call covers a whole pixel group
of a full 16-channel output block, so the stack-frame prologue is paid once
per (group, block) instead of once per 6-row tile. Partial channel blocks
fall back to per-tile `lw_nhwc_tile_rows16`. Same bias init, ascending ic
FMA order and store_row16 epilogue => bit-identical.

Verification:
- pw-microbench bit-identical YES; ctest 88/88 green.
- det-op-timer 960x960 tiny DET: pointwise total 23ms -> 15.2ms (-34%).

End-to-end (interleaved A/B vs HEAD build, 3 rounds, mean ocr_ms;
checksums identical on both sides):

| model  | workers | HEAD ms | opt ms | delta |
|--------|---------|---------|--------|-------|
| tiny   | 1       | 232.4   | 200.1  | -13.9% |
| tiny   | 4       | 81.8    | 70.0   | -14.4% |
| small  | 1       | 1169.6  | 1141.9 | -2.4%  |
| small  | 4       | 414.6   | 401.7  | -3.1%  |
| medium | 1       | 4548.1  | 4348.7 | -4.4%  |
| medium | 4       | 2208.5  | 2147.1 | -2.8%  |

checksums: tiny 46d99468540b5eb7, small 2ee4a78f9306c18b,
medium 3bdfd6741f2ffab7 (unchanged).

small/medium gains are smaller because small REC stays on the canonical
interpreter (known argmax mismatch) and medium REC runs through
lw_x64_rec_fast_plan; the grouped path mainly serves their DET stages.

## 2026-09-24 CTC 4 matmul L�vL,]n	

- �� src/runtime/ctc_head_parallel_internal.{c,h}CTC 4 argmax matmul 	 4 LW0�`
  �*��C /�z�2L8�h �M�8	 4 LWY�"^ 4 p>�
  E�L�{ 4 L�P
- recognizer 	�	 intra `lw_recognizer_set_intra_op_thread_count	0 session 
  x64 �M��ocr.c 	 worker pG��1w=8 �4w=2 �/recognizer	
- �e�executor.c $* CTC (�tiled/^ tiled	ctc_projection_internal.c��	
  ctc_row_probabilities 	�L�͝V�2L
- �R�binary }w�� channel_major_nchw �NCHW gL/�� rank-4 NCHW i
   �/(rank-3 �� [.., .., C]  �� channel-last ��� x64_rec_backend_nchw
  I 5 yK�1%semantic 155 Add max_abs=3.42	
- A/Bhead-build �� vs ,n3 n��G<checksum h�	
  tiny    1w 232.3->196.1ms (-15.6%)  4w 86.5->75.3ms (-13.0%)
  small   1w 1265.9->889.9ms (-29.7%) 4w 473.0->327.0ms (-30.9%)
  medium  1w 4791.6->4458.4ms (-7.0%) 4w 1885.2->1673.5ms (-11.2%)
- ctest 88/88 h�

�z�48SM�* 4 LW�A 14.4MB(medium)/9MB(small) C�panel-outer ��͒�
�C�U!�Xb����͒M�		bvL�Z%<'�v

## 2026-09-24 CTC 48 panel-outer ͒,An	

- lw_avx2_fma_packed_matmul_argmax_scores_f32 ��͒b16 	BLW�B
  L	 128 W�L�L '<�{Y
24KB	b/������W
  �C  inner �/�z���L argmax �	bG�vM ���� MATCH	
- 2L���120x192x18710	22.0 -> 7.2 ms/call-67%	C�ϧA� 432MB -> 14.4MB
- �0���4�L�vLvL��;�	tiny/small/medium �M� 1-2%
- -��K medium 6889-7584ms �G� �::h}j���6 profile q���
  Kb 4480ms ϧ	A/B �{������6 &MW DLL
- ctest 88/88 h�	!� checksum h�

## 2026-09-24 REC �r 1x1 pointwise w�� GvL,A n	

- x64_rec_backend_execute.cPOINTWISE �P	 12 � ��2/3/4/6 �  tile Gtd 12	
  0 recognizer intra `�� ��/�z��M �checksum h�	
- ��pixels*ic*oc >= 4M MAC  pixels >= 24 MG�M�Pe  
- A/B:��G<checksum h�	
  tiny    1w 182->117ms (-36%)   4w 65->53ms (-18%)
  small   1w 850->427ms (-49%)   4w 287->220ms (-23%)
  medium  1w 5640->2219ms (-60%) 4w 2315->1532ms (-34%):h�'� �	
- ctest 88/88 h�

iY��dense 3x3 convmedium node 11 � 41ms/L	q� scratch *Gdepthwise 	
S�X��Ldense G  per-worker scratchY\ n

## 2026-09-24 REC dense w��LGvL,A�n	

- x64_rec_backend_execute.cDENSE �P	��L0 intra `dense 8�/
  output_row_offset�e�h� �	 (oy+offset)*stride-pad �@��/��	LO�
  ��  tap /�z��M �checksum h�	
- per-worker scratchdense 8 scratch �ULp�s���'	 M workers*scratch_bytes
  owns_workspace ��instance_free �a��>	M1%� 2L
- A/B:��checksum h�	tiny 1w -9%small 1w -12%medium 1w -3%6 nG<
  :h��>	4w �,s� worker �`�8	
- ctest 88/88 h�

## 2026-09-24 REC depthwise LG + per-kind ����,A	n	

- �����Ϣ� LW_X64_REC_PROFILE=1run_backbone_ops �P��instance_run +>
   stderr Sp{�pw/dense/dw/bin/unary/reduce/pool/transpose/matmul/ctc	
  s���  medium 960 �2L�Kpw 283ms/L85%	dw 19.5matmul 11.3ctc 7
- DEPTHWISE �P	��LG8� output_row_offset� scratchM �	
- A/B� pointwise G HEAD��nchecksum h�	
  tiny 1w -9%small 1w -9%medium 1w -7% / 4w -11%dense+dw v6�	
- ctest 88/88 h�

iY��medium 960 �vL�L ~110ms	pw ~72ms2L 283ms8 �� 4.2x
&�/iU'�P	matmul ~13ms���*G	dw ~11msctc ~11msprobabilities
2L͗	

## 2026-09-24 REC matmul GvL,A�n	

- MATMUL �P$�M �Grank-2 q�C�	L4 L��� AVX2 LW>�18
  �&Yp�	�� rank-4 [1,B,M,K]x[1,B,K,N] ���	 B �y� *
  lw_scalar_matmul_f32 � rank-2 ��gL( lw_matmul_shared_f32M FMA 6)�	
- packed �CTC 4	�1 ctc_head :6vL(dG
- medium 960 � matmul 13 -> 7ms/LA/B� pointwise HEAD+ dense+dw+matmul /�
  checksum h�	tiny 1w -8%small 1w -13%medium 1w -16%4w �,s
- ctest 88/88 h�
