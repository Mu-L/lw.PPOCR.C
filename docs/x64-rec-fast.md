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
- GELU fusion safety checks for private single-consumer temporaries, plus elided intermediate tensors excluded from the NHWC workspace;
- a compiled physical-op table that executes fused semantic spans once and exposes per-kind profile timings;
- an independent physical-op table that owns packed weights, folded bias, epilogues, and semantic spans, with generic spans validated at execution time;
- plan-time Conv+BatchNorm folding plus Conv+ReLU and Conv+Add(+ReLU) epilogues for eligible Pointwise/Dense nodes;
- per-plan fusion counters (`fused_conv_bn`, `fused_conv_relu`, `fused_conv_bn_relu`, `fused_conv_add`, and `fused_conv_add_relu`) for regression and benchmark reporting;
- physical tensor lifetimes mapped from semantic spans, with workspace overlap validation before a plan is published;
- tile-first Dense scheduling that gathers border patches once per spatial tile and uses a single 6x16 accumulation tile;
- an AVX2/FMA Depthwise two-pixel interior kernel for full 32-channel blocks, with the existing border and tail fallback;
- a single-loop Depthwise dispatch that resumes the x2 interior kernel after border pixels; x1/x2 counters are collected inside the real kernel only for profiled runs, while timed runs pass stats=NULL;
- plan-time Dense K-offset metadata and a prepared AVX2/FMA entry point that removes per-call offset construction and K division/modulo from the hot loop, retaining only patch and partial-tile scratch;
- AVX2 NHWC support ops for prepared BatchNorm affine, spatial ReduceMean, and AveragePool/MaxPool kernels;
- deterministic comparison and paired AB/BA median timing against the canonical executor at REC width 960.
- v7 direct-NHWC preparation hooks: a private graph-input accessor and prepared backbone runner can consume preprocessed NHWC data without an initial NCHW-to-NHWC conversion when the model graph starts with an NHWC-capable node; graphs that require an NCHW prologue retain the existing NCHW entry path.

The fast path remains opt-in and experimental. The planner now classifies both the PP-OCRv6 Tiny and Medium REC graphs without an NHWC blocker. Fusion is conservative: it requires single-consumer, shape-compatible semantic chains and falls back to the original physical nodes when a pattern is not provably safe. The benchmark report emits `legacy_ms`, `fast_ms`, `speedup`, Pointwise/Dense/Depthwise/binary/unary/ReduceMean/Pool/Concat/BatchNorm node counts, `physical_op_count`, `elided_tensor_count`, profiled Depthwise x1/x2 dispatch counters (without a timed-run pre-walk), per-kind profile nanoseconds, conversion bytes, `unsupported_nhwc_node_count`, `argmax_mismatch`, and `max_abs <= 1e-4` correctness gates. A negative speedup is not promoted to the default path.

The prepared-NHWC hooks are internal and are not part of the public C ABI. Callers must query the graph-input accessor and treat a null pointer as an explicit fallback to lw_x64_rec_fast_run; the Tiny REC graph currently has an NCHW prologue, so it intentionally exercises that fallback.

The prototype is only built for x86/x64 when the existing experimental option is enabled. ARM64, LoongArch, WebAssembly, DET, and release builds remain on the legacy path.

The CTest entries are `nhwc_support_pool` (valid-window and channels<8 Pool regression), `layout_plan_direct_input`, and `x64_rec_fast_prototype`. It checks the layout conversion round trip, runs two warm-ups followed by nine alternating legacy/fast samples, and compares the complete REC output against `lw_execute_session_f32` with `max_abs <= 1e-4`.

Current local reference measurements (static width model, Windows x64 AVX2) are informational only. The v6 Tiny run below is a fresh post-optimization sample; it is not a release gate:

- Tiny REC960 (v6 local sample): zero mismatches (max_abs=2.72e-05, argmax_mismatch=0), four planned/runtime conversions, 40 elided GELU temporaries, about 3.7 MiB NHWC workspace, and a reported speedup ratio of about 0.69x in the latest local sample; the final profiled run reports 8330 Depthwise x2 and 19340 x1 dispatches. The physical lifetime validator passes and no fusion chain is present in this graph.
- Medium REC960: the previous `0.71x` / 21.6 MiB figures are retained only as a pre-v6 reference; rerun the Medium artifact before drawing a v6 performance conclusion.

These results confirm that planner coverage, physical lifetimes, Dense preparation, and Depthwise dispatch are improving, but the implementation is not ready for default-path integration: the current end-to-end ratio remains below 1.0x. The next stages are to connect the existing BGR-to-NHWC preprocessor to the prepared hook, then elide the terminal CTC logits/Softmax materialization and measure recognizer-level hotspots before any promotion decision.

## v12: compiled NCHW candidate

The experimental x64 REC backend now has a private lw_x64_rec_backend_compile_ex strategy
selector. The existing API remains the NHWC-compatible wrapper; the new NCHW strategy keeps
rank-4 tensors in their canonical NCHW storage and lowers Tiny REC pointwise/stem/depthwise
convolutions directly to the existing packed AVX2/FMA and NCHW kernels. BatchNorm affine and
channel broadcasts use channel-major helpers, while ReduceMean and pooling retain scalar-correct
NCHW fallbacks for the first candidate. GELU fusion is retained with the same private-temporary
safety rule.

This candidate is correctness-gated but not enabled by the production recognizer. The CTest
targets x64_rec_backend_nchw and x64_rec_layout_benchmark compare it against the canonical
executor and the existing NHWC candidate. A local Tiny REC960 Debug measurement (five measured
runs after two warmups) was:

| path | median end-to-end ms | relative to canonical | arena bytes |
| --- | ---: | ---: | ---: |
| canonical | 123.774 | 1.000x | — |
| compiled NHWC | 245.492 | 0.504x | 99,806,656 |
| compiled NCHW candidate | 115.950 | 1.067x | 99,806,656 |

The table uses the rotated-order benchmark (`order_rotated: true`); absolute values are machine-specific. The important result is that the NCHW candidate is
currently ahead of canonical on this workload while NHWC remains a negative control. Further
work should profile the NCHW operator mix and memory lifetime before any default-path promotion;
the private strategy API and candidate test fixtures are not part of the public C ABI.

The private target x64-rec-pointwise-layout-driver also measures every real Tiny
Pointwise node without inserting layout conversions. It reports the existing NHWC
kernel, packed NCHW FMA4, and the existing packed NCHW FMA8/8x8 candidate. On the
same local machine, representative large nodes were approximately 5.99 ms NHWC,
1.61 ms NCHW FMA4, and 1.45 ms NCHW FMA8 for a 48-to-96, 12x240 node. All FMA8
cases are checked against the FMA4 output before timing is accepted. The whole
layout benchmark rotates canonical/NHWC/NCHW execution order on each measured
round and reports order_rotated: true.

## v13: Release operator-mix profiling

The benchmark driver now accepts a strategy argument
(`x64-rec-backend-benchmark-driver rec.lwm [iterations] [nhwc|nchw]`), maps NCHW
physical ops into the existing per-kind slots, and emits per-op and per-conv-shape
breakdowns (`per_op_ns`, `conv_shapes`) plus the strategy tag. A second CTest
entry `x64_rec_backend_benchmark_nchw` runs the NCHW leg. Runtime profiling stays
outside the A/B timing loop (the 94bea36 design); the `lw_x64_rec_profile` struct
remains unwritten.

A local Release x64 sample (30 iterations, two warmups, rotated order) gives:

| strategy | canonical ms | backend ms | speedup |
| --- | ---: | ---: | ---: |
| NHWC | 18.789 | 18.368 | 1.023x |
| NCHW | 18.904 | 18.940 | 0.998x |

The Release numbers do not reproduce the Debug 1.067x NCHW figure: both compiled
strategies are at parity with canonical in Release, with NHWC slightly ahead.
The NCHW per-kind profile (per-op loop, cache state differs from the A/B loop)
is: pointwise 8.74 ms, unary 2.44, ctc 2.14, dense 1.85, binary 1.65, depthwise
0.95, transpose 0.27, reduce 0.25, pool 0.19. The scalar NCHW fallbacks
(reduce/pool/transpose/1x5 depthwise) total about 0.71 ms per frame. No FMA8
pointwise shape family exists in the Tiny graph (largest projections are
160-to-320), so the FMA8 selector never fires on this model. These figures are
the pre-optimization baseline for the CTC-elision and arena-lifetime stages.

## v14: lifetime-based arena reuse

The compiler now lowers the graph twice. The first pass discovers value
lifetimes (`producer`/`last_use` in physical-op space, GELU fusions, reshape
aliases); an offline greedy-by-size planner then places every live value at
the lowest non-overlapping 64-byte-aligned offset; the second pass embeds the
final offsets. Values never produced by a physical op (GELU fusion
temporaries, the CTC-excluded tail tensors, dead tensors) get no arena slot at
all (phantom sinks). The graph input is kept exclusive for the whole run so
the caller may rewrite it between runs. The planner mirrors the reference
`PlanWorkspace` design.

Tiny REC960 arena dropped from 99,806,656 bytes to 3,317,760 bytes for both
strategies. A local Release x64 sample (30 repeats, rotated order, same
session as the v13 table) gives:

| strategy | canonical ms | backend ms | speedup | arena bytes |
| --- | ---: | ---: | ---: | ---: |
| NHWC | 18.964 | 16.259 | 1.166x | 3,317,760 |
| NCHW | 18.964 | 17.208 | 1.102x | 3,317,760 |

The v13 NHWC/NCHW figures were 1.023x / 0.998x at 99.8 MiB: removing the
monotonic-arena cache pressure bought roughly 0.10-0.14x end-to-end. The
layout benchmark now asserts `arena_bytes <= 10 MiB` as a regression guard;
all correctness gates (`x64_rec_backend_nchw` 1e-3/1e-5, `contract`,
`execution` 1e-6, both benchmark entries) pass with the reuse planner.

## v15: CTC logits elision

The terminal CTC projection no longer materializes the [rows x classes] logit
tensor. `lw_avx2_fma_packed_matmul_argmax_scores_f32` (mirroring the reference
`MatMulArgMaxPackedAvx`) keeps only the per-row running max and its column
index across 16-wide packed panels (vector strict-greater replace, ties to the
lower index, scalar tail on the final partial panel) and writes
`best_indices` plus a per-row `scores` buffer. `lw_avx2_ctc_row_probabilities_f32`
(mirroring `SoftmaxMaximumProbability`) recomputes one row of logits per
emitted step — a single packed matvec — and folds the softmax denominator into
the same pass with the same 8-lane batches and per-lane accumulation order as
the previous stored-logits kernel, so emitted probabilities stay bit-comparable
(the 1e-6 execution gate still passes). Blank and repeated rows keep their
zero-initialized probability contract. The instance buffer shrinks from
rows*classes floats (~1.1 MiB) to rows floats; the shared canonical kernel is
untouched (the backend uses the two new kernels).

Local Release x64 sample (30 repeats): the NCHW `backend_ctc_ms` dropped from
2.14 ms to 1.19 ms, and the end-to-end ratios improved to 1.165x (NCHW,
backend benchmark) / 1.267x (NHWC, layout benchmark) on the same machine.

## v16: recognizer-level NCHW integration and measurement

The production recognizer now prefers the NCHW strategy at width 960 (still
gated by `LW_EXPERIMENTAL_AVX2_FAST_PATH`): `lw_x64_rec_backend_compile_ex`
with `LW_X64_REC_COMPILE_NCHW` is tried first and falls back to the NHWC
wrapper for graphs the NCHW strategy cannot lower. The preprocess branch picks
the planar `lw_rec_preprocess_bgr_u8` or the `_nhwc` variant from
`program->backend_layout`. Runtime-failure behavior is unchanged (error
returned, no silent canonical fallback). `lw_recognizer_clone` still runs
canonical at width 960. A test-only hook
`lw_recognizer_test_disable_x64_backend` (declared in `rec_internal.h` under
the fast-path gate) lets the new `x64-rec-recognizer-benchmark-driver` measure
both legs of the full recognizer path (preprocess + run + CTC decode) in one
binary; it reports rotated-order medians and compares decoded text. The CTest
entry is `x64_rec_recognizer_benchmark`.

Local Release x64 measurements after v15 (same machine):

| leg | ratio | notes |
| --- | ---: | --- |
| recognizer benchmark (NCHW) | 1.157x | text_match=true, 20 repeats |
| layout benchmark (NCHW) | 1.171x | 30 repeats, rotated order |
| backend benchmark (NCHW) | 1.165x | 30 repeats |
| layout benchmark (NHWC) | 1.267x | negative-control leg |

`full_ocr_golden_corpus` passes with the NCHW backend on the production
recognizer path (exact text match against the manifest, recognizer width 960),
and all nine `x64_rec*` CTest entries pass. This satisfies the proposed
promotion gate (>= 1.05x Release recognizer-level and layout speedup, golden
corpus text-exact, all gates green, arena at 3.3 MiB well below the 99.8 MiB
baseline).

The promotion is applied: `LW_EXPERIMENTAL_AVX2_FAST_PATH` now defaults to ON
(option description updated to the compiled NCHW REC backend). The option is
only wired for x86/x64 non-Emscripten builds, so ARM64, LoongArch, and web
builds keep the canonical path; the recognizer falls back to canonical for
widths other than 960, for graphs the NCHW strategy cannot lower (NHWC
wrapper), and on runtime failure. The full build (193 targets) and the
eleven REC-related CTest entries pass with the new default.
