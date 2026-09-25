# Experimental WASM compiled OCR

The browser and Node/WASM distributions still use the established executor by
default. `LW_WASM_COMPILED_REC=ON` is an opt-in experiment for the SIMD128 build;
it does not change the legacy scalar HTML. This first stage reuses the x64
physical REC compiler, its five adaptive width slots (192/320/480/640/960),
arena lifetime plan, fused epilogues, and CTC output elision. The compiled
executor selects a 4-pixel x 16-output-channel SIMD128 pointwise kernel and
panel-outer SIMD128 CTC projection with emitted-row probability recomputation.
Dense and depthwise now have SIMD128 physical kernels, selected by the
`LW_WASM_REC_SIMD_KERNELS` option (default `ON` when compiled REC is enabled).
Turning the option `OFF` preserves the portable physical kernels for an
equal-plan A/B without changing the model pack, lifetime plan, or CTC path.
Dense uses four SIMD128 vectors per OC16 block and retains the existing
six-pixel tile, K blocking, and partial sums. Full OC16 blocks now apply
bias/residual/activation in SIMD128; tails keep the prior scalar path.
Depthwise uses four-channel SIMD128 groups with the same OC32 packed weights
and tap order, including border and row-shard cases. Neither kernel changes
the public C ABI. The pack format remains OC16. Only the NHWC compiled layout
is supported on WASM; NCHW-only x64 kernels are not linked into its executor.
The compiled pointwise GELU epilogue and standalone Erf/GELU physical ops now
reuse the canonical SIMD128 Erf polynomial instead of per-element `erff`.
This is an experimental latency candidate, not a claimed speedup; the
three-way full-OCR CI comparison must establish its end-to-end effect while
the checked-in text checksum remains unchanged.
The CLS backbone attempts the same compiler; its fixed-point preprocessor
writes directly into the compiled NHWC input when available. Unsupported CLS
graphs still fall back to the normal session. In an experimental compiled REC
build, `LW_WASM_COMPILED_DET=ON` (default) additionally attempts NHWC compiled
DET when fixed-point DET preprocessing is enabled. DET Pointwise and
ConvTranspose use SIMD128 kernels; the fixed-point preprocessor writes NHWC
directly to the backend input. If DET compilation fails, the canonical DET
executor is used. The `LW_WASM_COMPILED_DET` option can be turned off for a
REC/CLS-only comparison. The normal browser/Node release build is unaffected.

There is no pthread requirement, fast-math flag, or relaxed-SIMD dependency.
The option requires `LW_WASM_SIMD128=ON`.

Normal push and pull-request CI keep the release HTML/SDK canonical, then build
experimental Node/WASM packages with SIMD kernels on and off. They compare
canonical vs compiled and compiled scalar vs compiled SIMD128 on the same
runner. To validate the experimental browser HTML and SDK too,
manually run `browser-wasm-sdk-and-html` with `compiled_rec=true`. Five warmed
full-OCR iterations per build use the same Tiny
models and 500x500 PPM with CLS enabled, matching the Web/Node Tiny golden
configuration. All three runs must match the exact text checksum in
`ci/web-ppocrv6-tiny.json`; timing and memory remain informational. The job
summary and `wasm-compiled-rec-comparison-*` artifact contain the three-way
measurements, including process RSS and WASM heap. The benchmark records all
three OCR texts before the reporter checks parity and the checked-in golden
checksum. A mismatch still fails CI, but the measurements and differing text
are preserved in the job summary instead of being lost at the first run.
After the uninstrumented A/B, CI also runs compiled SIMD128 in a separate
profile process and summarizes accumulated REC stage times (pointwise, dense,
depthwise, CTC, etc.). Those instrumented timings identify hotspots but must
not be compared to the full-OCR A/B latency. Only the experimental Node package
accepts `LW_X64_REC_PROFILE`, `LW_WASM_OCR_PROFILE`, and
`LW_REC_MEMORY_PROFILE` from its host environment; browser and canonical Node
packages do not change their environment policy.
The CI log must also confirm `widths=5/5`, `ctc=simd128`, and a compiled NHWC
DET marker for the sample's 512x512 resized input. This prevents a successful
32x32 initialization compile from hiding a fallback on the measured image.
The compiled-scalar control keeps existing SIMD128 Pointwise/CTC (and DET
ConvTranspose) kernels; it only disables the optional SIMD128 Dense,
Depthwise, and common-op dispatch. It is not a wholly scalar runtime.
REC preprocessing already writes into its active compiled backend input; DET
and CLS fixed-point preprocessing now also write directly into NHWC backend
input when those compiled programs are active. The optional SIMD128 DB bitmap
path preserves the strict `probability > threshold` rule and rejects NaN/Inf.

Push/PR CI also stages the Small and Medium runtime model packs and runs the
same Node/WASM canonical, compiled portable, and compiled SIMD128 full-OCR
comparison for each. Tiny uses five measured runs, Small three, and Medium
two, each following one warm-up. The report gates exact text against the
variant's checked-in checksum and line count, equal DET box counts, compiled backend markers,
zero REC line fallbacks, and SIMD128 WASM heap no more than 5 MiB above
canonical. Latency and process RSS remain informational on hosted runners.
The separate instrumented run prints DET/CLS/REC component times and DET/REC
physical-op breakdowns; those times are not mixed into the uninstrumented A/B.
The `*-profile-summary.json` artifacts retain those component times alongside
actual compiled DET/CLS arena and packed-constant bytes, plus each resident REC
width's owned/borrowed constants and arena/scratch capacity. Borrowed REC
constants are references to shared storage, not additional allocations.

This does **not** yet establish a performance win or complete coverage across
Tiny, Small, and Medium until that CI runs successfully. Keep
`LW_WASM_COMPILED_REC` off for releases until CI confirms exact text, backend
coverage, and end-to-end latency/memory improvement on all intended model
packs. The local native tests do not substitute for an Emscripten build.
