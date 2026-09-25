# Experimental WASM compiled REC

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
six-pixel tile, K blocking, partial sums, and scalar epilogue semantics.
Depthwise uses four-channel SIMD128 groups with the same OC32 packed weights
and tap order, including border and row-shard cases. Neither kernel changes
the public C ABI. The pack format remains OC16. Only the NHWC compiled layout
is supported on WASM; NCHW-only x64 kernels are not linked into its executor.
The compiled pointwise GELU epilogue and standalone Erf/GELU physical ops now
reuse the canonical SIMD128 Erf polynomial instead of per-element `erff`.
This is an experimental latency candidate, not a claimed speedup; the
three-way full-OCR CI comparison must establish its end-to-end effect while
the checked-in text checksum remains unchanged.
The CLS backbone attempts the
same compiler; unsupported CLS graphs fall back to the
normal session. DET remains on the existing WASM executor.

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
The CI log must also confirm `widths=5/5` and `ctc=simd128`, so matching
text cannot pass by silently using only the canonical executor.

This does **not** yet establish 100% compiled coverage, a REC-only benchmark,
or a performance win. Keep `LW_WASM_COMPILED_REC` off for releases until CI confirms the
five-width text contract, no canonical fallbacks in the intended model, and
end-to-end latency/memory improvements. Further WASM-specific CLS and DET
tuning are separate follow-up work.
