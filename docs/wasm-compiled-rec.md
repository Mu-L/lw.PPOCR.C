# Experimental WASM compiled REC

The browser and Node/WASM distributions still use the established executor by
default. `LW_WASM_COMPILED_REC=ON` is an opt-in experiment for the SIMD128 build;
it does not change the legacy scalar HTML. This first stage reuses the x64
physical REC compiler, its five adaptive width slots (192/320/480/640/960),
arena lifetime plan, fused epilogues, and CTC output elision. The compiled
executor selects a 4-pixel x 16-output-channel SIMD128 pointwise kernel and
portable NHWC dense/depthwise kernels. The pack format remains OC16. The CLS
backbone attempts the same compiler; unsupported CLS graphs fall back to the
normal session. DET remains on the existing WASM executor.

There is no pthread requirement, fast-math flag, or relaxed-SIMD dependency.
The option requires `LW_WASM_SIMD128=ON`.

To validate on GitHub Actions, manually run `browser-wasm-sdk-and-html` with
`compiled_rec=true`. The job builds the experimental HTML/SDK/Node package,
runs the normal browser golden tests, then builds a canonical Node/WASM package
on the same runner. Five warmed full-OCR iterations per build use the same Tiny
models and 500x500 PPM. Both runs must match the exact text checksum in
`ci/web-ppocrv6-tiny.json`; timing and memory remain informational. The job
summary and `wasm-compiled-rec-comparison-*` artifact contain the paired
measurements.

This does **not** yet establish 100% compiled coverage, a REC-only benchmark,
or a performance win. Keep the option off for releases until CI confirms the
five-width text contract, no canonical fallbacks in the intended model, and
end-to-end latency/memory improvements. WASM-specific dense, depthwise, CTC,
CLS, and DET tuning are separate follow-up work.
