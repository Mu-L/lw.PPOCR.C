# Local standalone HTML A/B

2026-09-30: Emscripten 4.0.15, Chromium 151, bundled 500x500 JPEG,
CLS enabled, adaptive REC capped at 960. The same browser runs each build
in a fresh context; revision order alternates. Three warm-ups precede five
measured calls. Image decode and engine initialization are excluded; timings
include the full `recognize()` call and Demo result materialization/UI update.

| Model | Canonical ms | Compiled ms | Paired speedup | Canonical heap MiB | Compiled heap MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny | 962.10 | 610.90 | 1.575x | 70.50 | 59.25 |
| Small | 4395.30 | 2663.50 | 1.684x | 134.44 | 132.38 |

Tiny used two alternating rounds and Small three. Latencies above are medians
of round medians; speedup is the median of paired ratios. Every call matched
the variant's checked-in Web golden. Linear heap capacity is **not** browser
process RSS. These local numbers are not comparable to hosted-runner Node PPM
measurements, nor evidence of mobile browser performance.

The improvement comes from enabling the existing compiled SIMD128 / 2x16 /
lazy-fallback backend, not a newly improved kernel. Global
`LW_WASM_COMPILED_REC` remains OFF. A forced two-IC Pointwise unroll regressed
Tiny full OCR (0.877x) and was discarded. LTO was inconclusive (1.021x) and
is not part of the recommended configuration.

## Reproduce

Activate Emscripten and a Python environment containing converter dependencies,
Playwright and Ninja. On this Windows setup:

```powershell
. ..\emsdk\emsdk_env.ps1
. .\build\wasm-tools\Scripts\Activate.ps1
```

Configure canonical and compiled builds explicitly:

```powershell
emcmake cmake -S . -B build/wasm-canonical -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DLW_BUILD_HTTP_DEMO=OFF `
  -DLW_WASM_COMPILED_REC=OFF -DLW_BUILD_WEB_MODEL_VARIANTS=ON
emcmake cmake -S . -B build/wasm-fast -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DLW_BUILD_HTTP_DEMO=OFF `
  -DLW_WASM_COMPILED_REC=ON -DLW_WASM_REC_LAZY_FALLBACK=ON `
  -DLW_WASM_POINTWISE_ROWS=2 -DLW_BUILD_WEB_MODEL_VARIANTS=ON
cmake --build build/wasm-canonical --target lw-ocr-html lw-ocr-html-small --parallel 8
cmake --build build/wasm-fast --target lw-ocr-html lw-ocr-html-small --parallel 8
python web/benchmark_ocr_html.py `
  --baseline build/wasm-canonical/ocr-demo.html `
  --candidate build/wasm-fast/ocr-demo.html `
  --sample models/ppocrv6-tiny/sample.jpg --golden ci/web-ppocrv6-tiny.json `
  --output build/wasm-fast/tiny-browser-ab.json
```

For Small use `ocr-demo-small.html` in both directories and
`ci/web-ppocrv6-small.json`. Default benchmark settings are three paired rounds,
three warm-ups and nine measured calls. Golden text/line count and repeatability
are gates; latency is informational. JSON retains every measured time and heap.

HTML now depends on the generated SDK file as well as its build target,
including model variants. Incremental runtime changes therefore update the
embedded WASM instead of silently leaving a stale HTML. Successful compiled
status lines emitted on stderr are matched strictly by browser regression
tests; unknown console errors and JavaScript page errors still fail tests.
