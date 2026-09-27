# lw.PPOCR.C v1.1.0

Native x64 performance and memory release / 原生 x64 性能与内存优化版。

## Highlights / 主要变化

- Compiled physical REC/DET execution, NHWC kernels, packed weights and fused
  epilogues reduce repeated dispatch, layout conversion and intermediate work
  on eligible AVX2 x64 hosts. Unsupported hosts retain their fallback paths.
- Compiled REC programs share immutable packed constants across adaptive widths
  and line workers while keeping mutable arenas instance-local.
- Persistent line workers and parallel crop processing reduce repeated request
  overhead. The maximum REC width remains 960 with adaptive per-line widths.
- Fix streaming-crop experimental builds under MSVC `/WX`; this does not
  enable streaming crops in release defaults.

## Performance evidence / 性能口径

The README snapshot is a local engineering measurement, not a measurement of
the tagged archive: Ryzen 7 7735H, Windows x64 Release/AVX2, bundled 500×500
sample, maximum REC width 960, one warm-up, three measured calls and three
fresh-process rounds. It includes resident widths and experimental dispatch
options documented in [the baseline](performance-baseline.md#compiled-rec-packed-constant-sharing-snapshot-2026-09-25).

| Model | 1 worker | 4 workers | Peak WS, 1 worker | Peak WS, 4 workers |
|---|---:|---:|---:|---:|
| Tiny | 112.68 ms | 54.09 ms | 84.3 MiB | 100.6 MiB |
| Small | 375.91 ms | 217.74 ms | 198.5 MiB | 233.8 MiB |
| Medium | 1,217.17 ms | 979.30 ms | 744.8 MiB | 763.2 MiB |

Constant-sharing A/B lowered local peak WS by about 16–18 MiB for Tiny,
76–78 MiB for Small and 334–337 MiB for Medium, with paired latency within
±2%. These comparisons isolate constant sharing, not the total speedup over
v1.0.0. Peak WS includes initialization and a standalone detector handle.

Run the [same-runner release comparison](release-sample-ocr-comparison.md)
against `v1.0.0` for the exact candidate's end-to-end speedup. Do not pool
local and hosted-runner timings or imply that all six cases use less memory
than v1.0.0. Model-specific text differences are reported explicitly;
checksum changes alone are not proof of improved accuracy.

## Compatibility and scope / 兼容性与支持范围

C ABI v1 and WASM Host ABI v1 remain frozen; this release does not change their
symbols, layouts or ownership rules. Tiny remains the sole stable model on
Windows x64, Linux x64 and modern browser WASM. Small and Medium remain Preview,
as do Android, Desktop Java/JNI and the optional architecture packages.
LWM v0.1 remains an internal Preview format. Model minimum-runtime metadata
remains an independent compatibility floor, not the package version.

WASM keeps the canonical stable default. `LW_WASM_COMPILED_REC=OFF` remains
the global default; experimental compiled SIMD128 Pointwise now defaults to
2×16. Its performance work targets Tiny and Small, not Medium mobile usage.
No pthread, relaxed-SIMD or fast-math requirement is introduced.

## Upgrade and verification / 升级与校验

Replace binaries, SDKs and model packs as a matching release set; verify the
SHA-256 files before use. High-level ABI v1 applications retain their ABI
contract, while low-level model/session APIs remain experimental. Rebuilding
consumers and testing the exact extracted package is recommended.
Android demo versionCode advances to 3; its SDK/demo support remains Preview.

The release workflow rebuilds and verifies all 18 primary assets before
publication, and produces build-provenance attestations. Create a new
annotated `v1.1.0` tag; tag signing is optional. Never move the published
`v1.0.0` tag. Exact-tag CI and artifact validation remain mandatory.

