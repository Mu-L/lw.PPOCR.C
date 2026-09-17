# lw.PPOCR.C v1.0.0

`v1.0.0` is the first stable release of the lightweight pure-C PP-OCR
runtime.

## Stable scope

- C ABI v1 is frozen, including high-level REC, CLS, DET, and full-OCR APIs,
  structure-prefix rules, ownership, UTF-8 text, status codes, and error
  semantics.
- WASM Host ABI v1 is frozen for the standalone browser SDK and the raw
  Node.js/WASM runtime.
- PP-OCRv6 Tiny is the stable runtime model and is the default in native,
  browser, and Node/WASM packages.
- Stable platform baselines are Windows x86-64, Linux x86-64, and modern
  browser WASM. The release workflow runs the complete CTest and package
  smoke-test matrix before publication.

## Preview attachments

The Release also carries separately validated artifacts for evaluation:
Small and Medium model packs/browser files, Android ARM64, Desktop Java/JNI,
Linux ARM64, and LoongArch64. These remain Preview and are not part of the
permanent 1.0 support promise. Their model converters, device coverage, and
performance claims must be evaluated independently.

## Format and supply chain

LWM v0.1 remains an internal Preview model format, not a stable interchange
format. Every primary download has a SHA-256 record; the publish workflow also
creates GitHub build-provenance attestations. The `v1.0.0` tag must be an
annotated tag whose signature is reported as verified by GitHub.

The stable Tiny model inputs and generated LWM assets are identified by these
SHA-256 values:

| Asset | SHA-256 |
|---|---|
| Tiny DET ONNX | `193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8` |
| Tiny CLS ONNX | `dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2` |
| Tiny REC ONNX | `9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6` |
| Tiny dictionary | `46e1b34ef45684cb46d75ac76d355341fe7f0a2c38d6ee02e63ae6b3878019fc` |
| Tiny DET LWM | `ba9164d371ac7003f90710c3106a344aeb906df2b0f1e7617fcf4608fa8cd66c` |
| Tiny CLS LWM | `d426c23f4758c9f21c5cbd0550ea9b90f1e7c0dd0289e1541e9754dee2e6ed61` |
| Tiny REC LWM | `59440146ae64068b70441f9c16b5878ccba21e75b41cb27220c8d9cc2d61e0ae` |

The frozen manifest checksums are: C ABI
`ce76163d17d445c3e8831c01fea6d4a733d38868741c144d1f9b93fb0ce0f209`, WASM
Host ABI `9d6c674fa912038c9796346c5db89be76ad6e554c62ecb48c1b301e892f6d977`,
and runtime contract `d391e5e2f73400916d15d3343d79b1e381515ba16ca634aa90f1daf3a3589f90`.

## Upgrade guidance

Do not mix binaries, SDK files, model packs, or dictionaries from different
releases. Applications should use the frozen high-level C ABI or WASM Host ABI;
the low-level model/session planning APIs remain experimental. Rebuild native
and managed consumers when changing release assets.
