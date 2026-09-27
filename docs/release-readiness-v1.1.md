# v1.1 release readiness

This is a release-preparation checklist, not a claim that v1.1.0 is already
published. The [v1.0 checklist](release-readiness-v1.0.md) remains historical.

## Scope

Preserve `ci/stable-release-scope.json`: Tiny-only stable model support,
frozen C ABI v1 / WASM Host ABI v1, and internal-preview LWM v0.1. Small,
Medium, Android and Java/JNI remain Preview. Do not promote them merely
because x64 performance improved. WASM compiled REC remains globally OFF.

## Evidence at preparation

Reviewed source baseline: `a8cf29b150955de86abc1e056545eaf817a03edf`.
Public GitHub status at review: native model-analysis, Tiny/Small/Medium
validation, Java/JNI, Android, browser WASM, x64 backend correctness and
sanitizers passed. The scheduled runtime-memory benchmark failed during
streaming build compilation. The same option combination reproduces an
unused local `status` declaration in `src/ppocr/ocr.c`; this preparation
fixes its preprocessor condition. Remote confirmation on the new commit is
still required.

Local preparation checks passed: 29 version/asset/model-pack unit tests,
seven focused native tests (C ABI, prefix compatibility, full OCR reference
and golden corpus, REC/DET backend contracts), the repaired MSVC streaming
benchmark build, release YAML parsing and whitespace checks. Frozen C ABI
and WASM Host ABI manifests are byte-identical to v1.0.0. These checks are
not a substitute for the full remote release matrix or exact archive smoke.

Performance evidence in README is explicitly a local source-build snapshot
with experiment options, not an exact v1.1.0 package benchmark. Keep text
differences visible and do not change goldens just to make a release green.

## Before creating the tag

- [ ] Push the preparation commit and require all release-related CI jobs to
  pass on that exact commit; rerun runtime-memory-benchmark to confirm the fix.
- [ ] Run `python tools/check_release_readiness.py --mode stable --version 1.1.0`
  and the versioning/release-asset tests. This validates metadata only.
- [ ] Run the manual x64 sample release comparison against `v1.0.0` on the
  exact candidate (Tiny/Small/Medium, 1/4 workers, maximum REC width 960).
  Review latency, peak WS, all text differences and raw reports.
- [ ] Keep `abi/c-abi-v1-candidate.json`, `abi/web-abi-v1-candidate.json`,
  public layouts and model/dictionary inputs unchanged.
- [ ] Confirm model packs declare revision 1.1.0 without silently upgrading
  Small/Medium analysis-only status or the 1.0.0 minimum-runtime floor.
- [ ] Create a new annotated tag only after the above checks. Signing is
  optional. Never overwrite an existing tag.

## Exact-tag release gate

The Release workflow must run native, browser/Node, Android, Java/JNI and
three model-pack build/validation jobs from the tag, then verify all 18
assets and checksums and generate attestations before publishing.
This preparation does not bypass those jobs. No local WASM or Android
toolchain installation is required.

After CI, test the exact downloaded Windows archive and HTML (including
clear/reselect image/PDF preview), and the Android APK where available.
Record any platform/device limitations rather than claiming universal coverage.
