# v1.0 release readiness

This document is the release gate for the first stable `lw.PPOCR.C` line. A
green CI run is necessary, but it is not sufficient for a 1.0 claim: the
public compatibility contracts and the model-support claims must agree with
the artifacts that are published.

## Current position

The repository is preparing the `v1.0.0` stable line. The Release workflow
builds and verifies the native, WASM, Android, Java/JNI, and Tiny/Small/Medium
model-pack artifacts, including SHA-256 records and GitHub build-provenance
attestations. The approved scope below intentionally keeps several optional
artifacts in Preview status.

The approved stable scope is machine-readable in
[`ci/stable-release-scope.json`](../ci/stable-release-scope.json). It currently
lists Tiny, C, and browser WASM as the only stable targets; its status is
`approved` for the current Tiny-only support decision. Small, Medium,
Android, Java/JNI, ARM64, and LoongArch64 remain explicitly Preview in that
scope. The approved LWM policy is `internal-preview`: Tiny model packs may
be shipped as implementation assets, but the LWM file format itself is not a
stable interchange contract until the policy is changed to `frozen`.

| Area | Current status | 1.0 requirement |
|---|---|---|
| C ABI | `frozen` | Frozen symbol allowlist, structure layouts, prefix rules, error/status semantics, and supported platform/compiler matrix. |
| LWM | Internal Preview | The approved 1.0 scope explicitly defers LWM interchange stability; model packs are implementation assets only. |
| PP-OCRv6 Tiny | `primary` runtime model | Re-run the full golden corpus and publish the exact model/dictionary checksums with the stable release. |
| PP-OCRv6 Small | `analysis-only` | Either promote through a production converter and independent golden evidence, or keep it preview-only and exclude it from the 1.0 support promise. |
| PP-OCRv6 Medium | `analysis-only` | Same decision as Small; current validation is explicitly analysis-only. |
| Windows/Linux/browser/Java | CI verified on stated baselines | Keep the claims limited to those baselines and preserve consumer/package smoke tests. |
| Linux ARM64/LoongArch64 | Experimental/manual validation | Decide whether these are supported 1.0 targets. If yes, require native customer hardware evidence; QEMU alone is not enough. |
| Android | ARM64 preview integration | Decide whether the AAR/API and signed demo APK are stable 1.0 interfaces or remain preview artifacts. |

## Required gates before `v1.0.0`

1. Commit the C ABI compatibility implementation and pass the complete CTest
   suite on every supported native job. The current Windows Release run is
   52/52. The prefix-compatibility test must
   cover short and extended high-level options, output metadata, and result
   structures; low-level experimental model/session options remain exact-size
   until a later ABI revision.
2. Preserve `abi/c-abi-v1-candidate.json` and
   `abi/web-abi-v1-candidate.json` as the frozen manifests (their historical
   filenames are retained for path compatibility). Do not change any listed
   symbol, enum value, structure size, field offset, or documented error
   meaning after the tag.
3. Resolve the LWM policy recorded in the stable scope contract. A stable
   release may either freeze the documented LWM v0.1 reader/writer subset
   (including rejection rules and checksum semantics), or use
   `internal-preview` and avoid presenting model-pack files as a stable
   interchange format. The gate enforces the selected policy.
4. Make the model catalog and release notes agree. Small and Medium must not be
   described as stable while their converters and validation tools report
   `analysis-only`. The runtime packager now requires an explicit
   `--runtime-status analysis-only` for those variants, so a future stable tag
   cannot upgrade their status implicitly.
   The stable scope contract is `approved` for the current Tiny-only decision.
5. Run the release workflows from the exact tag and verify the strict 18-asset
   manifest, every sidecar checksum, the Android checksum file, SBOMs, and
   artifact attestations. Do not reuse or move an existing public tag.
   The local preview gate is `python tools/check_release_readiness.py --mode
   preview`; after the final version and support decisions, run it again with
   `--mode stable --version 1.0.0` and require a zero exit status.
6. Run the installed-package smoke tests and at least one external consumer
   build for each stable binding (CMake/C, Java/JNI, browser SDK, and Android
   if promoted). Confirm that the documented examples use the final filenames.
7. Create a signed annotated tag and record the release commit, model checksums,
   ABI manifest checksum, LWM policy, and supported platform baselines in the
   release notes. The Release workflow now rejects a stable tag unless GitHub
   reports its annotated tag signature as verified. Unsigned tags are
   acceptable for previews but not the target 1.0 supply-chain claim.

## Recommended release scope

The lowest-risk 1.0 scope is:

- frozen C ABI v1;
- Tiny as the sole stable PP-OCRv6 runtime model;
- Windows x64, Linux x86_64, and modern browser WASM as stable targets;
- Java/JNI and Android explicitly labeled stable only if their public APIs and
  packaging contracts are frozen in the same release;
- Small, Medium, ARM64, and LoongArch64 remaining opt-in/preview until their
  converter, hardware, and accuracy evidence is promoted.

This scope does not prevent publishing richer Preview artifacts. It prevents a
preview-only model converter or an emulated architecture build from silently
becoming part of the permanent 1.0 compatibility promise.

## Preview-to-1.0 sequence

1. Commit the ABI prefix-compatibility hardening and frozen manifest changes.
2. Keep the approved LWM and Small/Medium support scope explicit in release
   notes and package manifests.
3. Freeze the final version metadata and documentation, run the tagged
   seven-job Release workflow, inspect all 18 primary assets, and then publish
   `v1.0.0`.
