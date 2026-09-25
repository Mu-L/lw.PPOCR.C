#!/usr/bin/env python3
"""Report all three WASM OCR variants before enforcing text parity."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid benchmark JSON: {path}")
    return value


def print_compiled_profile(path: Path) -> None:
    """Show stage shares from a separate, instrumented two-run OCR process."""
    snapshots = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("X64REC width="):
            continue
        fields = dict((key, float(value)) for key, value in
                      re.findall(r"([a-z]+)=([0-9]+(?:\.[0-9]+)?)", line))
        if "total" in fields and "ctc" in fields:
            snapshots.append(fields)
    print()
    print("### Compiled SIMD128 REC hotspot diagnostic")
    print()
    if not snapshots:
        print("No per-line REC timings were captured; check the profile log artifact.")
        return
    print(f"{len(snapshots)} REC invocations across one warm-up and one measured "
          "full OCR. These instrumented times are not part of the A/B latency above.")
    print()
    print("| Stage | Accumulated ms | Share of tracked REC time |")
    print("| --- | ---: | ---: |")
    totals = {key: sum(row.get(key, 0.0) for row in snapshots)
              for key in ("total", "pw", "dense", "dw", "bin", "unary",
                          "reduce", "pool", "transpose", "matmul", "ctc")}
    tracked = totals["total"] + totals["ctc"]
    labels = {"pw": "Pointwise", "dense": "Dense", "dw": "Depthwise",
              "bin": "Binary", "unary": "Unary", "reduce": "Reduce",
              "pool": "Pool", "transpose": "Transpose", "matmul": "MatMul",
              "ctc": "CTC"}
    stages = [(label, totals[key]) for key, label in labels.items()]
    other = max(0.0, totals["total"] - sum(totals[key] for key in labels if key != "ctc"))
    stages.append(("Other backbone", other))
    for label, milliseconds in sorted(stages, key=lambda item: item[1], reverse=True):
        print(f"| {label} | {milliseconds:.3f} | "
              f"{milliseconds / tracked * 100.0 if tracked else 0.0:.1f}% |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--scalar", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--profile-log", type=Path)
    parser.add_argument("--expected-text-sha256", required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    runs = {
        "Canonical": load(args.canonical),
        "Compiled scalar": load(args.scalar),
        "Compiled SIMD128": load(args.compiled),
    }
    canonical, scalar, simd = runs.values()
    issues = []
    for name, run in runs.items():
        if run.get("use_cls") is not True:
            issues.append(f"{name}: benchmark must enable CLS for the Tiny golden")
        lines = run.get("text_lines")
        if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
            issues.append(f"{name}: missing OCR text lines")
        else:
            digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
            if digest != run.get("text_sha256") or len(lines) != run.get("line_count"):
                issues.append(f"{name}: text lines do not match the reported count/checksum")
        for field in ("schema_version", "sample_sha256", "line_count"):
            if run.get(field) != canonical.get(field):
                issues.append(f"{name}: {field} differs from canonical")
        if run.get("text_sha256") != canonical.get("text_sha256"):
            issues.append(f"{name}: OCR text differs from canonical")
    if canonical.get("text_sha256") != args.expected_text_sha256:
        issues.append("Canonical: OCR text differs from the checked-in golden checksum")

    def mib(value: int) -> float:
        return value / 1048576

    print("## WASM full OCR: canonical / compiled scalar / compiled SIMD128")
    print()
    print(f"Sample SHA-256: `{canonical['sample_sha256']}`. "
          "CLS enabled; same runner, models, PPM input, warm-up and measured iterations.")
    print()
    print("| Metric | Canonical | Compiled scalar | Compiled SIMD128 |")
    print("| --- | ---: | ---: | ---: |")
    print("| Median OCR latency (ms) | " + " | ".join(
        f"{run['median_ms']:.3f}" for run in runs.values()) + " |")
    print("| Peak process RSS (MiB) | " + " | ".join(
        f"{mib(run['peak_rss_bytes']):.1f}" for run in runs.values()) + " |")
    print("| WASM heap after runs (MiB) | " + " | ".join(
        f"{mib(run['wasm_heap_bytes']):.1f}" for run in runs.values()) + " |")
    print("| OCR text SHA-256 | " + " | ".join(
        f"`{run['text_sha256']}`" for run in runs.values()) + " |")
    print()
    print("Speedup: canonical/compiled scalar "
          f"{canonical['median_ms'] / scalar['median_ms']:.3f}x; "
          f"compiled scalar/SIMD128 {scalar['median_ms'] / simd['median_ms']:.3f}x; "
          f"canonical/SIMD128 {canonical['median_ms'] / simd['median_ms']:.3f}x.")
    print("RSS delta vs canonical: compiled scalar "
          f"{mib(scalar['peak_rss_bytes'] - canonical['peak_rss_bytes']):+.1f} MiB; "
          f"SIMD128 {mib(simd['peak_rss_bytes'] - canonical['peak_rss_bytes']):+.1f} MiB.")
    print()
    if issues:
        print("**Text contract: FAILED.** Timing and memory are diagnostic only until resolved.")
        for issue in issues:
            print(f"- {issue}")
        print()
        for name, run in runs.items():
            print(f"<details><summary>{name} OCR lines</summary>")
            print()
            for index, line in enumerate(run.get("text_lines", [])):
                print(f"{index:02d}: {line.replace('|', '&#124;')}")
            print()
            print("</details>")
    else:
        print("**Text contract: PASS.** All three outputs match the golden checksum.")
    print("Latency and memory are informational, not hosted-runner pass/fail gates.")
    if args.profile_log is not None:
        print_compiled_profile(args.profile_log)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
