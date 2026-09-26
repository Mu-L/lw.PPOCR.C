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


def write_profile_json(log_path: Path, output_path: Path, variant: str) -> None:
    """Keep the measured component breakdown as machine-readable CI evidence."""
    snapshots: dict[str, list[dict[str, float]]] = {
        "LW_WASM_OCR_PROFILE ": [],
        "LW_WASM_DET_PROFILE ": [],
        "X64REC width=": [],
    }
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        for prefix, rows in snapshots.items():
            if line.startswith(prefix):
                rows.append({key: float(value) for key, value in
                             re.findall(r"([a-z_]+)=([0-9]+(?:\.[0-9]+)?)", line)})
                break
    if any(not rows for rows in snapshots.values()):
        raise ValueError("instrumented WASM run did not emit full OCR, DET, and REC profiles")
    for run in snapshots["LW_WASM_OCR_PROFILE "]:
        if (run.get("det_compiled") != 1 or run.get("det_fallback") != 0 or
                run.get("cls_compiled", 0) < 1 or run.get("cls_fallback") != 0 or
                run.get("rec_compiled", 0) < 1 or run.get("rec_fallback") != 0):
            raise ValueError("instrumented WASM run used a canonical OCR graph fallback")
    ocr = snapshots["LW_WASM_OCR_PROFILE "][-1]
    det = snapshots["LW_WASM_DET_PROFILE "][-1]
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    def memory_fields(prefix: str) -> list[dict[str, int]]:
        return [{key: int(value) for key, value in
                 re.findall(r"([a-z_]+)=([0-9]+)", line)}
                for line in lines if line.startswith(prefix)]

    cls_memory = memory_fields("LW_WASM_COMPILED_CLS ")
    det_memory = memory_fields("LW_WASM_COMPILED_DET input=512x512 ")
    rec_widths = memory_fields("REC_MEMORY width=")
    rec_shared = memory_fields("REC_MEMORY shared_arena=")
    if not cls_memory or not det_memory or not rec_widths or not rec_shared:
        raise ValueError("instrumented WASM run did not emit compiled memory metrics")
    report = {
        "schema_version": 1,
        "model_variant": variant,
        "full_ocr": {f"{name}_ms": ocr[name] for name in
                     ("total", "det_preprocess", "det_graph", "det_postprocess",
                      "crop", "cls", "rec")},
        "rec_compiled_lines": int(ocr["rec_compiled"]),
        "rec_fallback_lines": int(ocr["rec_fallback"]),
        "det_compiled_runs": int(ocr["det_compiled"]),
        "det_fallback_runs": int(ocr["det_fallback"]),
        "cls_compiled_runs": int(ocr["cls_compiled"]),
        "cls_fallback_runs": int(ocr["cls_fallback"]),
        "det_physical": {f"{name}_ms": det[name] for name in
                         ("total", "pointwise", "dense", "depthwise", "convtranspose",
                          "binary", "pool", "concat", "resize", "other")},
        "rec_invocations": snapshots["X64REC width="],
        "compiled_memory": {
            "cls_arena_bytes": cls_memory[-1]["arena_bytes"],
            "cls_packed_bytes": cls_memory[-1]["packed_bytes"],
            "det_arena_bytes": det_memory[-1]["arena_bytes"],
            "det_packed_bytes": det_memory[-1]["packed_bytes"],
            "rec_widths": rec_widths,
            "rec_shared": rec_shared[-1],
            "rec_unique_owned_constant_bytes": sum(row["owned"] for row in rec_widths),
        },
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


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
    by_width: dict[int, list[dict[str, float]]] = {}
    for row in snapshots:
        by_width.setdefault(int(row["width"]), []).append(row)
    print()
    print("| REC width | Invocations | Pointwise ms | CTC ms | Tracked ms | Pointwise share |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for width, rows in sorted(by_width.items()):
        pointwise = sum(row.get("pw", 0.0) for row in rows)
        ctc = sum(row.get("ctc", 0.0) for row in rows)
        tracked_width = sum(row["total"] + row["ctc"] for row in rows)
        print(f"| {width} | {len(rows)} | {pointwise:.3f} | {ctc:.3f} | "
              f"{tracked_width:.3f} | "
              f"{pointwise / tracked_width * 100.0 if tracked_width else 0.0:.1f}% |")


def print_full_ocr_profile(path: Path) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    def latest(prefix: str) -> dict[str, float] | None:
        candidates = [line for line in lines if line.startswith(prefix)]
        if not candidates:
            return None
        return {key: float(value) for key, value in
                re.findall(r"([a-z_]+)=([0-9]+(?:\.[0-9]+)?)", candidates[-1])}

    ocr = latest("LW_WASM_OCR_PROFILE ")
    det = latest("LW_WASM_DET_PROFILE ")
    print()
    print("### Compiled SIMD128 full-OCR component diagnostic")
    print()
    if ocr is None or det is None:
        print("Missing component timings; check the profile log artifact.")
        return
    print("Measured invocation only; instrumentation overhead is excluded from the A/B table.")
    print()
    print("| Component | Measured ms |")
    print("| --- | ---: |")
    for field, label in (("total", "Full OCR"),
                         ("det_preprocess", "DET preprocess"),
                         ("det_graph", "DET graph"),
                         ("det_postprocess", "DET postprocess"),
                         ("crop", "Crop"), ("cls", "CLS"), ("rec", "REC")):
        print(f"| {label} | {ocr.get(field, 0.0):.3f} |")
    print(f"REC compiled/fallback lines: {int(ocr.get('rec_compiled', 0))}/"
          f"{int(ocr.get('rec_fallback', 0))}.")
    print(f"DET compiled/fallback runs: {int(ocr.get('det_compiled', 0))}/"
          f"{int(ocr.get('det_fallback', 0))}; CLS: "
          f"{int(ocr.get('cls_compiled', 0))}/"
          f"{int(ocr.get('cls_fallback', 0))}.")
    print()
    print("| DET physical op | Measured ms |")
    print("| --- | ---: |")
    for field, label in (("pointwise", "Pointwise"), ("dense", "Dense"),
                         ("depthwise", "Depthwise"),
                         ("convtranspose", "ConvTranspose"),
                         ("binary", "Binary"), ("pool", "Pool"),
                         ("concat", "Concat"), ("resize", "Resize"),
                         ("other", "Other")):
        print(f"| {label} | {det.get(field, 0.0):.3f} |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--scalar", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--profile-log", type=Path)
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--expected-text-sha256", required=True)
    parser.add_argument("--expected-line-count", type=int, default=16)
    parser.add_argument("--model-variant", choices=("tiny", "small", "medium"),
                        default="tiny")
    args = parser.parse_args()
    if args.profile_json is not None:
        if args.profile_log is None:
            parser.error("--profile-json requires --profile-log")
        write_profile_json(args.profile_log, args.profile_json, args.model_variant)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    runs = {
        "Canonical": load(args.canonical),
        "Compiled scalar": load(args.scalar),
        "Compiled SIMD128": load(args.compiled),
    }
    canonical, scalar, simd = runs.values()
    issues = []
    memory_issues = []
    if args.expected_line_count < 1:
        parser.error("--expected-line-count must be positive")
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
        for field in ("schema_version", "sample_sha256", "line_count",
                      "detected_count", "model_variant"):
            if run.get(field) != canonical.get(field):
                issues.append(f"{name}: {field} differs from canonical")
        if run.get("model_variant") != args.model_variant:
            issues.append(f"{name}: model variant differs from requested workload")
        if run.get("line_count") != args.expected_line_count:
            issues.append(f"{name}: OCR line count differs from the checked-in golden")
        if run.get("text_sha256") != canonical.get("text_sha256"):
            issues.append(f"{name}: OCR text differs from canonical")
    if canonical.get("text_sha256") != args.expected_text_sha256:
        issues.append("Canonical: OCR text differs from the checked-in golden checksum")
    if simd.get("wasm_heap_bytes", 0) > canonical.get("wasm_heap_bytes", 0) + 5 * 1048576:
        memory_issues.append("Compiled SIMD128: WASM heap exceeds canonical by more than 5 MiB")

    def mib(value: int) -> float:
        return value / 1048576

    print(f"## WASM {args.model_variant.title()} full OCR: canonical / compiled scalar / compiled SIMD128")
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
    print("| DET box count | " + " | ".join(
        str(run.get("detected_count", "missing")) for run in runs.values()) + " |")
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
    if memory_issues:
        print("**WASM heap gate: FAILED.**")
        for issue in memory_issues:
            print(f"- {issue}")
    else:
        print("**WASM heap gate: PASS.** Compiled SIMD128 is within canonical +5 MiB.")
    print("Latency and process RSS are informational, not hosted-runner pass/fail gates.")
    if args.profile_log is not None:
        print_full_ocr_profile(args.profile_log)
        print_compiled_profile(args.profile_log)
    return 1 if issues or memory_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
