#!/usr/bin/env python3
"""Gate a same-runner 4x16/2x16 WASM pointwise A/B on OCR identity."""

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path


REC_WIDTHS = frozenset((192, 320, 480, 640, 960))


def load_run(path: Path) -> dict:
    run = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(run, dict) or run.get("schema_version") != 1:
        raise ValueError(f"invalid benchmark result: {path}")
    for field in ("median_ms", "peak_rss_bytes", "wasm_heap_bytes"):
        value = run.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{path}: invalid {field}")
    return run


def load_profile(path: Path, accepted_widths: frozenset[int] = REC_WIDTHS
                 ) -> dict[int, tuple[int, float, float, float]]:
    """Aggregate separately instrumented REC or CLS runs by width."""
    widths: dict[int, list[dict[str, float]]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("X64REC width="):
            continue
        fields = {key: float(value) for key, value in
                  re.findall(r"([a-z]+)=([0-9]+(?:\.[0-9]+)?)", line)}
        if all(key in fields for key in ("width", "total", "pw", "ctc")):
            width = int(fields["width"])
            if width in accepted_widths:
                widths.setdefault(width, []).append(fields)
            elif width not in REC_WIDTHS and width != 160:
                raise ValueError(f"{path}: unknown REC/CLS profile width {width}")
    if not widths:
        raise ValueError(f"{path}: no requested-width profile was captured")
    return {width: (len(rows),
                    sum(row["pw"] for row in rows),
                    sum(row["ctc"] for row in rows),
                    sum(row["total"] + row["ctc"] for row in rows))
            for width, rows in widths.items()}


def compare(four_paths: list[Path], two_paths: list[Path], golden_path: Path,
            expected_iterations: int, max_heap_growth_mib: float,
            four_profile: Path | None = None, two_profile: Path | None = None,
            expected_warmup: int = 1) -> str:
    if len(four_paths) != 2 or len(two_paths) != 2:
        raise ValueError("exactly two results per tile are required")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    four = [load_run(path) for path in four_paths]
    two = [load_run(path) for path in two_paths]
    reference = four[0]
    identity = ("wasm_backend", "model_variant", "use_cls", "sample_sha256",
                "line_count", "detected_count", "text_sha256", "text_lines")
    for path, run in zip(four_paths + two_paths, four + two):
        for field in identity:
            if run.get(field) != reference.get(field):
                raise ValueError(f"{path}: {field} differs from 4x16 reference")
        if run.get("iterations") != expected_iterations:
            raise ValueError(f"{path}: benchmark iteration count differs")
        if run.get("warmup_runs", 1) != expected_warmup:
            raise ValueError(f"{path}: benchmark warm-up count differs")
        if not isinstance(run.get("text_lines"), list) or any(
                not isinstance(line, str) for line in run["text_lines"]):
            raise ValueError(f"{path}: invalid OCR text lines")
        actual_digest = hashlib.sha256("\n".join(run["text_lines"]).encode("utf-8")).hexdigest()
        if actual_digest != run["text_sha256"]:
            raise ValueError(f"{path}: OCR text checksum does not match text lines")
    variant = golden.get("variant")
    if variant not in ("tiny", "small", "medium"):
        raise ValueError("golden must identify a supported model variant")
    if (reference["wasm_backend"] != "wasm128" or
            reference["model_variant"] != variant or
            reference["use_cls"] is not True or
            reference["line_count"] != golden["expected_line_count"] or
            reference["detected_count"] != golden["expected_line_count"] or
            reference["text_sha256"] != golden["expected_text_sha256"] or
            len(reference["text_lines"]) != reference["line_count"]):
        raise ValueError(f"4x16 reference does not match the {variant} OCR golden contract")
    if max(run["wasm_heap_bytes"] for run in two) > (
            max(run["wasm_heap_bytes"] for run in four) + max_heap_growth_mib * 1048576):
        raise ValueError("2x16 WASM heap exceeds 4x16 by more than the allowed growth")

    paired_ratios = [base["median_ms"] / candidate["median_ms"]
                     for base, candidate in zip(four, two)]
    rows = [f"## WASM Pointwise 4x16 vs 2x16 — {variant.title()} full OCR", "",
            f"Each result is a fresh process with {expected_warmup} warm-up run(s) and "
            f"{expected_iterations} measured OCR runs; run order is 4/2/2/4.", "",
            "| Pair | 4x16 ms | 2x16 ms | 4x16 / 2x16 | 4x16 heap MiB | 2x16 heap MiB |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for index, (base, candidate, ratio) in enumerate(zip(four, two, paired_ratios), 1):
        rows.append(f"| {index} | {base['median_ms']:.3f} | {candidate['median_ms']:.3f} | "
                    f"{ratio:.3f}x | {base['wasm_heap_bytes'] / 1048576:.1f} | "
                    f"{candidate['wasm_heap_bytes'] / 1048576:.1f} |")
    rows += ["", f"Median paired speedup: **{statistics.median(paired_ratios):.3f}x**. "
             "Latency and RSS are informational; OCR text, DET count and heap are gates.",
             "Text contract: **PASS**."]
    if (four_profile is None) != (two_profile is None):
        raise ValueError("both tile profiles are required for a width comparison")
    if four_profile is not None and two_profile is not None:
        profiled_four = load_profile(four_profile)
        profiled_two = load_profile(two_profile)
        if profiled_four.keys() != profiled_two.keys() or any(
                profiled_four[width][0] != profiled_two[width][0]
                for width in profiled_four):
            raise ValueError("REC width/profile invocation coverage differs between tiles")
        rows += ["", "### Instrumented REC width diagnostic", "",
                 "Width 160 is CLS and excluded from REC totals. "
                 "These values come from separate profiled runs and are not part of the "
                 "full-OCR latency A/B above.", "",
                 "| Width | Invocations | 4x16 PW ms/inv | 2x16 PW ms/inv | PW 4x16/2x16 | 4x16 PW share | 2x16 PW share | 4x16 CTC ms/inv | 2x16 CTC ms/inv |",
                 "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for width in sorted(profiled_four):
            count, pw_four, ctc_four, tracked_four = profiled_four[width]
            _, pw_two, ctc_two, tracked_two = profiled_two[width]
            rows.append(f"| {width} | {count} | {pw_four / count:.3f} | "
                        f"{pw_two / count:.3f} | {pw_four / pw_two if pw_two else 0.0:.3f}x | "
                        f"{pw_four / tracked_four * 100.0 if tracked_four else 0.0:.1f}% | "
                        f"{pw_two / tracked_two * 100.0 if tracked_two else 0.0:.1f}% | "
                        f"{ctc_four / count:.3f} | {ctc_two / count:.3f} |")
        total_four_pw = sum(row[1] for row in profiled_four.values())
        total_four_tracked = sum(row[3] for row in profiled_four.values())
        total_two_pw = sum(row[1] for row in profiled_two.values())
        total_two_tracked = sum(row[3] for row in profiled_two.values())
        rows += ["", "Overall Pointwise share of tracked REC time: "
                 f"4x16 **{total_four_pw / total_four_tracked * 100.0 if total_four_tracked else 0.0:.1f}%**, "
                 f"2x16 **{total_two_pw / total_two_tracked * 100.0 if total_two_tracked else 0.0:.1f}%**."]
        cls_four = load_profile(four_profile, frozenset((160,)))[160]
        cls_two = load_profile(two_profile, frozenset((160,)))[160]
        if cls_four[0] != cls_two[0]:
            raise ValueError("CLS profile invocation coverage differs between tiles")
        rows += [f"CLS width 160 (separate): {cls_four[0]} invocations; "
                 f"Pointwise 4x16 {cls_four[1] / cls_four[0]:.3f} ms/inv, "
                 f"2x16 {cls_two[1] / cls_two[0]:.3f} ms/inv."]
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--four", type=Path, action="append", required=True)
    parser.add_argument("--two", type=Path, action="append", required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--four-profile", type=Path)
    parser.add_argument("--two-profile", type=Path)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-heap-growth-mib", type=float, default=5.0)
    args = parser.parse_args()
    if (args.iterations < 1 or args.warmup < 1 or
            not math.isfinite(args.max_heap_growth_mib) or args.max_heap_growth_mib < 0):
        parser.error("iterations/warmup must be positive and heap growth limit finite/nonnegative")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    print(compare(args.four, args.two, args.golden, args.iterations,
                  args.max_heap_growth_mib, args.four_profile, args.two_profile,
                  args.warmup), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
