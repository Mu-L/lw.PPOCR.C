#!/usr/bin/env python3
"""Gate the official Medium REC hybrid executor before treating it as production-ready."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


WIDTHS = (192, 320, 480, 640, 960)


def executable(build_dir: Path, name: str) -> Path:
    candidates = [build_dir / "Release" / f"{name}.exe", build_dir / f"{name}.exe", build_dir / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing executable {name} in {build_dir}")


def run_json(command: list[str], label: str) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed ({completed.returncode}): {completed.stdout}\n{completed.stderr}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} did not emit JSON: {completed.stdout}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, help="Defaults to REPORT_DIR/converted-runtime")
    parser.add_argument("--sample", type=Path, help="Defaults to BUILD_DIR/models/sample.ppm")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    report_dir = args.report_dir.resolve()
    model_dir = args.model_dir.resolve() if args.model_dir else report_dir / "converted-runtime"
    sample = args.sample.resolve() if args.sample else build / "models" / "sample.ppm"
    rec_driver = executable(build, "rec-fast-prototype-driver")
    ocr_driver = executable(build, "full-ocr-profile-driver")
    rec = model_dir / "rec.lwm"
    numerical = []
    for width in WIDTHS:
        result = run_json([str(rec_driver), str(rec), str(width)], f"Medium REC{width}")
        if (result["width"] != width or result["mismatch"] != 0 or
                result["argmax_mismatch"] != 0 or not math.isfinite(result["max_abs"]) or
                result["max_abs"] > 1e-4):
            raise AssertionError(f"Medium REC{width} differs from canonical: {result}")
        numerical.append({key: result[key] for key in ("width", "legacy_ms", "fast_ms", "speedup", "max_abs", "mismatch", "argmax_mismatch")})
        print(f"Medium REC{width}: {result['speedup']:.3f}x, max_abs={result['max_abs']:.3g}, argmax_mismatch=0", flush=True)

    checksums = set()
    full_ocr = []
    for workers in (1, 4):
        command = [str(ocr_driver), str(model_dir / "det.lwm"), str(model_dir / "cls.lwm"), str(rec),
                   str(model_dir / "ppocr_keys.txt"), str(sample), "1", str(workers), "960"]
        result = run_json(command, f"Medium OCR {workers} workers")
        coverage = result["rec_backend_coverage"]
        if result["lines"] != 16 or coverage["compiled_lines"] != 16 or coverage["canonical_lines"] != 0:
            raise AssertionError(f"Medium OCR did not use the hybrid path for all lines ({workers} workers): {coverage}")
        checksums.add(result["output_checksum"])
        full_ocr.append({"workers": workers, "lines": result["lines"], "output_checksum": result["output_checksum"],
                         "total_ms": result["wall_nanoseconds"]["total"] / 1e6, "rec_backend_coverage": coverage})
        print(f"Medium OCR {workers} workers: {full_ocr[-1]['total_ms']:.3f} ms, compiled=16/16", flush=True)
    if len(checksums) != 1:
        raise AssertionError(f"Medium OCR text differs between 1 and 4 workers: {checksums}")

    report = {"schema_version": 1, "model": "ppocrv6-medium", "rec_widths": numerical, "full_ocr": full_ocr}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "medium-fast-path.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
