#!/usr/bin/env python3
"""Validate the experimental NHWC pointwise benchmark JSON and parity gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    args = parser.parse_args()
    completed = subprocess.run(
        [str(args.driver)], check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stdout + completed.stderr)
    report = json.loads(completed.stdout)
    assert report["schema_version"] == 1
    if report["status"] == "skipped":
        assert report["reason"] == "requires_avx2_fma"
        return 0
    assert report["status"] == "ok"
    assert report["backend"] == "avx2+fma"
    cases = report["cases"]
    assert len(cases) == 5
    for case in cases:
        assert case["max_abs"] <= 1.0e-4, case
        assert case["max_rel"] <= 1.0e-4, case
        assert case["output_mismatch_count"] == 0, case
        assert case["argmax_mismatch_count"] == 0, case
        assert case["nchw_ms"] > 0.0, case
        assert case["nhwc_ms"] > 0.0, case
    assert "promotion_gate" in report
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

