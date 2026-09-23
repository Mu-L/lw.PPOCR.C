#!/usr/bin/env python3
"""Validate the experimental NHWC dense correctness and A/B report."""

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
    records = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if len(records) == 1 and records[0].get("status") == "skipped":
        assert records[0]["reason"] == "requires_avx2_fma"
        return 0
    correctness = [record for record in records
                   if "max_abs" in record and "-hardswish" not in record["case"]]
    hardswish = [record for record in records
                 if "max_abs" in record and "-hardswish" in record["case"]]
    performance = [record for record in records if "perf_case" in record]
    cases = {
        "border-3x3-s1", "border-5x5-s2", "blocked-3x3-s1",
        "medium-rec-3x3-h6-w240", "det-2x2-s1-pads0011", "det-3x3-s2-pad1",
        "det-stem-3x3-s2-ic3", "det-graph-stem-16x32", "det-head-3x3-s1-64-16",
        "border-3x3-s1-asym",
    }
    assert len(correctness) == 5 * len(cases), records
    assert len(hardswish) == len(cases), records
    assert len(performance) == len(cases), records
    assert {record["case"] for record in correctness} == cases
    assert {record["case"].removesuffix("-hardswish") for record in hardswish} == cases
    assert {record["perf_case"] for record in performance} == cases
    assert {record["kc"] for record in correctness} == {0, 128, 256, 512, 1024}
    for record in correctness + hardswish:
        assert record["max_abs"] <= 1.0e-4, record
        if "kc" in record:
            assert record["scratch_bytes"] > 0, record
    for record in performance:
        assert record["scalar_ms"] > 0.0, record
        assert record["dense_ms"] > 0.0, record
        assert record["speedup"] > 0.0, record
        assert record["scratch_bytes"] > 0, record
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
