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
    correctness = [record for record in records if "max_abs" in record]
    performance = [record for record in records if "perf_case" in record]
    assert len(correctness) == 20, records
    assert len(performance) == 4, records
    assert {record["case"] for record in correctness} == {
        "border-3x3-s1", "border-5x5-s2", "blocked-3x3-s1", "medium-rec-3x3-h6-w240",
    }
    assert {record["perf_case"] for record in performance} == {
        "border-3x3-s1", "border-5x5-s2", "blocked-3x3-s1", "medium-rec-3x3-h6-w240",
    }
    assert {record["kc"] for record in correctness} == {0, 128, 256, 512, 1024}
    for record in correctness:
        assert record["max_abs"] <= 1.0e-4, record
        assert record["scratch_bytes"] > 0, record
    for record in performance:
        assert record["scalar_ms"] > 0.0, record
        assert record["dense_ms"] > 0.0, record
        assert record["speedup"] > 0.0, record
        assert record["scratch_bytes"] > 0, record
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
