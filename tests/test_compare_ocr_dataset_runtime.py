from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from tools.compare_ocr_dataset_runtime import (
    load_manifest,
    run_process_with_heartbeat,
    summarize,
    summarize_paired,
)


def report(mean: float, p95: float, rss: int, checksum: str = "abc") -> dict:
    return {
        "schema_version": 1,
        "resident_widths": False,
        "images": 2,
        "lines": 4,
        "workers": 1,
        "rec_target_width": 960,
        "ocr_ms": {"mean": mean, "p95": p95},
        "peak_rss_bytes": rss,
        "rss_sample_count": 2,
        "min_sampled_rss_bytes": rss - 1024,
        "max_sampled_rss_bytes": rss,
        "output_checksum": checksum,
    }


class CompareOcrDatasetRuntimeTests(unittest.TestCase):
    def test_process_heartbeat_returns_completed_process(self) -> None:
        returncode, stdout, stderr, elapsed = run_process_with_heartbeat(
            [sys.executable, "-c", "print('ok')"],
            label="heartbeat-smoke",
            timeout_seconds=10,
            heartbeat_seconds=10,
        )
        self.assertEqual(returncode, 0)
        self.assertEqual(stdout.strip(), "ok")
        self.assertEqual(stderr, "")
        self.assertGreaterEqual(elapsed, 0.0)

    def test_process_heartbeat_enforces_timeout(self) -> None:
        with self.assertRaises(RuntimeError):
            run_process_with_heartbeat(
                [sys.executable, "-c", "import time; time.sleep(2)"],
                label="heartbeat-timeout",
                timeout_seconds=1,
                heartbeat_seconds=10,
            )
    def test_summary_checks_contract_and_calculates_deltas(self) -> None:
        compact = report(100.0, 140.0, 100 * 1048576)
        resident = report(90.0, 120.0, 110 * 1048576)
        resident["resident_widths"] = True
        summary = summarize(compact, resident)
        self.assertAlmostEqual(summary["comparison"]["mean_speedup"], 100.0 / 90.0)
        self.assertAlmostEqual(summary["comparison"]["rss_delta_mib"], 10.0)
        self.assertAlmostEqual(summary["comparison"]["max_sampled_rss_delta_mib"], 10.0)
        self.assertEqual(summary["contract"]["lines"], 4)

    def test_paired_summary_uses_medians(self) -> None:
        compact = [report(100.0, 140.0, 100 * 1048576), report(120.0, 160.0, 102 * 1048576)]
        resident = [report(90.0, 120.0, 110 * 1048576), report(100.0, 140.0, 112 * 1048576)]
        for item in resident:
            item["resident_widths"] = True
        summary = summarize_paired(compact, resident, ["compact-first", "resident-first"])
        self.assertEqual(summary["paired"]["rounds"], 2)
        self.assertAlmostEqual(summary["compact"]["ocr_mean_ms"], 110.0)
        self.assertAlmostEqual(summary["resident"]["ocr_mean_ms"], 95.0)

    def test_summary_rejects_checksum_mismatch(self) -> None:
        compact = report(100.0, 140.0, 100 * 1048576)
        resident = report(90.0, 120.0, 110 * 1048576, checksum="different")
        with self.assertRaises(RuntimeError):
            summarize(compact, resident)

    def test_load_manifest_accepts_repeated_width_switch_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "width-switch.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_manifest_sha256": "a" * 64,
                        "entries": [
                            {"source_file": "small.jpg", "ppm_file": "small.ppm"},
                            {"source_file": "large.jpg", "ppm_file": "large.ppm"},
                            {"source_file": "small.jpg", "ppm_file": "small.ppm"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = load_manifest(path)
            self.assertEqual(
                [item["file"] for item in manifest["images"]],
                ["small.ppm", "large.ppm", "small.ppm"],
            )


if __name__ == "__main__":
    unittest.main()
