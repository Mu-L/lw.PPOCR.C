"""Pure report-contract tests for the x64 release sample comparison."""

from __future__ import annotations

import unittest

from tools.compare_release_sample_ocr import MIB, render_markdown, summarize_case, validate_payload


def run(mean_ms: float, peak_mib: float, checksum: str = "abc") -> dict:
    return {
        "schema_version": 1,
        "backend": "avx2",
        "image_width": 500,
        "image_height": 500,
        "lines": 16,
        "workers": 4,
        "rec_target_width": 960,
        "warmup": 2,
        "iterations": 5,
        "ocr_ms": {"mean": mean_ms, "p95": mean_ms * 1.2},
        "peak_rss_bytes": int(peak_mib * MIB),
        "rss_after_warmup_bytes": int((peak_mib - 20) * MIB),
        "output_checksum": checksum,
    }


class ReleaseSampleOcrTest(unittest.TestCase):
    def test_paired_latency_and_memory_summary(self) -> None:
        old = [run(200, 500), run(220, 510)]
        new = [run(100, 550), run(110, 560)]
        for payload in old + new:
            validate_payload(payload, 4, 960, 2, 5)
        case = summarize_case(
            "small", 4, old, new, ["baseline-candidate", "candidate-baseline"]
        )
        self.assertEqual(case["paired_speedup"], 2.0)
        self.assertEqual(case["peak_ws_delta_mib"], 50.0)
        self.assertTrue(case["output_match"])
        report = {
            "baseline_ref": "v1.0.0",
            "baseline_commit": "old",
            "candidate_commit": "new",
            "runner": {"platform": "Windows", "logical_cpus": 4},
            "sample_sha256": "sample",
            "rec_target_width": 960,
            "warmup": 2,
            "iterations": 5,
            "paired_rounds": 2,
            "cases": [case],
        }
        self.assertIn("| small | 4 |", render_markdown(report))

    def test_changed_text_is_reported_not_gated(self) -> None:
        case = summarize_case(
            "medium", 4, [run(200, 500)], [run(100, 550, "different")],
            ["baseline-candidate"],
        )
        self.assertFalse(case["output_match"])

    def test_rejects_wrong_worker_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "worker or REC width"):
            validate_payload(run(100, 500), 1, 960, 2, 5)


if __name__ == "__main__":
    unittest.main()
