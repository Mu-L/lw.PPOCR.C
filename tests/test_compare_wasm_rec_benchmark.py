"""The WASM A/B report must survive a text mismatch without weakening its gate."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "compare_wasm_rec_benchmark.py"


class WasmBenchmarkReportTest(unittest.TestCase):
    def run_report(self, simd_lines: list[str], expected_override: str | None = None
                   ) -> subprocess.CompletedProcess[str]:
        canonical_lines = ["识别结果", "OCR"]
        expected = hashlib.sha256("\n".join(canonical_lines).encode()).hexdigest()
        if expected_override is not None:
            expected = expected_override
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for name, lines, latency in (
                ("canonical", canonical_lines, 300.0),
                ("scalar", canonical_lines, 250.0),
                ("compiled", simd_lines, 200.0),
            ):
                path = Path(temporary) / f"{name}.json"
                path.write_text(json.dumps({
                    "schema_version": 1,
                    "sample_sha256": "fixture",
                    "line_count": len(lines),
                    "text_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
                    "text_lines": lines,
                    "median_ms": latency,
                    "peak_rss_bytes": 104857600,
                    "wasm_heap_bytes": 67108864,
                }), encoding="utf-8")
                paths.append(path)
            return subprocess.run(
                [sys.executable, str(SCRIPT), "--canonical", str(paths[0]),
                 "--scalar", str(paths[1]), "--compiled", str(paths[2]),
                 "--expected-text-sha256", expected],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                check=False,
            )

    def test_equal_text_reports_all_metrics(self) -> None:
        result = self.run_report(["识别结果", "OCR"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Text contract: PASS", result.stdout)
        self.assertIn("Median OCR latency", result.stdout)
        self.assertIn("Peak process RSS", result.stdout)
        self.assertIn("WASM heap", result.stdout)
        self.assertIn("1.250x", result.stdout)

    def test_mismatch_still_reports_metrics_then_fails(self) -> None:
        result = self.run_report(["识别结果", "0CR"])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Median OCR latency", result.stdout)
        self.assertIn("Text contract: FAILED", result.stdout)
        self.assertIn("Compiled SIMD128: OCR text differs", result.stdout)
        self.assertIn("0CR", result.stdout)

    def test_golden_mismatch_still_reports_metrics_then_fails(self) -> None:
        result = self.run_report(["识别结果", "OCR"], "0" * 64)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Median OCR latency", result.stdout)
        self.assertIn("Canonical: OCR text differs from the checked-in golden", result.stdout)


if __name__ == "__main__":
    unittest.main()
