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
    def run_report(self, simd_lines: list[str], expected_override: str | None = None,
                   use_cls: bool = True, profile_text: str | None = None
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
                    "use_cls": use_cls,
                    "sample_sha256": "fixture",
                    "line_count": len(lines),
                    "text_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
                    "text_lines": lines,
                    "median_ms": latency,
                    "peak_rss_bytes": 104857600,
                    "wasm_heap_bytes": 67108864,
                }), encoding="utf-8")
                paths.append(path)
            profile_args = []
            if profile_text is not None:
                profile = Path(temporary) / "profile.log"
                profile.write_text(profile_text, encoding="utf-8")
                profile_args = ["--profile-log", str(profile)]
            return subprocess.run(
                [sys.executable, str(SCRIPT), "--canonical", str(paths[0]),
                 "--scalar", str(paths[1]), "--compiled", str(paths[2]),
                 "--expected-text-sha256", expected, *profile_args],
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

    def test_cls_disabled_cannot_claim_golden_parity(self) -> None:
        result = self.run_report(["识别结果", "OCR"], use_cls=False)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("benchmark must enable CLS", result.stdout)
        self.assertIn("Median OCR latency", result.stdout)

    def test_profile_reports_hotspot_without_changing_text_gate(self) -> None:
        profile = "\n".join([
            "X64REC width=320 total=20.000 pw=12.000 dense=5.000 dw=2.000 "
            "bin=0.000 unary=0.000 reduce=0.000 pool=0.000 transpose=0.000 "
            "matmul=0.000 ctc=1.000",
            "X64REC width=320 total=20.000 pw=12.000 dense=5.000 dw=2.000 "
            "bin=0.000 unary=0.000 reduce=0.000 pool=0.000 transpose=0.000 "
            "matmul=0.000 ctc=1.000",
        ])
        result = self.run_report(["识别结果", "OCR"], profile_text=profile)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2 REC invocations", result.stdout)
        self.assertIn("| Pointwise | 24.000 | 57.1% |", result.stdout)
        self.assertIn("Text contract: PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
