"""The WASM A/B report must survive a text mismatch without weakening its gate."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "compare_wasm_rec_benchmark.py"


class WasmBenchmarkReportTest(unittest.TestCase):
    def run_report(self, simd_lines: list[str], expected_override: str | None = None,
                   use_cls: bool = True, profile_text: str | None = None,
                   simd_heap_bytes: int = 67108864,
                   simd_detected_count: int = 2,
                   expected_line_count: int = 2
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
                    "model_variant": "tiny",
                    "use_cls": use_cls,
                    "sample_sha256": "fixture",
                    "line_count": len(lines),
                    "detected_count": simd_detected_count if name == "compiled" else 2,
                    "text_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
                    "text_lines": lines,
                    "median_ms": latency,
                    "peak_rss_bytes": 104857600,
                    "wasm_heap_bytes": simd_heap_bytes if name == "compiled" else 67108864,
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
                 "--expected-text-sha256", expected,
                 "--expected-line-count", str(expected_line_count), *profile_args],
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
        self.assertIn("DET box count", result.stdout)
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

    def test_det_box_count_mismatch_fails(self) -> None:
        result = self.run_report(["识别结果", "OCR"], simd_detected_count=3)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("detected_count differs from canonical", result.stdout)

    def test_golden_line_count_mismatch_fails(self) -> None:
        result = self.run_report(["识别结果", "OCR"], expected_line_count=3)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("OCR line count differs from the checked-in golden", result.stdout)

    def test_heap_growth_over_five_mib_fails(self) -> None:
        result = self.run_report(["识别结果", "OCR"],
                                 simd_heap_bytes=67108864 + 6 * 1048576)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("WASM heap exceeds canonical by more than 5 MiB", result.stdout)

    def test_profile_reports_hotspot_without_changing_text_gate(self) -> None:
        profile = "\n".join([
            "LW_WASM_OCR_PROFILE total=100.000 det_preprocess=5.000 det_graph=30.000 "
            "det_postprocess=4.000 crop=3.000 cls=8.000 rec=50.000 "
            "det_compiled=1 det_fallback=0 cls_compiled=2 cls_fallback=0 "
            "rec_compiled=2 rec_fallback=0",
            "LW_WASM_DET_PROFILE total=30.000 pointwise=10.000 dense=8.000 "
            "depthwise=6.000 convtranspose=3.000 binary=1.000 pool=1.000 "
            "concat=0.500 resize=0.250 other=0.250",
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
        self.assertIn("| 320 | 2 | 24.000 | 2.000 | 42.000 | 57.1% |", result.stdout)
        self.assertIn("| DET graph | 30.000 |", result.stdout)
        self.assertIn("| ConvTranspose | 3.000 |", result.stdout)
        self.assertIn("REC compiled/fallback lines: 2/0", result.stdout)
        self.assertIn("DET compiled/fallback runs: 1/0; CLS: 2/0", result.stdout)
        self.assertIn("Text contract: PASS", result.stdout)

    def test_profile_json_keeps_measured_components(self) -> None:
        module_spec = importlib.util.spec_from_file_location("wasm_benchmark_report", SCRIPT)
        self.assertIsNotNone(module_spec)
        self.assertIsNotNone(module_spec.loader)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "profile.log"
            report = Path(temporary) / "profile.json"
            log.write_text(
                "LW_WASM_OCR_PROFILE total=100.000 det_preprocess=5.000 det_graph=30.000 "
                "det_postprocess=4.000 crop=3.000 cls=8.000 rec=50.000 "
                "det_compiled=1 det_fallback=0 cls_compiled=2 cls_fallback=0 "
                "rec_compiled=2 rec_fallback=0\n"
                "LW_WASM_DET_PROFILE total=30.000 pointwise=10.000 dense=8.000 "
                "depthwise=6.000 convtranspose=3.000 binary=1.000 pool=1.000 "
                "concat=0.500 resize=0.250 other=0.250\n"
                "LW_WASM_COMPILED_CLS layout=nhwc ops=100 unsupported=0 "
                "arena_bytes=4096 packed_bytes=512\n"
                "LW_WASM_COMPILED_DET input=512x512 layout=nhwc ops=200 unsupported=0 "
                "conversions=1 direct_input=1 arena_bytes=8192 packed_bytes=1024\n"
                "REC_MEMORY width=320 owned=2048 borrowed=0 borrowed_count=0 "
                "arena=16384 scratch=256\n"
                "REC_MEMORY shared_arena=16384 shared_scratch=256 "
                "ctc_workspace=128 medium_fast_shared=0\n"
                "X64REC width=320 total=20.000 pw=12.000 ctc=1.000\n",
                encoding="utf-8",
            )
            module.write_profile_json(log, report, "small")
            value = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(value["model_variant"], "small")
            self.assertEqual(value["full_ocr"]["det_graph_ms"], 30.0)
            self.assertEqual(value["det_physical"]["pointwise_ms"], 10.0)
            self.assertEqual(value["rec_compiled_lines"], 2)
            self.assertEqual(value["rec_fallback_lines"], 0)
            self.assertEqual(value["det_compiled_runs"], 1)
            self.assertEqual(value["cls_fallback_runs"], 0)
            self.assertEqual(len(value["rec_invocations"]), 1)
            self.assertEqual(value["compiled_memory"]["det_arena_bytes"], 8192)
            self.assertEqual(value["compiled_memory"]["cls_packed_bytes"], 512)
            self.assertEqual(value["compiled_memory"]["rec_unique_owned_constant_bytes"], 2048)
            log.write_text(log.read_text(encoding="utf-8").replace("cls_fallback=0",
                         "cls_fallback=1"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical OCR graph fallback"):
                module.write_profile_json(log, report, "small")


if __name__ == "__main__":
    unittest.main()
