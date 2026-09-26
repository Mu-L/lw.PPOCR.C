"""Identity and memory gates for the experimental Pointwise tile A/B."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "compare_wasm_pointwise_ab.py"


class PointwiseAbTest(unittest.TestCase):
    def test_contract_and_report(self) -> None:
        spec = importlib.util.spec_from_file_location("pointwise_ab", SCRIPT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        lines = ["文字", "OCR"]
        digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            golden = root / "golden.json"
            golden.write_text(json.dumps({"expected_line_count": 2,
                                          "expected_text_sha256": digest}), encoding="utf-8")
            paths = []
            for index, milliseconds in enumerate((100.0, 80.0, 82.0, 102.0)):
                path = root / f"run-{index}.json"
                path.write_text(json.dumps({
                    "schema_version": 1, "wasm_backend": "wasm128",
                    "model_variant": "tiny", "use_cls": True,
                    "sample_sha256": "sample", "iterations": 5,
                    "median_ms": milliseconds, "peak_rss_bytes": 100000000,
                    "wasm_heap_bytes": 70000000,
                    "line_count": 2, "detected_count": 2,
                    "text_sha256": digest, "text_lines": lines,
                }), encoding="utf-8")
                paths.append(path)
            report = module.compare([paths[0], paths[3]], [paths[1], paths[2]],
                                    golden, 5, 5.0)
            self.assertIn("4/2/2/4", report)
            self.assertIn("1.250x", report)
            self.assertIn("Text contract: **PASS**", report)
            four_profile = root / "four.log"
            two_profile = root / "two.log"
            four_profile.write_text("X64REC width=320 total=20.000 pw=12.000 ctc=1.000\n",
                                    encoding="utf-8")
            two_profile.write_text("X64REC width=320 total=18.000 pw=10.000 ctc=1.000\n",
                                   encoding="utf-8")
            profiled_report = module.compare([paths[0], paths[3]], [paths[1], paths[2]],
                                             golden, 5, 5.0, four_profile, two_profile)
            self.assertIn("| 320 | 1 | 12.000 | 10.000 | 1.200x | 57.1% | 52.6% |",
                          profiled_report)
            self.assertIn("4x16 **57.1%**, 2x16 **52.6%**", profiled_report)
            changed = json.loads(paths[2].read_text(encoding="utf-8"))
            changed["text_lines"] = ["文字", "0CR"]
            paths[2].write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "text_lines differs"):
                module.compare([paths[0], paths[3]], [paths[1], paths[2]],
                               golden, 5, 5.0)
            changed["text_lines"] = lines
            changed["wasm_heap_bytes"] += 6 * 1048576
            paths[2].write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "WASM heap exceeds"):
                module.compare([paths[0], paths[3]], [paths[1], paths[2]],
                               golden, 5, 5.0)


if __name__ == "__main__":
    unittest.main()
