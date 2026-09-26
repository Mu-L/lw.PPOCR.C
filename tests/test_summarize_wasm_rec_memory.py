"""The memory diagnostic must use real profile fields without hiding heap growth."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "summarize_wasm_rec_memory.py"


class WasmRecMemorySummaryTest(unittest.TestCase):
    def test_retained_components_and_heap_delta(self) -> None:
        spec = importlib.util.spec_from_file_location("wasm_rec_memory", SCRIPT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "profile.log"
            log.write_text("REC_MEMORY canonical_workspace=1048576 canonical_input=524288 "
                           "canonical_output=262144 canonical_planned_workspace=4194304\n",
                           encoding="utf-8")
            retained = module.canonical_rec_bytes(log)
            report = module.render("small", {"wasm_heap_bytes": 100 * 1048576},
                                   {"wasm_heap_bytes": 120 * 1048576},
                                   {"compiled_memory": {
                                       "det_arena_bytes": 1048576, "det_packed_bytes": 524288,
                                       "cls_arena_bytes": 262144, "cls_packed_bytes": 131072,
                                       "rec_unique_owned_constant_bytes": 2097152,
                                       "rec_shared": {"shared_arena": 4194304,
                                                      "shared_scratch": 0,
                                                      "ctc_workspace": 524288}}}, retained)
            self.assertIn("Canonical REC session workspace | 1.00", report)
            self.assertIn("Canonical REC state retained: **1.75 MiB**", report)
            self.assertIn("workspace planned for a possible fallback: **4.00 MiB**", report)
            self.assertIn("**+20.00 MiB**", report)
            self.assertIn("high-water size", report)
            log.write_text("REC_MEMORY shared_arena=1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing canonical REC"):
                module.canonical_rec_bytes(log)


if __name__ == "__main__":
    unittest.main()
