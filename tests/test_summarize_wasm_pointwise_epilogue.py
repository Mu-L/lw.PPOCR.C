"""The epilogue report must exclude CLS and rank real REC work."""

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "summarize_wasm_pointwise_epilogue.py"


class EpilogueReportTest(unittest.TestCase):
    def test_rec_only_and_elapsed_ranking(self) -> None:
        spec = importlib.util.spec_from_file_location("wasm_pw_epilogue", SCRIPT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = module.summarize(
            "WASM_PW_EPI width=160 kind=bias calls=2 macs=99 elapsed=30.000\n"
            "WASM_PW_EPI width=320 kind=plain calls=3 macs=100 elapsed=12.000\n"
            "WASM_PW_EPI width=480 kind=hardswish calls=1 macs=200 elapsed=8.000\n",
            "small",
        )
        self.assertIn("Excluded 2 CLS Pointwise calls", report)
        self.assertIn("| plain | 3 | 100 | 12.000 | 60.0% |", report)
        self.assertIn("| hardswish | 1 | 200 | 8.000 | 40.0% |", report)
        self.assertNotIn("| bias |", report)
        with self.assertRaisesRegex(ValueError, "no Tiny/Small REC"):
            module.summarize("WASM_PW_EPI width=160 kind=bias calls=1 macs=1 elapsed=1.000", "tiny")


if __name__ == "__main__":
    unittest.main()
