import unittest
from tools.summarize_wasm_fine_widths import render


class FineWidthsSummaryTest(unittest.TestCase):
    def report(self, passed):
        run = dict(median_ms=100, ready_ms=200, cls_reconfigure_ms=50,
                   first_ocr_ms=110, heap_bytes=1048576, text_lines=["OEMODM"])
        return dict(runs={"baseline": [run], "candidate": [dict(run, text_lines=["OEM ODM"])]},
                    text_contract_pass=passed, paired_speedup=1.1, warmup=3,
                    iterations=5, sample_sha256="abc", browser_version="151")

    def test_text_failure_is_visible(self):
        text = render(self.report(False), "Small")
        self.assertIn("FAIL — diagnostic only", text)
        self.assertIn("OEMODM", text)
        self.assertIn("OEM ODM", text)
        self.assertIn("not process RSS", text)

    def test_success(self):
        self.assertIn("**PASS**", render(self.report(True), "Tiny"))

    def test_reviewed_difference_is_not_silently_equal(self):
        report = self.report(True)
        report["texts_identical"] = False
        self.assertIn("not byte-identical A/B", render(report, "Small"))


if __name__ == "__main__":
    unittest.main()
