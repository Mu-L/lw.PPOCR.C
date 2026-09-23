"""Tests for the native OCR text-diff parser."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.compare_ocr_text_output import texts


class CompareOcrTextOutputTest(unittest.TestCase):
    def test_extracts_ordered_unicode_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ocr.txt"
            output.write_text(
                "lines=2 elapsed_ms=1\n"
                "0 text=纯臻营养护发素 rec=0.9\n"
                "1 text=产品信息/参数 rec=0.8\n",
                encoding="utf-8",
            )
            self.assertEqual(texts(output), ["纯臻营养护发素", "产品信息/参数"])


if __name__ == "__main__":
    unittest.main()
