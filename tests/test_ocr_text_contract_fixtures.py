"""Keep readable OCR fixtures in sync with native and Web SHA contracts."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OcrTextContractFixturesTest(unittest.TestCase):
    def test_small_and_medium_contracts(self) -> None:
        for variant in ("small", "medium"):
            with self.subTest(variant=variant):
                fixture = ROOT / "ci" / "fixtures" / f"ppocrv6-{variant}-full-ocr.txt"
                lines = fixture.read_text(encoding="utf-8").splitlines()
                native = json.loads(
                    (ROOT / "ci" / f"ppocrv6-{variant}-validation.json").read_text(
                        encoding="utf-8"
                    )
                )
                web = json.loads(
                    (ROOT / "ci" / f"web-ppocrv6-{variant}.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(len(lines), native["full_ocr"]["expected_lines"])
                self.assertTrue(all(lines))
                digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
                self.assertEqual(digest, native["full_ocr"]["expected_text_sha256"])
                self.assertEqual(digest, web["expected_text_sha256"])


if __name__ == "__main__":
    unittest.main()
