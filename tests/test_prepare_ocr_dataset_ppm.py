from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.generate_ocr_dataset import generate_dataset
from tools.prepare_ocr_dataset_ppm import prepare_dataset


class PrepareOcrDatasetPpmTests(unittest.TestCase):
    def test_prepare_preserves_manifest_identity_and_ppm_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            generate_dataset(
                dataset,
                count=3,
                seed=123,
                image_format="png",
                text_pool=(("project", "OCR test"),),
            )
            output = root / "ppm"
            report = prepare_dataset(dataset, output)
            self.assertEqual(len(report["images"]), 3)
            self.assertRegex(report["source_manifest_sha256"], r"^[0-9a-f]{64}$")
            list_lines = (output / "images.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(list_lines), 3)
            for entry, listed in zip(report["images"], list_lines):
                self.assertEqual(Path(listed).name, entry["file"])
                with Image.open(listed) as image:
                    self.assertEqual(image.format, "PPM")
                    self.assertEqual(image.size, (entry["width"], entry["height"]))


if __name__ == "__main__":
    unittest.main()
