from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_ocr_width_switch_manifest import build_width_switch_manifest


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class BuildOcrWidthSwitchManifestTests(unittest.TestCase):
    def test_orders_available_buckets_low_high_and_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "metadata.json"
            ppm_root = root / "ppm"
            ppm_root.mkdir()
            source_records = []
            ppm_records = []
            for index, width in enumerate((180, 350, 700), start=1):
                source_file = f"img-{index:03d}.png"
                ppm_file = f"{index:04d}.ppm"
                (ppm_root / ppm_file).write_bytes(b"P6\n1 1\n255\n\0\0\0")
                source_records.append(
                    {"file": source_file, "lines": [{"natural_width_at_height_48": width}]}
                )
                ppm_records.append({"source_file": source_file, "file": ppm_file})
            write_json(dataset, {"version": 1, "images": source_records})
            source_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
            ppm_manifest = ppm_root / "benchmark-manifest.json"
            write_json(
                ppm_manifest,
                {"schema_version": 1, "source_manifest_sha256": source_hash, "images": ppm_records},
            )
            output_list = root / "switch" / "images.txt"
            output_manifest = root / "switch" / "manifest.json"
            report = build_width_switch_manifest(
                dataset, ppm_manifest, output_list, output_manifest, cycles=2
            )
            self.assertEqual(report["available_buckets"], [192, 480, 960])
            self.assertEqual(report["bucket_order"], [192, 960, 480])
            self.assertEqual(
                [entry["expected_bucket"] for entry in report["entries"]],
                [192, 960, 480, 192, 960, 480],
            )
            self.assertEqual(len(output_list.read_text(encoding="utf-8").splitlines()), 6)

    def test_rejects_mismatched_prepared_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "metadata.json"
            write_json(dataset, {"version": 1, "images": []})
            ppm_manifest = root / "benchmark-manifest.json"
            write_json(
                ppm_manifest,
                {"schema_version": 1, "source_manifest_sha256": "0" * 64, "images": []},
            )
            with self.assertRaises(ValueError):
                build_width_switch_manifest(
                    dataset, ppm_manifest, root / "images.txt", root / "switch.json"
                )


if __name__ == "__main__":
    unittest.main()