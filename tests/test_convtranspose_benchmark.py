from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import unittest


class ConvTransposeBenchmarkTest(unittest.TestCase):
    def test_profile_geometry_is_machine_readable_and_parity_checked(self) -> None:
        completed = subprocess.run(
            [ARGS.driver, "16", "16", "128", "128", "2"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(
            (report["input_channels"], report["output_channels"],
             report["input_height"], report["input_width"]),
            (16, 16, 128, 128),
        )
        self.assertEqual((report["output_height"], report["output_width"]), (256, 256))
        self.assertEqual(report["iterations"], 2)
        self.assertIn(report["simd_backend"], {"scalar", "sse2", "avx2", "neon"})
        for field in ("scalar_ms", "simd_ms", "speedup", "max_abs_error"):
            self.assertTrue(math.isfinite(report[field]), (field, report))
            self.assertGreaterEqual(report[field], 0.0, (field, report))
        self.assertLessEqual(report["max_abs_error"], 1.0e-3, report)
        self.assertRegex(report["scalar_checksum"], re.compile(r"^0x[0-9a-f]{16}$"))
        self.assertRegex(report["simd_checksum"], re.compile(r"^0x[0-9a-f]{16}$"))
        self.assertEqual(report["scalar_checksum"], report["simd_checksum"], report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    unittest.main(argv=[__file__], verbosity=2)
