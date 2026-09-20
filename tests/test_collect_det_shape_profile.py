from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tools import collect_det_shape_profile as profile


class CollectDetShapeProfileTests(unittest.TestCase):
    def test_collect_groups_real_convolution_geometry(self) -> None:
        reports = {
            (1, 960): {
                "schema_version": 1,
                "workers": 1,
                "rec_target_width": 960,
                "output_checksum": "abc",
                "det_convolution_nodes": [
                    {
                        "node": 2, "operation": "Conv", "nanoseconds": 1200000,
                        "invocations": 2, "input": [1, 64, 32, 32],
                        "weights": [128, 64, 3, 3], "output": [1, 128, 16, 16],
                        "group": 1, "kernel": [3, 3], "strides": [2, 2],
                        "dilations": [1, 1], "pads": [1, 1, 1, 1],
                    },
                    {
                        "node": 3, "operation": "ConvTranspose", "nanoseconds": 800000,
                        "invocations": 2, "input": [1, 128, 16, 16],
                        "weights": [128, 64, 2, 2], "output": [1, 64, 32, 32],
                        "group": 1, "kernel": [2, 2], "strides": [2, 2],
                        "dilations": [1, 1], "pads": [0, 0, 0, 0],
                    },
                ],
            },
        }

        def fake_run(*args, **kwargs):
            return reports[(args[7], args[8])]

        with patch.object(profile, "run_profile", side_effect=fake_run):
            result = profile.collect(
                Path("driver"), Path("det"), Path("cls"), Path("rec"),
                Path("dict"), Path("image"), (960,), (1,), 1,
            )

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(len(result["det_shapes"]), 2)
        self.assertEqual({entry["operation"] for entry in result["det_shapes"]}, {"Conv", "ConvTranspose"})
        conv = next(entry for entry in result["det_shapes"] if entry["operation"] == "Conv")
        self.assertEqual(conv["key"], "Conv 64->128-h32-w32-k3x3-s2x2")
        self.assertEqual(conv["nodes"], [2])
        self.assertAlmostEqual(conv["total_milliseconds"], 1.2)
        self.assertEqual(conv["samples"][0]["invocations"], 2)

    def test_render_markdown_contains_operation_and_sample(self) -> None:
        report = {
            "widths": [960], "workers": [1], "iterations": 1,
            "det_shapes": [{
                "key": "Conv 64->128-h32-w32-k3x3-s2x2",
                "nodes": [2], "total_milliseconds": 1.2,
                "samples": [{"workers": 1, "target_width": 960,
                             "milliseconds_per_invocation": 0.6}],
            }],
        }
        text = profile.render_markdown(report)
        self.assertIn("Conv 64->128-h32-w32-k3x3-s2x2", text)
        self.assertIn("1/960: 0.600", text)


if __name__ == "__main__":
    unittest.main()
