from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tools import collect_rec_shape_profile as profile


class CollectRecShapeProfileTests(unittest.TestCase):
    def test_collect_groups_only_pointwise_convolutions(self) -> None:
        reports = {
            192: {
                "width": 192,
                "conv_nodes": [
                    {
                        "node": 7,
                        "nanoseconds": 1200000,
                        "invocations": 2,
                        "input": [1, 96, 6, 48],
                        "weights": [192, 96, 1, 1],
                        "output": [1, 192, 6, 48],
                        "group": 1,
                        "kernel": [1, 1],
                        "strides": [1, 1],
                    },
                    {
                        "node": 8,
                        "nanoseconds": 9999999,
                        "invocations": 2,
                        "input": [1, 96, 6, 48],
                        "weights": [96, 96, 3, 3],
                        "output": [1, 96, 6, 48],
                        "group": 1,
                        "kernel": [3, 3],
                        "strides": [1, 1],
                    },
                ],
            }
        }

        def fake_run(driver: Path, model: Path, width: int, iterations: int) -> dict:
            self.assertEqual(width, 192)
            self.assertEqual(iterations, 1)
            return reports[width]

        with patch.object(profile, "run_profile", side_effect=fake_run):
            result = profile.collect(Path("driver"), Path("rec.lwm"), (192,), 1)

        self.assertEqual(len(result["conv1x1_shapes"]), 1)
        entry = result["conv1x1_shapes"][0]
        self.assertEqual(entry["key"], "96x192-h6-w48")
        self.assertEqual(entry["nodes"], [7])
        self.assertAlmostEqual(entry["total_milliseconds"], 1.2)

    def test_render_markdown_contains_shape_and_samples(self) -> None:
        report = {
            "model": "rec.lwm",
            "widths": [192],
            "iterations": 1,
            "conv1x1_shapes": [
                {
                    "key": "96x192-h6-w48",
                    "nodes": [7],
                    "total_milliseconds": 1.2,
                    "samples": [{"width": 192, "milliseconds_per_invocation": 0.6}],
                }
            ],
        }
        text = profile.render_markdown(report)
        self.assertIn("96x192-h6-w48", text)
        self.assertIn("192: 0.600", text)


if __name__ == "__main__":
    unittest.main()
