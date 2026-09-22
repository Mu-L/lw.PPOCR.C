from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort


def make_bgr_source(width: int, height: int, stride: int) -> tuple[np.ndarray, bytes]:
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    raw = bytearray([0xEE] * (stride * height))
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                value = (y * 53 + x * 29 + channel * 71 + 17) % 256
                pixels[y, x, channel] = value
                raw[y * stride + x * 3 + channel] = value
    return pixels, bytes(raw)


def round_to_short(value: float) -> int:
    """CvRoundToShort: round-half-even to an int16 coefficient."""
    rounded = math.floor(value + 0.5)
    if rounded - value == 0.5 and int(rounded) & 1:
        rounded -= 1.0
    return int(rounded)


def preprocess_reference(source: np.ndarray) -> tuple[np.ndarray, int]:
    """OpenCV-parity fixed-point keep-aspect resize + RecNorm LUT + -1 pad."""
    source_height, source_width, _ = source.shape
    window_width = source_width
    if 80 * window_width > 320 * source_height:
        window_width = (320 * source_height) // 80
    actual_width = min(160, (80 * window_width + source_height - 1) // source_height)
    scale = 2048
    offsets = np.empty(actual_width, dtype=np.int32)
    coefficients = np.empty((actual_width, 2), dtype=np.int32)
    for x in range(actual_width):
        coordinate = (x + 0.5) * window_width / actual_width - 0.5
        source_index = int(np.floor(coordinate))
        fraction = coordinate - source_index
        if source_index < 0:
            source_index = 0
            fraction = 0.0
        if source_index >= window_width - 1:
            source_index = window_width - 1
            fraction = 0.0
        offsets[x] = source_index
        coefficients[x, 0] = round_to_short((1.0 - fraction) * scale)
        coefficients[x, 1] = round_to_short(fraction * scale)
    output = np.full((3, 80, 160), -1.0, dtype=np.float32)
    for output_y in range(80):
        coordinate = (output_y + 0.5) * source_height / 80.0 - 0.5
        source_y = int(np.floor(coordinate))
        fraction = coordinate - source_y
        beta0 = round_to_short((1.0 - fraction) * scale)
        beta1 = round_to_short(fraction * scale)
        source_y0 = min(max(source_y, 0), source_height - 1)
        source_y1 = min(max(source_y + 1, 0), source_height - 1)
        rows = []
        for row_index in (source_y0, source_y1):
            row = np.empty(actual_width * 3, dtype=np.int32)
            for x in range(actual_width):
                sx = offsets[x]
                sx1 = min(sx + 1, window_width - 1)
                c0 = coefficients[x, 0]
                c1 = coefficients[x, 1]
                for channel in range(3):
                    row[x * 3 + channel] = (
                        int(source[row_index, sx, channel]) * c0
                        + int(source[row_index, sx1, channel]) * c1
                    )
            rows.append(row)
        for x in range(actual_width):
            for channel in range(3):
                h0 = rows[0][x * 3 + channel]
                h1 = rows[1][x * 3 + channel]
                value = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2
                value = min(max(value, 0), 255)
                output[channel, output_y, x] = np.float32(value * (2.0 / 255.0) - 1.0)
    return output, actual_width

class ClsPipelineReferenceTest(unittest.TestCase):
    driver: Path
    lwm_model: Path
    onnx_model: Path

    def run_driver(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.driver), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )

    def test_preprocess_and_public_api_match_references(self) -> None:
        width, height, stride = 7, 5, 24
        pixels, raw = make_bgr_source(width, height, stride)
        expected_input, expected_width = preprocess_reference(pixels)
        reference = ort.InferenceSession(
            str(self.onnx_model), providers=["CPUExecutionProvider"]
        )
        probabilities = reference.run(
            None,
            {reference.get_inputs()[0].name: expected_input[np.newaxis, ...]},
        )[0][0]
        expected_label = int(np.argmax(probabilities))
        expected_score = float(probabilities[expected_label])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.bgr"
            output_path = root / "preprocessed.f32"
            source_path.write_bytes(raw)
            preprocessed = self.run_driver(
                [
                    "preprocess",
                    str(source_path),
                    str(width),
                    str(height),
                    str(stride),
                    str(output_path),
                ]
            )
            self.assertEqual(
                preprocessed.returncode, 0, preprocessed.stdout + preprocessed.stderr
            )
            actual_input = np.fromfile(output_path, dtype="<f4").reshape(
                expected_input.shape
            )
            unicode_dir = root / "分类模型"
            unicode_dir.mkdir()
            unicode_model = unicode_dir / "方向分类.lwm"
            shutil.copyfile(self.lwm_model, unicode_model)
            classified = self.run_driver(
                [
                    "pipeline",
                    str(unicode_model),
                    str(source_path),
                    str(width),
                    str(height),
                    str(stride),
                ]
            )
        np.testing.assert_allclose(actual_input, expected_input, rtol=0.0, atol=1.0e-6)
        self.assertIn(f"resized_width={expected_width}", preprocessed.stdout)
        self.assertEqual(classified.returncode, 0, classified.stdout + classified.stderr)
        match = re.search(
            r"label=(\d+) score=([^ ]+) orientation=(\d+) resized_width=(\d+)",
            classified.stdout,
        )
        self.assertIsNotNone(match, classified.stdout)
        assert match is not None
        self.assertEqual(int(match.group(1)), expected_label)
        self.assertAlmostEqual(float(match.group(2)), expected_score, places=5)
        self.assertEqual(int(match.group(3)), expected_label * 180)
        self.assertEqual(int(match.group(4)), expected_width)

    def test_short_line_uses_full_source_without_padding(self) -> None:
        width, height, stride = 14, 8, 14 * 3
        pixels, _ = make_bgr_source(width, height, stride)
        expected, expected_width = preprocess_reference(pixels)
        # Keep-aspect: a 14x8 crop maps to ceil(80*14/8)=140 columns and the
        # trailing 20 columns stay at the -1 padding value.
        self.assertEqual(expected_width, 140)
        self.assertEqual(expected.shape, (3, 80, 160))
        self.assertTrue(np.all(np.isfinite(expected[:, :, :expected_width])))
        self.assertTrue(np.all(expected[:, :, expected_width:] == -1.0))

    def test_left_four_h_window_ignores_tail_pixels(self) -> None:
        width, height, stride = 64, 8, 64 * 3
        pixels, raw = make_bgr_source(width, height, stride)
        altered = bytearray(raw)
        for y in range(height):
            for x in range(4 * height, width):
                for channel in range(3):
                    altered[y * stride + x * 3 + channel] ^= 0xFF
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.bgr"
            second_path = root / "second.bgr"
            first_output = root / "first.f32"
            second_output = root / "second.f32"
            first_path.write_bytes(raw)
            second_path.write_bytes(altered)
            for source_path, output_path in (
                (first_path, first_output),
                (second_path, second_output),
            ):
                result = self.run_driver(
                    [
                        "preprocess",
                        str(source_path),
                        str(width),
                        str(height),
                        str(stride),
                        str(output_path),
                    ]
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            np.testing.assert_array_equal(
                np.fromfile(first_output, dtype="<f4"),
                np.fromfile(second_output, dtype="<f4"),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--lwm-model", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    ClsPipelineReferenceTest.driver = arguments.driver
    ClsPipelineReferenceTest.lwm_model = arguments.lwm_model
    ClsPipelineReferenceTest.onnx_model = arguments.onnx_model
    unittest.main(argv=[__file__], verbosity=2)
