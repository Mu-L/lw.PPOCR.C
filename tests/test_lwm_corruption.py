from __future__ import annotations

import argparse
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from converter.lwm_v0 import CHECKSUM_OFFSET, fnv1a64


INSPECT: Path
MODEL: Path


def replace_checksum(data: bytearray) -> None:
    struct.pack_into("<Q", data, CHECKSUM_OFFSET, 0)
    struct.pack_into("<Q", data, CHECKSUM_OFFSET, fnv1a64(data))


class CorruptModelTests(unittest.TestCase):
    def run_variant(self, data: bytes, expected: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.lwm"
            path.write_bytes(data)
            result = subprocess.run(
                [str(INSPECT), str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(expected, result.stderr)

    def test_bad_magic(self) -> None:
        data = bytearray(MODEL.read_bytes())
        data[0] ^= 0xFF
        self.run_variant(data, "invalid_format")

    def test_truncated_file(self) -> None:
        self.run_variant(MODEL.read_bytes()[:-8], "out_of_bounds")

    def test_section_offset_out_of_bounds(self) -> None:
        data = bytearray(MODEL.read_bytes())
        struct.pack_into("<Q", data, 48, (1 << 64) - 8)
        self.run_variant(data, "out_of_bounds")

    def test_content_checksum_mismatch(self) -> None:
        data = bytearray(MODEL.read_bytes())
        data[-1] ^= 1
        self.run_variant(data, "checksum_mismatch")

    def test_tensor_index_out_of_bounds_with_valid_checksum(self) -> None:
        data = bytearray(MODEL.read_bytes())
        tensor_count = struct.unpack_from("<I", data, 16)[0]
        node_offset = struct.unpack_from("<Q", data, 56)[0]
        struct.pack_into("<I", data, node_offset + 8, tensor_count)
        replace_checksum(data)
        self.run_variant(data, "out_of_bounds")

    def test_header_reserved_field_is_rejected(self) -> None:
        data = bytearray(MODEL.read_bytes())
        struct.pack_into("<Q", data, 136, 1)
        self.run_variant(data, "invalid_format")

    def test_unsupported_header_flags_are_rejected(self) -> None:
        data = bytearray(MODEL.read_bytes())
        struct.pack_into("<I", data, 12, 0)
        replace_checksum(data)
        self.run_variant(data, "unsupported")

    def test_tensor_reserved_field_is_rejected(self) -> None:
        data = bytearray(MODEL.read_bytes())
        tensor_offset = struct.unpack_from("<Q", data, 48)[0]
        struct.pack_into("<I", data, tensor_offset + 44, 1)
        replace_checksum(data)
        self.run_variant(data, "invalid_format")

    def test_node_reserved_field_is_rejected(self) -> None:
        data = bytearray(MODEL.read_bytes())
        node_offset = struct.unpack_from("<Q", data, 56)[0]
        struct.pack_into("<I", data, node_offset + 68, 1)
        replace_checksum(data)
        self.run_variant(data, "invalid_format")

    def test_deterministic_mutations_fail_closed(self) -> None:
        original = MODEL.read_bytes()
        rng = random.Random(0x4C574C4D)
        offsets = [0, 1, 16, 24, 48, 56, 64, 72, CHECKSUM_OFFSET]
        offsets.extend(rng.randrange(len(original)) for _ in range(16))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.lwm"
            for index, offset in enumerate(offsets):
                data = bytearray(original)
                data[offset] ^= 1 << (index % 8)
                path.write_bytes(data)
                result = subprocess.run(
                    [str(INSPECT), str(path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=30,
                )
                self.assertGreaterEqual(result.returncode, 0, f"mutation {offset} crashed")
                self.assertNotEqual(result.returncode, 0, f"mutation {offset} accepted")
                self.assertTrue(result.stderr.strip(), f"mutation {offset} had no diagnostic")

    def test_utf8_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "中文模型.lwm"
            path.write_bytes(MODEL.read_bytes())
            result = subprocess.run(
                [str(INSPECT), str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"nodes":159', result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    INSPECT = args.inspect
    MODEL = args.model
    unittest.main(argv=[__file__, *remaining])
