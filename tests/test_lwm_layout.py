from __future__ import annotations

import argparse
import json
import struct
import unittest
from pathlib import Path


def fnv1a64(data: bytes, checksum_offset: int, checksum_size: int) -> int:
    value = 0xCBF29CE484222325
    for index, byte in enumerate(data):
        if checksum_offset <= index < checksum_offset + checksum_size:
            byte = 0
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


class LwmLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(ARGUMENTS.manifest.read_text(encoding="utf-8"))
        self.operator_parameters = self.manifest["operator_parameters"]

    def test_manifest_schema(self) -> None:
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(self.manifest["format"], "LWM")
        self.assertEqual(self.manifest["format_major"], 0)
        self.assertEqual(self.manifest["format_minor"], 1)
        self.assertEqual(len(self.operator_parameters), 25)

    def test_model_layout(self) -> None:
        for model_path in ARGUMENTS.models:
            with self.subTest(model=model_path.name):
                self.check_model(model_path)

    def check_model(self, model_path: Path) -> None:
        data = model_path.read_bytes()
        header_size = self.manifest["header_size"]
        tensor_size = self.manifest["tensor_record_size"]
        node_size = self.manifest["node_record_size"]
        alignment = self.manifest["section_alignment"]
        checksum_offset = self.manifest["checksum_offset"]
        checksum_size = self.manifest["checksum_size"]
        self.assertGreaterEqual(len(data), header_size)
        self.assertEqual(data[:4], b"LWM0")
        self.assertEqual(struct.unpack_from("<H", data, 4)[0], self.manifest["format_major"])
        self.assertEqual(struct.unpack_from("<H", data, 6)[0], self.manifest["format_minor"])
        self.assertEqual(struct.unpack_from("<I", data, 8)[0], header_size)
        self.assertEqual(struct.unpack_from("<I", data, 12)[0], self.manifest["header_flags"]["no_memory_plan"])
        tensor_count, node_count = struct.unpack_from("<II", data, 16)
        input_count, output_count = struct.unpack_from("<II", data, 24)
        self.assertGreater(input_count, 0)
        self.assertGreater(output_count, 0)
        input_offset, output_offset, tensor_offset, node_offset = struct.unpack_from(
            "<QQQQ", data, 32
        )
        param_offset, param_size, string_offset, string_size = struct.unpack_from(
            "<QQQQ", data, 64
        )
        weight_offset, weight_size, file_size, workspace_size = struct.unpack_from(
            "<QQQQ", data, 96
        )
        stored_checksum = struct.unpack_from("<Q", data, checksum_offset)[0]
        self.assertEqual(file_size, len(data))
        self.assertEqual(workspace_size, 0)
        self.assertNotEqual(stored_checksum, 0)
        self.assertEqual(
            stored_checksum,
            fnv1a64(data, checksum_offset, checksum_size),
        )

        offsets = (input_offset, output_offset, tensor_offset, node_offset, param_offset, string_offset, weight_offset)
        for offset in offsets:
            self.assertEqual(offset % alignment, 0)
            self.assertGreaterEqual(offset, header_size)
        self.assertEqual(string_size, 0)
        self.assertLessEqual(input_offset + input_count * 4, output_offset)
        self.assertLessEqual(output_offset + output_count * 4, tensor_offset)
        self.assertLessEqual(tensor_offset + tensor_count * tensor_size, node_offset)
        self.assertLessEqual(node_offset + node_count * node_size, param_offset)
        self.assertLessEqual(param_offset + param_size, string_offset)
        self.assertLessEqual(string_offset + string_size, weight_offset)
        self.assertLessEqual(weight_offset + weight_size, file_size)

        max_inputs = self.manifest["limits"]["max_node_inputs"]
        max_outputs = self.manifest["limits"]["max_node_outputs"]
        for index in range(node_count):
            node = node_offset + index * node_size
            op, input_arity, output_arity, flags = struct.unpack_from("<HHHH", data, node)
            self.assertIn(str(op), self.operator_parameters)
            self.assertLessEqual(input_arity, max_inputs)
            self.assertGreater(output_arity, 0)
            self.assertLessEqual(output_arity, max_outputs)
            self.assertEqual(flags, 0)
            self.assertEqual(struct.unpack_from("<I", data, node + 68)[0], 0)
            parameter_offset = struct.unpack_from("<Q", data, node + 56)[0]
            parameter_size = struct.unpack_from("<I", data, node + 64)[0]
            expected_size = self.operator_parameters[str(op)]["size"]
            self.assertEqual(parameter_size, expected_size)
            if expected_size == 0:
                self.assertEqual(parameter_offset, 0)
            else:
                self.assertEqual(parameter_offset % alignment, 0)
                self.assertGreaterEqual(parameter_offset, param_offset)
                self.assertLessEqual(parameter_offset + parameter_size, param_offset + param_size)
                self.assertEqual(struct.unpack_from("<H", data, parameter_offset)[0], 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", dest="models", type=Path, action="append", required=True)
    return parser.parse_args()


ARGUMENTS = parse_args()

if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
