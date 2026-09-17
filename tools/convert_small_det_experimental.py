#!/usr/bin/env python3
"""Convert a fixed-shape PP-OCRv6 Small DET analysis prototype to LWM."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.lwm_v0 import _prepare_det_model, _write_model
from tools.probe_rec_shape_metadata import _instrument


def concretize_shapes(model: onnx.ModelProto, height: int, width: int) -> onnx.ModelProto:
    """Attach shapes observed from one deterministic fixed-shape ORT run."""
    concretized = copy.deepcopy(model)
    input_name = concretized.graph.input[0].name
    output_names = list(dict.fromkeys(
        output for node in concretized.graph.node for output in node.output if output
    ))
    instrumented = _instrument(concretized, output_names)
    sample = np.random.default_rng(0).standard_normal(
        (1, 3, height, width), dtype=np.float32
    )
    values = ort.InferenceSession(
        instrumented.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(output_names, {input_name: sample})
    values_by_name = dict(zip(output_names, values))
    shapes = {name: np.asarray(value).shape for name, value in values_by_name.items()}
    input_shape = concretized.graph.input[0].type.tensor_type.shape
    for axis, dimension in enumerate((1, 3, height, width)):
        input_shape.dim[axis].ClearField("dim_param")
        input_shape.dim[axis].dim_value = dimension
    for value in (*concretized.graph.value_info, *concretized.graph.output):
        shape = shapes.get(value.name)
        if shape is None:
            continue
        del value.type.tensor_type.shape.dim[:]
        for dimension in shape:
            value.type.tensor_type.shape.dim.add(dim_value=int(dimension))
    return concretized


def transfer_shapes(model: onnx.ModelProto, shape_source: onnx.ModelProto) -> onnx.ModelProto:
    """Copy concrete value shapes onto an internally lowered graph."""
    source_values = {
        value.name: value
        for value in (*shape_source.graph.input, *shape_source.graph.value_info,
                      *shape_source.graph.output)
    }
    transferred = copy.deepcopy(model)
    for value in (*transferred.graph.input, *transferred.graph.value_info,
                  *transferred.graph.output):
        source = source_values.get(value.name)
        if source is None or not source.type.tensor_type.HasField("shape"):
            continue
        del value.type.tensor_type.shape.dim[:]
        for dimension in source.type.tensor_type.shape.dim:
            value.type.tensor_type.shape.dim.add(
                dim_value=int(dimension.dim_value) if dimension.HasField("dim_value") else -1
            )
    return transferred


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument(
        "--dynamic", action="store_true",
        help="keep dynamic spatial dimensions for an experimental OCR-pipeline graph",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--runtime-status",
        choices=("analysis-only", "production"),
        default="analysis-only",
        help="status recorded in the conversion report (production is used by the stable wrapper)",
    )
    args = parser.parse_args(argv)
    if args.height <= 0 or args.width <= 0:
        raise SystemExit("--height and --width must be positive")

    source = onnx.load(str(args.model), load_external_data=True)
    onnx.checker.check_model(source, full_check=True)
    converted, _ = _prepare_det_model(source)
    if args.dynamic:
        # The lowered Resize form is an internal LWM representation, so it is
        # intentionally not passed through ONNX shape inference again.
        static_model = converted
        inferred = onnx.shape_inference.infer_shapes(
            source, strict_mode=True, data_prop=False
        )
    else:
        source_static = concretize_shapes(source, args.height, args.width)
        static_model = transfer_shapes(converted, source_static)
        inferred = static_model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    info = _write_model(static_model, args.output, inferred)
    report_status = args.runtime_status
    if report_status == "analysis-only" and args.dynamic:
        report_status = "dynamic-analysis-only"
    report = {
        "schema_version": 1,
        "tool": "tools/convert_small_det.py" if args.runtime_status == "production" else "tools/convert_small_det_experimental.py",
        "status": report_status,
        "model": str(args.model),
        "height": args.height,
        "width": args.width,
        "dynamic": args.dynamic,
        "output": str(args.output),
        "conversion": {
            **info.__dict__,
            "checksum": f"0x{info.checksum:016x}",
        },
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
