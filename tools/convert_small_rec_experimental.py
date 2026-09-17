#!/usr/bin/env python3
"""Convert an analysis-only PP-OCRv6 Small REC prototype to LWM.

This is intentionally an analysis tool, not a release converter. The default
mode materializes Shape-derived metadata at one fixed input width.
The optional ``--dynamic`` mode removes the narrow Shape/Slice metadata
subgraph used only as Reshape control and retains one unresolved output
dimension for the runtime to infer. Both modes are experimental and must be
compared with ONNX Runtime before any model support claim is made.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, shape_inference

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.lwm_v0 import _materialize_slice_inputs, _normalize_same_upper, _write_model
from converter.ppocr_contracts import PP_OCRV6_REC_WIDTHS, PP_OCRV6_SMALL_REC_SHA256
from tools.probe_rec_shape_metadata import _instrument, metadata_nodes, staticize
from tools.probe_rec_shape_metadata import probe
from tools.validate_small_rec_dynamic_rule import validate_report


def normalize_padding(model: onnx.ModelProto) -> onnx.ModelProto:
    converted = copy.deepcopy(model)
    rewritten = []
    for node in converted.graph.node:
        if node.op_type in ("Conv", "MaxPool"):
            rewritten.append(_normalize_same_upper(node))
        else:
            rewritten.append(copy.deepcopy(node))
    del converted.graph.node[:]
    converted.graph.node.extend(rewritten)
    return converted


def lower_dynamic_metadata(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove only the Small REC Shape/Slice values used as Reshape controls.

    This is intentionally a graph-specific lowering. It refuses to remove a
    metadata value consumed by an ordinary numeric operator, which prevents a
    future model graph from silently taking this experimental path.
    """
    converted = copy.deepcopy(model)
    metadata_outputs = {
        output
        for node in metadata_nodes(converted)
        for output in node["outputs"]
    }
    control_values = set(metadata_outputs)
    changed = True
    while changed:
        changed = False
        for node in converted.graph.node:
            if node.op_type in ("Identity", "Squeeze", "Concat", "Unsqueeze", "Cast") and any(
                name in control_values for name in node.input if name
            ):
                for output in node.output:
                    if output and output not in control_values:
                        control_values.add(output)
                        changed = True

    retained_nodes = []
    for node in converted.graph.node:
        if node.op_type in (
            "Identity", "Squeeze", "Concat", "Unsqueeze", "Cast", "Shape", "Slice"
        ) and any(output in control_values for output in node.output if output):
            continue
        control_inputs = [name for name in node.input if name in control_values]
        if node.op_type == "Reshape" and control_inputs:
            if len(node.input) < 2 or node.input[1] not in control_values:
                raise ValueError(f"unexpected dynamic Reshape controls: {node.name}")
            replacement = copy.deepcopy(node)
            del replacement.input[1:]
            retained_nodes.append(replacement)
            continue
        if control_inputs:
            if node.op_type in (
                "Identity", "Squeeze", "Concat", "Unsqueeze", "Cast", "Shape", "Slice"
            ):
                continue
            raise ValueError(
                f"dynamic metadata value reaches numeric operator {node.op_type} ({node.name})"
            )
        retained_nodes.append(copy.deepcopy(node))
    del converted.graph.node[:]
    converted.graph.node.extend(retained_nodes)

    # Specialize only batch=1, while retaining the symbolic REC width. The
    # production recognizer already supplies batch-one tensors.
    input_shape = converted.graph.input[0].type.tensor_type.shape
    input_shape.dim[0].ClearField("dim_param")
    input_shape.dim[0].dim_value = 1
    return converted


def dynamic_shape_inference(
    model: onnx.ModelProto,
    widths: tuple[int, int] = (320, 640),
    width_symbol: str = "LW_SMALL_REC_WIDTH",
) -> onnx.ModelProto:
    """Attach observed shapes while marking only width-varying axes dynamic."""
    inferred = copy.deepcopy(model)
    output_names = list(dict.fromkeys(
        output for node in inferred.graph.node for output in node.output if output
    ))
    output_ops = {
        output: node.op_type
        for node in inferred.graph.node
        for output in node.output
        if output
    }
    instrumented = _instrument(inferred, output_names)
    session = ort.InferenceSession(
        instrumented.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    observed: dict[str, list[tuple[int, ...]]] = {name: [] for name in output_names}
    for width in widths:
        sample = np.zeros((1, 3, 48, width), dtype=np.float32)
        values = session.run(output_names, {session.get_inputs()[0].name: sample})
        for name, value in zip(output_names, values):
            observed[name].append(tuple(np.asarray(value).shape))
    for name, shapes in observed.items():
        if len(set(shapes)) > 1 and any(len(shape) != len(shapes[0]) for shape in shapes):
            raise ValueError(f"dynamic shape rank changed for {name}: {shapes}")
        varying_axes = [
            axis
            for axis, dimension in enumerate(shapes[0])
            if any(shape[axis] != dimension for shape in shapes[1:])
        ]
        if output_ops.get(name) == "Reshape" and len(varying_axes) > 1:
            raise ValueError(
                f"dynamic Reshape shape has more than one varying axis for {name}: {shapes}"
            )
    value_info = {
        value.name: value
        for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
    }
    for name, shapes in observed.items():
        value = value_info.get(name)
        if value is None:
            continue
        del value.type.tensor_type.shape.dim[:]
        for axis, dimension in enumerate(shapes[0]):
            dimension_info = value.type.tensor_type.shape.dim.add()
            if any(shape[axis] != dimension for shape in shapes[1:]):
                dimension_info.dim_param = width_symbol
            else:
                dimension_info.dim_value = int(dimension)
    input_shape = inferred.graph.input[0].type.tensor_type.shape
    input_shape.dim[0].ClearField("dim_param")
    input_shape.dim[0].dim_value = 1
    input_shape.dim[3].ClearField("dim_value")
    input_shape.dim[3].dim_param = width_symbol
    return inferred


def concretize_shapes(model: onnx.ModelProto, width: int) -> onnx.ModelProto:
    """Attach runtime-observed shapes so the fixed-width LWM plan is bounded."""
    concretized = copy.deepcopy(model)
    input_name = concretized.graph.input[0].name
    output_names = list(dict.fromkeys(
        output for node in concretized.graph.node for output in node.output if output
    ))
    instrumented = _instrument(concretized, output_names)
    sample = np.random.default_rng(0).standard_normal((1, 3, 48, width), dtype=np.float32)
    values = ort.InferenceSession(
        instrumented.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(output_names, {input_name: sample})
    values_by_name = dict(zip(output_names, values))
    shapes = {name: np.asarray(value).shape for name, value in values_by_name.items()}
    input_shape = concretized.graph.input[0].type.tensor_type.shape
    for axis, dimension in enumerate((1, 3, 48, width)):
        input_shape.dim[axis].ClearField("dim_param")
        input_shape.dim[axis].dim_value = dimension
    for value in (*concretized.graph.value_info, *concretized.graph.output):
        shape = shapes.get(value.name)
        if shape is None:
            continue
        del value.type.tensor_type.shape.dim[:]
        for dimension in shape:
            value.type.tensor_type.shape.dim.add(dim_value=int(dimension))
    value_info = {
        value.name: value
        for value in (
            *concretized.graph.input,
            *concretized.graph.value_info,
            *concretized.graph.output,
        )
    }
    constant_names = {item.name for item in concretized.graph.initializer}
    retained_nodes = []
    folded_initializers = []
    for node in concretized.graph.node:
        outputs = [name for name in node.output if name]
        can_fold = bool(outputs) and all(
            name in constant_names for name in node.input if name
        )
        can_fold = can_fold and all(
            value_info.get(name) is not None
            and value_info[name].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            for name in outputs
        )
        if not can_fold:
            retained_nodes.append(node)
            continue
        for name in outputs:
            folded_initializers.append(numpy_helper.from_array(
                np.asarray(values_by_name[name]), name=name
            ))
            constant_names.add(name)
    del concretized.graph.node[:]
    concretized.graph.node.extend(retained_nodes)
    concretized.graph.initializer.extend(folded_initializers)
    return concretized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--width", type=int)
    mode.add_argument("--dynamic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--runtime-status",
        choices=("analysis-only", "production"),
        default="analysis-only",
        help="status recorded in the conversion report (production is used by the stable wrapper)",
    )
    args = parser.parse_args(argv)
    if args.width is not None and args.width <= 0:
        raise SystemExit("--width must be positive")

    static_report = None
    dynamic_contract = None
    if args.dynamic:
        digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
        if digest != PP_OCRV6_SMALL_REC_SHA256:
            raise SystemExit(
                "dynamic Small REC conversion requires the pinned PP-OCRv6 Small asset; "
                f"expected SHA-256 {PP_OCRV6_SMALL_REC_SHA256}, got {digest}"
            )
        dynamic_contract = validate_report(probe(args.model, list(PP_OCRV6_REC_WIDTHS)))
        source = onnx.load(str(args.model), load_external_data=True)
        inferred = dynamic_shape_inference(source)
        model = normalize_padding(_materialize_slice_inputs(lower_dynamic_metadata(source)))
        width = None
    else:
        with tempfile.TemporaryDirectory(prefix="lw-small-rec-") as directory:
            static_path = Path(directory) / "static.onnx"
            static_report = staticize(args.model, args.width, static_path)
            static_model = concretize_shapes(
                onnx.load(str(static_path), load_external_data=True), args.width
            )
            model = normalize_padding(_materialize_slice_inputs(static_model))
            inferred = static_model
            width = args.width
    args.output.parent.mkdir(parents=True, exist_ok=True)
    info = _write_model(model, args.output, inferred)

    report_status = args.runtime_status
    if report_status == "analysis-only" and args.dynamic:
        report_status = "dynamic-analysis-only"
    report = {
        "schema_version": 1,
        "tool": "tools/convert_small_rec.py" if args.runtime_status == "production" else "tools/convert_small_rec_experimental.py",
        "status": report_status,
        "model": str(args.model),
        "width": width,
        "dynamic": bool(args.dynamic),
        "output": str(args.output),
        "staticization": static_report,
        "dynamic_contract": dynamic_contract,
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
