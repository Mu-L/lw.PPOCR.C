#!/usr/bin/env python3
"""Convert the pinned PP-OCRv6 Medium REC graph at fixed or dynamic widths.

This is an analysis-only converter. It materializes the Medium Shape/Slice
metadata for one input width, or applies the narrowly verified dynamic
metadata lowering, then writes LWM v0.1 and emits a report that must be checked
against ONNX Runtime. It does not change the production model package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.lwm_v0 import _materialize_slice_inputs, _write_model
from converter.ppocr_contracts import PP_OCRV6_MEDIUM_REC_SHA256, PP_OCRV6_REC_WIDTHS
from tools.convert_small_rec_experimental import (
    concretize_shapes,
    dynamic_shape_inference,
    lower_dynamic_metadata,
    normalize_padding,
)
from tools.probe_rec_shape_metadata import staticize


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(
    model_path: Path, width: int | None, dynamic: bool, output_path: Path,
    runtime_status: str = "analysis-only",
) -> dict[str, object]:
    digest = sha256(model_path)
    if digest != PP_OCRV6_MEDIUM_REC_SHA256:
        raise ValueError(
            "Medium REC conversion requires the pinned asset; "
            f"expected {PP_OCRV6_MEDIUM_REC_SHA256}, got {digest}"
        )
    if not dynamic and width not in PP_OCRV6_REC_WIDTHS:
        raise ValueError(f"Medium REC width must be one of {PP_OCRV6_REC_WIDTHS}")
    source = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(source, full_check=True)
    static_report = None
    if dynamic:
        inferred = dynamic_shape_inference(
            source,
            widths=(320, 640),
            width_symbol="LW_MEDIUM_REC_WIDTH",
        )
        lowered = normalize_padding(_materialize_slice_inputs(lower_dynamic_metadata(source)))
        input_dims = [
            dimension.dim_param or dimension.dim_value
            for dimension in inferred.graph.input[0].type.tensor_type.shape.dim
        ]
        output_dims = [
            dimension.dim_param or dimension.dim_value
            for dimension in inferred.graph.output[0].type.tensor_type.shape.dim
        ]
        if input_dims != [1, 3, 48, "LW_MEDIUM_REC_WIDTH"] or output_dims != [
            1,
            "LW_MEDIUM_REC_WIDTH",
            18710,
        ]:
            raise ValueError(
                "Medium dynamic REC shape contract changed: "
                f"input={input_dims!r}, output={output_dims!r}"
            )
        width = None
    else:
        with tempfile.TemporaryDirectory(prefix="lw-medium-rec-") as directory:
            static_path = Path(directory) / "static.onnx"
            static_report = staticize(model_path, width, static_path)
            static_model = concretize_shapes(
                onnx.load(str(static_path), load_external_data=True), width
            )
            lowered = normalize_padding(_materialize_slice_inputs(static_model))
            inferred = static_model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = _write_model(lowered, output_path, inferred)
    return {
        "schema_version": 1,
        "tool": "tools/convert_medium_rec.py" if runtime_status == "production" else "tools/convert_medium_rec_experimental.py",
        "status": runtime_status if runtime_status == "production" else ("dynamic-analysis-only" if dynamic else "analysis-only"),
        "model": str(model_path),
        "model_sha256": digest,
        "width": width,
        "dynamic": dynamic,
        "dynamic_width_symbol": "LW_MEDIUM_REC_WIDTH" if dynamic else None,
        "output": str(output_path),
        "staticization": static_report,
        "conversion": {
            **info.__dict__,
            "checksum": f"0x{info.checksum:016x}",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--width", type=int, choices=PP_OCRV6_REC_WIDTHS)
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
    report = convert(args.model, args.width, args.dynamic, args.output, args.runtime_status)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
