#!/usr/bin/env python3
"""Convert the pinned PP-OCRv6 Medium DET graph at fixed or dynamic shapes.

This is an analysis-only wrapper around the existing DET lowering. It pins the
Medium asset identity and never changes the production converters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.lwm_v0 import _prepare_det_model, _write_model
from converter.ppocr_contracts import PP_OCRV6_MEDIUM_DET_SHA256
from tools.convert_small_det_experimental import concretize_shapes, transfer_shapes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(
    model_path: Path, height: int, width: int, dynamic: bool, output_path: Path,
    runtime_status: str = "analysis-only",
) -> dict[str, object]:
    digest = sha256(model_path)
    if digest != PP_OCRV6_MEDIUM_DET_SHA256:
        raise ValueError(
            "Medium DET conversion requires the pinned asset; "
            f"expected {PP_OCRV6_MEDIUM_DET_SHA256}, got {digest}"
        )
    source = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(source, full_check=True)
    lowered, _ = _prepare_det_model(source)
    if dynamic:
        static_model = lowered
        inferred = onnx.shape_inference.infer_shapes(source, strict_mode=True, data_prop=False)
    else:
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        source_static = concretize_shapes(source, height, width)
        static_model = transfer_shapes(lowered, source_static)
        inferred = static_model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = _write_model(static_model, output_path, inferred)
    return {
        "schema_version": 1,
        "tool": "tools/convert_medium_det.py" if runtime_status == "production" else "tools/convert_medium_det_experimental.py",
        "status": runtime_status if runtime_status == "production" else ("dynamic-analysis-only" if dynamic else "analysis-only"),
        "model": str(model_path),
        "model_sha256": digest,
        "height": None if dynamic else height,
        "width": None if dynamic else width,
        "dynamic": dynamic,
        "output": str(output_path),
        "conversion": {
            **info.__dict__,
            "checksum": f"0x{info.checksum:016x}",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--runtime-status",
        choices=("analysis-only", "production"),
        default="analysis-only",
        help="status recorded in the conversion report (production is used by the stable wrapper)",
    )
    args = parser.parse_args(argv)
    if not args.dynamic and (args.height is None or args.width is None):
        parser.error("--height and --width are required unless --dynamic is used")
    report = convert(
        args.model,
        args.height or 0,
        args.width or 0,
        args.dynamic,
        args.output,
        args.runtime_status,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
