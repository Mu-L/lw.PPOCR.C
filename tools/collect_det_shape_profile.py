#!/usr/bin/env python3
"""Collect resolved DET Conv/ConvTranspose shapes from full-OCR profiles.

This tool is diagnostic only. It runs the existing full-ocr-profile-driver and
aggregates real detector geometry, call counts, and elapsed time. It never
changes runtime dispatch or model assets.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

DEFAULT_WIDTHS = (960,)
DEFAULT_WORKERS = (1,)


def run_profile(
    driver: Path,
    det_model: Path,
    cls_model: Path,
    rec_model: Path,
    dictionary: Path,
    image: Path,
    iterations: int,
    workers: int,
    target_width: int,
    det_threads: int | None = None,
) -> dict[str, Any]:
    command = [
        str(driver), str(det_model), str(cls_model), str(rec_model),
        str(dictionary), str(image), str(iterations), str(workers),
        str(target_width),
    ]
    if det_threads is not None:
        command.append(str(det_threads))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"DET profile failed for workers={workers}, width={target_width}:\n"
            f"{completed.stdout[-3000:]}\n{completed.stderr[-3000:]}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("DET profile produced no JSON")
    try:
        report = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError("DET profile output is not JSON") from error
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise RuntimeError("unsupported full OCR profile schema")
    if report.get("workers") != workers or report.get("rec_target_width") != target_width:
        raise RuntimeError("full OCR profile identity does not match requested case")
    nodes = report.get("det_convolution_nodes")
    if not isinstance(nodes, list):
        raise RuntimeError("full OCR profile lacks det_convolution_nodes")
    return report


def _shape_key(node: dict[str, Any]) -> str:
    fields = (
        "operation", "input", "weights", "output", "group", "kernel",
        "strides", "dilations", "pads",
    )
    return json.dumps({field: node.get(field) for field in fields}, sort_keys=True, separators=(",", ":"))


def _display_key(node: dict[str, Any]) -> str:
    input_shape = node["input"]
    output_shape = node["output"]
    kernel = node["kernel"]
    strides = node["strides"]
    return (
        f"{node['operation']} {input_shape[1]}->{output_shape[1]}"
        f"-h{input_shape[2]}-w{input_shape[3]}"
        f"-k{kernel[0]}x{kernel[1]}-s{strides[0]}x{strides[1]}"
    )


def collect(
    driver: Path,
    det_model: Path,
    cls_model: Path,
    rec_model: Path,
    dictionary: Path,
    image: Path,
    widths: Iterable[int],
    workers: Iterable[int],
    iterations: int,
    det_threads: int | None = None,
) -> dict[str, Any]:
    widths = tuple(widths)
    workers = tuple(workers)
    grouped: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    for worker_count in workers:
        for width in widths:
            report = run_profile(
                driver, det_model, cls_model, rec_model, dictionary, image,
                iterations, worker_count, width, det_threads,
            )
            cases.append({
                "workers": worker_count,
                "target_width": width,
                "output_checksum": report.get("output_checksum"),
            })
            for node in report["det_convolution_nodes"]:
                if not isinstance(node, dict):
                    continue
                required = ("input", "weights", "output", "group", "kernel", "strides", "dilations", "pads")
                if any(field not in node for field in required):
                    continue
                if not all(isinstance(node[field], list) for field in ("input", "weights", "output")):
                    continue
                key = _shape_key(node)
                item = grouped.setdefault(key, {
                    "key": _display_key(node),
                    "operation": node["operation"],
                    "shape": {field: node[field] for field in required},
                    "nodes": [],
                    "samples": [],
                })
                item["nodes"].append(int(node["node"]))
                invocations = int(node["invocations"])
                nanoseconds = int(node["nanoseconds"])
                item["samples"].append({
                    "workers": worker_count,
                    "target_width": width,
                    "node": int(node["node"]),
                    "nanoseconds": nanoseconds,
                    "invocations": invocations,
                    "milliseconds_per_invocation": (
                        nanoseconds / invocations / 1_000_000.0 if invocations else 0.0
                    ),
                })
    entries: list[dict[str, Any]] = []
    for item in grouped.values():
        total_nanoseconds = sum(int(sample["nanoseconds"]) for sample in item["samples"])
        total_invocations = sum(int(sample["invocations"]) for sample in item["samples"])
        entries.append({
            **item,
            "nodes": sorted(set(item["nodes"])),
            "total_nanoseconds": total_nanoseconds,
            "total_invocations": total_invocations,
            "total_milliseconds": total_nanoseconds / 1_000_000.0,
        })
    entries.sort(key=lambda entry: entry["total_nanoseconds"], reverse=True)
    return {
        "schema_version": 1,
        "driver": str(driver),
        "models": {
            "det": str(det_model), "cls": str(cls_model), "rec": str(rec_model),
            "dictionary": str(dictionary), "image": str(image),
        },
        "iterations": iterations,
        "widths": list(widths),
        "workers": list(workers),
        "det_threads": det_threads,
        "cases": cases,
        "det_shapes": entries,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DET convolution shape profile",
        "",
        f"Widths: `{', '.join(str(width) for width in report['widths'])}`; "
        f"workers: `{', '.join(str(worker) for worker in report['workers'])}`; "
        f"iterations: `{report['iterations']}`",
        "",
        "| Shape | Nodes | Total ms | Samples (workers/width: ms per invocation) |",
        "|---|---|---:|---|",
    ]
    for entry in report["det_shapes"]:
        samples = ", ".join(
            f"{sample['workers']}/{sample['target_width']}: "
            f"{sample['milliseconds_per_invocation']:.3f}"
            for sample in entry["samples"]
        )
        lines.append(
            f"| `{entry['key']}` | `{','.join(str(node) for node in entry['nodes'])}` | "
            f"{entry['total_milliseconds']:.3f} | {samples} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--det-model", type=Path, required=True)
    parser.add_argument("--cls-model", type=Path, required=True)
    parser.add_argument("--rec-model", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--width", type=int, action="append", dest="widths")
    parser.add_argument("--workers", type=int, action="append")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--det-threads", type=int)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    widths = tuple(args.widths or DEFAULT_WIDTHS)
    workers = tuple(args.workers or DEFAULT_WORKERS)
    if not widths or any(width < 7 for width in widths):
        parser.error("widths must be at least 7")
    if not workers or any(worker <= 0 for worker in workers):
        parser.error("workers must be positive")
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    if args.det_threads is not None and args.det_threads <= 0:
        parser.error("det-threads must be positive")
    report = collect(
        args.driver, args.det_model, args.cls_model, args.rec_model,
        args.dictionary, args.image, widths, workers, args.iterations,
        args.det_threads,
    )
    markdown = render_markdown(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
