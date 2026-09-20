#!/usr/bin/env python3
"""Collect shape-aware REC Conv1x1 profiles for reproducible kernel work.

The native rec-profile-driver already reports resolved tensor shapes for one
REC width. This helper runs that driver for every supported width and emits a
single report grouped by the actual Conv1x1 shape. It is diagnostic only: it
does not change runtime dispatch or model assets.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_WIDTHS = (192, 320, 480, 640, 960)


def run_profile(driver: Path, model: Path, width: int, iterations: int) -> dict[str, Any]:
    completed = subprocess.run(
        [str(driver), str(model), str(width), str(iterations)],
        check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=3600,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"REC profile failed for width {width}:\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"REC profile produced no JSON for width {width}")
    report = json.loads(lines[-1])
    if not isinstance(report, dict) or report.get("width") != width:
        raise RuntimeError(f"invalid REC profile for width {width}")
    if not isinstance(report.get("conv_nodes"), list):
        raise RuntimeError(f"REC profile lacks conv_nodes for width {width}")
    return report


def collect(driver: Path, model: Path, widths: tuple[int, ...], iterations: int) -> dict[str, Any]:
    reports = [run_profile(driver, model, width, iterations) for width in widths]
    grouped: dict[str, dict[str, Any]] = {}
    for report in reports:
        width = int(report["width"])
        for node in report["conv_nodes"]:
            shape = node.get("input")
            output = node.get("output")
            weights = node.get("weights")
            if not (
                isinstance(shape, list) and len(shape) == 4
                and isinstance(output, list) and len(output) == 4
                and isinstance(weights, list) and len(weights) == 4
            ):
                continue
            if node.get("group") != 1 or node.get("kernel") != [1, 1]:
                continue
            key = f"{shape[1]}x{output[1]}-h{shape[2]}-w{shape[3]}"
            item = grouped.setdefault(key, {
                "shape": {
                    "input": shape, "weights": weights, "output": output,
                    "group": node["group"], "kernel": node["kernel"],
                    "strides": node["strides"],
                },
                "nodes": [], "samples": [],
            })
            item["nodes"].append(node["node"])
            item["samples"].append({
                "width": width, "node": node["node"],
                "nanoseconds": node["nanoseconds"],
                "invocations": node["invocations"],
                "milliseconds_per_invocation": (
                    float(node["nanoseconds"]) / float(node["invocations"]) / 1_000_000.0
                    if node["invocations"] else 0.0
                ),
            })
    entries = []
    for key, item in grouped.items():
        total_ns = sum(int(sample["nanoseconds"]) for sample in item["samples"])
        entries.append({
            "key": key, "shape": item["shape"],
            "nodes": sorted(set(item["nodes"])), "samples": item["samples"],
            "total_nanoseconds": total_ns,
            "total_milliseconds": total_ns / 1_000_000.0,
        })
    entries.sort(key=lambda entry: entry["total_nanoseconds"], reverse=True)
    return {
        "schema_version": 1, "driver": str(driver), "model": str(model),
        "iterations": iterations, "widths": list(widths),
        "conv1x1_shapes": entries,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# REC Conv1x1 shape profile", "",
        f"Model: `{report['model']}`",
        f"Widths: `{', '.join(str(width) for width in report['widths'])}`; "
        f"iterations: `{report['iterations']}`", "",
        "| Shape | Nodes | Total ms | Samples (width: ms/invocation) |",
        "|---|---|---:|---|",
    ]
    for entry in report["conv1x1_shapes"]:
        samples = ", ".join(
            f"{sample['width']}: {sample['milliseconds_per_invocation']:.3f}"
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--width", type=int, action="append", dest="widths")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    widths = tuple(args.widths or DEFAULT_WIDTHS)
    if not widths or any(width < 7 for width in widths):
        parser.error("widths must be at least 7")
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    report = collect(args.driver, args.model, widths, args.iterations)
    markdown = render_markdown(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
