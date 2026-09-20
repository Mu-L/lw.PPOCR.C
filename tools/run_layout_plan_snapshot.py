"""Run and validate the analysis-only NHWC layout planner snapshot."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SUMMARY_RE = re.compile(
    r"layout_plan tensor_count=(?P<tensor_count>\d+) "
    r"node_count=(?P<node_count>\d+) "
    r"nhwc_nodes=(?P<nhwc_nodes>\d+) "
    r"nchw_nodes=(?P<nchw_nodes>\d+) "
    r"conversions=(?P<conversions>\d+) "
    r"islands=(?P<islands>\d+) "
    r"direct_input=(?P<direct_input>[01]) "
    r"width=(?P<width>\d+)"
)
NODE_RE = re.compile(
    r"^(?P<index>\d+) (?P<op>\S+) +(?P<layout>NCHW|NHWC) "
    r"inputs=(?P<inputs>[^ ]*) output=(?P<output>\d+)$"
)


def executable(build_dir: Path, name: str) -> Path:
    candidates = [build_dir / name]
    if sys.platform == "win32":
        candidates.insert(0, build_dir / "Release" / f"{name}.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"planner executable not found under {build_dir}: {name}")


def parse_output(stdout: str) -> dict[str, Any]:
    summary_match = SUMMARY_RE.search(stdout)
    if summary_match is None:
        raise ValueError("layout planner summary line is missing")
    summary: dict[str, Any] = {
        key: int(value) for key, value in summary_match.groupdict().items()
    }
    nodes: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        match = NODE_RE.match(line)
        if match is None:
            continue
        inputs = [] if match.group("inputs") == "" else [
            int(value) for value in match.group("inputs").split(",")
        ]
        nodes.append(
            {
                "index": int(match.group("index")),
                "op": match.group("op"),
                "layout": match.group("layout"),
                "inputs": inputs,
                "output": int(match.group("output")),
            }
        )
    if len(nodes) != summary["node_count"]:
        raise ValueError(
            f"layout planner node dump has {len(nodes)} nodes, "
            f"expected {summary['node_count']}"
        )
    for expected_index, node in enumerate(nodes):
        if node["index"] != expected_index:
            raise ValueError("layout planner node indexes are not contiguous")
    return {"summary": summary, "nodes": nodes}


def load_contract(path: Path, variant: str) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported layout planner contract schema")
    snapshots = contract.get("variants")
    if not isinstance(snapshots, dict) or variant not in snapshots:
        raise ValueError(f"layout planner contract has no variant: {variant}")
    expected = snapshots[variant]
    if not isinstance(expected, dict):
        raise ValueError(f"invalid snapshot contract for variant: {variant}")
    return expected


def check_expected(summary: dict[str, Any], expected: dict[str, Any], variant: str) -> None:
    fields = (
        "tensor_count",
        "node_count",
        "nhwc_nodes",
        "nchw_nodes",
        "conversions",
        "islands",
        "width",
    )
    for field in fields:
        if field not in expected:
            raise ValueError(f"contract missing {variant}.{field}")
        if summary[field] != expected[field]:
            raise ValueError(
                f"{variant} layout snapshot mismatch for {field}: "
                f"{summary[field]} != {expected[field]}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", required=True, choices=("tiny", "small", "medium"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct-input", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_dir = args.build_dir.resolve()
    model = args.model.resolve()
    contract_path = (root / args.contract).resolve() if not args.contract.is_absolute() else args.contract.resolve()
    if not model.is_file():
        raise FileNotFoundError(f"layout snapshot model not found: {model}")
    driver = executable(build_dir, "layout-plan-driver")
    command = [str(driver), str(model), "960"]
    if args.direct_input:
        command.append("direct-nhwc")
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"layout planner exited with {completed.returncode}")
    parsed = parse_output(completed.stdout)
    expected = load_contract(contract_path, args.variant)
    check_expected(parsed["summary"], expected, args.variant)
    report = {
        "schema_version": 1,
        "variant": args.variant,
        "model": model.name,
        "direct_input": args.direct_input,
        **parsed,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())