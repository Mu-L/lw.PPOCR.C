#!/usr/bin/env python3
"""Summarize Tiny/Small WASM Pointwise epilogue work from an instrumented run."""

import argparse
import re
import sys
from pathlib import Path


REC_WIDTHS = {192, 320, 480, 640, 960}
EPILOGUES = {"plain", "bias", "hardswish", "residual", "resgelu", "postgelu", "other"}
PATTERN = re.compile(
    r"^WASM_PW_EPI width=(\d+) kind=([a-z]+) calls=(\d+) macs=(\d+) elapsed=([0-9]+(?:\.[0-9]+)?)$"
)


def summarize(log: str, variant: str) -> str:
    totals = {kind: [0, 0, 0.0] for kind in EPILOGUES}
    cls_calls = 0
    for line in log.splitlines():
        if not line.startswith("WASM_PW_EPI "):
            continue
        match = PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed Pointwise epilogue profile: {line}")
        width, kind, calls, macs, elapsed = match.groups()
        width = int(width)
        if kind not in EPILOGUES:
            raise ValueError(f"unknown Pointwise epilogue kind: {kind}")
        if width == 160:
            cls_calls += int(calls)
            continue
        if width not in REC_WIDTHS:
            raise ValueError(f"unknown REC profile width: {width}")
        row = totals[kind]
        row[0] += int(calls)
        row[1] += int(macs)
        row[2] += float(elapsed)
    measured = [(kind, *values) for kind, values in totals.items() if values[0] > 0]
    if not measured:
        raise ValueError("no Tiny/Small REC Pointwise epilogue profile was captured")
    total_ms = sum(row[3] for row in measured)
    rows = [f"### {variant.title()} WASM REC Pointwise epilogues", "",
            "Instrumented diagnostic only; width 160 is CLS and excluded from REC totals. "
            f"Excluded {cls_calls} CLS Pointwise calls.", "",
            "| Epilogue | Calls | MACs | Elapsed ms | Time share |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for kind, calls, macs, elapsed in sorted(measured, key=lambda row: row[3], reverse=True):
        rows.append(f"| {kind} | {calls} | {macs} | {elapsed:.3f} | "
                    f"{elapsed / total_ms * 100.0 if total_ms else 0.0:.1f}% |")
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--variant", choices=("tiny", "small"), required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    print(summarize(args.log.read_text(encoding="utf-8", errors="replace"), args.variant), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
