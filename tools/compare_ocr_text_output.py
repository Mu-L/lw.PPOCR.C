#!/usr/bin/env python3
"""Show per-line OCR text differences between two lw-ocr-ppm outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


LINE_RE = re.compile(r"^(?P<index>\d+) text=(?P<text>.*?) rec=")


def texts(path: Path) -> list[str]:
    result: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE_RE.match(line)
        if match is not None:
            result.append(match.group("text"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    reference = texts(args.reference)
    candidate = texts(args.candidate)
    print(f"reference_lines={len(reference)}")
    print(f"candidate_lines={len(candidate)}")
    if not reference or not candidate:
        print("no OCR result lines parsed from one or both outputs")
        return 2
    different = 0
    for index in range(max(len(reference), len(candidate))):
        lhs = reference[index] if index < len(reference) else "<missing>"
        rhs = candidate[index] if index < len(candidate) else "<missing>"
        if lhs != rhs:
            different += 1
            print(f"[{index}]\n  reference: {lhs}\n  candidate: {rhs}")
    print(f"different_lines={different}")
    return 0 if different == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
