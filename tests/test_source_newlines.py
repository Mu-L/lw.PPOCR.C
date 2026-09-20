#!/usr/bin/env python3
"""Check that tracked C and header sources end with a newline."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    missing: list[Path] = []
    for directory in ("src", "include", "tests"):
        base = args.root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in {".c", ".h"} or not path.is_file():
                continue
            data = path.read_bytes()
            if data and not data.endswith(b"\n"):
                missing.append(path.relative_to(args.root))

    if missing:
        for path in missing:
            print(f"missing final newline: {path}")
        return 1
    print("all C/H sources have a final newline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())