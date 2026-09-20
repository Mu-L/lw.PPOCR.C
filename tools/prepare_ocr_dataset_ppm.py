#!/usr/bin/env python3
"""Materialize a project OCR dataset as a bounded PPM image list.

The generated PPM files and list are benchmark-only artifacts.  They belong
under build-local-data/ (ignored by Git) and retain the source manifest hash so
that a benchmark cannot silently mix images from another corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from tools.evaluate_ocr_dataset import read_dataset
except ModuleNotFoundError:  # Direct ``python tools/...`` invocation.
    from evaluate_ocr_dataset import read_dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_dataset(
    dataset: Path,
    output: Path,
    start: int = 0,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    manifest_path, manifest, warnings = read_dataset(dataset)
    if start < 0:
        raise ValueError("start must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    selected = manifest["images"][start:]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("dataset selection is empty")
    output = output.resolve()
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"output directory is not empty; use --force: {output}")
    output.mkdir(parents=True, exist_ok=True)
    image_root = manifest_path.parent
    entries: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        source = (image_root / str(record["file"])).resolve()
        target = output / f"{index:04d}-{Path(str(record['file'])).stem}.ppm"
        with Image.open(source) as image:
            image.convert("RGB").save(target, format="PPM")
        entries.append(
            {
                "file": target.name,
                "source_file": record["file"],
                "width": int(record["width"]),
                "height": int(record["height"]),
                "sha256": sha256_file(target),
            }
        )
    list_path = output / "images.txt"
    list_path.write_text(
        "".join(f"{(output / entry['file']).resolve()}\n" for entry in entries),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "source_manifest": manifest_path.name,
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_dataset": manifest.get("generator", {}).get("corpus_id"),
        "start": start,
        "images": entries,
        "warnings": warnings,
        "list": list_path.name,
    }
    (output / "benchmark-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        report = prepare_dataset(args.dataset, args.output, args.start, args.limit, args.force)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest_sha256": report["source_manifest_sha256"],
                "images": len(report["images"]),
                "list": str((args.output / report["list"]).resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
