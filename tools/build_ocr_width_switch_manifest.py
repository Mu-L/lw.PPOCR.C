#!/usr/bin/env python3
"""Build a deterministic image-list for adaptive REC width switching.

The OCR runtime chooses a REC width from the geometry of each detected text
line. This helper does not alter images or runtime; it orders prepared PPM
files so a same-process benchmark exercises several width buckets repeatedly.
Generated lists and reports belong under ``build-local-data`` and must not be
committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any


WIDTH_BUCKETS = (192, 320, 480, 640, 960)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field} must be a positive number")
    return float(value)


def adaptive_bucket(line_width: float) -> int:
    """Mirror the runtime's five adaptive REC width buckets."""
    for bucket in WIDTH_BUCKETS:
        if line_width <= bucket:
            return bucket
    return WIDTH_BUCKETS[-1]


def _line_width_hint(record: dict[str, Any]) -> float:
    lines = record.get("lines")
    if not isinstance(lines, list):
        raise ValueError(f"image record lacks lines: {record.get('file')!r}")
    widths = []
    for line in lines:
        if isinstance(line, dict) and "natural_width_at_height_48" in line:
            widths.append(_positive_number(line["natural_width_at_height_48"], "line width"))
    if not widths:
        raise ValueError(f"image record has no natural line width: {record.get('file')!r}")
    # A median line is a stable image-level representative. The runtime still
    # chooses a width independently for every detected crop; this hint only
    # orders requests so mixed-width images are interleaved predictably.
    return float(median(widths))


def _load_json(path: Path, expected_schema: int) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    schema = value.get("schema_version", value.get("version")) if isinstance(value, dict) else None
    if not isinstance(value, dict) or schema != expected_schema:
        raise ValueError(f"unsupported manifest schema: {path}")
    return value


def _ppm_entries(ppm_manifest_path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    manifest = _load_json(ppm_manifest_path, 1)
    source_hash = manifest.get("source_manifest_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("benchmark manifest lacks source_manifest_sha256")
    entries = manifest.get("images")
    if not isinstance(entries, list) or not entries:
        raise ValueError("benchmark manifest has no images")
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("benchmark image entry must be an object")
        source_file = entry.get("source_file")
        ppm_file = entry.get("file")
        if not isinstance(source_file, str) or not source_file:
            raise ValueError("benchmark image entry lacks source_file")
        if not isinstance(ppm_file, str) or not ppm_file:
            raise ValueError("benchmark image entry lacks file")
        if source_file in result:
            raise ValueError(f"duplicate source_file in benchmark manifest: {source_file}")
        result[source_file] = {
            "source_file": source_file,
            "ppm_file": ppm_file,
            "sha256": entry.get("sha256"),
        }
    return source_hash, result


def build_width_switch_manifest(
    dataset_metadata: Path,
    ppm_manifest: Path,
    output_list: Path,
    output_manifest: Path,
    cycles: int = 3,
    minimum_buckets: int = 2,
) -> dict[str, Any]:
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    if minimum_buckets < 2 or minimum_buckets > len(WIDTH_BUCKETS):
        raise ValueError("minimum_buckets must be between 2 and 5")

    dataset = _load_json(dataset_metadata, 1)
    dataset_images = dataset.get("images")
    if not isinstance(dataset_images, list) or not dataset_images:
        raise ValueError("dataset metadata has no images")
    source_hash, ppm_entries = _ppm_entries(ppm_manifest)
    actual_hash = sha256_file(dataset_metadata)
    if actual_hash != source_hash:
        raise ValueError("dataset metadata and PPM benchmark manifest do not match")

    by_bucket: dict[int, list[dict[str, Any]]] = {bucket: [] for bucket in WIDTH_BUCKETS}
    ppm_root = ppm_manifest.parent
    for source_record in dataset_images:
        if not isinstance(source_record, dict):
            raise ValueError("dataset image record must be an object")
        source_file = source_record.get("file")
        if not isinstance(source_file, str) or source_file not in ppm_entries:
            raise ValueError(f"dataset image is missing from PPM manifest: {source_file!r}")
        hint = _line_width_hint(source_record)
        bucket = adaptive_bucket(hint)
        ppm_record = ppm_entries[source_file]
        ppm_path = (ppm_root / str(ppm_record["ppm_file"])).resolve()
        if not ppm_path.is_file():
            raise ValueError(f"prepared PPM image does not exist: {ppm_path}")
        by_bucket[bucket].append(
            {
                "source_file": source_file,
                "ppm_file": str(ppm_record["ppm_file"]),
                "ppm_path": str(ppm_path),
                "line_width_hint": hint,
                "expected_bucket": bucket,
                "sha256": ppm_record.get("sha256"),
            }
        )

    available = [bucket for bucket in WIDTH_BUCKETS if by_bucket[bucket]]
    if len(available) < minimum_buckets:
        raise ValueError(
            f"dataset contains only {len(available)} adaptive width buckets; "
            f"need at least {minimum_buckets}"
        )

    order: list[int] = []
    left = 0
    right = len(available) - 1
    while left <= right:
        order.append(available[left])
        left += 1
        if left <= right:
            order.append(available[right])
            right -= 1

    selected_by_bucket = {bucket: by_bucket[bucket][0] for bucket in available}
    sequence: list[dict[str, Any]] = []
    for cycle in range(cycles):
        for bucket in order:
            sequence.append({"cycle": cycle, **selected_by_bucket[bucket]})

    output_list = output_list.resolve()
    output_manifest = output_manifest.resolve()
    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_list.write_text(
        "".join(f"{item['ppm_path']}\n" for item in sequence),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "purpose": "adaptive-rec-width-switch",
        "source_manifest_sha256": actual_hash,
        "ppm_manifest": str(ppm_manifest.resolve()),
        "cycles": cycles,
        "available_buckets": available,
        "bucket_order": order,
        "entries": sequence,
        "list": str(output_list),
    }
    output_manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="source metadata.json")
    parser.add_argument("--ppm-manifest", type=Path, required=True)
    parser.add_argument("--output-list", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--minimum-buckets", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        report = build_width_switch_manifest(
            args.dataset,
            args.ppm_manifest,
            args.output_list,
            args.output_manifest,
            args.cycles,
            args.minimum_buckets,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest_sha256": report["source_manifest_sha256"],
                "buckets": report["available_buckets"],
                "entries": len(report["entries"]),
                "list": report["list"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())