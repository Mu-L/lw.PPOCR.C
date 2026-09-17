#!/usr/bin/env python3
"""Validate the shared PP-OCRv6 model catalog and all referenced assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VARIANTS = ("tiny", "small", "medium")
RUNTIME_STATUSES = frozenset(("primary", "supported", "analysis-only"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must stay inside models/")
    return value


def check_asset(root: Path, descriptor: dict[str, Any], field: str) -> str:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{field} must be an asset descriptor")
    relative = safe_path(descriptor.get("path"), f"{field}.path")
    expected = descriptor.get("sha256")
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        raise ValueError(f"{field}.sha256 must be a lowercase SHA-256")
    path = root / relative
    if not path.is_file():
        raise ValueError(f"missing model asset: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {field}: {actual} != {expected}")
    return relative


def validate_catalog(catalog_path: Path) -> dict[str, Any]:
    root = catalog_path.parent
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        raise ValueError("model catalog schema_version must be 1")
    if catalog.get("family") != "PP-OCRv6":
        raise ValueError("model catalog family must be PP-OCRv6")
    shared = catalog.get("shared_assets")
    if not isinstance(shared, dict) or set(shared) != {"cls", "small_rec_dictionary"}:
        raise ValueError("shared_assets must contain cls and small_rec_dictionary")
    shared_paths = {name: check_asset(root, descriptor, f"shared_assets.{name}") for name, descriptor in shared.items()}
    expected_users = {"cls": {"tiny", "small", "medium"}, "small_rec_dictionary": {"small", "medium"}}
    for name, descriptor in shared.items():
        users = descriptor.get("used_by")
        if not isinstance(users, list) or set(users) != expected_users[name]:
            raise ValueError(f"shared_assets.{name}.used_by does not match the catalog contract")
    variants = catalog.get("variants")
    if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
        raise ValueError("variants must contain tiny, small, and medium")
    resolved: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        entry = variants[variant]
        if not isinstance(entry, dict):
            raise ValueError(f"variants.{variant} must be an object")
        runtime_status = entry.get("runtime_status")
        if runtime_status not in RUNTIME_STATUSES:
            raise ValueError(
                f"variants.{variant}.runtime_status must be one of "
                f"{sorted(RUNTIME_STATUSES)}"
            )
        assets = {role: check_asset(root, entry.get(role), f"variants.{variant}.{role}") for role in ("det", "rec")}
        for role in ("cls", "dictionary"):
            reference = entry.get(role)
            if isinstance(reference, str) and reference.startswith("shared:"):
                shared_name = reference.removeprefix("shared:")
                if shared_name not in shared_paths:
                    raise ValueError(f"variants.{variant}.{role} references unknown shared asset: {shared_name}")
                assets[role] = shared_paths[shared_name]
            elif role == "dictionary" and isinstance(reference, dict):
                assets[role] = check_asset(root, reference, f"variants.{variant}.{role}")
            else:
                raise ValueError(f"variants.{variant}.{role} must reference a shared asset or dictionary descriptor")
        resolved[variant] = assets
    if resolved["small"]["dictionary"] != resolved["medium"]["dictionary"]:
        raise ValueError("Small and Medium must use the same dictionary")
    if len({resolved[variant]["cls"] for variant in VARIANTS}) != 1:
        raise ValueError("Tiny, Small, and Medium must use the same CLS asset")
    return {
        "status": "ok",
        "variants": list(VARIANTS),
        "runtime_status": {variant: variants[variant]["runtime_status"] for variant in VARIANTS},
        "resolved": resolved,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(validate_catalog(args.catalog), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
