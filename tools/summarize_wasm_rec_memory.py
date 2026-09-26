#!/usr/bin/env python3
"""Explain compiled WASM OCR heap growth using measured retained allocations."""

import argparse
import json
import re
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def canonical_rec_bytes(log_path: Path) -> dict[str, int]:
    matches = [line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
               if line.startswith("REC_MEMORY canonical_workspace=")]
    if not matches:
        raise ValueError(f"missing canonical REC memory profile in {log_path}")
    values = {key: int(value) for key, value in
              re.findall(r"([a-z_]+)=([0-9]+)", matches[-1])}
    required = ("canonical_workspace", "canonical_input", "canonical_output")
    if any(key not in values for key in required):
        raise ValueError(f"incomplete canonical REC memory profile in {log_path}")
    return values


def render(variant: str, canonical: dict, compiled: dict, profile: dict,
           retained: dict[str, int]) -> str:
    memory = profile["compiled_memory"]
    shared = memory["rec_shared"]
    quantities = {
        "Canonical REC session workspace": retained["canonical_workspace"],
        "Canonical REC input": retained["canonical_input"],
        "Canonical REC output/CTC": retained["canonical_output"],
        "Compiled DET arena": memory["det_arena_bytes"],
        "Compiled DET packed constants": memory["det_packed_bytes"],
        "Compiled CLS arena": memory["cls_arena_bytes"],
        "Compiled CLS packed constants": memory["cls_packed_bytes"],
        "Compiled REC shared arena": shared["shared_arena"],
        "Compiled REC shared scratch": shared["shared_scratch"],
        "Compiled REC CTC workspace": shared["ctc_workspace"],
        "Compiled REC unique owned constants": memory["rec_unique_owned_constant_bytes"],
    }
    canonical_retained = sum(value for name, value in quantities.items()
                             if name.startswith("Canonical REC"))
    compiled_known = sum(value for name, value in quantities.items()
                         if name.startswith("Compiled"))
    heap_delta = compiled["wasm_heap_bytes"] - canonical["wasm_heap_bytes"]
    rows = [f"### {variant.title()} compiled WASM retained-memory diagnostic", "",
            "| Allocation | MiB |", "| --- | ---: |"]
    rows.extend(f"| {name} | {value / 1048576:.2f} |"
                for name, value in quantities.items())
    rows += ["", f"Canonical REC state retained: **{canonical_retained / 1048576:.2f} MiB**; "
             f"known compiled allocations: **{compiled_known / 1048576:.2f} MiB**; "
             f"observed compiled − canonical WASM heap: **{heap_delta / 1048576:+.2f} MiB**.",
             f"Canonical REC workspace planned for a possible fallback: "
             f"**{retained.get('canonical_planned_workspace', 0) / 1048576:.2f} MiB** "
             "(not counted as retained unless the canonical session exists).",
             "These are retained-allocation components, not additive heap-growth accounting: "
             "the WASM heap records its high-water size, allocator fragmentation and "
             "temporary peak allocations. Do not infer recovered heap from `free()` alone."]
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("tiny", "small", "medium"), required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--profile-log", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    print(render(args.variant, load_json(args.canonical), load_json(args.compiled),
                 load_json(args.profile_json), canonical_rec_bytes(args.profile_log)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
