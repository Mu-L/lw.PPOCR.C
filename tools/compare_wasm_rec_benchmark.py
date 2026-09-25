#!/usr/bin/env python3
"""Report a paired canonical/compiled WASM full-OCR experiment."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--baseline-label", default="Canonical")
    parser.add_argument("--candidate-label", default="Compiled REC")
    parser.add_argument("--title", default="Experimental WASM compiled REC: paired full OCR")
    args = parser.parse_args()
    canonical = json.loads(args.canonical.read_text(encoding="utf-8"))
    compiled = json.loads(args.compiled.read_text(encoding="utf-8"))
    for key in ("schema_version", "sample_sha256", "line_count", "text_sha256"):
        if canonical[key] != compiled[key]:
            raise SystemExit(f"WASM benchmark contract mismatch for {key}")
    speedup = canonical["median_ms"] / compiled["median_ms"]
    memory_delta = (compiled["peak_rss_bytes"] - canonical["peak_rss_bytes"]) / 1048576
    print(f"## {args.title}")
    print()
    print(f"Sample SHA-256: `{canonical['sample_sha256']}`; "
          f"text SHA-256: `{canonical['text_sha256']}`. ")
    print("Both builds ran on the same CI runner with the same models and sample.")
    print()
    print(f"| Metric | {args.baseline_label} | {args.candidate_label} | Change |")
    print("| --- | ---: | ---: | ---: |")
    print(f"| Median OCR latency | {canonical['median_ms']:.3f} ms | "
          f"{compiled['median_ms']:.3f} ms | {speedup:.3f}x speedup |")
    print(f"| Peak process RSS | {canonical['peak_rss_bytes'] / 1048576:.1f} MiB | "
          f"{compiled['peak_rss_bytes'] / 1048576:.1f} MiB | "
          f"{memory_delta:+.1f} MiB |")
    print(f"| WASM heap after runs | {canonical['wasm_heap_bytes'] / 1048576:.1f} MiB | "
          f"{compiled['wasm_heap_bytes'] / 1048576:.1f} MiB | n/a |")
    print()
    print("Latency and memory are informational; exact OCR text is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
