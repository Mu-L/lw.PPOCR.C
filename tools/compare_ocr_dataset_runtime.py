#!/usr/bin/env python3
"""Compare compact and resident same-process OCR dataset benchmarks."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from statistics import median
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError(f"unsupported benchmark manifest: {path}")
    images = report.get("images")
    # Width-switch manifests intentionally use ``entries`` because one source
    # image can occur once per cycle. Normalize that schema here so the same
    # paired benchmark runner can validate a repeated request sequence.
    if images is None and isinstance(report.get("entries"), list):
        images = []
        for index, entry in enumerate(report["entries"]):
            if not isinstance(entry, dict):
                raise ValueError(f"benchmark width-switch entry {index} is not an object")
            ppm_file = entry.get("ppm_file")
            source_file = entry.get("source_file")
            if not isinstance(ppm_file, str) or not ppm_file:
                raise ValueError(f"benchmark width-switch entry {index} lacks ppm_file")
            if not isinstance(source_file, str) or not source_file:
                raise ValueError(f"benchmark width-switch entry {index} lacks source_file")
            images.append({"file": ppm_file, "source_file": source_file})
        report = {**report, "images": images}
    if not isinstance(images, list) or not images:
        raise ValueError(f"benchmark manifest has no images: {path}")
    if not isinstance(report.get("source_manifest_sha256"), str):
        raise ValueError(f"benchmark manifest lacks source manifest hash: {path}")
    return report



def run_process_with_heartbeat(
    command: list[str],
    *,
    label: str,
    timeout_seconds: int,
    heartbeat_seconds: int,
) -> tuple[int, str, str, float]:
    print(f"[benchmark] START {label}", flush=True)
    started = time.monotonic()
    next_heartbeat = started + heartbeat_seconds
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    while True:
        returncode = process.poll()
        if returncode is not None:
            break
        now = time.monotonic()
        elapsed = now - started
        if elapsed >= timeout_seconds:
            process.kill()
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"dataset benchmark timed out after {elapsed:.1f}s: {label}\n"
                f"stdout tail:\n{stdout[-4000:]}\n"
                f"stderr tail:\n{stderr[-4000:]}"
            )
        if now >= next_heartbeat:
            print(
                f"[benchmark] RUNNING {label}: {elapsed / 60.0:.1f} min elapsed",
                flush=True,
            )
            next_heartbeat = now + heartbeat_seconds
        time.sleep(1.0)
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr, time.monotonic() - started

def run_benchmark(
    executable: Path,
    arguments: list[str],
    expected_resident: bool,
    expected_images: int,
    expected_workers: int,
    expected_width: int,
    *,
    label: str,
    timeout_seconds: int,
    heartbeat_seconds: int,
) -> dict[str, Any]:
    returncode, stdout, stderr, elapsed = run_process_with_heartbeat(
        [str(executable), *arguments],
        label=label,
        timeout_seconds=timeout_seconds,
        heartbeat_seconds=heartbeat_seconds,
    )
    if returncode != 0:
        raise RuntimeError(
            f"dataset benchmark failed: {executable}\n"
            f"case: {label}\n"
            f"stdout tail:\n{stdout[-4000:]}\n"
            f"stderr tail:\n{stderr[-4000:]}"
        )
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"dataset benchmark produced no JSON: {executable}")
    try:
        report = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"dataset benchmark output is not JSON: {executable}\n"
            f"stdout tail:\n{stdout[-4000:]}"
        ) from error
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise RuntimeError(f"unsupported dataset benchmark JSON: {executable}")
    expected = {
        "resident_widths": expected_resident,
        "images": expected_images,
        "workers": expected_workers,
        "rec_target_width": expected_width,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(
                f"dataset benchmark contract mismatch for {key}: "
                f"{report.get(key)!r} != {value!r}"
            )
    if not isinstance(report.get("output_checksum"), str):
        raise RuntimeError("dataset benchmark lacks output_checksum")
    if not isinstance(report.get("lines"), int) or report["lines"] < 0:
        raise RuntimeError("dataset benchmark has invalid line count")
    for key in ("mean", "p95"):
        value = report.get("ocr_ms", {}).get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise RuntimeError(f"dataset benchmark has invalid ocr_ms.{key}")
    peak = report.get("peak_rss_bytes")
    if not isinstance(peak, int) or peak <= 0:
        raise RuntimeError("dataset benchmark has invalid peak_rss_bytes")
    sample_count = report.get("rss_sample_count")
    min_sampled = report.get("min_sampled_rss_bytes")
    max_sampled = report.get("max_sampled_rss_bytes")
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise RuntimeError("dataset benchmark has invalid rss_sample_count")
    if not isinstance(min_sampled, int) or min_sampled <= 0:
        raise RuntimeError("dataset benchmark has invalid min_sampled_rss_bytes")
    if not isinstance(max_sampled, int) or max_sampled < min_sampled:
        raise RuntimeError("dataset benchmark has invalid max_sampled_rss_bytes")
    print(
        f"[benchmark] DONE {label}: {elapsed:.1f}s wall, "
        f"mean={float(report['ocr_ms']['mean']):.3f} ms/image, "
        f"p95={float(report['ocr_ms']['p95']):.3f} ms/image, "
        f"peak_rss={peak / 1048576.0:.2f} MiB",
        flush=True,
    )
    return report

def summarize(compact: dict[str, Any], resident: dict[str, Any]) -> dict[str, Any]:
    if compact["lines"] != resident["lines"] or compact["output_checksum"] != resident["output_checksum"]:
        raise RuntimeError("OCR output contract differs between compact and resident")
    compact_mean = float(compact["ocr_ms"]["mean"])
    resident_mean = float(resident["ocr_ms"]["mean"])
    compact_p95 = float(compact["ocr_ms"]["p95"])
    resident_p95 = float(resident["ocr_ms"]["p95"])
    compact_rss = int(compact["peak_rss_bytes"])
    resident_rss = int(resident["peak_rss_bytes"])
    compact_sampled_rss = int(compact["max_sampled_rss_bytes"])
    resident_sampled_rss = int(resident["max_sampled_rss_bytes"])
    return {
        "compact": {
            "ocr_mean_ms": compact_mean,
            "ocr_p95_ms": compact_p95,
            "peak_rss_mib": compact_rss / 1048576.0,
            "max_sampled_rss_mib": compact_sampled_rss / 1048576.0,
        },
        "resident": {
            "ocr_mean_ms": resident_mean,
            "ocr_p95_ms": resident_p95,
            "peak_rss_mib": resident_rss / 1048576.0,
            "max_sampled_rss_mib": resident_sampled_rss / 1048576.0,
        },
        "comparison": {
            "mean_speedup": compact_mean / resident_mean,
            "mean_latency_change_percent": (resident_mean / compact_mean - 1.0) * 100.0,
            "p95_latency_change_percent": (resident_p95 / compact_p95 - 1.0) * 100.0,
            "rss_delta_mib": (resident_rss - compact_rss) / 1048576.0,
            "max_sampled_rss_delta_mib": (resident_sampled_rss - compact_sampled_rss) / 1048576.0,
        },
        "contract": {
            "match": True,
            "compact_checksum": compact["output_checksum"],
            "resident_checksum": resident["output_checksum"],
            "lines": compact["lines"],
            "images": compact["images"],
            "workers": compact["workers"],
            "rec_target_width": compact["rec_target_width"],
        },
    }


def summarize_paired(
    compact_runs: list[dict[str, Any]], resident_runs: list[dict[str, Any]], orders: list[str]
) -> dict[str, Any]:
    if not compact_runs or len(compact_runs) != len(resident_runs) or len(orders) != len(compact_runs):
        raise RuntimeError("paired dataset runs must have equal non-zero sample counts")
    samples = [summarize(compact, resident) for compact, resident in zip(compact_runs, resident_runs)]
    compact = {
        "ocr_mean_ms": median([sample["compact"]["ocr_mean_ms"] for sample in samples]),
        "ocr_p95_ms": median([sample["compact"]["ocr_p95_ms"] for sample in samples]),
        "peak_rss_mib": median([sample["compact"]["peak_rss_mib"] for sample in samples]),
        "max_sampled_rss_mib": median([sample["compact"]["max_sampled_rss_mib"] for sample in samples]),
    }
    resident = {
        "ocr_mean_ms": median([sample["resident"]["ocr_mean_ms"] for sample in samples]),
        "ocr_p95_ms": median([sample["resident"]["ocr_p95_ms"] for sample in samples]),
        "peak_rss_mib": median([sample["resident"]["peak_rss_mib"] for sample in samples]),
        "max_sampled_rss_mib": median([sample["resident"]["max_sampled_rss_mib"] for sample in samples]),
    }
    comparison = {
        "mean_speedup": compact["ocr_mean_ms"] / resident["ocr_mean_ms"],
        "mean_latency_change_percent": (resident["ocr_mean_ms"] / compact["ocr_mean_ms"] - 1.0) * 100.0,
        "p95_latency_change_percent": (resident["ocr_p95_ms"] / compact["ocr_p95_ms"] - 1.0) * 100.0,
        "rss_delta_mib": resident["peak_rss_mib"] - compact["peak_rss_mib"],
        "max_sampled_rss_delta_mib": resident["max_sampled_rss_mib"] - compact["max_sampled_rss_mib"],
    }
    base_contract = samples[0]["contract"]
    for sample in samples[1:]:
        if sample["contract"]["compact_checksum"] != base_contract["compact_checksum"] or \
                sample["contract"]["resident_checksum"] != base_contract["resident_checksum"]:
            raise RuntimeError("paired dataset checksums are not deterministic")
    return {
        "schema_version": 1,
        "compact": compact,
        "resident": resident,
        "comparison": comparison,
        "contract": base_contract,
        "paired": {
            "rounds": len(samples),
            "orders": orders,
            "samples": [
                {"order": order, **sample} for order, sample in zip(orders, samples)
            ],
        },
    }


def render_markdown(
    summary: dict[str, Any], manifest: dict[str, Any], candidate_label: str = "Resident"
) -> str:
    compact = summary["compact"]
    resident = summary["resident"]
    comparison = summary["comparison"]
    contract = summary["contract"]
    paired = summary.get("paired", {})
    return "\n".join(
        [
            f"# Compact vs {candidate_label} OCR dataset runtime",
            "",
            f"Manifest SHA-256: `{manifest['source_manifest_sha256']}`",
            f"Images: `{contract['images']}`; workers: `{contract['workers']}`; REC width: `{contract['rec_target_width']}`",
            "",
            "| Profile | OCR mean (ms/image) | OCR P95 (ms/image) | Peak RSS (MiB) | Max sampled RSS (MiB) |",
            "|---|---:|---:|---:|---:|",
            f"| Compact | {compact['ocr_mean_ms']:.3f} | {compact['ocr_p95_ms']:.3f} | {compact['peak_rss_mib']:.3f} | {compact['max_sampled_rss_mib']:.3f} |",
            f"| {candidate_label} | {resident['ocr_mean_ms']:.3f} | {resident['ocr_p95_ms']:.3f} | {resident['peak_rss_mib']:.3f} | {resident['max_sampled_rss_mib']:.3f} |",
            "",
            f"Mean speedup: **{comparison['mean_speedup']:.3f}x**",
            f"Mean latency change: **{comparison['mean_latency_change_percent']:+.2f}%**",
            f"P95 latency change: **{comparison['p95_latency_change_percent']:+.2f}%**",
            f"RSS delta: **{comparison['rss_delta_mib']:+.3f} MiB**",
            f"Max sampled RSS delta: **{comparison['max_sampled_rss_delta_mib']:+.3f} MiB**",
            f"Paired rounds: `{paired.get('rounds', 1)}`",
            f"Checksum: `{contract['compact_checksum']}` vs `{contract['resident_checksum']}`",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-driver", type=Path, required=True)
    parser.add_argument("--resident-driver", type=Path, required=True)
    parser.add_argument(
        "--candidate-resident-widths",
        choices=("true", "false"),
        default="true",
        help="Expected resident_widths field for the candidate driver (default: true)",
    )
    parser.add_argument("--candidate-label", default="Resident")
    parser.add_argument("--case-label", default="dataset", help="Human-readable label used in progress logs")
    parser.add_argument("--process-timeout-seconds", type=int, default=3600, help="Timeout for one benchmark executable invocation")
    parser.add_argument("--heartbeat-seconds", type=int, default=60, help="Progress heartbeat interval while a benchmark process runs")
    parser.add_argument("--det", type=Path, required=True)
    parser.add_argument("--cls", type=Path, required=True)
    parser.add_argument("--rec", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--target-width", type=int, default=960)
    parser.add_argument("--paired-rounds", type=int, default=1)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    candidate_resident_widths = args.candidate_resident_widths == "true"
    if args.warmup <= 0 or args.iterations <= 0 or args.paired_rounds <= 0:
        parser.error("warmup, iterations and paired-rounds must be positive")
    if args.process_timeout_seconds <= 0:
        parser.error("process-timeout-seconds must be positive")
    if args.heartbeat_seconds < 10:
        parser.error("heartbeat-seconds must be at least 10")
    manifest = load_manifest(args.benchmark_manifest)
    expected_images = len(manifest["images"])
    benchmark_args = [
        str(args.det), str(args.cls), str(args.rec), str(args.dictionary), str(args.image_list),
        str(args.warmup), str(args.iterations), str(args.workers), str(args.target_width),
    ]
    compact_runs: list[dict[str, Any]] = []
    resident_runs: list[dict[str, Any]] = []
    orders: list[str] = []
    for round_index in range(args.paired_rounds):
        round_number = round_index + 1
        compact_first = round_index % 2 == 0
        order = "compact-first" if compact_first else "resident-first"
        orders.append(order)
        print(
            f"[benchmark] ROUND {round_number}/{args.paired_rounds}: "
            f"{args.case_label}, order={order}",
            flush=True,
        )
        compact_label = (
            f"{args.case_label} round {round_number}/{args.paired_rounds} Compact"
        )
        candidate_label = (
            f"{args.case_label} round {round_number}/{args.paired_rounds} "
            f"{args.candidate_label}"
        )

        def run_compact() -> dict[str, Any]:
            return run_benchmark(
                args.compact_driver,
                benchmark_args,
                False,
                expected_images,
                args.workers,
                args.target_width,
                label=compact_label,
                timeout_seconds=args.process_timeout_seconds,
                heartbeat_seconds=args.heartbeat_seconds,
            )

        def run_candidate() -> dict[str, Any]:
            return run_benchmark(
                args.resident_driver,
                benchmark_args,
                candidate_resident_widths,
                expected_images,
                args.workers,
                args.target_width,
                label=candidate_label,
                timeout_seconds=args.process_timeout_seconds,
                heartbeat_seconds=args.heartbeat_seconds,
            )

        if compact_first:
            compact_runs.append(run_compact())
            resident_runs.append(run_candidate())
        else:
            resident_runs.append(run_candidate())
            compact_runs.append(run_compact())
    summary = summarize_paired(compact_runs, resident_runs, orders)
    markdown = render_markdown(summary, manifest, args.candidate_label)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8", newline="\n")
    if not args.json_output and not args.markdown_output:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
