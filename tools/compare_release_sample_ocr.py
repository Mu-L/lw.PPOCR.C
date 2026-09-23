#!/usr/bin/env python3
"""Compare two native releases on the same PP-OCRv6 sample and model assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


MODELS = ("tiny", "small", "medium")
WORKERS = (1, 4)
ASSET_NAMES = ("det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt")
MIB = 1048576.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def validate_payload(
    payload: dict[str, Any], workers: int, width: int, warmup: int, iterations: int
) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported OCR benchmark schema")
    if payload.get("workers") != workers or payload.get("rec_target_width") != width:
        raise ValueError("OCR benchmark worker or REC width contract mismatch")
    if payload.get("warmup") != warmup or payload.get("iterations") != iterations:
        raise ValueError("OCR benchmark warm-up or iteration contract mismatch")
    if payload.get("image_width") != 500 or payload.get("image_height") != 500:
        raise ValueError("OCR benchmark did not use the bundled 500x500 sample")
    if not isinstance(payload.get("backend"), str) or not payload["backend"]:
        raise ValueError("OCR benchmark did not report its SIMD backend")
    if not isinstance(payload.get("output_checksum"), str) or not payload["output_checksum"]:
        raise ValueError("OCR benchmark did not report a text checksum")
    positive_number(payload.get("lines"), "lines")
    positive_number(payload.get("ocr_ms", {}).get("mean"), "ocr_ms.mean")
    positive_number(payload.get("ocr_ms", {}).get("p95"), "ocr_ms.p95")
    positive_number(payload.get("peak_rss_bytes"), "peak_rss_bytes")
    positive_number(payload.get("rss_after_warmup_bytes"), "rss_after_warmup_bytes")


def run_driver(
    driver: Path,
    assets: Path,
    sample: Path,
    warmup: int,
    iterations: int,
    workers: int,
    width: int,
    timeout: int,
    raw_path: Path,
) -> dict[str, Any]:
    command = [
        str(driver),
        *(str(assets / name) for name in ASSET_NAMES),
        str(sample),
        str(warmup),
        str(iterations),
        str(workers),
        str(width),
    ]
    print(f"[release-compare] START {raw_path.parent.name}/{raw_path.stem}", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    next_heartbeat = started + 30.0
    while process.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            process.kill()
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"benchmark timed out after {elapsed:.1f}s: {driver}\n"
                f"stdout tail:\n{stdout[-4000:]}\nstderr tail:\n{stderr[-4000:]}"
            )
        if time.monotonic() >= next_heartbeat:
            print(f"[release-compare] RUNNING {raw_path.stem}: {elapsed:.0f}s", flush=True)
            next_heartbeat = time.monotonic() + 30.0
        time.sleep(1.0)
    stdout, stderr = process.communicate()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(stdout, encoding="utf-8", newline="\n")
    if stderr:
        raw_path.with_suffix(".stderr.txt").write_text(
            stderr, encoding="utf-8", newline="\n"
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"benchmark failed ({process.returncode}): {driver}\n{stderr}"
        )
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"benchmark returned no JSON: {driver}")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"benchmark returned invalid JSON: {driver}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark JSON is not an object: {driver}")
    validate_payload(payload, workers, width, warmup, iterations)
    print(
        f"[release-compare] DONE {raw_path.stem}: "
        f"{payload['ocr_ms']['mean']:.3f} ms, "
        f"{payload['peak_rss_bytes'] / MIB:.1f} MiB peak WS",
        flush=True,
    )
    return payload


def summarize_case(
    model: str,
    workers: int,
    baseline_runs: list[dict[str, Any]],
    candidate_runs: list[dict[str, Any]],
    orders: list[str],
) -> dict[str, Any]:
    if not baseline_runs or len(baseline_runs) != len(candidate_runs) or len(orders) != len(baseline_runs):
        raise ValueError("paired OCR benchmark runs are incomplete")
    for label, runs in (("baseline", baseline_runs), ("candidate", candidate_runs)):
        if len({(run["lines"], run["output_checksum"]) for run in runs}) != 1:
            raise ValueError(f"{label} OCR output changed between paired rounds")
        if len({run["backend"] for run in runs}) != 1:
            raise ValueError(f"{label} SIMD backend changed between paired rounds")
    if any(a["backend"] != b["backend"] for a, b in zip(baseline_runs, candidate_runs)):
        raise ValueError("SIMD backend differs on the same runner")

    def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "mean_ms": median(run["ocr_ms"]["mean"] for run in runs),
            "p95_ms": median(run["ocr_ms"]["p95"] for run in runs),
            "peak_ws_mib": median(run["peak_rss_bytes"] / MIB for run in runs),
            "warmup_rss_mib": median(run["rss_after_warmup_bytes"] / MIB for run in runs),
            "lines": runs[0]["lines"],
            "output_checksum": runs[0]["output_checksum"],
            "backend": runs[0]["backend"],
        }

    baseline = aggregate(baseline_runs)
    candidate = aggregate(candidate_runs)
    return {
        "model": model,
        "workers": workers,
        "baseline": baseline,
        "candidate": candidate,
        "paired_speedup": median(
            a["ocr_ms"]["mean"] / b["ocr_ms"]["mean"]
            for a, b in zip(baseline_runs, candidate_runs)
        ),
        "peak_ws_delta_mib": candidate["peak_ws_mib"] - baseline["peak_ws_mib"],
        "warmup_rss_delta_mib": candidate["warmup_rss_mib"] - baseline["warmup_rss_mib"],
        "output_match": (
            baseline["lines"] == candidate["lines"]
            and baseline["output_checksum"] == candidate["output_checksum"]
        ),
        "rounds": [
            {"order": order, "baseline": a, "candidate": b}
            for order, a, b in zip(orders, baseline_runs, candidate_runs)
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = [
        "# x64 sample OCR: current vs release",
        "",
        f"Baseline: `{report['baseline_ref']}` (`{report['baseline_commit']}`); "
        f"candidate: `{report['candidate_commit']}`.",
        f"Runner: `{report['runner']['platform']}`; logical CPUs: "
        f"`{report['runner']['logical_cpus']}`.",
        f"Sample SHA-256: `{report['sample_sha256']}`; REC width: "
        f"`{report['rec_target_width']}`; warm-up: `{report['warmup']}`; "
        f"iterations: `{report['iterations']}`; paired rounds: `{report['paired_rounds']}`.",
        "",
        "| Model | Workers | Release ms | Current ms | Speedup | Release peak WS MiB | Current peak WS MiB | Δ peak MiB | Text |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in report["cases"]:
        old, new = case["baseline"], case["candidate"]
        rows.append(
            f"| {case['model']} | {case['workers']} | {old['mean_ms']:.3f} | "
            f"{new['mean_ms']:.3f} | {case['paired_speedup']:.3f}× | "
            f"{old['peak_ws_mib']:.1f} | {new['peak_ws_mib']:.1f} | "
            f"{case['peak_ws_delta_mib']:+.1f} | "
            f"{'same' if case['output_match'] else 'different'} |"
        )
    rows += [
        "",
        "Each revision uses the same `lw-ocr-benchmark` source, model files, "
        "dictionary and 500×500 PPM on one Windows x64 runner. "
        "Each measurement is a fresh process; run order alternates by round.",
        "`Release ms` and `Current ms` are medians of per-process OCR means; "
        "speedup is the median of paired release/current ratios. "
        "Peak WS is the process peak working set, including model initialization "
        "and the benchmark's standalone detector handle. Positive Δ means "
        "the current revision uses more memory.",
        "Text differences are reported, not silently treated as equal. "
        "Latency and memory are informational and are not CI gates on a hosted runner.",
        "",
    ]
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-driver", type=Path, required=True)
    parser.add_argument("--candidate-driver", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--paired-rounds", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if not (1 <= args.warmup <= 20 and 1 <= args.iterations <= 100
            and 1 <= args.paired_rounds <= 10 and 30 <= args.timeout_seconds <= 3600):
        parser.error("warmup, iterations, paired rounds, or timeout is out of range")
    baseline_driver = args.baseline_driver.resolve(strict=True)
    candidate_driver = args.candidate_driver.resolve(strict=True)
    sample = args.sample.resolve(strict=True)
    assets_root = args.assets_root.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_assets: dict[str, dict[str, str]] = {}
    for model in MODELS:
        model_dir = assets_root / model
        model_assets[model] = {
            name: file_sha256((model_dir / name).resolve(strict=True))
            for name in ASSET_NAMES
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_ref": args.baseline_ref,
        "baseline_commit": args.baseline_commit,
        "candidate_commit": args.candidate_commit,
        "runner": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "sample_sha256": file_sha256(sample),
        "model_assets_sha256": model_assets,
        "rec_target_width": 960,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "paired_rounds": args.paired_rounds,
        "cases": [],
    }
    for model in MODELS:
        for workers in WORKERS:
            old_runs: list[dict[str, Any]] = []
            new_runs: list[dict[str, Any]] = []
            orders: list[str] = []
            for round_index in range(args.paired_rounds):
                order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
                orders.append("-".join(order))
                for label in order:
                    driver = baseline_driver if label == "baseline" else candidate_driver
                    raw_path = output / "raw" / model / f"w{workers}" / f"round-{round_index + 1}-{label}.json"
                    payload = run_driver(
                        driver, assets_root / model, sample, args.warmup,
                        args.iterations, workers, 960, args.timeout_seconds, raw_path,
                    )
                    (old_runs if label == "baseline" else new_runs).append(payload)
            report["cases"].append(summarize_case(model, workers, old_runs, new_runs, orders))
            (output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
            )
            (output / "report.md").write_text(
                render_markdown(report), encoding="utf-8", newline="\n"
            )
    # The Markdown is already persisted as UTF-8 and published by the workflow.
    # Windows CI may give Python a cp1252 stdout, which cannot encode Delta/multiplication signs.
    print("[release-compare] report.json and report.md written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
