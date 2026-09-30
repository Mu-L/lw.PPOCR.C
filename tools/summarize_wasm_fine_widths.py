"""Browser-only fine-width A/B summary; failed text contracts are diagnostic."""
import argparse
import json
import statistics
import sys
from pathlib import Path


def render(report, model):
    runs = report["runs"]
    def median(label, key):
        return statistics.median(item[key] for item in runs[label])
    passed = report.get("text_contract_pass") is True
    lines = [
        f"## {model}: five versus thirteen REC widths",
        "",
        f"Text contract: **{'PASS' if passed else 'FAIL — diagnostic only'}**.",
        f"Paired ratio: {report['paired_speedup']:.3f}x. "
        "Latency is informational; this does not certify corpus accuracy.",
        "",
        "| Metric | Five widths | Thirteen widths |",
        "| --- | ---: | ---: |",
    ]
    if report.get("texts_identical") is False:
        lines.insert(3, "Outputs differ; PASS means each build matches its own reviewed golden, not byte-identical A/B.")
    for title, key, divisor in (
        ("Full OCR median ms", "median_ms", 1),
        ("Page navigation to ready ms", "ready_ms", 1),
        ("CLS engine rebuild ms", "cls_reconfigure_ms", 1),
        ("First OCR ms", "first_ocr_ms", 1),
        ("Linear heap MiB (not process RSS)", "heap_bytes", 1048576),
    ):
        lines.append(f"| {title} | {median('baseline', key)/divisor:.2f} | "
                     f"{median('candidate', key)/divisor:.2f} |")
    lines.extend(["", f"Sample SHA-256: `{report['sample_sha256']}`. "
                  f"Chromium: `{report['browser_version']}`. "
                  f"Warm-up: {report['warmup']}; calls per round: {report['iterations']}."])
    baseline = runs["baseline"][0].get("text_lines", [])
    candidate = runs["candidate"][0].get("text_lines", [])
    for index, (before, after) in enumerate(zip(baseline, candidate)):
        if before != after:
            lines.extend(["", f"Line {index}: `{before}` → `{after}`."])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for model in ("tiny", "small"):
        path = args.directory / (model + "-ab.json")
        if path.is_file():
            print(render(json.loads(path.read_text(encoding="utf-8")), model.title()))
        else:
            print(f"## {model.title()}\n\nNo completed report; inspect earlier job errors.")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
