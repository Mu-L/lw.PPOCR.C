"""Rank physical REC ops and real adjacent windows, never across invocations.

Instrumented Node timings identify candidates; they are not browser E2E A/B.
Adjacency is not a proof of data dependency, single-consumer ownership or
fusion eligibility. Pair/triple windows overlap and their totals cannot be
added together as potential savings.
"""
import argparse
import contextlib
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def fields(line):
    result = {}
    for token in line.split()[1:]:
        key, value = token.split("=", 1)
        if key in result:
            raise ValueError("Duplicate profile field: " + key)
        result[key] = value
    return result


def parse_log(text, skip_ocr_runs=0):
    traces, active, seen, boundaries = [], None, set(), []
    for line in text.splitlines():
        if line.startswith("LW_WASM_OCR_PROFILE "):
            if active is not None:
                raise ValueError("OCR boundary inside an unfinished REC invocation")
            boundaries.append(len(traces))
        if not line.startswith("WASM_REC_OP"):
            continue
        data = fields(line)
        identity = (int(data["width"]), int(data["invocation"]))
        if line.startswith("WASM_REC_OP_BEGIN "):
            if active is not None or identity in seen or min(identity) < 1:
                raise ValueError("Overlapping, repeated or invalid invocation")
            seen.add(identity)
            active = dict(width=identity[0], invocation=identity[1],
                          expected=int(data["ops"]), ops=[])
            if active["expected"] < 1:
                raise ValueError("Empty physical program")
        elif active is None or identity != (active["width"], active["invocation"]):
            raise ValueError("Profile entry outside its invocation")
        elif line.startswith("WASM_REC_OP_END "):
            if len(active["ops"]) != active["expected"]:
                raise ValueError("Incomplete physical program trace")
            traces.append(active)
            active = None
        elif line.startswith("WASM_REC_OP "):
            index, elapsed = int(data["index"]), float(data["elapsed"])
            if index != len(active["ops"]) or index >= active["expected"]:
                raise ValueError("Missing, duplicate or unordered op index")
            if int(data["status"]) != 0 or elapsed < 0 or not math.isfinite(elapsed):
                raise ValueError("Failed op or invalid duration")
            active["ops"].append(dict(index=index, semantic=int(data["semantic"]),
                kind=data["kind"], elapsed_ms=elapsed,
                metadata={k: v for k, v in data.items() if k not in
                    {"width", "invocation", "index", "semantic", "span", "kind", "elapsed", "status"}}))
        else:
            raise ValueError("Unknown physical profile marker")
    if active is not None:
        raise ValueError("Truncated profile invocation")
    if skip_ocr_runs:
        if skip_ocr_runs < 0 or len(boundaries) <= skip_ocr_runs or boundaries[-1] != len(traces):
            raise ValueError("Cannot skip OCR warm-up without complete full-OCR boundaries")
        traces = traces[boundaries[skip_ocr_runs - 1]:]
    if not traces:
        raise ValueError("No complete physical REC traces")
    return traces


def summarize(traces, limit=15):
    windows = {}
    by_kind = defaultdict(float)
    by_width = defaultdict(lambda: dict(invocations=0, elapsed_ms=0.0))
    for trace in traces:
        by_width[trace["width"]]["invocations"] += 1
        for op in trace["ops"]:
            by_kind[op["kind"]] += op["elapsed_ms"]
            by_width[trace["width"]]["elapsed_ms"] += op["elapsed_ms"]
    for length, title in [(1, "ops"), (2, "pairs"), (3, "triples")]:
        totals = {}
        for trace in traces:
            for start in range(len(trace["ops"]) - length + 1):
                group = trace["ops"][start:start + length]
                identity = (trace["width"], tuple(
                    (op["index"], op["semantic"], op["kind"]) for op in group))
                if identity not in totals:
                    totals[identity] = dict(width=trace["width"],
                        indices=[op["index"] for op in group],
                        semantics=[op["semantic"] for op in group],
                        kinds=[op["kind"] for op in group],
                        metadata=[op["metadata"] for op in group], calls=0, elapsed_ms=0.0)
                totals[identity]["calls"] += 1
                totals[identity]["elapsed_ms"] += sum(op["elapsed_ms"] for op in group)
        ordered = sorted(totals.values(), key=lambda x: (-x["elapsed_ms"], x["width"], x["indices"]))
        for row in ordered:
            row["mean_ms"] = row["elapsed_ms"] / row["calls"]
        windows[title] = ordered[:limit]
    return dict(schema_version=1, instrumented=True, trace_count=len(traces),
        total_backbone_ms=sum(by_kind.values()),
        by_kind=dict(sorted(by_kind.items(), key=lambda item: -item[1])),
        by_width=dict(sorted(by_width.items())), **windows)


def render(report):
    lines = ["# WASM REC physical-op profile", "",
        f"Complete REC invocations: {report['trace_count']}. "
        f"Instrumented backbone total: {report['total_backbone_ms']:.3f} ms.", "",
        f"Excluded whole-OCR warm-ups: {report.get('skipped_ocr_warmups', 0)}.", "",
        "Diagnostic only: not browser E2E. CLS is excluded; CTC head, DET, CLS "
        "and crop timing remain in the separate stage profile.", "",
        "Adjacent windows overlap. Their elapsed time is NOT estimated fusion "
        "savings, and adjacency alone does not prove fusion eligibility.", "",
        "| Physical kind | Total ms | Backbone share |", "| --- | ---: | ---: |"]
    for kind, elapsed in report["by_kind"].items():
        share = elapsed / report["total_backbone_ms"] if report["total_backbone_ms"] else 0
        lines.append(f"| {kind} | {elapsed:.3f} | {share:.2%} |")
    for key, title in [("ops", "Top individual physical ops"),
                       ("pairs", "Top adjacent pairs"), ("triples", "Top adjacent triples")]:
        lines.extend(["", "## " + title, "",
            "| Width | Indices | Semantic nodes | Pattern | Calls | Total ms | Mean ms |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: |"])
        for row in report[key]:
            indices = "/".join(map(str, row["indices"]))
            semantics = "/".join(map(str, row["semantics"]))
            lines.append(f"| {row['width']} | {indices} | {semantics} | "
                f"{' -> '.join(row['kinds'])} | {row['calls']} | "
                f"{row['elapsed_ms']:.3f} | {row['mean_ms']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--skip-ocr-runs", type=int, default=0,
                        help="Exclude whole OCR warm-ups using LW_WASM_OCR_PROFILE boundaries")
    parser.add_argument("--stage-output", type=Path,
                        help="Also validate/write existing DET/CLS/REC/CTC/memory profile JSON")
    parser.add_argument("--variant", choices=["tiny", "small"], default="tiny")
    args = parser.parse_args()
    if args.limit < 1 or args.skip_ocr_runs < 0:
        parser.error("limit must be positive and skip-ocr-runs nonnegative")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = summarize(parse_log(args.log.read_text(encoding="utf-8"), args.skip_ocr_runs), args.limit)
    report["source_log"] = str(args.log)
    report["skipped_ocr_warmups"] = args.skip_ocr_runs
    text = render(report)
    if args.stage_output:
        # Reuse the established stage/memory parser instead of another profiler.
        try:
            from .compare_wasm_rec_benchmark import print_full_ocr_profile, write_profile_json
        except ImportError:
            from compare_wasm_rec_benchmark import print_full_ocr_profile, write_profile_json
        args.stage_output.parent.mkdir(parents=True, exist_ok=True)
        write_profile_json(args.log, args.stage_output, args.variant)
        stage = json.loads(args.stage_output.read_text(encoding="utf-8"))
        if len(stage["rec_invocations"]) < report["trace_count"]:
            raise ValueError("CTC stage profile has fewer invocations than the physical trace")
        retained = stage["rec_invocations"][-report["trace_count"]:]
        report["ctc_total_ms"] = sum(row["ctc"] for row in retained)
        text += (f"\nCTC head across the same {report['trace_count']} retained REC calls: "
                 f"{report['ctc_total_ms']:.3f} ms (outside backbone/pair totals).\n")
        stage_text = io.StringIO()
        with contextlib.redirect_stdout(stage_text):
            print_full_ocr_profile(args.log)
        text += "\n## Last instrumented full OCR (not browser A/B)\n\n" + stage_text.getvalue()
    for path, content in [(args.output, json.dumps(report, indent=2) + "\n"), (args.markdown, text)]:
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
