import unittest
from tools.summarize_wasm_rec_ops import parse_log, render, summarize


def trace(width, invocation, kinds, times):
    lines = [f"WASM_REC_OP_BEGIN width={width} invocation={invocation} ops={len(kinds)}"]
    for index, (kind, elapsed) in enumerate(zip(kinds, times)):
        lines.append(f"WASM_REC_OP width={width} invocation={invocation} index={index} "
                     f"semantic={index + 10} span=1 kind={kind} elapsed={elapsed} status=0")
    lines.append(f"WASM_REC_OP_END width={width} invocation={invocation}")
    return "\n".join(lines)


class RecOpsSummaryTest(unittest.TestCase):
    def test_windows_do_not_cross_invocations_or_widths(self):
        log = trace(192, 1, ["pointwise", "depthwise"], [1, 2]) + "\n" + \
              trace(320, 2, ["pointwise", "dense"], [5, 7])
        report = summarize(parse_log(log))
        self.assertEqual(report["pairs"][0]["elapsed_ms"], 12)
        self.assertEqual(len(report["pairs"]), 2)
        self.assertEqual(report["triples"], [])
        self.assertEqual(report["total_backbone_ms"], 15)

    def test_rank_by_elapsed_not_calls(self):
        log = "\n".join([trace(192, i, ["depthwise"], [1]) for i in range(1, 4)])
        log += "\n" + trace(320, 4, ["pointwise"], [10])
        report = summarize(parse_log(log))
        self.assertEqual(report["ops"][0]["kinds"], ["pointwise"])
        self.assertEqual(report["ops"][1]["calls"], 3)

    def test_triples_are_diagnostic_not_savings(self):
        report = summarize(parse_log(trace(192, 1, ["pointwise", "depthwise", "pointwise"], [1, 2, 3])))
        self.assertEqual(report["triples"][0]["elapsed_ms"], 6)
        self.assertIn("NOT estimated fusion savings", render(report))

    def test_reject_incomplete_or_failed_traces(self):
        good = trace(192, 1, ["pointwise", "depthwise"], [1, 2])
        for bad in ["", good.rsplit("\n", 1)[0], good.replace("status=0", "status=1"),
                    good.replace("index=1", "index=0"), good.replace("elapsed=2", "elapsed=nan"),
                    good + "\n" + good]:
            with self.subTest(log=bad), self.assertRaises(ValueError):
                parse_log(bad)

    def test_skip_whole_ocr_warmup(self):
        log = trace(192, 1, ["dense"], [100]) + "\nLW_WASM_OCR_PROFILE total=100\n"
        log += trace(320, 2, ["pointwise"], [2]) + "\nLW_WASM_OCR_PROFILE total=2\n"
        report = summarize(parse_log(log, skip_ocr_runs=1))
        self.assertEqual(report["trace_count"], 1)
        self.assertEqual(report["total_backbone_ms"], 2)
        with self.assertRaises(ValueError):
            parse_log(trace(192, 1, ["dense"], [100]), skip_ocr_runs=1)


if __name__ == "__main__":
    unittest.main()
