"""Recognize successful compiled-backend status lines emitted on stderr."""
import re
import argparse
from pathlib import Path

_WIDTHS = re.compile(r"LW_WASM_COMPILED_REC widths=(\d+)/(\d+)")


_STATUS = re.compile(
    r"(?:LW_WASM_COMPILED_REC widths=\d+/\d+|"
    r"LW_WASM_COMPILED_REC ctc=simd128|"
    r"LW_WASM_REC_LAZY_FALLBACK full_coverage=[01] canonical_retained=[01]|"
    r"LW_WASM_COMPILED_CLS layout=nhwc ops=\d+ unsupported=0 "
    r"arena_bytes=\d+ packed_bytes=\d+|"
    r"LW_WASM_COMPILED_DET input=\d+x\d+ layout=nhwc ops=\d+ "
    r"unsupported=0 conversions=\d+ direct_input=[01] "
    r"arena_bytes=\d+ packed_bytes=\d+)"
)


def is_compiled_status(message):
    widths = _WIDTHS.fullmatch(message)
    if widths:
        compiled, total = map(int, widths.groups())
        return total > 0 and compiled == total
    return _STATUS.fullmatch(message) is not None


def assert_compiled_widths(log, expected):
    matches = _WIDTHS.findall(log)
    if not matches or any(int(a) != expected or int(b) != expected for a, b in matches):
        raise AssertionError(f"Expected full compiled coverage {expected}/{expected}, got {matches}")
    return expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate compiled REC width coverage")
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-widths", type=int, required=True)
    args = parser.parse_args()
    if args.expected_widths < 1:
        parser.error("expected-widths must be positive")
    assert_compiled_widths(args.log.read_text(encoding="utf-8"), args.expected_widths)
    print(f"Compiled REC widths: {args.expected_widths}/{args.expected_widths}")
