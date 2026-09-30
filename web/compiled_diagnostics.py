"""Recognize successful compiled-backend status lines emitted on stderr."""
import re


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
    return _STATUS.fullmatch(message) is not None
