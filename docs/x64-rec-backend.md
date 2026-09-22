# x64 REC compiled backend (v9 bootstrap)

The v9 path is a separate, opt-in backend contract. It is intentionally not
part of the public C ABI and does not replace the existing x64_rec_fast_*
prototype yet.

lw_x64_rec_backend_compile() probes a model/width and returns
LW_X64_REC_COMPILE_OK only when the complete graph is already suitable for an
all-or-nothing backend:

- graph input is direct NHWC;
- no generic physical nodes remain;
- no runtime layout conversions are required;
- no unsupported NHWC nodes remain.

Any failed condition returns LW_X64_REC_COMPILE_UNSUPPORTED; callers must
fall back to the canonical recognizer as a whole. There is no per-node fallback
inside this backend contract.

A successful compile creates a ref-counted model descriptor. Width-specific
instances borrow that descriptor and own their session/arena state. The input
accessor is intended for lw_rec_preprocess_bgr_u8_nhwc() so preprocessing can
write directly into the backend input arena.

The current PP-OCRv6 Tiny graph still has an NCHW stem and terminal CTC
integration is not compiled into this bootstrap. It therefore correctly
returns UNSUPPORTED instead of claiming a partially compiled backend. The
next v9 increment will lower the stem into dedicated NHWC Dense/Conv operations,
then add the CTC projection epilogue and recognizer integration.