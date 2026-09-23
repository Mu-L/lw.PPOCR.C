# x64 sample OCR release comparison

Run the manual `x64 sample OCR release comparison` GitHub Actions workflow to
compare the selected branch/commit with a published release tag. The default
baseline is `v1.0.0`. This is an informational performance and memory report,
not a release gate.

The workflow checks out both source revisions on **one Windows x64 runner** and
builds `lw-ocr-benchmark` with the same Visual Studio toolchain. It verifies
that the benchmark source, PPM loader, SIMD-reporting source, and public C ABI
header are byte-identical across the revisions. If the harness changes, the
workflow stops rather than silently compare different measurement methods.
Both builds use the same explicit CMake arguments; revision-specific Runtime
feature defaults remain intact. This is a product-version comparison, not an
equal-kernel-policy experiment.

The current checkout converts Tiny, Small, and Medium once. Both executables
receive the **same** DET/CLS/REC LWM files, dictionary, and bundled 500×500
sample PPM. This intentionally isolates Runtime changes; it does not compare
different model-pack releases. REC target width is fixed at 960. Each model is
measured with 1 and 4 OCR workers, for six cases total.

Within each case, the workflow runs fresh baseline/current processes in
alternating order. Defaults are two warm-up OCR calls, five measured calls per
process, and three paired rounds. The report shows the median of per-process
OCR mean latency, the median paired release/current speedup, process peak
working set, and its current-minus-release difference. The peak includes model
initialization and the benchmark's standalone detector handle; it is not a
per-operator workspace figure. `rss_after_warmup_bytes`, per-round samples,
model/sample SHA-256, text checksum, and both commit IDs are retained in the
JSON artifact. A text difference is displayed explicitly but is not mistaken
for performance parity.

From the Actions page, select **x64 sample OCR release comparison**, choose the
branch or commit to run, and keep `baseline_tag=v1.0.0` unless comparing with a
newer published tag. The job summary contains the six-row table; download the
`x64-sample-release-comparison-*` artifact for `report.json`, `report.md`, and
all raw benchmark outputs. Hosted-runner CPU placement and load can change, so
latency and memory are not threshold-gated. Compare only results paired within
the same run; repeat the workflow before drawing conclusions from small deltas.
