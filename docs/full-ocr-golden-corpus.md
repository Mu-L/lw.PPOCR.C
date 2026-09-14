# Full-OCR golden corpus

The v0.1 full-OCR regression gate is versioned in
`tests/fixtures/ocr-golden-corpus.json`. The manifest pins the SHA-256 of the
source image, DET/CLS/REC ONNX models, and recognition dictionary so a model or
fixture change cannot silently reuse stale expectations.

See [OCR orientation and reading-order contract](ocr-orientation-contract.md) for the
coordinate, crop-local rotation, CLS, and reading-order semantics asserted by
the manifest.

`full_ocr_golden_corpus` runs seven deterministic cases through the public
three-model OCR pipeline:

- the original 500x500 image with direction classification enabled;
- the same image with direction classification disabled;
- integer-defined nearest-neighbor resizing to 375x375 and 750x750;
- the source on a 500x600 white canvas;
- a byte-exact 90-degree counterclockwise rotation;
- a 640x192 white image with no text.

The derived pixels do not depend on Pillow or OpenCV resizing implementations.
The test defines nearest-neighbor indices with integer arithmetic and uses NumPy
only to apply those indices, rotations, and canvas operations. Pillow decodes
the one bundled JPEG, as it does in the existing reference tests.

Every case freezes the exact UTF-8 result and reading order, detected count,
DET resize shape, classifier labels, and applied rotations. Scores use explicit
lower bounds instead of exact floating-point comparison. Every quadrilateral
must remain within the source image and have nonzero area. The original image
also compares all eight coordinates of all 16 boxes with a two-pixel tolerance.
The blank case freezes the valid zero-line/zero-text result.

This corpus broadens pipeline regression coverage across scale, aspect ratio,
empty input, classifier options, and rotated text, but it is derived from one
redistributable source image. It is not a general OCR accuracy benchmark.
Future additions should prefer independently sourced redistributable images,
record their hashes, and review changed expectations separately from runtime
optimizations.

## Project-owned generated datasets

For repeatable model comparison, use the repository-owned generator instead of
depending on an external evaluator or an external image corpus. The generator
records the seed, Pillow version, font file names and SHA-256 values, image
dimensions, image SHA-256 values, text categories, font settings, orientation,
rotation, and bounded non-overlapping line boxes in `metadata.json`. If a
requested density does not fit a particular canvas, the generator records the
requested and placed line counts instead of painting one text line over another.
The manifest also includes a corpus identifier (`lw-ppocr-c-project-v1`) and
orientation policy so a changed project text pool cannot silently reuse an
older baseline. It also stores a SHA-256 of the exact text pool. The default
corpus uses only the orientations currently
covered by the CLS contract: `0` and `180` degrees.
The default text pool is project-specific: it covers PP-OCRv6 model profiles,
DET/CLS/REC, LWM and WASM, Android/Java and C ABI integration, SIMD backends,
PDF handling, reading order, and TXT/JSON export.

Generated images and reports belong under `build-local-data/`, which is ignored
by Git. The generator, manifest schema, evaluator, and unit tests are tracked;
the generated image files are not. This keeps CI and release packages small
while allowing any developer to reproduce the same local dataset with the
same seed, Pillow version, and fonts.

The parser, generator, pairwise comparator, multi-model summary, and regression
checker are included in the CMake `model_analysis` test. Run it with the
configuration appropriate for the generator, for example:

```bash
ctest --test-dir build -C Release -R model_analysis --output-on-failure
```

Vertical `90/-90` degree text is intentionally an optional stress corpus rather
than part of the core regression baseline. Generate it explicitly with
`--include-vertical`; its manifest receives the separate
`lw-ppocr-c-project-v1-vertical` identifier.

Generate the default 100-image local corpus on Windows with CJK fonts:

```bash
python tools/generate_ocr_dataset.py \
  --output build-local-data/lw-generated-ocr \
  --count 100 \
  --seed 20260907 \
  --font C:/Windows/Fonts/msyh.ttc \
  --font C:/Windows/Fonts/simsun.ttc
```

Evaluate that local corpus with the project-native OCR driver:

```bash
python tools/evaluate_ocr_dataset.py \
  --dataset build-local-data/lw-generated-ocr \
  --driver build/Release/lw-ocr-ppm.exe \
  --detector build/models/det.lwm \
  --classifier build/models/cls.lwm \
  --recognizer build/models/rec.lwm \
  --dictionary models/ppocrv6-tiny/ppocr_keys.txt \
  --model-name ppocrv6-tiny \
  --rec-max-width 960 \
  --output build-local-data/tiny-generated-ocr-960-core-full.json
```

For the Small profile, keep the shared Tiny CLS model and switch only DET,
REC, and the dictionary:

```bash
python tools/evaluate_ocr_dataset.py \
  --dataset build-local-data/lw-generated-ocr \
  --driver build/Release/lw-ocr-ppm.exe \
  --detector build-model-foundation/ppocrv6-small-det-dynamic.lwm \
  --classifier build/models/cls.lwm \
  --recognizer build-model-foundation/ppocrv6-small-rec-dynamic.lwm \
  --dictionary models/ppocrv6-shared/PP-OCRv6_small_rec_dict.txt \
  --model-name ppocrv6-small \
  --rec-max-width 960 \
  --output build-local-data/small-generated-ocr-960-core-full.json
```

For the Medium profile, use the converted Medium DET/REC pair with the same
Tiny CLS model and the shared Small/Medium recognition dictionary:

```bash
python tools/evaluate_ocr_dataset.py \
  --dataset build-local-data/lw-generated-ocr \
  --driver build/Release/lw-ocr-ppm.exe \
  --detector build-model-foundation/medium-analysis/full-ocr/medium-det-dynamic.lwm \
  --classifier build/models/cls.lwm \
  --recognizer build-model-foundation/medium-analysis/full-ocr/medium-rec-dynamic.lwm \
  --dictionary models/ppocrv6-shared/PP-OCRv6_small_rec_dict.txt \
  --model-name ppocrv6-medium \
  --rec-max-width 960 \
  --output build-local-data/medium-generated-ocr-960-core-full.json
```

The three profiles therefore differ only in DET/REC capacity and dictionary
where applicable: Tiny uses its Tiny dictionary, while Small and Medium share
`PP-OCRv6_small_rec_dict.txt` and reuse the Tiny CLS model. Keep the REC width
at 960 for an apples-to-apples comparison; changing it creates a different
performance/accuracy experiment.

The evaluator verifies image hashes and dimensions, converts JPEG input to
temporary PPM, invokes the existing native driver, and matches predicted boxes
to generated boxes by one-to-one greedy IoU. It reports detection
precision/recall/F1, matched-box IoU, CER on matched lines, exact reference-line
rate, missing lines, and extra lines. The report also contains grouped metrics
by text category, orientation, and canvas size; category/orientation groups
report matched-line recall, IoU, Exact Line Rate, and CER, while canvas groups
retain full detection precision and F1. It is a local model-analysis tool, not
a release gate. Generated images must not be added to Git; any future
checked-in fixture still needs an independent provenance and redistribution
review.

To compare two reports from the same generated corpus, use the project-owned
report comparator. It refuses to compare different manifest hashes, seeds,
REC widths, or IoU thresholds:

```bash
python tools/compare_ocr_dataset_reports.py \
  --baseline build-local-data/tiny-generated-ocr-960-core-full.json \
  --candidate build-local-data/small-generated-ocr-960-core-full.json \
  --output build-local-data/tiny-vs-small-core.json
```

The comparison records candidate-minus-baseline deltas for the overall report
and every category, orientation, and canvas group.

For a compact multi-profile table, pass the reports to the summary tool. The
first report is the baseline and every other report must use the same manifest,
REC width, and IoU threshold:

```bash
python tools/summarize_ocr_dataset_reports.py \
  --report Tiny=build-local-data/tiny-generated-ocr-960-core-full.json \
  --report Small=build-local-data/small-generated-ocr-960-core-full.json \
  --report Medium=build-local-data/medium-generated-ocr-960-core-full.json \
  --output build-local-data/tiny-small-medium-summary.json
```

When a model or kernel change needs an explicit no-regression check, use the
regression tool with absolute metric tolerances (`0.01` means one percentage
point). All three thresholds are required so that a CI gate cannot silently
inherit an unsuitable policy:

```bash
python tools/check_ocr_dataset_regression.py \
  --baseline build-local-data/tiny-generated-ocr-960-core-full.json \
  --candidate build-local-data/small-generated-ocr-960-core-full.json \
  --max-f1-drop 0.01 \
  --max-exact-drop 0.02 \
  --max-cer-increase 0.01 \
  --output build-local-data/tiny-vs-small-regression.json
```

The command exits nonzero only when a threshold is exceeded. It always checks
that both reports use the same generated manifest, REC width, and IoU matching
threshold before evaluating accuracy deltas.

For example, compare Medium against both smaller profiles:

```bash
python tools/compare_ocr_dataset_reports.py \
  --baseline build-local-data/tiny-generated-ocr-960-core-full.json \
  --candidate build-local-data/medium-generated-ocr-960-core-full.json \
  --output build-local-data/tiny-vs-medium-core.json

python tools/compare_ocr_dataset_reports.py \
  --baseline build-local-data/small-generated-ocr-960-core-full.json \
  --candidate build-local-data/medium-generated-ocr-960-core-full.json \
  --output build-local-data/small-vs-medium-core.json
```

For reference, a local run on the 20260907 core corpus (614 reference lines)
produced the following non-gating snapshot:

| Profile | Detection F1 | Exact Line Rate | CER |
|---|---:|---:|---:|
| Tiny | 99.35% | 58.14% | 3.65% |
| Small | 99.92% | 78.66% | 2.01% |
| Medium | 100.00% | 80.62% | 1.26% |

These numbers are for model selection and regression investigation, not a
release promise. Re-run the commands above after changing a model, converter,
dictionary, font, or runtime setting; do not silently update a baseline when a
metric changes.
