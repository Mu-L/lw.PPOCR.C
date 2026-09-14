# OCR orientation contract

This document defines the orientation and reading-order semantics of the
public full-OCR pipeline. It is intentionally separate from the model format
and from the C ABI: it describes what the existing result fields mean.

## Coordinate and crop semantics

The detector reports each text region as eight image-space coordinates in the
order `top-left`, `top-right`, `bottom-right`, `bottom-left`. Coordinates are
measured in the source image after any caller-owned image preparation. The
runtime does not rotate or mutate the caller's pixels.

Before recognition, the quadrilateral is perspective-cropped. A very tall
crop is rotated into the horizontal orientation expected by REC. This is a
crop-local canonicalization step; it is not a claim that the source page has
been globally rotated upright.

The crop and the detector box remain independent. A crop-local 90-degree
canonicalization must not be reported as a CLS result, and it must not change
the detector box coordinates returned to the caller.

## CLS semantics

CLS is an optional crop-local classifier. It can report only the two model
labels currently supported by the pipeline:

| Result field | Meaning |
| --- | --- |
| `classification_label` | Model label (`0` or `1`) |
| `classification_score` | Selected softmax score for that label |
| `applied_rotation_degrees` | Actual 180-degree correction applied to the crop (`0` or `180`) |

`applied_rotation_degrees` is not a page orientation estimate. It records the
correction applied after crop canonicalization. The runtime never rotates
caller-owned image memory. A future page-orientation classifier must use a new
contract instead of changing these fields.

## Reading order

Reading order is applied after detection and is independent of CLS. The
current explicit modes are:

- `horizontal_ltr`: horizontal lines, left to right and top to bottom;
- `vertical_rtl`: vertical columns, right to left and top to bottom;
- `vertical_ltr`: vertical columns, left to right and top to bottom.

There is no implicit `AUTO` mode in the stable contract. Applications that
need automatic selection must inspect the page or provide their own policy and
then choose one of the explicit modes.

## Regression contract

`tests/fixtures/ocr-golden-corpus.json` carries an
`orientation_contract` object. The Golden test validates that every case
declares its source orientation and explicit reading order, and that every
reported CLS rotation is one of the supported values. This catches accidental
changes such as treating a tall-crop rotation as a 180-degree CLS result or
silently introducing an undocumented reading-order default.

The checked-in corpus is a deterministic seed corpus derived from the bundled
sample image. It is a semantic/runtime regression gate, not a general OCR
accuracy benchmark. Independently sourced vertical-writing and perspective
images should be added only after their provenance and expected text have
been manually reviewed.

Run the focused gate with:

```bash
ctest --test-dir build -C Release -R full_ocr_golden_corpus --output-on-failure
```