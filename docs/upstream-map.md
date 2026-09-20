# NHWC fast-path upstream map

The analysis-only layout planner is an algorithmic reimplementation. It does not
copy source code from the upstream project and does not change the public C ABI
or the LWM format.

| Local file | Upstream reference | Upstream commit | Migration type | Notes |
| --- | --- | --- | --- | --- |
| `src/runtime/layout_planner.c` | `OnnxSharp/LayoutPlanner.cs` in `sdcb/SimdPaddleOCR` | `e6921a9c2186ae607160224d4f9f148fbf0784af` | algorithmic reimplementation | Conservative NHWC capability and island analysis only; no execution path is enabled. |
| `tests/layout_plan_driver.c` | Layout-plan diagnostics in the same upstream runtime | `e6921a9c2186ae607160224d4f9f148fbf0784af` | independent test/diagnostic | Prints per-node layout decisions and conversion counts for REC960 snapshots. |

The upstream project is Apache-2.0 licensed. This planner contains no copied
Apache-2.0 implementation text; any future intrinsic/kernel translation must be
added here with its exact source commit and corresponding notice review.

The experimental NHWC pointwise kernel is an independent implementation of the repository design document; it does not copy source code from SimdPaddleOCR and remains outside the production target.
