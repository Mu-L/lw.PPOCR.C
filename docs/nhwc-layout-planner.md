# NHWC layout planner snapshots

The current planner is analysis-only. It never changes tensor storage, the
executor dispatch, the public C ABI, or the LWM format. It answers one narrow
question: which REC graph nodes could form a future NHWC fast-path island at
`[1, 3, 48, 960]` under the conservative capability rules?

The checked-in contract is [`ci/nhwc-layout-planner.json`](../ci/nhwc-layout-planner.json). The planner now treats layout availability as a physical twin cache: requiring an already available NCHW/NHWC twin does not increment the conversion count twice. Neutral rank-4 tensors (`C == 1` or `H == W == 1`) do not start islands or require conversions. Resize eligibility reads IEEE-754 float scales and only accepts unchanged N/C dimensions with positive integer spatial scales. ReduceMean is limited to keepdims H/W reduction, and Concat is limited to non-constant rank-4 channel concatenation. Direct NHWC graph input is enabled only when the first consumer can start a dense Conv or ConvTranspose NHWC segment.
The report tool invokes the private `layout-plan-driver`, validates the exact
summary counters, and stores every node's selected layout in JSON:

```powershell
python tools/run_layout_plan_snapshot.py `
  --build-dir build `
  --model build/models/rec.lwm `
  --variant tiny `
  --contract ci/nhwc-layout-planner.json `
  --output build/tiny-layout-plan.json
```

Current REC960 snapshots from the checked-in Tiny model and the locally
converted Small/Medium validation models are:

| Variant | Tensors | Nodes | NHWC nodes | NCHW nodes | Islands | Conversions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny | 274 | 159 | 127 | 32 | 8 | 4 |
| Small | 563 | 336 | 206 | 130 | 12 | 4 |
| Medium | 595 | 355 | 221 | 134 | 14 | 4 |

The Small and Medium workflows reuse the `converted-runtime/rec.lwm` produced
by their existing model validation jobs. A snapshot mismatch fails the job
before any SIMD kernel work is attempted. These numbers are planning evidence,
not a performance claim; the next phase must still benchmark each candidate
kernel against the current production NCHW path.
