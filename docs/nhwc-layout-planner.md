# NHWC layout planner snapshots

The current planner is analysis-only. It never changes tensor storage, the
executor dispatch, the public C ABI, or the LWM format. It answers one narrow
question: which REC graph nodes could form a future NHWC fast-path island at
`[1, 3, 48, 960]` under the conservative capability rules?

The checked-in contract is [`ci/nhwc-layout-planner.json`](../ci/nhwc-layout-planner.json).
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
| Tiny | 274 | 159 | 121 | 38 | 5 | 10 |
| Small | 563 | 336 | 200 | 136 | 8 | 18 |
| Medium | 595 | 355 | 238 | 117 | 9 | 19 |

The Small and Medium workflows reuse the `converted-runtime/rec.lwm` produced
by their existing model validation jobs. A snapshot mismatch fails the job
before any SIMD kernel work is attempted. These numbers are planning evidence,
not a performance claim; the next phase must still benchmark each candidate
kernel against the current production NCHW path.