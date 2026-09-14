# LWM compatibility policy

This policy defines how a runtime and a `.lwm` model file may evolve. The
loader treats model bytes as untrusted input and validates the header, version,
flags, table ranges, tensor/node references, parameter records, dimensions,
size arithmetic, and content checksum before exposing a model handle.

## Current format

The repository currently emits and validates **LWM v0.1**. The format is
little-endian, uses fixed-width records, and keeps reserved fields zero. The
canonical graph and FP32 weights are portable deployment data; they do not
contain CPU-specific prepared buffers.

## Compatibility rules

| Change | Policy |
| --- | --- |
| Same major/minor | Existing valid files remain readable; only the documented v0.1 contract is allowed today. |
| New major version | Breaking layout or semantic change; older runtimes reject it with `LW_STATUS_UNSUPPORTED_VERSION`. |
| New minor version | Additive change only; a runtime may accept it only after explicitly implementing the added records/features. |
| Unknown required feature | Reject with a stable status; never guess a tensor, operator, parameter, or workspace meaning. |
| Unknown optional metadata | Ignore only when the record is explicitly marked optional and its byte range is validated. |
| Old valid model on newer runtime | Must remain readable unless the model uses a deliberately retired feature. |
| Newer model on older runtime | Must fail closed with `LW_STATUS_UNSUPPORTED_VERSION` or another precise format status. |

A version number alone is not a permission to reinterpret unknown bytes. The
validator remains the authority for all offsets, counts, operator IDs, and
parameter versions.

## Runtime-local optimizations

Prepared execution, packed Conv1x1/Conv3x3 weights, FMA dispatch, NHWC/layout
choices, cache decisions, and worker scheduling remain Runtime-local. They must
not be serialized into LWM v0.1. A model converted once must remain usable by
Windows x64, Linux ARM64, LoongArch64, and WASM without embedding a CPU-specific
kernel choice in the file.

The runtime may derive these structures during session creation or preparation,
subject to the configured workspace and memory limits. Reusing them across
sessions is an implementation detail and does not change the model contract.

## Release and test requirements

Every format change must update the converter, validator, model metadata tests,
and the compatibility manifest together. Before release, run at least:

```bash
ctest --test-dir build -C Release -R 'lwm_loader|lwm_corruption|model_analysis' --output-on-failure
```

The public C ABI candidate and the cross-product version snapshot are separate
contracts. They must not be inferred from the LWM version, and changing one
requires an explicit compatibility review for the others.