# Validation and release evidence

## Current status

At preparation on **17 November 2026**, and after re-pinning to Swiss AI's `apertus-1-5` vLLM branch (ROCm 10.0.0) on **18 November 2026**, this bundle has **not built an image or run inference on a GPU**. The build recipe, target architecture, and smoke scripts are implementation work, not evidence of a working deployment. Update this document with actual results as each stage completes; never infer a later stage from an earlier one.

## Validation stages

| Stage | Evidence required | Initial status |
| --- | --- | --- |
| Source checks | Build-wrapper argument validation, script syntax, manifest consistency, and inspection of the dry-run command | Passed: 18 offline tests and Python syntax checks; see ../source-checks.json |
| Image build | Complete successful build log, resolved dependencies, final image digest | Not run |
| Runtime imports | `scripts/check_runtime.py` runs inside that exact image; reports installed versions, architecture availability, and parser availability | Not run |
| GPU text | Model loads on the intended AMD GPU and produces sensible text through the API | Not run |
| Tools and streaming | Correct structured tool calls, complete streamed responses, and successful tool-result continuation | Not run |
| Image and audio | Correct responses to known media fixtures through the documented multimodal API | Not run |
| Tensor parallelism | Successful model load and requests with the advertised TP configuration | Not run |
| Concurrency | Recorded overlapping requests complete with the stated context sizes and no unexplained failures | Not run |

## Establish the runtime first

By default `scripts/build.py` uses AMD's prebuilt `rocm/vllm-dev:base` at the digest in `versions.json`; `--build-base` builds it from the pinned source's `docker/Dockerfile.rocm_base` instead. If you pass another base with `--base-image`, use its real digest. Preserve the full build command.

The image deliberately tolerates two `pip check` conflicts (see `known_dependency_conflicts` in `versions.json`): vLLM requires `transformers>=5.16.1,<5.21.0` and `tokenizers>=0.23.2`, while the pinned Transformers is 5.14.0.dev0 and needs `tokenizers<=0.23.0`. Treat any vLLM import, tokenization or model-load error that involves Transformers or tokenizers as possibly caused by this, and record it. `--dry-run` previews commands and performs no image validation. The default `gfx950` target controls compilation; it does not prove that the build or host driver works on MI355X.

For RunPod MI300X testing, compile with `--gpu-arch gfx942` and identify the result as an MI300X run. Validate MI355X separately with `gfx950`.

Run `scripts/check_runtime.py` in the built container before downloading a large checkpoint. An import failure, missing Apertus 1.5 architecture, wrong PyTorch build, or unavailable parser is a failed prerequisite. A passing import check still does not prove that GPU kernels or model inference work.

Use the 8B checkpoint for initial API validation where practical. Then test the exact checkpoint and configuration intended for the release. A passing 8B run does not establish that 70B, quantization, long contexts, or TP=2 work.

## Exercise the advertised behavior

Use `scripts/smoke.py` according to its CLI help for the checks it implements. Record other checks separately; the presence of a smoke script does not imply that it covers every row above.

For text, inspect answer quality as well as HTTP success. For tool calls, use a harmless deterministic function fixture, validate its name and JSON arguments, return a fixed result, and verify the continuation. Test the documented non-thinking tool endpoint independently from a thinking endpoint.

For streaming, reconstruct the answer or tool call from deltas and check completion and parsing. For images and audio, keep a small reproducible fixture with an expected observable answer. Text-only success provides no evidence about either encoder. Keep the original precision baseline before assessing quantized weights or KV cache.

For TP=2, record the two devices and verify real requests after initialization. For concurrency, state simultaneous request count, input and output lengths, context limit, precision, and failures. Report latency and throughput only from measured runs, including warm-up treatment and test duration.

## Attach a reproducible result record

Each report should include the date; GPU model, count, memory, and topology; host driver and ROCm versions; container digest and base digest; Python, PyTorch, Transformers, and vLLM versions; all source SHAs; model identifier and revision; tokenizer/encoder revisions; launch command; precision; TP size; context and batching limits; test command; pass/fail counts; and relevant error logs.

Record only the capabilities actually exercised. Keep credentials and private input data out of reports. An experimental image becomes a hardware-tested release only when its published claims can be traced to this evidence.
