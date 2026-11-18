# Validation and release evidence

## Current status

On **18 November 2026** an image built from this repository at commit `e6c3d7a` ran on one **AMD Instinct MI350X** (RunPod). Apertus 1.5 8B served text, tool calls and image input. Audio input failed because the image lacked vLLM's audio extra. Commit `1539243` adds the extra and a build-time audio check; an image built from it ran on a second MI350X the same day and **audio passed**, with text, tool call and image results unchanged. Nothing else below has been run. Update this document with actual results as each stage completes; never infer a later stage from an earlier one.

## Validation stages

| Stage | Evidence required | Status |
| --- | --- | --- |
| Source checks | Build-wrapper argument validation, script syntax, manifest consistency, and inspection of the dry-run command | Passed: offline tests and Python syntax checks; see ../source-checks.json |
| Image build | Complete successful build log, resolved dependencies, final image digest | Passed for `e6c3d7a` and `1539243` (`--gpu-arch "gfx942;gfx950"`, AMD prebuilt base, 18 Nov 2026) |
| Runtime imports | `scripts/check_runtime.py` runs inside that exact image | Passed at build time, and with `--require-gpu` on the MI350X (`1539243` adds `vllm.audio_support`, passed) |
| GPU text | Model loads on the intended AMD GPU and produces sensible text through the API | Passed: 8B on MI350X (English and German answers correct) |
| Tools and streaming | Correct structured tool calls, complete streamed responses, and successful tool-result continuation | Partial: one tool call correct; streaming and tool-result continuation not run |
| Image and audio | Correct responses to known media fixtures through the documented multimodal API | Passed on `1539243`: image (one synthetic fixture) and audio (one synthetic speech clip, `input_audio` and `audio_url`). Audio had failed on `e6c3d7a` (`Please install vllm[audio]`) |
| Tensor parallelism | Successful model load and requests with the advertised TP configuration | Not run |
| Concurrency | Recorded overlapping requests complete with the stated context sizes and no unexplained failures | Partial: 16 and 64 concurrent 512-token requests, no errors |
| MI300X (gfx942) | Any of the above on MI300X | Not run |
| 70B | Any of the above with Apertus 1.5 70B | Not run |

## Result: MI350X, 18 November 2026

| Item | Value |
| --- | --- |
| GPU | 1 × AMD Instinct MI350X, gfx950, 288 GB (RunPod Secure Cloud) |
| Host driver | amdgpu 7.1.3 (`rocm-smi`) |
| Image | built from this repository at `e6c3d7a`; engine digest `sha256:4e2e60f72546aad52875009a75ad4f5ad882f9e430e9aff2097119ae8fe4972f` |
| Base | `rocm/vllm-dev:base@sha256:ee81d993f614a9924149c909525be35eec1bcc036458a6047cc90786e5f5b18f` |
| Versions | Python 3.12.16, PyTorch 2.13.0+rocm10.0.0, Transformers 5.14.0.dev0 (`3797303`), vLLM `640bad3` (reports `0.23.1rc1.dev6474+g640bad371`) |
| Model | `swiss-ai/Apertus-v1.5-8B` at `1519de0ae7f3d114879291810a58259966f64c99`, bfloat16, TP=1 |
| Launch | `vllm serve swiss-ai/Apertus-v1.5-8B --max-model-len 16384 --enable-auto-tool-choice --tool-call-parser apertus` (defaults otherwise; weights preloaded, `HF_HUB_OFFLINE=1`) |

| Check | Result |
| --- | --- |
| `check_runtime.py --require-gpu` | passed (all checks, GPU FP32 matmul on gfx950) |
| bf16 matmul 8192×8192 (PyTorch) | about 1,131 TFLOPS |
| vLLM with a small control model (Qwen2.5-0.5B-Instruct) | served; attention backend `ROCM_ATTN`, AITER kernels loaded |
| Model load | weights 2 s, 17.23 GiB on the GPU; server ready about 3 minutes after container start |
| Chat, English | "The capital of Switzerland is Bern, and the language most widely spoken there is German." |
| Chat, German ("17 mal 23?") | "17 mal 23 ergibt 391." (correct) |
| Tool call (`get_weather`, `tool_choice: auto`) | `get_weather({"city": "Zurich"})`, finish reason `tool_calls` |
| Image (224×224 solid red PNG, "What colour?") | "Red" |
| Audio (synthetic 16 kHz WAV, `input_audio` and `audio_url`) | HTTP 500 `Please install vllm[audio] for audio support` |
| Throughput, 512 output tokens, `ignore_eos` | 1 request about 245 tok/s; 16 concurrent about 2,700 tok/s; 64 concurrent about 9,300 tok/s; no errors (measured through the RunPod HTTP proxy, includes network time, no warm-up control) |

Logged warnings, with correct results despite them: `hipblasLt` `HIPBLAS_STATUS_INTERNAL_ERROR` for some GEMM shapes during graph capture; xIELU uses the PyTorch fallback (the fused kernel is CUDA-only); aiter recommends disabling NUMA balancing on the host.

## Result: MI350X audio re-test, 18 November 2026

Same GPU type, host driver, base, versions, model revision and launch command as above, on a new Pod. Image: built from this repository at `1539243` with the 8B weights preloaded; engine digest `sha256:a2323e1dc840f5fa3d300bf94a81c220c3fd8415b45f3b86fd2116ba471bb7ab`.

| Check | Result |
| --- | --- |
| `check_runtime.py --require-gpu` | passed (all checks including `vllm.audio_support`, GPU FP32 matmul on gfx950) |
| Server ready | about 3 minutes after the container started |
| Audio, `input_audio` (synthetic TTS WAV, 16 kHz mono: "The meeting is moved to Thursday at three o'clock.", "Transcribe this audio exactly.") | "the meeting is moved to thursday at 3 o'clock" |
| Audio, `audio_url` data URL, same clip | "the meeting is moved to thursday at 3 o'clock" |
| Audio question ("On which day is the meeting?") | "Thursday" |
| Chat English, chat German, tool call, image | same answers as the first run |

Additional logged warning: `torchcodec` is not installed, so vLLM decodes audio with its fallback backend; the answers above were correct with it. Throughput and concurrency were not re-measured.

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
