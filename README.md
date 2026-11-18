# Apertus 1.5 on ROCm — community integration kit

**Experimental. Built and run on an AMD Instinct MI350X (gfx950) on
18 November 2026: Apertus 1.5 8B served text, tool calls, image and audio
input (audio after the fix in `1539243`). MI300X, 70B, tensor
parallelism and long context are untested. No prebuilt image is published.**
Details: [validation](docs/VALIDATION.md).

This repository prepares a shareable ROCm build for Apertus 1.5. It builds
Swiss AI's vLLM branch for Apertus 1.5 with that revision's own ROCm Dockerfiles,
then installs the matching Transformers source. It is a starting point for
integration and hardware validation, not a claim of verified MI355X support.

## Versions

| Component | Pin |
| --- | --- |
| vLLM | `swiss-ai/vllm` branch `apertus-1-5` at `640bad3` (29 Oct 2026), the head of [vLLM PR #50496](https://github.com/vllm-project/vllm/pull/50496); based on upstream main between v0.33.0rc3 and v0.34.0 |
| ROCm | 10.0.0 (AMD TheRock SDK), PyTorch `2.13.0+rocm10.0.0`, from AMD's prebuilt `rocm/vllm-dev:base` (29 Oct 2026, pinned by digest), which matches the pinned source's `docker/Dockerfile.rocm_base` |
| Transformers | `swiss-ai/transformers` at `3797303` (5.14.0.dev0) |

**Known dependency conflict.** vLLM at this revision declares
`transformers>=5.16.1,<5.21.0`, but no Transformers revision fits that range and
matches vLLM's Apertus 1.5 code. That code reads
`Apertus1p5Config.vision_tokenizer_config` / `audio_tokenizer_config`, which the
pinned 5.14.0.dev0 revision has. The Apertus 1.5 model merged into
`huggingface/transformers` on 17 Nov 2026 (5.24.0.dev0) renamed them to
`vision_config` / `audio_config` and changed the vision tokenizer's `encode()`
result, so vLLM's code cannot load the model with it. The pinned revision also
refuses to import unless `huggingface-hub<2.0` and `tokenizers<=0.23.0`, so the
image pins both; vLLM declares `tokenizers>=0.23.2`, a second known conflict.
The image build allows exactly these two `pip check` conflicts (plus an unrelated
Ubuntu `pygobject` note) and fails on any other. vLLM, the Apertus 1.5 model
class and the Apertus tool and reasoning parsers import with these versions;
whether vLLM serves correctly with them is unverified until a GPU run. Re-pin both when Swiss AI updates PR #50496 to the merged
Transformers model. Details: `known_dependency_conflicts` in `versions.json`.

## What is included

- Exact Swiss vLLM and Transformers commits in `versions.json`.
- A source-build wrapper on AMD's matching prebuilt ROCm 10 base (pinned by digest),
  or on a base built from the pinned source.
- A thin runtime Dockerfile that preserves the base's ROCm PyTorch installation.
- Import/GPU diagnostics and basic OpenAI API smoke checks.
- Source-only CI and clear validation/publication guidance.
- Apache 2.0 license and upstream attribution.

The exact source pins originally came from the [Swiss AI release recipe](https://github.com/swiss-ai/model-launch/blob/main/images/vllm_apertus_1.5_release/Dockerfile),
which still pins the July vLLM revision on CUDA.

## Get the source

```bash
git clone https://github.com/Makersclub-Team/brandbot.apertus-vllm-rocm.git
cd brandbot.apertus-vllm-rocm
```

## RunPod testing

Select the compilation target for the actual GPU:

| GPU | `--gpu-arch` |
| --- | --- |
| AMD Instinct MI300X | `gfx942` |
| AMD Instinct MI355X | `gfx950` |

The build examples below use `gfx950`. For an MI300X Pod, pass
`--gpu-arch gfx942` to both commands. Confirm the device in the Pod before
choosing a target; a CPU image label such as `amd64` does not identify its GPU
runtime.

Ordinary RunPod Pods do not support a Docker daemon or Docker Compose inside the
Pod. Run this build wrapper on a host with Docker, then deploy the resulting
image through a Pod template. Inside that Pod, run the diagnostic script and
vLLM directly. This project does not yet provide an installer for adapting an
arbitrary existing Pod environment; Python source and compiled ROCm extensions
must be compatible.

One RunPod test has been completed: a single MI350X (gfx950) with the 8B model,
recorded in [validation](docs/VALIDATION.md). MI300X (gfx942) is untested; record
MI300X and MI350X/MI355X validation separately. See [RunPod GPU types](https://docs.runpod.io/references/gpu-types)
and [Pod limitations](https://docs.runpod.io/pods/overview).

## Build a local experimental image

Use Linux x86-64 with Git, Python 3.10+, Docker, and Buildx. The wrapper uses
Docker's default builder so the second build can access the intermediate local
image. Building the source stack needs substantial CPU, RAM, disk, and network
access. A GPU is not used to validate inference during compilation.

By default the wrapper builds on AMD's prebuilt ROCm 10 base,
`rocm/vllm-dev:base`, pinned by digest in `versions.json`. Its image labels list
the same components as the pinned source's `docker/Dockerfile.rocm_base` (ROCm
10.0.0, PyTorch 2.13.0, Triton `669b31ac`, AITER v0.1.26, flash-attention
`0e60e394`, MORI v1.2.3.post1). It then builds the upstream `vllm-openai` target,
which compiles vLLM's GPU kernels for `--gpu-arch`, and this repository's runtime
layer. Preview the plan, then build:

```bash
python3 scripts/build.py --tag apertus-vllm-rocm:experimental --gpu-arch gfx950 --dry-run

python3 scripts/build.py --tag apertus-vllm-rocm:experimental --gpu-arch gfx950 --max-jobs 32
```

`--build-base` builds the base from the pinned source instead (several more hours:
it compiles Triton, AITER and flash-attention). `--base-image repository@sha256:...`
uses any other base; a digest makes the selected bytes explicit, it does not
establish that the base is compatible.

The wrapper fetches the pinned source into `.build/vllm`. It does not push an
image. The final layer installs Transformers with `--no-deps`; missing dependencies,
any dependency conflict other than the documented one, or import failures stop the
build instead of silently replacing ROCm PyTorch with another build. Investigate
and pin any required dependency additions.

The resulting `.build/build-report.json` records source pins, which base was used,
and the local image ID. Upstream build steps still resolve some packages dynamically, so
this is not yet a hermetic or bit-for-bit reproducible build. Record the resolved
dependencies for any release.

The Python-only/precompiled route may eventually reduce build work. This initial
recipe uses a source build to avoid inventing a compatible ROCm wheel/base pair.

## Inspect the runtime on the AMD host

```bash
mkdir -p results
docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --ipc=host \
  -v "$PWD/results:/results" \
  --entrypoint python3 \
  apertus-vllm-rocm:experimental \
  /opt/apertus/check_runtime.py \
  --require-gpu --output /results/runtime.json
```

The ROCm PyTorch API uses `torch.cuda` names for AMD devices; the diagnostic checks
`torch.version.hip` to distinguish the runtime. Permissions on `/dev/kfd` and
`/dev/dri` depend on the host configuration.

## Serve the 8B model

Accept the model repository's current access conditions using your own account
and provide `HF_TOKEN` in your environment if required. Set
`APERTUS_MODEL_REVISION` to the exact accepted model commit before recording a
reproducible validation result. Model weights are downloaded into a separate volume.

```bash
docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --ipc=host \
  -e HF_TOKEN \
  -p 127.0.0.1:8000:8000 \
  -v apertus-model-cache:/root/.cache/huggingface \
  apertus-vllm-rocm:experimental \
  swiss-ai/Apertus-v1.5-8B \
  --revision "$APERTUS_MODEL_REVISION" \
  --served-model-name apertus-1.5-8b \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --chat-template-content-format string \
  --enable-auto-tool-choice --tool-call-parser apertus
```

These are initial validation settings, not performance recommendations. For 70B,
change the model ID and configure tensor parallelism for the actual GPU memory.
Validate text, then streaming and tools, before increasing context or concurrency.

```bash
python3 scripts/smoke.py --model apertus-1.5-8b \
  --output results/chat-and-stream.json

python3 scripts/smoke.py --model apertus-1.5-8b --tools \
  --output results/chat-stream-tools.json
```

For a separate thinking-mode endpoint, replace the automatic tool flags with
`--reasoning-parser apertus --default-chat-template-kwargs.enable_thinking true`.
Combining tools and thinking depends on the exact parser/template versions; it is
outside this starter package's automated smoke checks. One image and one audio
fixture were checked on an MI350X (see [validation](docs/VALIDATION.md)); long
context, tensor parallelism, quantization, and load behavior still require
separate testing.

## Share the work

This repository shares the experimental source and validation tools. Publish an OCI image
under a versioned tag once the build and corresponding checks have actually run;
provide its immutable digest and evidence. Avoid a `stable` or `validated-mi355x`
label until it is supported by recorded tests. See [sharing](docs/SHARING.md).

Keep functional model fixes upstream in [Swiss AI's fork](https://github.com/swiss-ai/vllm)
and [vLLM PR #50496](https://github.com/vllm-project/vllm/pull/50496). When released
upstream support covers the tested configuration, simplify or retire the custom
layer. This package does not claim ownership of the Apertus implementation.

## Local source checks

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
```

These checks do not download weights, build a container, or validate a GPU.

## License

Original packaging: Apache 2.0. Upstream code and image dependencies retain their
licenses. See [ATTRIBUTION.md](ATTRIBUTION.md) and [LICENSE](LICENSE).
