# Apertus 1.5 on ROCm — community integration kit

**Experimental source package. No container build or GPU inference has been run
for this project. No prebuilt image is published.**

This repository prepares a shareable ROCm build for Apertus 1.5. It reuses Swiss
AI's pinned vLLM implementation and that revision's own ROCm Dockerfile, then
installs the matching Transformers source. It is a starting point for integration
and hardware validation, not a claim of verified MI355X support.

## What is included

- Exact Swiss vLLM and Transformers commits in `versions.json`.
- A source-build wrapper with an explicit ROCm base-image digest requirement.
- A thin runtime Dockerfile that preserves the base's ROCm PyTorch installation.
- Import/GPU diagnostics and basic OpenAI API smoke checks.
- Source-only CI and clear validation/publication guidance.
- Apache 2.0 license and upstream attribution.

There is deliberately no selected ROCm base digest: the source branch uses an
older ROCm build layout, and an arbitrary current base may be incompatible.
Selecting and validating that dependency combination is the next engineering
step. The exact source pins come from the [Swiss AI release recipe](https://github.com/swiss-ai/model-launch/blob/main/images/vllm_apertus_1.5_release/Dockerfile).

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

No RunPod GPU test has been completed for this project. Record MI300X and MI355X
validation separately. See [RunPod GPU types](https://docs.runpod.io/references/gpu-types)
and [Pod limitations](https://docs.runpod.io/pods/overview).

## Build a local experimental image

Use Linux x86-64 with Git, Python 3.10+, Docker, and Buildx. The wrapper uses
Docker's default builder so the second build can access the intermediate local
image. Building the source stack needs substantial CPU, RAM, disk, and network
access. A GPU is not used to validate inference during compilation.

1. Select a ROCm **development** base compatible with the pinned vLLM source,
   including its PyTorch and compiler requirements. Inspect the upstream
   [`docker/Dockerfile.rocm_base`](https://github.com/swiss-ai/vllm/blob/a601a9d998ddeb488f0c17e8512874b116aa7658/docker/Dockerfile.rocm_base).
2. Resolve the actual image digest and set `APERTUS_ROCM_BASE_IMAGE` to its full
   `repository@sha256:...` reference. A digest makes the selected bytes explicit;
   it does not establish that the base is compatible.
3. Preview the plan, then build:

```bash
python3 scripts/build.py \
  --base-image "$APERTUS_ROCM_BASE_IMAGE" \
  --tag apertus-vllm-rocm:experimental \
  --gpu-arch gfx950 \
  --dry-run

python3 scripts/build.py \
  --base-image "$APERTUS_ROCM_BASE_IMAGE" \
  --tag apertus-vllm-rocm:experimental \
  --gpu-arch gfx950 \
  --max-jobs 8
```

The wrapper fetches the pinned source into `.build/vllm`, builds the upstream
`vllm-openai` target, and adds this repository's runtime layer. It does not push an
image. The final layer installs Transformers with `--no-deps`; missing dependencies
or import failures stop the build instead of silently replacing ROCm PyTorch with
another build. Investigate and pin any required dependency additions.

The resulting `.build/build-report.json` records source pins, base digest, and
local image ID. Upstream build steps still resolve some packages dynamically, so
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
outside this starter package's automated smoke checks. Image/audio correctness,
long context, tensor parallelism, quantization, and load behavior also require
separate testing; see [validation](docs/VALIDATION.md).

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
