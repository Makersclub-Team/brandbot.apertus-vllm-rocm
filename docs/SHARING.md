# Sharing this integration

## What can be shared now

This project is a **community, experimental Apertus 1.5 integration for vLLM on ROCm**. Its build recipe and validation tools can be published as source before GPU testing is complete, provided that the README and release notes clearly state the validation status. At preparation on 17 November 2026, no container build or GPU inference run has been completed for this bundle. It is not an official Swiss AI, vLLM, or AMD release.

Keep the public repository small: the version manifest, build wrapper, container layer, launch examples, smoke checks, and validation reports belong here. The model implementation should continue to come from its pinned upstream repository. This makes the integration easier to review and allows it to retire once ordinary upstream releases cover the same use case.

## Source and image releases are different milestones

An initial source release may be labelled `experimental` or `source-only`. Record the exact dependency revisions and the checks that actually ran. A successful dry run only demonstrates that the wrapper can construct the intended build command; it does not demonstrate compilation or inference.

The wrapper requires a real, compatible ROCm base image by digest through `--base-image`. Its defaults are `--tag apertus-vllm-rocm:experimental`, `--gpu-arch gfx950`, and `--max-jobs 8`; `--dry-run` previews the work. Select and verify the base before attempting a build. A digest ensures repeatability of that input but does not establish compatibility with the pinned vLLM source.

After a successful build, an OCI image may be published as an **unvalidated experimental build**, accompanied by its digest and build log. Describe a release as tested on a specific GPU only after the relevant runtime tests in [VALIDATION.md](VALIDATION.md) pass on that hardware. Record MI300X (`gfx942`) and MI355X (`gfx950`) results separately. State separately whether 8B, 70B, multimodal requests, tensor parallelism, and concurrency were tested. Compilation for `gfx950` alone is not MI355X validation.

## Distribute the runtime without model weights

The public image should contain the inference runtime and integration code. Keep model checkpoints, downloaded encoder weights, Hugging Face caches, and customer data outside the image. Users supply the appropriate model access and mount or download weights at runtime. Publish the tested model identifier and revision so others can reproduce the result.

Do not include access tokens in Docker arguments, image layers, example commands, logs, or release artifacts. Runtime credentials stay with the operator. A source release does not require access to a user's model account.

## License and attribution

Use the repository's Apache 2.0 license for its original integration code and documentation. Preserve upstream copyright notices and any applicable `NOTICE` files when redistributing upstream code. Identify Swiss AI's Apertus implementation and vLLM as upstream projects; publishing this wrapper does not transfer their authorship.

Dependencies, encoders, model weights, and base-image components retain their own licenses and access conditions. Record those components and source revisions in the release manifest. An Apache 2.0 license on the wrapper does not relicense every component inside an OCI image or the weights downloaded later.

## Contributing back

Send narrowly scoped model or ROCm runtime fixes to [swiss-ai/vllm](https://github.com/swiss-ai/vllm), and coordinate changes intended for standard vLLM with its [existing Apertus 1.5 integration](https://github.com/vllm-project/vllm/pull/50496). Keep packaging changes separate from model behavior changes. Swiss AI's current [model-launch documentation](https://github.com/swiss-ai/model-launch/blob/main/docs/building-images.md) assigns maintained images to a private images repository; access to submit a PR there is not assumed. A public integration repository and reproducible reports remain useful while maintainers decide how to adopt the work.
