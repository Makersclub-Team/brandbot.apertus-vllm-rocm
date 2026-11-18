# Attribution and component licenses

This project provides original integration packaging for a community Apertus 1.5
ROCm build. It does not claim to be an official Swiss AI, AMD, or vLLM release.
The packaging in this repository is offered under Apache License 2.0; see LICENSE.

The model implementation belongs to its upstream contributors. The build fetches
it at exact commits instead of copying a second model implementation into this
repository.

| Component | Source | License reference |
| --- | --- | --- |
| vLLM and Swiss AI integration | https://github.com/swiss-ai/vllm/tree/640bad371eed16edfe40665721e8db028481bdbc | https://github.com/swiss-ai/vllm/blob/640bad371eed16edfe40665721e8db028481bdbc/LICENSE |
| Matching Transformers integration | https://github.com/swiss-ai/transformers/tree/3797303dda74844e3d1f8977ff5518bb91f818b4 | https://github.com/swiss-ai/transformers/blob/3797303dda74844e3d1f8977ff5518bb91f818b4/LICENSE |
| Official launch recipe used as a technical reference | https://github.com/swiss-ai/model-launch/blob/main/images/vllm_apertus_1.5_release/Dockerfile | https://github.com/swiss-ai/model-launch/blob/main/LICENSE |
| Apertus model weights, downloaded separately by the user | https://huggingface.co/swiss-ai/Apertus-v1.5-8B and https://huggingface.co/swiss-ai/Apertus-v1.5-70B | Refer to each model's current license and access conditions |

Preserve upstream copyright, license, and applicable NOTICE files in distributions.
Mark modified upstream files and describe the changes when contributing patches.
The image's ROCm, PyTorch, tokenizer, operating-system, and other dependencies
retain their own licenses; this document is not a complete binary-image inventory.
Before distributing a built image, collect its resolved package/license inventory
and carry required notices forward. Do not describe the entire image as licensed
solely under this repository's license.

Upstream contribution: https://github.com/vllm-project/vllm/pull/50496

No model weights, Hugging Face tokens, proprietary application code, or downloaded
model caches are included in this source bundle.
