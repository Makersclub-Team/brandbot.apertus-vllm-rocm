# SPDX-License-Identifier: Apache-2.0
# Original integration packaging. Upstream components retain their licenses.
# The input is the local ROCm engine image built by scripts/build.py.
ARG APERTUS_ENGINE_IMAGE
FROM ${APERTUS_ENGINE_IMAGE}

ARG APERTUS_VLLM_COMMIT
ARG APERTUS_TRANSFORMERS_COMMIT
ARG APERTUS_TRANSFORMERS_REPOSITORY=https://github.com/swiss-ai/transformers.git

LABEL org.opencontainers.image.title="Community Apertus 1.5 ROCm integration" \
      org.opencontainers.image.description="Experimental build recipe; GPU inference not validated" \
      org.apertus.vllm.commit="${APERTUS_VLLM_COMMIT}" \
      org.apertus.transformers.commit="${APERTUS_TRANSFORMERS_COMMIT}" \
      org.apertus.validation="unvalidated"

USER root
WORKDIR /opt/apertus

# Install only the pinned Transformers source. Preserve the base's ROCm torch.
# Missing/incompatible Python dependencies must be fixed explicitly; pip check
# below fails instead of silently replacing the GPU stack. The pinned revision
# (5.14.0.dev0) refuses to import unless huggingface-hub<2.0 and
# tokenizers<=0.23.0, so both are pinned to its ranges. That leaves two known
# conflicts with what vLLM declares, both allowed (known_dependency_conflicts in
# versions.json): transformers>=5.16.1,<5.21.0 and tokenizers>=0.23.2. The
# pygobject/pycairo line is an Ubuntu system package and unrelated. Any other
# pip check line fails.
RUN git clone --filter=blob:none --no-checkout \
      "${APERTUS_TRANSFORMERS_REPOSITORY}" transformers-src \
    && git -C transformers-src fetch origin "${APERTUS_TRANSFORMERS_COMMIT}" \
    && git -C transformers-src checkout --detach "${APERTUS_TRANSFORMERS_COMMIT}" \
    && test "$(git -C transformers-src rev-parse HEAD)" = "${APERTUS_TRANSFORMERS_COMMIT}" \
    && uv pip install --system ./transformers-src \
         "huggingface-hub>=1.31.0,<2.0" "tokenizers>=0.22.0,<=0.23.0" \
    && { python3 -m pip check > /tmp/pip-check.txt 2>&1 || true; } \
    && cat /tmp/pip-check.txt \
    && if grep -v -E '^No broken requirements found\.$|^vllm [^ ]+ (has requirement|requires) transformers[^,]*,? ?[^ ]*, but you have transformers 5\.14\.0\.dev0\.?$|^vllm [^ ]+ (has requirement|requires) tokenizers>=0\.23\.2, but you have tokenizers 0\.2(2\.[0-9]+|3\.0)\.?$|^pygobject [^ ]+ requires pycairo, which is not installed\.$' /tmp/pip-check.txt | grep -q .; then \
         echo "Unexpected dependency conflicts (see above)" >&2; exit 1; \
       fi

COPY scripts/check_runtime.py /opt/apertus/check_runtime.py
COPY LICENSE ATTRIBUTION.md /opt/apertus/integration-notices/
COPY versions.json /opt/apertus/source-versions.json

# Build-time imports verify the Python integration only, without requiring GPUs.
RUN python3 /opt/apertus/check_runtime.py \
      --output /opt/apertus/build-import-check.json

ENTRYPOINT ["vllm", "serve"]
CMD ["--help"]
