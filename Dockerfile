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
# below fails instead of silently replacing the GPU stack.
RUN git clone --filter=blob:none --no-checkout \
      "${APERTUS_TRANSFORMERS_REPOSITORY}" transformers-src \
    && git -C transformers-src fetch origin "${APERTUS_TRANSFORMERS_COMMIT}" \
    && git -C transformers-src checkout --detach "${APERTUS_TRANSFORMERS_COMMIT}" \
    && test "$(git -C transformers-src rev-parse HEAD)" = "${APERTUS_TRANSFORMERS_COMMIT}" \
    && uv pip install --system --no-deps ./transformers-src \
    && python3 -m pip check

COPY scripts/check_runtime.py /opt/apertus/check_runtime.py
COPY LICENSE ATTRIBUTION.md /opt/apertus/integration-notices/
COPY versions.json /opt/apertus/source-versions.json

# Build-time imports verify the Python integration only, without requiring GPUs.
RUN python3 /opt/apertus/check_runtime.py \
      --output /opt/apertus/build-import-check.json

ENTRYPOINT ["vllm", "serve"]
CMD ["--help"]
