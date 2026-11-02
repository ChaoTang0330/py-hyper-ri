# syntax=docker/dockerfile:1

# Keep the two bases on the same PyTorch, CUDA, cuDNN and Python versions.
ARG PYTORCH_RUNTIME_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca
ARG PYTORCH_DEVEL_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel@sha256:b574d4ccf6d8856a5d87dcadc667aa4f95dc18d337ef3a28d02b7b01897d7081

FROM ${PYTORCH_DEVEL_IMAGE} AS development

LABEL org.opencontainers.image.source="https://github.com/ChaoTang0330/py-hyper-ri" \
      org.opencontainers.image.licenses="BSD-3-Clause"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_CONSTRAINT=/opt/pytorch-constraints.txt

# Capture this base's versions without pulling the runtime image.
RUN python - <<'PY'
from importlib.metadata import distributions
from pathlib import Path

names = {"torch", "torchvision", "torchaudio", "triton"}
constraints = []
for dist in distributions():
    name = dist.metadata["Name"]
    normalized_name = name.lower().replace("_", "-").replace(".", "-")
    if normalized_name in names or normalized_name.startswith(("nvidia-", "cuda-")):
        constraints.append(f"{name}=={dist.version}")

Path("/opt/pytorch-constraints.txt").write_text(
    "\n".join(sorted(constraints)) + "\n", encoding="utf-8"
)
PY

WORKDIR /workspace/py-hyper-ri
COPY . .
RUN python -m pip install -e ".[all]" \
    && python -m pip check \
    && hyri --help

ENTRYPOINT ["/bin/bash"]

# This final stage is also the default when no --target is supplied.
FROM ${PYTORCH_RUNTIME_IMAGE} AS runtime

LABEL org.opencontainers.image.source="https://github.com/ChaoTang0330/py-hyper-ri" \
      org.opencontainers.image.licenses="BSD-3-Clause"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_CONSTRAINT=/opt/pytorch-constraints.txt

# Capture this base's versions without pulling the development image.
RUN python - <<'PY'
from importlib.metadata import distributions
from pathlib import Path

names = {"torch", "torchvision", "torchaudio", "triton"}
constraints = []
for dist in distributions():
    name = dist.metadata["Name"]
    normalized_name = name.lower().replace("_", "-").replace(".", "-")
    if normalized_name in names or normalized_name.startswith(("nvidia-", "cuda-")):
        constraints.append(f"{name}=={dist.version}")

Path("/opt/pytorch-constraints.txt").write_text(
    "\n".join(sorted(constraints)) + "\n", encoding="utf-8"
)
PY

# pip builds and installs a wheel from a temporary source mount. The writable
# mount lets setuptools create build metadata; those writes are discarded.
RUN --mount=type=bind,target=/build,rw \
    python -m pip install "/build[cuda,ms]" \
    && python -m pip check \
    && hyri --help

COPY LICENSE CITATION.cff /usr/local/share/pyHyperRI/
WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
