# Build with --build-arg TRAINING_BASE_IMAGE=<PyTorch CUDA image>@sha256:<verified digest>.
# No experiment config, Vertex hardware, or mutable DINOv3 weight download is baked in.
ARG TRAINING_BASE_IMAGE
FROM ${TRAINING_BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
RUN python -m pip install --no-cache-dir uv==0.12.3
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra train --extra data --extra cloud --extra telemetry
ENV PATH="/opt/venv/bin:${PATH}"
ENTRYPOINT ["python", "-m", "defect_platform.trainer.runner"]
