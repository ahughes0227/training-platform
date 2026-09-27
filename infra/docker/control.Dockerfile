# Build with --build-arg PYTHON_BASE_IMAGE=python:3.12-slim@sha256:<verified digest>.
# The base digest and uv.lock are both part of runtime certification evidence.
ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
RUN python -m pip install --no-cache-dir uv==0.12.3
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra agent --extra cloud --extra data --extra telemetry
ENV PATH="/opt/venv/bin:${PATH}"
EXPOSE 8080
CMD ["uvicorn", "defect_platform.control.api:app", "--host", "0.0.0.0", "--port", "8080"]
