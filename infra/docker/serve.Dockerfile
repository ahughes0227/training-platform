ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
RUN python -m pip install --no-cache-dir uv==0.12.3
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra train --extra cloud --extra serve --extra telemetry
ENV PATH="/opt/venv/bin:${PATH}"
CMD ["ray", "start", "--head", "--dashboard-host", "0.0.0.0", "--block"]
