ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
RUN python -m pip install --no-cache-dir uv==0.12.3
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY infra/docker/mlflow_entrypoint.py ./mlflow_entrypoint.py
RUN uv sync --frozen --no-dev --extra mlflow-server
ENV PATH="/opt/venv/bin:${PATH}"
EXPOSE 8080
CMD ["python", "/app/mlflow_entrypoint.py"]
