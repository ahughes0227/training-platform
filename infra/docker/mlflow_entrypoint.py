"""Start MLflow with a Cloud SQL socket and a GCS artifact destination."""

from __future__ import annotations

import os
import sys
from urllib.parse import quote


def main() -> None:
    user = os.environ["DB_USER"]
    password = quote(os.environ["DB_PASSWORD"], safe="")
    database = os.environ["DB_NAME"]
    socket = os.environ["DB_SOCKET"]
    artifacts = os.environ["MLFLOW_ARTIFACTS_DESTINATION"]
    backend = f"postgresql+psycopg://{user}:{password}@/{database}?host={quote(socket, safe='/')}"
    os.execv(
        sys.executable,
        [
            sys.executable, "-m", "mlflow", "server", "--host", "0.0.0.0", "--port", "8080",
            "--backend-store-uri", backend, "--serve-artifacts",
            "--artifacts-destination", artifacts,
        ],
    )


if __name__ == "__main__":
    main()
