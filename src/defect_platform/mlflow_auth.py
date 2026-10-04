"""Short-lived Google identity token for an IAM-protected MLflow Cloud Run URL."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from collections.abc import Iterator


_token_lock = threading.RLock()


@contextmanager
def mlflow_tracking_auth(tracking_uri: str) -> Iterator[None]:
    """Set a fresh token for one bounded MLflow operation, then restore env."""
    use_iam = os.getenv("DEFECT_MLFLOW_IAM_AUTH") == "1" or ".run.app" in tracking_uri
    if not tracking_uri.startswith("https://") or not use_iam:
        yield
        return
    try:
        import google.auth.transport.requests
        from google.oauth2 import id_token
    except ImportError as exc:
        raise RuntimeError("Cloud Run MLflow authentication requires google-auth") from exc
    with _token_lock:
        old = os.environ.get("MLFLOW_TRACKING_TOKEN")
        audience = os.getenv("DEFECT_MLFLOW_IAM_AUDIENCE", tracking_uri.rstrip("/"))
        token = id_token.fetch_id_token(google.auth.transport.requests.Request(), audience)
        if not isinstance(token, str) or not token:
            raise RuntimeError("Cloud Run MLflow authentication returned an empty identity token")
        os.environ["MLFLOW_TRACKING_TOKEN"] = token
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("MLFLOW_TRACKING_TOKEN", None)
            else:
                os.environ["MLFLOW_TRACKING_TOKEN"] = old
