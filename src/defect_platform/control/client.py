"""CLI client for the IAM-protected durable Cloud Run control API."""

from __future__ import annotations

import json
import os
from urllib import error, parse, request

from defect_platform.contracts import RunRecord


class ControlAPIClient:
    def __init__(self, base_url: str, *, token_provider=None, opener=None):
        if not base_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError("control service URL must use HTTPS or local loopback")
        self.base_url = base_url.rstrip("/")
        self.token_provider = token_provider
        self.opener = opener or request.urlopen

    def _token(self) -> str | None:
        if self.token_provider is not None:
            return self.token_provider(self.base_url)
        if not (".run.app" in self.base_url or os.getenv("DEFECT_CONTROL_IAM_AUTH") == "1"):
            return None
        try:
            import google.auth.transport.requests
            from google.oauth2 import id_token
        except ImportError as exc:
            raise RuntimeError("Cloud Run authentication requires the cloud dependency group") from exc
        return id_token.fetch_id_token(google.auth.transport.requests.Request(), self.base_url)

    def _call(self, method: str, path: str, payload: dict | None = None):
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        token = self._token()
        if token:
            headers["Authorization"] = "Bearer " + token
        http_request = request.Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with self.opener(http_request, timeout=30) as response:
                return json.loads(response.read())
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"Control service returned HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Cannot reach the control service: {exc.reason}") from exc

    def submit(self, payload: dict) -> tuple[RunRecord, bool]:
        result = self._call("POST", "/runs", payload)
        return RunRecord.model_validate(result["run"]), bool(result["created"])

    def get(self, run_id: str) -> RunRecord:
        result = self._call("GET", "/runs/" + parse.quote(run_id, safe=""))
        return RunRecord.model_validate(result)

    def list(self, object_slug: str | None = None) -> list[RunRecord]:
        path = "/runs"
        if object_slug:
            path += "?" + parse.urlencode({"object_slug": object_slug})
        return [RunRecord.model_validate(value) for value in self._call("GET", path)]
