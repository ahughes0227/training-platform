"""CLI client for the IAM-protected durable Cloud Run control API."""

from __future__ import annotations

import json
import os
from urllib import error, parse, request

from defect_platform.contracts import RunRecord
from defect_platform.control.setup import IntakeSession


class ControlAPIError(RuntimeError):
    """Structured error returned by the control API."""

    def __init__(
        self, code: str, message: str, *, retryable: bool = False, status: int | None = None
    ):
        super().__init__(message)
        self.code, self.message, self.retryable, self.status = code, message, retryable, status


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
            raise RuntimeError(
                "Cloud Run authentication requires the cloud dependency group"
            ) from exc
        return id_token.fetch_id_token(google.auth.transport.requests.Request(), self.base_url)

    def _call(self, method: str, path: str, payload: dict | None = None):
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        token = self._token()
        if token:
            headers["Authorization"] = "Bearer " + token
        http_request = request.Request(
            self.base_url + path, data=body, headers=headers, method=method
        )
        try:
            with self.opener(http_request, timeout=30) as response:
                return json.loads(response.read())
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            try:
                parsed = json.loads(detail).get("error", {})
            except (ValueError, AttributeError):
                parsed = {}
            raise ControlAPIError(
                parsed.get("code", "HTTP_ERROR"),
                parsed.get("message", detail),
                retryable=bool(parsed.get("retryable", exc.code >= 500)),
                status=exc.code,
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Cannot reach the control service: {exc.reason}") from exc

    def submit(self, payload: dict) -> tuple[RunRecord, bool]:
        result = self._call("POST", "/runs", payload)
        return RunRecord.model_validate(result["run"]), bool(result["created"])

    def submit_references(self, payload: dict) -> tuple[RunRecord, bool]:
        """Submit an agent request containing IDs/checksums only."""
        result = self._call("POST", "/runs/reference", payload)
        return RunRecord.model_validate(result["run"]), bool(result["created"])

    def get(self, run_id: str) -> RunRecord:
        # The compact projection carries the RunRecord identity fields, so this
        # remains one request and is compatible with older control services.
        result = self._call("GET", "/runs/" + parse.quote(run_id, safe=""))
        return RunRecord.model_validate(result)

    def list(self, object_slug: str | None = None) -> list[RunRecord]:
        path = "/runs"
        if object_slug:
            path += "?" + parse.urlencode({"object_slug": object_slug, "detail": "true"})
        else:
            path += "?detail=true"
        return [RunRecord.model_validate(value) for value in self._call("GET", path)]

    def start_intake(
        self, request_text: str, *, model: str | None = None, ttl_seconds: int | None = None
    ) -> dict:
        """Start a server-side setup session and process its first compact turn."""
        payload: dict[str, str | int] = {"request": request_text}
        if model is not None:
            payload["model"] = model
        if ttl_seconds is not None:
            payload["ttl_seconds"] = ttl_seconds
        return self._call("POST", "/intake/sessions", payload)

    def get_schema(self, schema_id: str, *, version: str = "1") -> dict:
        """Fetch a registered schema by ID; callers can cache its checksum."""
        path = (
            "/schemas/"
            + parse.quote(schema_id, safe="")
            + "?"
            + parse.urlencode({"version": version})
        )
        return self._call("GET", path)

    def get_intake(self, session_id: str) -> IntakeSession:
        result = self._call("GET", "/intake/sessions/" + parse.quote(session_id, safe=""))
        # GET returns the session directly; turn/patch responses wrap it.
        return IntakeSession.model_validate(result.get("session", result))

    def intake_turn(self, session_id: str, message: str, *, expected_revision: int) -> dict:
        path = "/intake/sessions/" + parse.quote(session_id, safe="") + "/turn"
        return self._call(
            "POST", path, {"message": message, "expected_revision": expected_revision}
        )

    def intake_patch(self, session_id: str, patch: list[dict], *, expected_revision: int) -> dict:
        path = "/intake/sessions/" + parse.quote(session_id, safe="") + "/patch"
        return self._call("POST", path, {"patch": patch, "expected_revision": expected_revision})

    # Explicit aliases keep the API discoverable for integrations that call
    # this feature "setup" rather than "intake".
    start_setup_session = start_intake
    get_setup_session = get_intake
    setup_turn = intake_turn

    def events(self, run_id: str, *, cursor: int = 0, limit: int = 100) -> dict:
        path = (
            "/runs/"
            + parse.quote(run_id, safe="")
            + "/events?"
            + parse.urlencode({"cursor": cursor, "limit": limit})
        )
        return self._call("GET", path)

    def plan_batch(self, payload: dict) -> dict:
        return self._call("POST", "/batches/plan", payload)

    def submit_batch(self, payload: dict) -> dict:
        return self._call("POST", "/batches", payload)
