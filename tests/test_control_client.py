from __future__ import annotations

import json
from datetime import datetime, timezone

from defect_platform.control.client import ControlAPIClient


def test_remote_cli_uses_one_authenticated_durable_api_for_submit_and_status():
    calls = []
    record = {
        "run_id": "run-1", "object_slug": "panel", "experiment_id": "exp-1",
        "dataset_version_id": "ds-1", "runtime_id": "runtime-1", "state": "submitted",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    class Response:
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def read(self): return json.dumps(self.data).encode()

    def opener(http_request, timeout):
        calls.append((http_request.get_method(), http_request.full_url,
                      http_request.get_header("Authorization"), http_request.data))
        if http_request.get_method() == "POST":
            return Response({"run": record, "created": True})
        return Response(record)

    client = ControlAPIClient("https://control.example.run.app", token_provider=lambda _: "token-1", opener=opener)
    run, created = client.submit({"idempotency_key": "key"})
    assert created and run.run_id == "run-1"
    assert client.get("run-1").run_id == "run-1"
    assert calls[0][0:3] == ("POST", "https://control.example.run.app/runs", "Bearer token-1")
    assert calls[1][1] == "https://control.example.run.app/runs/run-1"
