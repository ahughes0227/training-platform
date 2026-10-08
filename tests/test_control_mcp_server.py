from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

from defect_platform.control.agent_tools import AgentToolbox, ReadOnlyStoreBackend
from defect_platform.control.mcp_server import McpServer
from defect_platform.control.store import SQLiteRunStore


def _server(tmp_path):
    return McpServer(AgentToolbox(projects_root=tmp_path / "projects",
                                  backend=ReadOnlyStoreBackend(
                                      lambda: SQLiteRunStore(tmp_path / "runs.db"))))


def _request(method, params=None, request_id=1):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def test_initialize_list_and_call(tmp_path):
    server = _server(tmp_path)
    init = server.handle(_request("initialize", {"protocolVersion": "2025-03-26",
                                                 "capabilities": {}, "clientInfo": {"name": "t"}}))
    assert init["result"]["protocolVersion"] == "2025-03-26"
    assert "tools" in init["result"]["capabilities"]
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    tools = {tool["name"]: tool for tool in server.handle(_request("tools/list"))["result"]["tools"]}
    assert tools["launch_run"]["annotations"]["readOnlyHint"] is False
    assert tools["plan_run"]["annotations"]["readOnlyHint"] is True
    assert "plan_id" in tools["launch_run"]["inputSchema"]["required"]

    created = server.handle(_request("tools/call", {"name": "create_project", "arguments": {
        "display_name": "Bracket", "classes": ["bent", "rust"]}}))["result"]
    assert created["isError"] is False
    assert created["structuredContent"]["project"] == "bracket"
    assert json.loads(created["content"][0]["text"])["project"] == "bracket"


def test_errors_follow_json_rpc_and_tool_error_conventions(tmp_path):
    server = _server(tmp_path)
    refused = server.handle(_request("tools/call", {"name": "view_project",
                                                    "arguments": {"project": "missing"}}))
    assert refused["result"]["isError"] is True
    assert "project not found" in refused["result"]["content"][0]["text"]
    assert server.handle(_request("tools/call", {"name": "build_image"}))["error"]["code"] == -32602
    assert server.handle(_request("resources/list"))["error"]["code"] == -32601
    assert server.handle({"id": 3})["error"]["code"] == -32600

    out = io.StringIO()
    server.serve(io.StringIO("not json\n\n" + json.dumps(_request("ping", request_id=9)) + "\n"), out)
    responses = [json.loads(line) for line in out.getvalue().splitlines()]
    assert responses[0]["error"]["code"] == -32700
    assert responses[1] == {"jsonrpc": "2.0", "id": 9, "result": {}}


def test_stdio_module_entry_point(tmp_path):
    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src), "DEFECT_PROJECTS_ROOT": str(tmp_path / "projects"),
           "DEFECT_STATE_DATABASE_URL": f"sqlite://{tmp_path / 'runs.db'}"}
    env.pop("DEFECT_CONTROL_SERVICE_URL", None)
    env.pop("DEFECT_MLFLOW_TRACKING_URI", None)
    messages = [_request("initialize", {"protocolVersion": "2025-06-18"}, 1),
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                _request("tools/call", {"name": "list_projects", "arguments": {}}, 2)]
    result = subprocess.run([sys.executable, "-m", "defect_platform.control.mcp_server"],
                            input="".join(json.dumps(m) + "\n" for m in messages),
                            capture_output=True, text=True, env=env, cwd=tmp_path, timeout=60,
                            check=True)
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert [response["id"] for response in responses] == [1, 2]
    assert responses[1]["result"]["structuredContent"] == {"projects": []}
