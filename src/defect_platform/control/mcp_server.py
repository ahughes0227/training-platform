"""Model Context Protocol server exposing the agent tools over stdio.

Messages are newline-delimited JSON-RPC 2.0, as the MCP stdio transport specifies.
The server implements ``initialize``, ``ping``, ``tools/list``, and ``tools/call``
without the MCP SDK, so it runs with the core dependencies only:

    python -m defect_platform.control.mcp_server
"""

from __future__ import annotations

import json
import logging
import sys
from typing import IO, Any

from defect_platform.control.agent_tools import AgentToolbox, ToolError, toolbox_from_env

log = logging.getLogger(__name__)

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "defect-training-platform", "version": "1"}

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602


class McpServer:
    def __init__(self, toolbox: AgentToolbox) -> None:
        self.toolbox = toolbox

    def handle(self, message: Any) -> dict[str, Any] | None:
        """Return the JSON-RPC response for one message, or None for notifications."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" \
                or not isinstance(message.get("method"), str):
            return _error(message.get("id") if isinstance(message, dict) else None,
                          INVALID_REQUEST, "invalid JSON-RPC request")
        if "id" not in message:
            return None
        request_id, method = message["id"], message["method"]
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return _error(request_id, INVALID_PARAMS, "params must be an object")
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0]
            return _result(request_id, {
                "protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": "Use plan_run before launch_run. The control plane enforces spend "
                                "caps, certified image digests, and reviewed class catalogs."})
        if method == "ping":
            return _result(request_id, {})
        if method == "tools/list":
            return _result(request_id, {"tools": [_describe(tool) for tool in self.toolbox.tools()]})
        if method == "tools/call":
            name, arguments = params.get("name"), params.get("arguments") or {}
            if not isinstance(name, str) or name not in {tool.name for tool in self.toolbox.tools()}:
                return _error(request_id, INVALID_PARAMS, f"unknown tool: {name}")
            if not isinstance(arguments, dict):
                return _error(request_id, INVALID_PARAMS, "arguments must be an object")
            try:
                value = self.toolbox.call(name, arguments)
            except ToolError as exc:
                return _result(request_id, {"content": [{"type": "text", "text": str(exc)}],
                                            "isError": True})
            except Exception:
                log.exception("agent tool %s failed", name)
                return _result(request_id, {"content": [{"type": "text",
                                                         "text": f"{name} failed unexpectedly"}],
                                            "isError": True})
            return _result(request_id, {
                "content": [{"type": "text", "text": json.dumps(value, indent=2, default=str)}],
                "structuredContent": json.loads(json.dumps(value, default=str)), "isError": False})
        return _error(request_id, METHOD_NOT_FOUND, f"method not found: {method}")

    def serve(self, reader: IO[str], writer: IO[str]) -> None:
        for line in reader:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                response = _error(None, PARSE_ERROR, "parse error")
            else:
                response = self.handle(message)
            if response is not None:
                writer.write(json.dumps(response) + "\n")
                writer.flush()


def _describe(tool) -> dict[str, Any]:
    return {"name": tool.name, "title": tool.skill, "description": tool.description,
            "inputSchema": tool.input_schema(),
            "annotations": {"readOnlyHint": not tool.mutation, "destructiveHint": False,
                            "idempotentHint": tool.idempotent, "openWorldHint": tool.open_world}}


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def main() -> None:
    # Stdout carries protocol messages only; diagnostics go to stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    McpServer(toolbox_from_env()).serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
