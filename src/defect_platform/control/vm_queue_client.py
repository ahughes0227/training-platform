"""Client for the protected local VM queue Unix socket."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from defect_platform.control.vm_queue_api import MAX_MESSAGE_BYTES


class VMQueueRemoteError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class VMQueueClient:
    def __init__(self, socket_path: str | Path, *, timeout: float = 30.0,
                 max_message_bytes: int = MAX_MESSAGE_BYTES):
        self.socket_path = str(socket_path)
        self.timeout = timeout
        self.max_message_bytes = max_message_bytes

    def call(self, op: str, **values: Any) -> Any:
        payload = json.dumps({"op": op, **values}, separators=(",", ":")).encode() + b"\n"
        if len(payload) > self.max_message_bytes:
            raise ValueError(f"request exceeds maximum message size ({self.max_message_bytes} bytes)")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(self.timeout)
            try:
                conn.connect(self.socket_path)
                conn.sendall(payload)
                response = self._read_response(conn)
            except OSError as exc:
                raise RuntimeError(f"cannot communicate with VM queue service at {self.socket_path}: {exc}") from exc
        try:
            result = json.loads(response)
        except json.JSONDecodeError as exc:
            raise RuntimeError("VM queue service returned invalid JSON") from exc
        if not isinstance(result, dict) or not isinstance(result.get("ok"), bool):
            raise RuntimeError("VM queue service returned an invalid response")  # noqa: TRY004
        if not result["ok"]:
            error = result.get("error") or {}
            raise VMQueueRemoteError(str(error.get("code", "error")), str(error.get("message", "queue operation failed")))
        return result.get("result")

    def _read_response(self, conn: socket.socket) -> bytes:
        chunks = bytearray()
        while len(chunks) <= self.max_message_bytes:
            piece = conn.recv(min(65536, self.max_message_bytes + 1 - len(chunks)))
            if not piece:
                break
            newline = piece.find(b"\n")
            if newline >= 0:
                chunks.extend(piece[:newline])
                break
            chunks.extend(piece)
        if len(chunks) > self.max_message_bytes:
            raise RuntimeError("VM queue response exceeds maximum message size")
        return bytes(chunks)
