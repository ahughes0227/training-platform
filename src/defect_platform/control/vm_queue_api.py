"""Authenticated local API for the Compute Engine VM training queue.

The socket protocol deliberately carries no actor identity. The server derives every
principal from Linux peer credentials and allows worker observations only from an
explicitly mapped worker principal.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import struct
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from defect_platform.control.queue_contracts import QueueIntent, RetryClassification

MAX_MESSAGE_BYTES = 1_048_576


@dataclass(frozen=True)
class Peer:
    pid: int
    uid: int
    gid: int


@dataclass(frozen=True)
class Principal:
    name: str
    role: str


class InvalidRequestError(ValueError):
    """Malformed user or worker payload, distinct from queue state conflicts."""


class VMQueueController(Protocol):
    def enqueue(self, intent: Any, *, idempotency_key: str, requester: str) -> Any: ...
    def enqueue_batch(self, intents: list[Any], *, idempotency_key: str, requester: str) -> Any: ...
    def list_queue(self) -> Any: ...
    def status(self) -> Any: ...
    def pause(self, *, reason: str, actor: str) -> Any: ...
    def resume(self, *, actor: str) -> Any: ...
    def cancel_waiting(self, entry_id: str, *, actor: str) -> Any: ...
    def request_active_cancel(self, entry_id: str, *, actor: str) -> Any: ...
    def retry(self, entry_id: str, *, expected_revision: int, reason: str, actor: str,
              classification: RetryClassification) -> Any: ...
    def skip(self, entry_id: str, *, expected_revision: int, reason: str, actor: str) -> Any: ...
    def record_observation(self, observation: dict[str, Any], *, worker_id: str) -> Any: ...
    def backup(self, destination: Path) -> Any: ...
    def verify_restore(self, source: Path) -> Any: ...


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "model_dump_json"):
        return json.loads(value.model_dump_json())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class VMQueueAPI:
    """Operation dispatcher. `resolve_peer` must map kernel credentials, not body data."""

    def __init__(self, controller: VMQueueController,
                 resolve_peer: Callable[[Peer], Principal | None]):
        self.controller = controller
        self.resolve_peer = resolve_peer

    def dispatch(self, body: Mapping[str, Any], peer: Peer) -> dict[str, Any]:
        principal = self.resolve_peer(peer)
        if principal is None or principal.role not in {"operator", "worker"}:
            raise PermissionError("local peer is not authorized for queue access")
        if not isinstance(body, Mapping):
            raise InvalidRequestError("request must be a JSON object")
        op = body.get("op")
        if not isinstance(op, str):
            raise InvalidRequestError("request operation is required")
        if op == "observe":
            self._require_role(principal, "worker")
            observation = body.get("observation")
            if not isinstance(observation, dict):
                raise InvalidRequestError("observation must be an object")
            value = self.controller.record_observation(observation, worker_id=principal.name)
        else:
            self._require_role(principal, "operator")
            actor = principal.name
            if op == "enqueue":
                intent = self._intent(self._object(body, "intent"))
                key = self._text(body, "idempotency_key")
                if key != intent.idempotency_key:
                    raise InvalidRequestError("idempotency key must match the prepared intent")
                value = self.controller.enqueue(intent,
                    idempotency_key=key, requester=actor)
            elif op == "enqueue_batch":
                intents = body.get("intents")
                if not isinstance(intents, list) or not intents or len(intents) > 100:
                    raise InvalidRequestError("intents must contain between 1 and 100 prepared requests")
                if any(not isinstance(item, dict) for item in intents):
                    raise InvalidRequestError("each prepared request must be an object")
                prepared = [self._intent(item) for item in intents]
                if len({item.idempotency_key for item in prepared}) != len(prepared):
                    raise InvalidRequestError("each prepared request needs a distinct idempotency key")
                value = self.controller.enqueue_batch(prepared,
                    idempotency_key=self._text(body, "idempotency_key"), requester=actor)
            elif op == "list":
                value = self.controller.list_queue()
            elif op == "status":
                value = self.controller.status()
            elif op == "pause":
                value = self.controller.pause(reason=self._text(body, "reason"), actor=actor)
            elif op == "resume":
                value = self.controller.resume(actor=actor)
            elif op == "cancel_waiting":
                value = self.controller.cancel_waiting(self._text(body, "entry_id"), actor=actor)
            elif op == "cancel_active":
                value = self.controller.request_active_cancel(self._text(body, "entry_id"), actor=actor)
            elif op == "retry":
                value = self.controller.retry(self._text(body, "entry_id"),
                    expected_revision=self._revision(body), reason=self._text(body, "reason"), actor=actor,
                    classification=RetryClassification(self._text(body, "classification")))
            elif op == "skip":
                value = self.controller.skip(self._text(body, "entry_id"),
                    expected_revision=self._revision(body), reason=self._text(body, "reason"), actor=actor)
            elif op == "backup":
                value = self.controller.backup(self._path(body, "destination"))
            elif op == "restore_check":
                value = self.controller.verify_restore(self._path(body, "source"))
            else:
                raise ValueError(f"unknown queue operation: {op}")
        return {"ok": True, "result": _jsonable(value)}

    @staticmethod
    def _require_role(principal: Principal, required: str) -> None:
        if principal.role != required:
            raise PermissionError(f"operation requires {required} credentials")

    @staticmethod
    def _text(body: Mapping[str, Any], key: str) -> str:
        value = body.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 512:
            raise InvalidRequestError(f"{key} must be a non-empty string of at most 512 characters")
        return value

    @classmethod
    def _object(cls, body: Mapping[str, Any], key: str) -> dict[str, Any]:
        value = body.get(key)
        if not isinstance(value, dict):
            raise InvalidRequestError(f"{key} must be an object")
        return value

    @staticmethod
    def _intent(value: dict[str, Any]) -> QueueIntent:
        try:
            return QueueIntent.model_validate(value)
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

    @staticmethod
    def _revision(body: Mapping[str, Any]) -> int:
        value = body.get("expected_revision")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidRequestError("expected_revision must be a non-negative integer")
        return value

    @staticmethod
    def _path(body: Mapping[str, Any], key: str) -> Path:
        raw = VMQueueAPI._text(body, key)
        path = Path(raw)
        if not path.is_absolute():
            raise InvalidRequestError(f"{key} must be an absolute path")
        return path


def linux_peer_credentials(connection: socket.socket) -> Peer:
    """Read credentials supplied by the Linux kernel for this connected peer."""
    if not hasattr(socket, "SO_PEERCRED"):
        raise RuntimeError("Linux SO_PEERCRED is required for production queue service")
    raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    pid, uid, gid = struct.unpack("3i", raw)
    return Peer(pid=pid, uid=uid, gid=gid)


class UnixQueueServer:
    """Small bounded newline-JSON Unix socket server with restrictive socket mode."""

    def __init__(self, api: VMQueueAPI, socket_path: str | Path, *, mode: int = 0o660,
                 peer_reader: Callable[[socket.socket], Peer] = linux_peer_credentials,
                 group_id: int | None = None):
        self.api = api
        self.socket_path = Path(socket_path)
        self.mode = mode
        self.peer_reader = peer_reader
        self.group_id = group_id
        self._socket: socket.socket | None = None
        self._stop = threading.Event()

    def serve_forever(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        if self.socket_path.exists() or self.socket_path.is_symlink():
            info = self.socket_path.lstat()
            if not stat.S_ISSOCK(info.st_mode):
                raise RuntimeError(f"refusing to replace non-socket path: {self.socket_path}")
            raise RuntimeError(f"queue socket already exists; verify and remove stale socket explicitly: {self.socket_path}")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket = server
        try:
            server.bind(str(self.socket_path))
            if self.group_id is not None:
                os.chown(self.socket_path, -1, self.group_id)
            os.chmod(self.socket_path, self.mode)
            server.listen(16)
            server.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    conn, _ = server.accept()
                except TimeoutError:
                    continue
                threading.Thread(target=self._serve_connection, args=(conn,), daemon=True).start()
        finally:
            server.close()
            self._socket = None
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass

    def stop(self) -> None:
        self._stop.set()

    def _serve_connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                peer = self.peer_reader(conn)
                data = self._read_line(conn)
                body = json.loads(data)
                result = self.api.dispatch(body, peer)
                response = result
            except PermissionError as exc:
                response = {"ok": False, "error": {"code": "forbidden", "message": str(exc)}}
            except (InvalidRequestError, json.JSONDecodeError) as exc:
                response = {"ok": False, "error": {"code": "invalid_request", "message": str(exc)}}
            except (KeyError, LookupError) as exc:
                response = {"ok": False, "error": {"code": "not_found", "message": str(exc)}}
            except ValueError as exc:
                response = {"ok": False, "error": {"code": "conflict", "message": str(exc)[:2000]}}
            except Exception as exc:  # noqa: BLE001 - keep the service alive and return a bounded error
                response = {"ok": False, "error": {"code": "conflict", "message": str(exc)[:2000]}}
            encoded = json.dumps(response, separators=(",", ":")).encode() + b"\n"
            conn.sendall(encoded)

    @staticmethod
    def _read_line(conn: socket.socket) -> bytes:
        chunks = bytearray()
        while len(chunks) <= MAX_MESSAGE_BYTES:
            piece = conn.recv(min(65536, MAX_MESSAGE_BYTES + 1 - len(chunks)))
            if not piece:
                break
            newline = piece.find(b"\n")
            if newline >= 0:
                chunks.extend(piece[:newline])
                break
            chunks.extend(piece)
        if len(chunks) > MAX_MESSAGE_BYTES:
            raise InvalidRequestError("request exceeds maximum message size")
        if not chunks:
            raise InvalidRequestError("empty request")
        return bytes(chunks)
