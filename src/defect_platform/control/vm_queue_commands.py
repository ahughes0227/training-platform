"""Operator CLI and service entry point for the local VM training queue."""
# Typer uses parameter defaults to define the command-line schema.
# ruff: noqa: B008

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
import yaml

from defect_platform.control.queue_contracts import QueueIntent, RetryClassification
from defect_platform.control.vm_queue_api import Peer, Principal, UnixQueueServer, VMQueueAPI
from defect_platform.control.vm_queue_client import VMQueueClient

vm_queue_app = typer.Typer(help="Enqueue and inspect Compute Engine training requests.", no_args_is_help=True)
log = logging.getLogger(__name__)


def _client() -> VMQueueClient:
    import os
    path = os.environ.get("DEFECT_VM_QUEUE_SOCKET", "/run/defect-platform/queue.sock")
    return VMQueueClient(path)


def _load_prepared(path: Path) -> list[QueueIntent]:
    """Read and validate finite prepared YAML locally; service never opens caller paths."""
    candidates = sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml")) if path.is_dir() else [path]
    if not candidates:
        raise ValueError(f"no YAML request files found in {path}")
    intents: list[QueueIntent] = []
    for candidate in candidates:
        if candidate.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"prepared requests must be YAML files: {candidate}")
        value = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        raw = value if isinstance(value, list) else [value]
        if not raw or any(not isinstance(item, dict) for item in raw):
            raise ValueError(f"{candidate} must contain a request object or non-empty list of objects")
        intents.extend(QueueIntent.model_validate(item) for item in raw)
        if len(intents) > 100:
            raise ValueError("one enqueue operation is limited to 100 prepared requests")
    if len({intent.idempotency_key for intent in intents}) != len(intents):
        raise ValueError("each prepared request in a batch needs a distinct idempotency_key")
    return intents


def _batch_key(intents: list[QueueIntent]) -> str:
    encoded = json.dumps([intent.config_sha256 for intent in intents], separators=(",", ":"))
    return "batch-" + hashlib.sha256(encoded.encode()).hexdigest()


def _show(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


@vm_queue_app.command("enqueue")
def enqueue(path: Path = typer.Argument(..., exists=True, readable=True),
            socket_path: Path | None = typer.Option(None, "--socket")) -> None:
    """Append one prepared YAML request or a bounded directory/YAML batch."""
    intents = _load_prepared(path)
    client = VMQueueClient(socket_path) if socket_path else _client()
    if len(intents) == 1:
        intent = intents[0]
        result = client.call("enqueue", idempotency_key=intent.idempotency_key,
                             intent=intent.model_dump(mode="json"))
    else:
        result = client.call("enqueue_batch", idempotency_key=_batch_key(intents),
                             intents=[item.model_dump(mode="json") for item in intents])
    _show(result)


@vm_queue_app.command("list")
def list_queue(socket_path: Path | None = typer.Option(None, "--socket")) -> None:
    _show((VMQueueClient(socket_path) if socket_path else _client()).call("list"))


@vm_queue_app.command("status")
def status(socket_path: Path | None = typer.Option(None, "--socket")) -> None:
    _show((VMQueueClient(socket_path) if socket_path else _client()).call("status"))


@vm_queue_app.command("pause")
def pause(reason: str = typer.Option(..., "--reason"),
          socket_path: Path | None = typer.Option(None, "--socket")) -> None:
    _show((VMQueueClient(socket_path) if socket_path else _client()).call("pause", reason=reason))


@vm_queue_app.command("resume")
def resume(socket_path: Path | None = typer.Option(None, "--socket")) -> None:
    _show((VMQueueClient(socket_path) if socket_path else _client()).call("resume"))


@vm_queue_app.command("cancel")
def cancel(entry_id: str, active: bool = typer.Option(False, "--active"),
           socket_path: Path | None = typer.Option(None, "--socket")) -> None:
    op = "cancel_active" if active else "cancel_waiting"
    _show((VMQueueClient(socket_path) if socket_path else _client()).call(op, entry_id=entry_id))


@vm_queue_app.command("retry")
def retry(entry_id: str, reason: str = typer.Option(..., "--reason"),
          classification: RetryClassification = typer.Option(..., "--classification"),
          expected_revision: int = typer.Option(..., "--revision"),
          socket_path: Path | None = typer.Option(None, "--socket")) -> None:
    _show((VMQueueClient(socket_path) if socket_path else _client()).call(
        "retry", entry_id=entry_id, reason=reason, expected_revision=expected_revision,
        classification=classification.value))


@vm_queue_app.command("skip")
def skip(entry_id: str, reason: str = typer.Option(..., "--reason"),
         expected_revision: int = typer.Option(..., "--revision"),
         socket_path: Path | None = typer.Option(None, "--socket")) -> None:
    _show((VMQueueClient(socket_path) if socket_path else _client()).call(
        "skip", entry_id=entry_id, reason=reason, expected_revision=expected_revision))


@vm_queue_app.command("resolve-interrupted")
def resolve_interrupted(entry_id: str,
                        config: Path = typer.Option(..., "--config", exists=True, readable=True),
                        expected_revision: int = typer.Option(..., "--revision"),
                        reason: str = typer.Option(..., "--reason")) -> None:
    """Admin-only recovery after controller proves the exact host slot is clear."""
    if os.geteuid() != 0:
        raise typer.BadParameter("interrupted recovery requires the host administrator")
    from defect_platform.control.queue_controller import create_vm_queue_controller

    controller = create_vm_queue_controller(config)
    result = controller.resolve_interrupted(entry_id, expected_revision=expected_revision,
        reason=reason, actor=f"admin-uid:{os.geteuid()}")
    _show(result.model_dump(mode="json") if hasattr(result, "model_dump") else result)


@vm_queue_app.command("backup")
def backup(destination: Path = typer.Argument(...),
           socket_path: Path | None = typer.Option(None, "--socket")) -> None:
    """Create an application-consistent backup at an explicitly configured protected path."""
    if not destination.is_absolute():
        raise typer.BadParameter("destination must be an absolute path")
    _show((VMQueueClient(socket_path) if socket_path else _client()).call("backup", destination=str(destination)))


@vm_queue_app.command("restore-check")
def restore_check(source: Path = typer.Argument(..., exists=True, readable=True),
                  socket_path: Path | None = typer.Option(None, "--socket")) -> None:
    """Inspect an authorized backup without replacing live queue state."""
    if not source.is_absolute():
        raise typer.BadParameter("source must be an absolute path")
    _show((VMQueueClient(socket_path) if socket_path else _client()).call("restore_check", source=str(source)))


def serve_local_queue(*, controller: Any, socket_path: str | Path,
                      principal_resolver: Callable[[Peer], Principal | None],
                      peer_reader=None, socket_mode: int = 0o660,
                      socket_group_id: int | None = None) -> None:
    """Run the local service; caller supplies reviewed config and principal mapping."""
    kwargs = {"peer_reader": peer_reader} if peer_reader is not None else {}
    UnixQueueServer(VMQueueAPI(controller, principal_resolver), socket_path,
                    mode=socket_mode, group_id=socket_group_id, **kwargs).serve_forever()


def principal_resolver_for_uids(operator_uids: set[int], worker_uids: set[int]) -> Callable[[Peer], Principal | None]:
    """Build a role map from trusted service configuration and kernel peer UID."""
    if operator_uids & worker_uids:
        raise ValueError("operator and worker UID lists must be disjoint")

    def resolve(peer: Peer) -> Principal | None:
        if peer.uid in operator_uids:
            return Principal(f"uid:{peer.uid}", "operator")
        if peer.uid in worker_uids:
            return Principal(f"worker-uid:{peer.uid}", "worker")
        return None

    return resolve


def worker_entrypoint(worker: Any, *, tick_interval_seconds: float,
                      stop_event: threading.Event | None = None) -> None:
    """Run periodic worker reconciliation independently of the submitting CLI."""
    if tick_interval_seconds <= 0:
        raise ValueError("worker tick interval must be explicitly configured and positive")
    stop_event = stop_event or threading.Event()
    while not stop_event.is_set():
        try:
            result = worker.tick()
            if getattr(result, "action", None) in {"held", "error"}:
                log.warning("queue worker held: %s", getattr(result, "reason", "no reason reported"))
        except Exception:
            log.exception("queue worker tick failed; durable state will be reconciled on the next tick")
        stop_event.wait(tick_interval_seconds)


@vm_queue_app.command("serve")
def serve(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    """Start the protected local Unix socket service using operator config."""
    from defect_platform.control.queue_controller import create_vm_queue_controller

    controller = create_vm_queue_controller(config)
    policy = controller.config
    operator_uids = set(policy.allowed_operator_uids)
    worker_uids = set(policy.worker_uids)
    serve_local_queue(controller=controller, socket_path=policy.socket_path,
        principal_resolver=principal_resolver_for_uids(operator_uids, worker_uids),
        socket_group_id=policy.socket_group_id)


@vm_queue_app.command("worker")
def worker(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    """Run the supervised periodic VM queue worker using operator config."""
    from defect_platform.control.queue_admission import load_queue_service_config
    from defect_platform.control.queue_controller import create_vm_queue_worker

    service_config = load_queue_service_config(config)
    worker_instance = create_vm_queue_worker(config)
    stop = threading.Event()
    try:
        worker_entrypoint(worker_instance, tick_interval_seconds=service_config.tick_interval_seconds,
                          stop_event=stop)
    except KeyboardInterrupt:
        stop.set()
