from __future__ import annotations

import threading
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import defect_platform.control.vm_queue_commands as queue_commands
from defect_platform.control import queue_controller
from defect_platform.control.queue_contracts import QueueIntent
from defect_platform.control.vm_queue_api import Peer
from defect_platform.control.vm_queue_commands import (
    _batch_key,
    _load_prepared,
    principal_resolver_for_uids,
    vm_queue_app,
    worker_entrypoint,
)


def prepared(key: str = "request-1") -> dict:
    digest = "a" * 64
    return {
        "experiment": {
            "experiment_id": "exp-1", "object_slug": "panel", "dataset_version_id": "panel-v1",
            "runtime_id": "runtime-1", "model": {"weights_uri": "gs://weights/model.safetensors",
                "weights_sha256": "b" * 64},
        },
        "dataset": {"version_id": "panel-v1", "object_slug": "panel", "root_uri": "gs://data/panel-v1",
            "manifest_uri": "gs://data/panel-v1/manifest.json", "shard_uris": {"train": ["gs://data/shard"]},
            "sample_counts": {"train": 2}, "sha256": "c" * 64, "source_snapshot_uri": "gs://data/source",
            "semantic_manifest_uri": "gs://data/panel-v1/semantic.json", "semantic_sha256": "d" * 64},
        "runtime": {"runtime_id": "runtime-1", "source_commit": "e" * 40, "image_tag": "train:v1",
            "image_digest": f"registry.example/train@sha256:{digest}", "runtime_version": "1",
            "python_version": "3.12", "pytorch_version": "2.6", "cuda_version": "12.4",
            "validation": {"trainer": True, "container_gpu": True, "vertex_gpu": True,
                "gcs_read": True, "gcs_write": True}, "certified": True,
            "certified_at": datetime.now(UTC).isoformat()},
        "classes": ["crack", "dent"],
        "job": {"profile_id": "gpu-profile", "project_id": "project", "zone": "us-central1-a",
            "instance_id": "123", "gpu_identity": "gpu-0", "image_digest": f"registry.example/train@sha256:{digest}",
            "estimated_runtime_seconds": 3600, "per_run_cost_usd": 12.0,
            "max_runtime_seconds": 7200,
            "storage_peak_bytes": 1000, "storage_peak_inodes": 100, "max_attempts": 2},
        "idempotency_key": key,
    }


def test_prepared_yaml_loads_typed_requests_and_batch_key_is_stable(tmp_path):
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump([prepared("request-a"), prepared("request-b")]), encoding="utf-8")
    intents = _load_prepared(path)
    assert len(intents) == 2
    assert all(isinstance(item, QueueIntent) for item in intents)
    assert _batch_key(intents) == _batch_key(intents)


def test_prepared_yaml_rejects_duplicate_keys_and_non_request_shape(tmp_path):
    path = tmp_path / "requests.yaml"
    path.write_text(yaml.safe_dump([prepared("same"), prepared("same")]), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct idempotency_key"):
        _load_prepared(path)
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list"):
        _load_prepared(path)


def test_cli_requires_finite_local_yaml_input():
    result = CliRunner().invoke(vm_queue_app, ["enqueue", "/no/such/request.yaml"])
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_service_role_map_uses_configured_uid_and_rejects_ambiguous_roles():
    resolve = principal_resolver_for_uids({1001}, {1002})
    assert resolve(Peer(4, 1001, 20)).role == "operator"
    assert resolve(Peer(5, 1002, 20)).role == "worker"
    assert resolve(Peer(6, 1003, 20)) is None
    with pytest.raises(ValueError, match="disjoint"):
        principal_resolver_for_uids({1001}, {1001})


def test_worker_loop_ticks_and_stops_with_interruptible_event():
    stop = threading.Event()

    class Worker:
        calls = 0
        def tick(self):
            self.calls += 1
            stop.set()
            return SimpleNamespace(action="idle", reason=None)

    worker = Worker()
    worker_entrypoint(worker, tick_interval_seconds=0.01, stop_event=stop)
    assert worker.calls == 1
    with pytest.raises(ValueError, match="positive"):
        worker_entrypoint(worker, tick_interval_seconds=0)


def test_serve_command_uses_controller_config_and_uid_authority(tmp_path, monkeypatch):
    config_file = tmp_path / "service.yaml"
    config_file.write_text("unused", encoding="utf-8")
    controller = SimpleNamespace(config=SimpleNamespace(
        socket_path="/run/queue.sock", socket_group_id=2200,
        allowed_operator_uids=[1001], worker_uids=[1002]))
    monkeypatch.setattr(queue_controller, "create_vm_queue_controller", lambda path: controller)
    called = {}
    monkeypatch.setattr(queue_commands, "serve_local_queue", lambda **kwargs: called.update(kwargs))
    result = CliRunner().invoke(vm_queue_app, ["serve", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert called["socket_path"] == "/run/queue.sock"
    assert called["socket_group_id"] == 2200
    assert called["principal_resolver"](Peer(5, 1001, 30)).role == "operator"
    assert called["principal_resolver"](Peer(6, 1002, 30)).role == "worker"
    assert called["principal_resolver"](Peer(7, 9999, 30)) is None


def test_worker_command_uses_explicit_service_tick_interval(tmp_path, monkeypatch):
    config_file = tmp_path / "service.yaml"
    config_file.write_text("unused", encoding="utf-8")
    instance = SimpleNamespace(tick=lambda: None)
    monkeypatch.setattr(queue_controller, "create_vm_queue_worker", lambda path: instance)
    from defect_platform.control import queue_admission
    monkeypatch.setattr(queue_admission, "load_queue_service_config",
                        lambda path: SimpleNamespace(tick_interval_seconds=3.5))
    called = {}
    monkeypatch.setattr(queue_commands, "worker_entrypoint",
                        lambda worker, **kwargs: called.update(worker=worker, **kwargs))
    result = CliRunner().invoke(vm_queue_app, ["worker", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert called["worker"] is instance
    assert called["tick_interval_seconds"] == 3.5


def test_interrupted_recovery_requires_admin_and_calls_controller_factory(tmp_path, monkeypatch):
    config_file = tmp_path / "service.yaml"
    config_file.write_text("unused", encoding="utf-8")
    calls = {}
    controller = SimpleNamespace(resolve_interrupted=lambda entry_id, **kwargs:
                                 calls.update(entry_id=entry_id, **kwargs) or {"state": "waiting"})
    monkeypatch.setattr(queue_controller, "create_vm_queue_controller", lambda path: controller)
    monkeypatch.setattr(queue_commands.os, "geteuid", lambda: 0)
    result = CliRunner().invoke(vm_queue_app, ["resolve-interrupted", "entry-1", "--config",
        str(config_file), "--revision", "7", "--reason", "VM process tree independently inspected"])
    assert result.exit_code == 0, result.output
    assert calls == {"entry_id": "entry-1", "expected_revision": 7,
        "reason": "VM process tree independently inspected", "actor": "admin-uid:0"}


def test_interrupted_recovery_rejects_non_admin_without_factory_effect(tmp_path, monkeypatch):
    config_file = tmp_path / "service.yaml"
    config_file.write_text("unused", encoding="utf-8")
    monkeypatch.setattr(queue_commands.os, "geteuid", lambda: 1001)
    calls = []
    monkeypatch.setattr(queue_controller, "create_vm_queue_controller", lambda path: calls.append(path))
    result = CliRunner().invoke(vm_queue_app, ["resolve-interrupted", "entry-1", "--config",
        str(config_file), "--revision", "7", "--reason", "reviewed"])
    assert result.exit_code != 0
    assert "host administrator" in result.output
    assert calls == []
