"""Queue authority/controller integration with real transactions and fake services."""
import io
import json
import tarfile
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_control import fixtures

from defect_platform.control.queue_admission import (
    QueueAdmission,
    QueueServiceConfig,
    VMReadiness,
)
from defect_platform.control.queue_contracts import (
    FailurePolicy,
    QueueIntent,
    QueueOperatorConfig,
    VMJobProfile,
)
from defect_platform.control.queue_controller import AttemptTracker, QueueController
from defect_platform.control.queue_store import SQLiteQueueStore
from defect_platform.control.queue_worker import QueueWorker
from defect_platform.control.vm_executor import (
    DiskCapacity,
    ExecutionObservation,
    ExecutionResult,
    HostIdentity,
    VMExecutor,
)
from defect_platform.dataset import load_dataset_semantics
from defect_platform.semantics import (
    canonical_bytes,
    make_training_manifest,
    policy_for_config,
    serialize_manifest,
)
from defect_platform.trainer.weights import write_bundle_integrity


def configured(tmp_path):
    runtime, experiment, dataset, _, _, catalogs = fixtures(tmp_path)
    semantic = load_dataset_semantics(dataset)
    # Preserve semantics while representing the already published immutable GCS version.
    dataset = dataset.model_copy(update={
        "root_uri": "gs://datasets/version", "manifest_uri": "gs://datasets/version/manifest",
        "semantic_manifest_uri": "gs://datasets/version/semantics.json",
        "source_snapshot_uri": "gs://datasets/version/source",
        "shard_uris": {"train": ["gs://datasets/version/train.tar"]}})
    experiment = experiment.model_copy(update={"catalog_sha256": semantic.catalog.sha256})
    now = datetime(2026, 9, 29, tzinfo=UTC)
    profile = VMJobProfile(profile_id="gpu", project_id="p", zone="z", instance_id="123",
        gpu_identity="GPU-1", image_digest=runtime.image_digest,
        estimated_runtime_seconds=30, max_runtime_seconds=60, per_run_cost_usd=1,
        storage_peak_bytes=1000, storage_peak_inodes=10, max_attempts=2)
    limits = QueueOperatorConfig(revision=1, max_backlog=10, campaign_cost_cap_usd=20,
        campaign_authorization_id="campaign-reviewed", cost_meter_started_at=now,
        vm_hourly_cost_usd=1, reserved_headroom_bytes=1000, reserved_headroom_inodes=100,
        usable_capacity_bytes=100000, observed_free_bytes=90000, observed_free_inodes=10000,
        capacity_observed_at=now, max_start_delay_seconds=60, stalled_work_seconds=120,
        recovery_bound_seconds=120, max_capacity_observation_age_seconds=120,
        backup_evidence_ref="verified-off-vm-policy", retry_backoff_seconds=30)
    readiness = VMReadiness(project_id="p", zone="z", instance_id="123", boot_id="boot",
        gpu_identity="GPU-1", driver_version="observed-driver", toolkit_version="observed-toolkit",
        engine_version="observed-engine", compatible_digests=[runtime.image_digest],
        observed_at=now, expires_at=now + timedelta(hours=1), evidence_ref="gpu-probe",
        metadata_denial_evidence_ref="metadata-denial-probe",
        credential_scope_evidence_ref="grant-probe", boot_disk_evidence_ref="boot-disk-probe")
    config = QueueServiceConfig(scope="vm", storage_path=tmp_path / "state",
        socket_path=tmp_path / "queue.sock", catalog_root="gs://approved/catalogs",
        allowed_operator_uids=[500], worker_uids=[600], socket_group_id=600, state_group_id=601,
        worker_id="worker", tick_interval_seconds=10, lease_seconds=60,
        operator=limits, readiness=readiness, approved_runtimes={runtime.runtime_id: runtime},
        approved_datasets={dataset.version_id: dataset}, approved_profiles={profile.profile_id: profile},
        approved_experiments={experiment.experiment_id: experiment},
        experiment_profiles={experiment.experiment_id: profile.profile_id},
        output_root_uri="gs://artifacts/attempts", mlflow_tracking_uri="https://tracking.example",
        mlflow_experiment_name="reviewed", credential_path=tmp_path / "credentials.json",
        trainer_uid=700, trainer_gid=700, gpu_devices=["0"], network_name="defect-trainers",
        memory_bytes=1024000, pids_limit=100, vm_cost_upper_bound_hourly_usd=1,
        cost_authorized_from=now, cost_authorized_until=now + timedelta(hours=1),
        backup_root_uri="gs://backup/queue", backup_interval_seconds=60, backup_local_keep=2)
    admission = QueueAdmission(config, catalogs, semantic_loader=lambda _: semantic)
    controller = QueueController(config=config, store=SQLiteQueueStore(tmp_path / "state/queue.sqlite"),
                                 admission=admission, clock=lambda: now)
    intent = QueueIntent(experiment=experiment, dataset=dataset, runtime=runtime,
                         classes=semantic.catalog.labels, job=profile, idempotency_key="request")
    return controller, intent, now


def test_controller_rejects_self_asserted_authority_before_persisting(tmp_path):
    controller, intent, _ = configured(tmp_path)
    intent.runtime = intent.runtime.model_copy(update={"source_commit": "c" * 40})
    with pytest.raises(ValueError, match="protected"):
        controller.enqueue(intent, idempotency_key="request", requester="uid-500")
    assert controller.store.list_entries("vm") == []


def test_controller_rechecks_revocation_and_boot_identity(tmp_path):
    controller, intent, now = configured(tmp_path)
    controller.enqueue(intent, idempotency_key="request", requester="uid-500")
    with pytest.raises(ValueError, match="VM/boot"):
        controller.admission.validate(intent, now=now, identity=HostIdentity("p", "z", "123", "newboot"))
    controller.admission.refresh = lambda: controller.config.model_copy(update={"approved_runtimes": {}})
    with pytest.raises(ValueError, match="protected"):
        controller.admission.validate(intent, now=now)


def test_controller_atomic_idempotent_batch_controls_and_cost_cursor(tmp_path):
    controller, intent, now = configured(tmp_path)
    first = controller.enqueue(intent, idempotency_key="request", requester="uid-500")
    assert controller.enqueue(intent, idempotency_key="request", requester="uid-500").run_id == first.run_id
    changed = intent.model_copy(update={"experiment": intent.experiment.model_copy(update={"epochs": 99})})
    with pytest.raises(ValueError, match="protected|idempotency"):
        controller.enqueue(changed, idempotency_key="request", requester="uid-500")
    controller.pause(reason="maintenance", actor="uid-500")
    assert controller.status()["state"] == "paused"
    controller.resume(actor="uid-500")
    assert controller.status()["state"] == "running"
    controller.record_exposure(now + timedelta(seconds=60))
    spent = controller.status()["spent_usd"]
    controller.record_exposure(now + timedelta(seconds=60))
    assert controller.status()["spent_usd"] == spent == pytest.approx(1 / 60)
    assert controller.list_queue()["entries"][0]["run_id"] == first.run_id


def test_untrusted_prices_and_catalog_order_cannot_shrink_reservations(tmp_path):
    controller, intent, now = configured(tmp_path)
    intent.job = intent.job.model_copy(update={"per_run_cost_usd": 0.001})
    with pytest.raises(ValueError, match="protected"):
        controller.admission.validate(intent, now=now)
    intent.job = controller.config.approved_profiles["gpu"]
    intent.classes.reverse()
    with pytest.raises(ValueError, match="catalog"):
        controller.admission.validate(intent, now=now)


def test_profile_cost_bound_and_nonfinite_campaign_rejected(tmp_path):
    controller, _, _ = configured(tmp_path)
    payload = controller.config.model_dump(mode="json")
    payload["approved_profiles"]["gpu"]["per_run_cost_usd"] = 0.001
    with pytest.raises(ValueError, match="maximum VM runtime"):
        QueueServiceConfig.model_validate(payload)
    payload["operator"]["campaign_cost_cap_usd"] = float("inf")
    with pytest.raises(ValueError):
        QueueServiceConfig.model_validate(payload)


def test_cli_exposes_queue_under_existing_training_commands():
    from typer.testing import CliRunner

    from defect_platform.cli import app
    result = CliRunner().invoke(app, ["train", "queue", "--help"])
    assert result.exit_code == 0
    for command in ("enqueue", "serve", "worker", "pause", "retry", "backup", "restore-check"):
        assert command in result.output


def claimed(controller, intent, now):
    controller.enqueue(intent, idempotency_key=intent.idempotency_key, requester="uid-500")
    return controller.store.claim_next(scope="vm", worker_id="worker", lease_seconds=60,
        boot_id="boot", observed_free_bytes=90000, observed_free_inodes=10000,
        capacity_evidence_ref="statfs", capacity_observed_at=now, now=now)


def test_lost_tracking_create_response_cannot_create_duplicate_run(tmp_path):
    controller, intent, now = configured(tmp_path)
    claim = claimed(controller, intent, now)
    class Client:
        creates = 0
        def get_experiment_by_name(self, _): return SimpleNamespace(experiment_id="1")
        def search_runs(self, *args, **kwargs): return []
        def create_run(self, *args, **kwargs):
            self.creates += 1
            raise TimeoutError("response lost after commit")
    client = Client()
    tracker = AttemptTracker(controller.config, client=client)
    tracker._auth = nullcontext
    journal = tmp_path / "tracking" / "attempt.json"
    with pytest.raises(TimeoutError):
        tracker.prepare(claim.entry, claim.attempt, journal)
    with pytest.raises(ValueError, match="outcome remains unknown"):
        tracker.prepare(claim.entry, claim.attempt, journal)
    assert client.creates == 1 and json.loads(journal.read_text())["state"] == "create_intent"


def test_complete_output_gate_rejects_partial_and_tampered_attempts(tmp_path, monkeypatch):
    controller, intent, now = configured(tmp_path)
    claim = claimed(controller, intent, now)
    class Tracker:
        def prepare(self, entry, attempt, path): return "mlflow-attempt"
        def verify(self, run_id, attempt, fingerprint): assert run_id == "mlflow-attempt"
    class Artifacts:
        calls = 0
        def verify(self, directory, destination): self.calls += 1
    controller.tracker = Tracker()
    artifacts = Artifacts()
    controller.artifact_verifier = artifacts
    monkeypatch.setattr("defect_platform.control.queue_controller.require_protected_file", lambda _: None)
    controller.config.credential_path.write_text("{}")
    spec = controller.prepare_execution(claim.entry, claim.attempt, HostIdentity("p", "z", "123", "boot"))
    assert spec.command[2] == "defect_platform.trainer.runner"
    assert spec.inputs[0].readonly
    original = spec.inputs[0].source.read_bytes()
    # Reconciliation prepares identical input; it never changes the active request.
    controller.prepare_execution(claim.entry, claim.attempt, HostIdentity("p", "z", "123", "boot"))
    assert spec.inputs[0].source.read_bytes() == original
    with pytest.raises(FileNotFoundError):
        controller.verify_outputs(claim.entry, claim.attempt, spec)
    semantic = controller.admission.semantic_loader(intent.dataset)
    model = make_training_manifest(semantic, intent.experiment, dataset_sha256=intent.dataset.sha256,
        runtime_image_digest=intent.runtime.image_digest, runtime_source_commit=intent.runtime.source_commit)
    directory = spec.output_dir / "model"
    (directory / "backbone").mkdir(parents=True)
    (directory / "semantics.json").write_bytes(serialize_manifest(model))
    (directory / "class_mapping.json").write_bytes(canonical_bytes({"classes": intent.classes}))
    for name in ("model.pt", "inference_config.json", "backbone/config.json", "backbone/model.safetensors"):
        (directory / name).write_bytes(b"fixture-opaque-bundle-member")
    write_bundle_integrity(directory)
    policy = policy_for_config(intent.experiment).model_dump(mode="json")
    keys = ("optimizer", "loss", "focal_gamma", "horizontal_flip_probability", "class_weights", "seed", "max_review_error_rate")
    metrics = {"mcc": 0.5, "macro_f1": 0.7, "accuracy": 0.8, "confusion": [[1, 0], [0, 1]],
               "per_class": {label: {} for label in intent.classes}}
    report = {"classes": intent.classes, "best_epoch": 1, "history": [{"epoch": 1}],
        "validation": metrics, "test": metrics, "training_config": {key: policy[key] for key in keys},
        "semantic_sha256": model.sha256, "device": "cuda", "visible_gpu_count": 1,
        "mlflow_run_id": "mlflow-attempt", "output_uri": json.loads(original)["output_uri"]}
    (spec.output_dir / "evaluation.json").write_bytes(canonical_bytes(report))
    assert controller.verify_outputs(claim.entry, claim.attempt, spec)["outputs_verified"]
    assert artifacts.calls == 1
    (directory / "model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        controller.verify_outputs(claim.entry, claim.attempt, spec)
    assert artifacts.calls == 1


def test_off_vm_backup_contains_recovery_state_without_credentials(tmp_path):
    controller, intent, now = configured(tmp_path)
    claimed(controller, intent, now)
    controller.config.credential_path.write_text("private-key-should-never-be-archived")
    class Blob:
        def __init__(self): self.data = None
        def upload_from_filename(self, path, *, if_generation_match):
            assert if_generation_match == 0
            self.data = Path(path).read_bytes()
        def open(self, mode): return io.BytesIO(self.data)
    class Bucket:
        def __init__(self): self.blobs = {}
        def blob(self, key):
            self.blobs[key] = Blob()
            return self.blobs[key]
    class Client:
        def __init__(self): self.selected = Bucket()
        def bucket(self, name): assert name == "backup"; return self.selected
    client = Client()
    controller.backup_client = client
    first = controller.maintain_backup(now)
    assert first.startswith("gs://backup/queue/")
    assert controller.maintain_backup(now + timedelta(seconds=10)) == first
    assert len(client.selected.blobs) == 1
    archive = next(iter(client.selected.blobs.values())).data
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        names = bundle.getnames()
        assert "queue.sqlite" in names and "operator-config.json" in names
        assert not any("credentials" in name or "scratch" in name for name in names)
        for member in bundle.getmembers():
            assert b"private-key-should-never-be-archived" not in bundle.extractfile(member).read()
    assert len(list((controller.storage_path / "backups/uploaded").glob("*.tar.gz"))) == 1


def test_administrator_recovery_requires_empty_observed_gpu_and_same_instance(tmp_path, monkeypatch):
    controller, intent, now = configured(tmp_path)
    claim = claimed(controller, intent, now)
    with pytest.raises(PermissionError):
        controller.resolve_interrupted(claim.entry.entry_id,
            expected_revision=claim.entry.revision, reason="boot changed", actor="uid-500")
    monkeypatch.setattr("defect_platform.control.queue_controller.os.geteuid", lambda: 0)
    class Executor:
        identity = HostIdentity("p", "z", "123", "newboot")
        def __init__(self): self.workloads = [{"container_id": "unbound"}]
        def assert_host_identity(self): return self.identity
        def inspect(self, *args): return ExecutionObservation("absent")
        def scan_gpu_workloads(self): return self.workloads
    executor = Executor()
    with pytest.raises(ValueError, match="not proven empty"):
        controller.resolve_interrupted(claim.entry.entry_id,
            expected_revision=claim.entry.revision, reason="boot changed", actor="admin-uid:0", executor=executor)
    assert controller.store.get_entry(claim.entry.entry_id).revision == claim.entry.revision
    executor.workloads = []
    recovered = controller.resolve_interrupted(claim.entry.entry_id,
        expected_revision=claim.entry.revision, reason="old boot/container loss verified",
        actor="admin-uid:0", executor=executor)
    assert recovered.state.value == "blocked" and recovered.container_exit_code is None
    assert controller.store.get_attempt(claim.attempt.attempt_id).state.value == "interrupted"
    assert recovered.outputs_verified is None


def integrated_worker(tmp_path, monkeypatch, policy=FailurePolicy.PAUSE):
    controller, intent, now = configured(tmp_path)
    controller.config.operator.failure_policy = policy
    controller.store = SQLiteQueueStore(tmp_path / "policy.sqlite")
    controller.store.initialize(scope="vm", operator_config=controller.config.operator,
                                actor="worker", now=now)
    monkeypatch.setattr("defect_platform.control.queue_controller.require_protected_file", lambda _: None)
    controller.config.credential_path.write_text("{}")
    class Tracker:
        def __init__(self): self.failures = []
        def prepare(self, entry, attempt, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"run_id": attempt.attempt_id}))
            return attempt.attempt_id
        def fail(self, run_id, *, canceled): self.failures.append((run_id, canceled))
    controller.tracker = Tracker()
    controller.maintain_backup = lambda _: "verified-backup-fixture"
    class Executor:
        def __init__(self):
            self.containers = {}
            self.creates = 0
            self.starts = 0
            self.lose_start_response = False
            self.terminate = False
        def assert_host_identity(self): return HostIdentity("p", "z", "123", "boot")
        def assert_readiness(self, *args, **kwargs): return {}
        def disk_capacity(self, path): return DiskCapacity(path, 90000, 10000, 100000)
        def scan_gpu_workloads(self):
            return [{"container_id": cid} for cid, value in self.containers.items() if value["state"] == "running"]
        def container_name(self, run_id, attempt_id): return VMExecutor.container_name(run_id, attempt_id)
        def create(self, spec):
            self.creates += 1
            cid = "c" * 63 + str(self.creates)
            self.containers[cid] = {"state": "created", "exit": None, "spec": spec}
            return ExecutionResult(cid, self.inspect(spec.run_id, spec.attempt_id, cid), ("docker", "create"))
        def inspect(self, run_id, attempt_id, container_id=None):
            cid = container_id or next((cid for cid, value in self.containers.items()
                if value["spec"].run_id == run_id and value["spec"].attempt_id == attempt_id), None)
            if cid is None or cid not in self.containers:
                return ExecutionObservation("absent")
            value = self.containers[cid]
            return ExecutionObservation(value["state"], cid, value["exit"], value["spec"].image_digest,
                {"defect-platform.managed": "true", "defect-platform.run-id": run_id,
                 "defect-platform.attempt-id": attempt_id}, process_id="1234" if value["state"] == "running" else None)
        def start(self, cid, *, run_id, attempt_id):
            self.starts += 1
            if self.lose_start_response:
                raise TimeoutError("start response lost before execution")
            self.containers[cid]["state"] = "running"
            return ExecutionResult(cid, self.inspect(run_id, attempt_id, cid), ("docker", "start"))
        def cancel(self, cid, *, run_id, attempt_id):
            if self.containers[cid]["state"] == "created":
                del self.containers[cid]
                return ExecutionResult(cid, ExecutionObservation("absent"), ("docker", "rm"))
            if self.terminate:
                self.containers[cid].update(state="exited", exit=143)
            return ExecutionResult(cid, self.inspect(run_id, attempt_id, cid), ("docker", "stop"))
        def logs(self, cid): return "fixture training log"
    executor = Executor()
    worker = QueueWorker(store=controller.store, executor=executor, authority=controller,
        scope="vm", worker_id="worker", lock_path=controller.storage_path / "worker.lock", clock=lambda: now)
    first = controller.enqueue(intent, idempotency_key=intent.idempotency_key, requester="uid-500")
    second = controller.enqueue(intent.model_copy(update={"idempotency_key": "second"}),
                                idempotency_key="second", requester="uid-500")
    return controller, executor, worker, first, second


def test_cancel_does_not_release_slot_until_actual_termination(tmp_path, monkeypatch):
    controller, executor, worker, first, second = integrated_worker(tmp_path, monkeypatch)
    assert worker.tick().action == "started"
    controller.request_active_cancel(first.entry_id, actor="uid-500")
    assert worker.tick().action == "canceling"
    assert executor.creates == 1 and controller.store.get_entry(second.entry_id).state.value == "waiting"
    executor.terminate = True
    assert worker.tick().action == "canceled"
    canceled = controller.store.get_entry(first.entry_id)
    assert canceled.container_exit_code == 143
    assert canceled.mlflow_finalized is True and canceled.outputs_verified is False
    assert controller.tracker.failures[-1][1] is True
    assert worker.tick().action == "started" and executor.creates == 2


def test_cancel_created_container_never_replays_start_or_invents_exit(tmp_path, monkeypatch):
    controller, executor, worker, first, _second = integrated_worker(tmp_path, monkeypatch)
    executor.lose_start_response = True
    assert worker.tick().action == "unknown"
    assert executor.starts == 1
    controller.request_active_cancel(first.entry_id, actor="uid-500")
    assert worker.tick().action == "canceled"
    canceled = controller.store.get_entry(first.entry_id)
    attempt = controller.store.get_attempt(canceled.active_attempt_id)
    assert canceled.container_exit_code is None and attempt.exit_code is None
    assert canceled.mlflow_finalized is True and canceled.outputs_verified is False
    assert controller.tracker.failures[-1][1] is True
    assert executor.starts == 1 and not executor.containers
    executor.lose_start_response = False
    assert worker.tick().action == "started" and executor.creates == 2


@pytest.mark.parametrize("policy", [FailurePolicy.PAUSE, FailurePolicy.CONTINUE])
def test_worker_failure_policy_reconciles_tracking_before_continuing(tmp_path, monkeypatch, policy):
    controller, executor, worker, first, _second = integrated_worker(tmp_path, monkeypatch, policy)
    assert worker.tick().action == "started"
    cid = controller.store.get_attempt(controller.store.get_entry(first.entry_id).active_attempt_id).container_id
    executor.containers[cid].update(state="exited", exit=1)
    result = worker.tick()
    assert controller.tracker.failures[-1][1] is False
    resolved = controller.store.get_entry(first.entry_id)
    assert resolved.mlflow_finalized is True and resolved.outputs_verified is False
    if policy == FailurePolicy.PAUSE:
        assert result.action == "blocked" and resolved.state.value == "blocked"
        assert worker.tick().action in {"held", "blocked"} and executor.creates == 1
    else:
        assert result.action == "failed" and resolved.state.value == "failed"
        assert worker.tick().action == "started" and executor.creates == 2
    assert resolved.release_eligible is False
