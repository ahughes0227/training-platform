import json
from datetime import UTC, datetime
from pathlib import Path

from defect_platform.contracts import (
    CertifiedRuntime,
    DatasetVersion,
    ExperimentConfig,
    ModelSpec,
    ValidationResults,
)
from defect_platform.control.queue_contracts import (
    QueueIntent,
    QueueOperatorConfig,
    QueuePolicyState,
    VMJobProfile,
)
from defect_platform.control.queue_store import SQLiteQueueStore
from defect_platform.control.queue_worker import QueueWorker
from defect_platform.control.vm_executor import (
    BindMount,
    ContainerSpec,
    DiskCapacity,
    ExecutionObservation,
    ExecutionResult,
    HostIdentity,
)


class NeverStore:
    def list_entries(self, scope):
        return []

    def claim_next(self, **kwargs):
        raise AssertionError("unsafe dispatch must be held before queue claim")


class Executor:
    expected_identity = HostIdentity("p", "z", "i", "boot")

    def __init__(self, current=None, workloads=None):
        self.current = current or self.expected_identity
        self.workloads = workloads or []

    def assert_host_identity(self):
        if self.current != self.expected_identity:
            raise RuntimeError("VM or boot identity changed")
        return self.current

    def scan_gpu_workloads(self):
        return self.workloads


class Authority:
    storage_path = None

    def record_exposure(self, now):
        pass

    def maintain_backup(self, now):
        return "backup"

    def heartbeat(self, result):
        pass

    def validate_executor(self, executor, identity):
        pass


def worker(tmp_path, executor):
    return QueueWorker(store=NeverStore(), executor=executor, authority=Authority(),
                       scope="vm-gpu", worker_id="worker-a",
                       lock_path=tmp_path / "worker.lock",
                       clock=lambda: datetime.now(UTC))


def test_worker_holds_before_claim_when_unbound_gpu_workload_exists(tmp_path):
    result = worker(tmp_path, Executor(workloads=[{"container_id": "unknown", "kind": "container-gpu"}])).tick()
    assert result.action == "held"
    assert "unbound container" in result.reason


def test_worker_holds_if_gpu_observer_is_unavailable(tmp_path):
    class FailedObserver(Executor):
        def scan_gpu_workloads(self):
            raise OSError("nvidia-smi unavailable")

    result = worker(tmp_path, FailedObserver()).tick()
    assert result.action == "held"
    assert "GPU workload observation failed" in result.reason


def test_worker_fences_changed_boot_before_reading_queue(tmp_path):
    changed = HostIdentity("p", "z", "i", "new-boot")
    result = worker(tmp_path, Executor(current=changed)).tick()
    assert result.action == "held"
    assert "identity fence failed" in result.reason


def test_worker_runs_persisted_create_start_terminal_finalize_lifecycle(tmp_path):
    now = datetime(2026, 9, 29, tzinfo=UTC)
    digest = "a" * 64
    image = f"us-docker.pkg.dev/p/r/trainer@sha256:{digest}"
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    dataset = DatasetVersion(version_id="v1", object_slug="gear", root_uri="gs://data/v1",
                             manifest_uri="gs://data/v1/manifest.json",
                             shard_uris={"train": ["gs://data/train"]}, sample_counts={"train": 4},
                             sha256=digest, source_snapshot_uri="gs://data/v1/source")
    experiment = ExperimentConfig(experiment_id="exp-v1", object_slug="gear", dataset_version_id="v1",
                                  runtime_id="runtime-1", model=ModelSpec(weights_uri="gs://w/model.pt",
                                                                           weights_sha256=digest))
    runtime = CertifiedRuntime(runtime_id="runtime-1", source_commit="b" * 40, image_tag="v1",
                               image_digest=image, runtime_version="1", python_version="3.12",
                               pytorch_version="2", cuda_version="12",
                               validation=ValidationResults(trainer=True, container_gpu=True,
                                                            vertex_gpu=True, gcs_read=True, gcs_write=True),
                               certified=True, certified_at=now)
    profile = VMJobProfile(profile_id="profile", project_id="p", zone="z", instance_id="i",
                           gpu_identity="gpu-0", image_digest=image, estimated_runtime_seconds=30,
                           max_runtime_seconds=60, per_run_cost_usd=1, storage_peak_bytes=100,
                           storage_peak_inodes=2, max_attempts=1)
    intent = QueueIntent(experiment=experiment, dataset=dataset, runtime=runtime,
                         classes=["defect", "ok"], job=profile, idempotency_key="key")
    config = QueueOperatorConfig(
        revision=1, max_backlog=5, campaign_cost_cap_usd=10, campaign_authorization_id="approved",
        cost_meter_started_at=now, vm_hourly_cost_usd=1, reserved_headroom_bytes=100,
        reserved_headroom_inodes=10, usable_capacity_bytes=10_000, observed_free_bytes=5000,
        observed_free_inodes=500, capacity_observed_at=now, max_start_delay_seconds=300,
        stalled_work_seconds=300, recovery_bound_seconds=600,
        max_capacity_observation_age_seconds=300, retry_backoff_seconds=60,
        backup_evidence_ref="backup-policy")
    entry = store.enqueue_batch(scope="vm", intents=[intent], operator_config=config,
                                expected_policy_revision=1, actor="operator", now=now,
                                batch_idempotency_key="batch-a",
                                observed_free_bytes=5000, observed_free_inodes=500,
                                capacity_evidence_ref="enqueue-statfs",
                                capacity_observed_at=now)[0]
    identity = HostIdentity("p", "z", "i", "boot")

    class FakeExecutor:
        def __init__(self):
            self.engine = "docker"
            self.state, self.cid = "absent", "c" * 64
            self.create_calls = self.start_calls = 0
            self.lose_create_ack = True
            self.lose_start_ack = True
        def assert_host_identity(self): return identity
        def disk_capacity(self, path): return DiskCapacity(path, 5000, 500, 10_000)
        def scan_gpu_workloads(self): return []
        def container_name(self, run_id, attempt_id): return f"defect-{run_id}-{attempt_id}"
        def labels(self, run_id, attempt_id):
            return {"defect-platform.managed": "true", "defect-platform.run-id": run_id,
                    "defect-platform.attempt-id": attempt_id}
        def create(self, spec):
            self.create_calls += 1; self.state = "created"
            if self.lose_create_ack:
                self.lose_create_ack = False
                return ExecutionResult(None, ExecutionObservation("unknown", error="connection reset"),
                                       ("docker", "create"), "create response lost")
            return ExecutionResult(self.cid, ExecutionObservation("created", self.cid, image=image,
                                                                 labels=self.labels(spec.run_id, spec.attempt_id)),
                                   ("docker", "create"), "create response reconciled by deterministic identity")
        def start(self, container_id, *, run_id, attempt_id):
            self.start_calls += 1; self.state = "running"
            if self.lose_start_ack:
                self.lose_start_ack = False
                return ExecutionResult(self.cid, ExecutionObservation("unknown", error="connection reset"),
                                       ("docker", "start"), "start response lost")
            return ExecutionResult(self.cid, ExecutionObservation("running", self.cid, image=image,
                                                                 labels=self.labels(run_id, attempt_id)),
                                   ("docker", "start"))
        def inspect(self, run_id, attempt_id, container_id=None):
            return ExecutionObservation(self.state, self.cid if self.state != "absent" else None,
                                        0 if self.state == "exited" else None, image,
                                        {"defect-platform.managed": "true",
                                         "defect-platform.run-id": run_id,
                                         "defect-platform.attempt-id": attempt_id},
                                        process_id="4321" if self.state == "running" else None)
        def logs(self, container_id): return "trainer done\n"
        def remove_verified_attempt(self, run_id, attempt_id, container_id, image_digest):
            self.state = "absent"
            return ExecutionResult(container_id, ExecutionObservation("absent"),
                                   ("docker", "rm", "--", container_id))

    executor = FakeExecutor()

    class StoreAuthority:
        storage_path = tmp_path
        def record_exposure(self, observed_at):
            policy = store.policy("vm")
            store.record_exposure(scope="vm", expected_policy_revision=policy["revision"], additional_cost_usd=0,
                                  actor="worker", now=observed_at, evidence_ref="exposure-meter")
        def record_capacity(self, capacity, observed_at): return "statfs-measured"
        def maintain_backup(self, observed_at): return "backup-evidence"
        def heartbeat(self, result): pass
        def validate_executor(self, executor, host): pass
        def persist_worker_evidence(self, claimed, attempt, event, evidence):
            path = tmp_path / f"{attempt.attempt_id}-{event}.json"
            path.write_text(json.dumps(evidence))
            return str(path)
        def prepare_execution(self, claimed, attempt, host):
            base = tmp_path / "attempt-files" / attempt.attempt_id
            base.mkdir(parents=True, exist_ok=True)
            request = base / "request.json"
            request.write_text(json.dumps({"run_id": claimed.run_id}))
            output, scratch = base / "output", base / "scratch"
            output.mkdir(exist_ok=True); scratch.mkdir(exist_ok=True)
            return ContainerSpec(
                run_id=claimed.run_id, attempt_id=attempt.attempt_id, image_digest=image,
                command=("python", "-m", "defect_platform.trainer.runner", "--request",
                         "/defect/input/request.json", "--output-dir", "/defect/output"),
                output_dir=output, scratch_dir=scratch,
                inputs=(BindMount(request, "/defect/input/request.json"),),
                uid_gid="10001:10001", gpu_devices=("0",), network_mode="defect-trainer-isolated",
                memory_bytes=1024, pids_limit=100, timeout_seconds=60,
                metadata_isolation_verified=True)
        def persist_logs(self, claimed, attempt, text):
            path = tmp_path / f"{attempt.attempt_id}.log"
            path.write_text(text)
            return str(path)
        def read_prepared_execution(self, claimed, attempt):
            return self.prepare_execution(claimed, attempt, identity)
        def cleanup_verified_attempt(self, claimed, attempt, spec):
            return None
        def verify_outputs(self, claimed, attempt, spec):
            return {"outputs_verified": True, "mlflow_finalized": True,
                    "evidence_ref": "finalization-evidence"}

    worker_instance = QueueWorker(store=store, executor=executor, authority=StoreAuthority(),
                                  scope="vm", worker_id="worker", lock_path=tmp_path / "worker.lock",
                                  clock=lambda: now)
    # Lost create and start acknowledgments both become unknown; deterministic
    # inspection recovers the exact container without repeating either operation.
    first_tick = worker_instance.tick()
    assert first_tick.action == "unknown", first_tick.reason
    second_tick = worker_instance.tick()
    assert second_tick.action == "unknown", second_tick
    recovered = worker_instance.tick()
    assert recovered.action == "active", recovered
    assert executor.create_calls == executor.start_calls == 1
    dataset_b = dataset.model_copy(update={"version_id": "v2", "root_uri": "gs://data/v2",
        "manifest_uri": "gs://data/v2/manifest.json", "source_snapshot_uri": "gs://data/v2/source",
        "shard_uris": {"train": ["gs://data/v2/train"]}})
    experiment_b = experiment.model_copy(update={"experiment_id": "exp-v2", "dataset_version_id": "v2"})
    intent_b = QueueIntent(experiment=experiment_b, dataset=dataset_b, runtime=runtime,
                          classes=["defect", "ok"], job=profile, idempotency_key="key-b")
    queued_b = store.enqueue_batch(scope="vm", intents=[intent_b], operator_config=config,
                                   expected_policy_revision=1, actor="operator", now=now,
                                   batch_idempotency_key="batch-b",
                                   observed_free_bytes=5000, observed_free_inodes=500,
                                   capacity_evidence_ref="enqueue-statfs-b",
                                   capacity_observed_at=now)[0]
    assert queued_b.state.value == "waiting"
    assert executor.create_calls == 1  # appending work does not alter or replace active A
    policy = store.policy("vm")
    store.set_policy(scope="vm", expected_policy_revision=policy["revision"],
                     state=QueuePolicyState.PAUSED, actor="operator", now=now)
    executor.state = "exited"
    assert worker_instance.tick().action == "completed"
    assert executor.create_calls == 1  # the next request waits until A is finalized
    assert worker_instance.tick().action == "idle"
    assert store.get_entry(queued_b.entry_id).state.value == "waiting"
    policy = store.policy("vm")
    store.set_policy(scope="vm", expected_policy_revision=policy["revision"],
                     state=QueuePolicyState.RUNNING, actor="operator", now=now)
    assert worker_instance.tick().action == "started"
    assert executor.create_calls == 2
    completed = store.get_entry(entry.entry_id)
    attempt = store.get_attempt(completed.active_attempt_id)
    assert completed.state.value == "completed"
    assert attempt.external_outcome.value == "terminal"
    assert attempt.process_id == "4321"
    assert Path(attempt.log_ref).read_text() == "trainer done\n"
