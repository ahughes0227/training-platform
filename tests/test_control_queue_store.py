from __future__ import annotations

import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from defect_platform.contracts import (
    CertifiedRuntime,
    DatasetVersion,
    ExperimentConfig,
    ModelSpec,
    ValidationResults,
)
from defect_platform.control.queue_contracts import (
    ExternalOutcome,
    QueueIntent,
    QueueOperatorConfig,
    QueuePolicyState,
    QueueState,
    RetryClassification,
    VMJobProfile,
)
from defect_platform.control.queue_store import (
    QueueAdmissionError,
    QueueConflict,
    SQLiteQueueStore,
)

NOW = datetime(2026, 9, 29, tzinfo=UTC)
HASH = "a" * 64
IMAGE = f"registry.example/train@sha256:{HASH}"


def make_intent(key: str, *, version: str = "v1", cost: float = 10.0,
                disk_bytes: int = 100, attempts: int = 1) -> QueueIntent:
    dataset = DatasetVersion(
        version_id=version, object_slug="gear", root_uri=f"gs://data/{version}",
        manifest_uri=f"gs://data/{version}/manifest.json", shard_uris={"train": ["gs://data/train"]},
        sample_counts={"train": 10}, sha256=HASH, source_snapshot_uri=f"gs://data/{version}/source")
    experiment = ExperimentConfig(
        experiment_id=f"exp-{version}", object_slug="gear", dataset_version_id=version,
        runtime_id="runtime-1", model=ModelSpec(weights_uri="gs://weights/model.pt", weights_sha256=HASH))
    runtime = CertifiedRuntime(
        runtime_id="runtime-1", source_commit="b" * 40, image_tag="v1", image_digest=IMAGE,
        runtime_version="1", python_version="3.12", pytorch_version="2", cuda_version="12",
        validation=ValidationResults(trainer=True, container_gpu=True, vertex_gpu=True,
                                     gcs_read=True, gcs_write=True), certified=True, certified_at=NOW)
    job = VMJobProfile(
        profile_id="operator-profile-v1", project_id="train-project", zone="us-central1-a",
        instance_id="123456789", gpu_identity="nvidia-l4-1", image_digest=IMAGE,
        estimated_runtime_seconds=3600, max_runtime_seconds=7200,
        per_run_cost_usd=cost, storage_peak_bytes=disk_bytes, storage_peak_inodes=5,
        max_attempts=attempts)
    return QueueIntent(experiment=experiment, dataset=dataset, runtime=runtime,
                       classes=["defect", "ok"], job=job, idempotency_key=key)


def operator_config(**overrides) -> QueueOperatorConfig:
    fields = {
        "revision": 1, "max_backlog": 20, "campaign_cost_cap_usd": 100.0,
        "campaign_authorization_id": "approved-budget-revision-9",
        "cost_meter_started_at": NOW, "vm_hourly_cost_usd": 1.0,
        "reserved_headroom_bytes": 100, "reserved_headroom_inodes": 10,
        "usable_capacity_bytes": 100_000, "observed_free_bytes": 10_000,
        "observed_free_inodes": 1000, "capacity_observed_at": NOW,
        "max_start_delay_seconds": 3600, "stalled_work_seconds": 3600,
        "recovery_bound_seconds": 7200, "retry_backoff_seconds": 0,
        "max_capacity_observation_age_seconds": 300,
        "backup_evidence_ref": "operator-reviewed-backup-policy",
    }
    fields.update(overrides)
    return QueueOperatorConfig(**fields)


def enqueue(store, *intents, config=None, observed_at=NOW,
            free_bytes=10_000, free_inodes=1000, batch_key=None):
    return store.enqueue_batch(scope="vm-123", intents=list(intents),
                               batch_idempotency_key=batch_key or "test:" + ",".join(
                                   item.idempotency_key for item in intents),
                               operator_config=config or operator_config(),
                               expected_policy_revision=1, actor="uid:operator",
                               observed_free_bytes=free_bytes, observed_free_inodes=free_inodes,
                               capacity_evidence_ref="statfs:enqueue",
                               capacity_observed_at=observed_at, now=observed_at)


def test_finite_batch_is_fifo_idempotent_and_atomic(tmp_path):
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    a, b = make_intent("key-a", version="v1"), make_intent("key-b", version="v2")
    entries = enqueue(store, a, b, batch_key="stable-batch")
    assert [e.sequence for e in entries] == [1, 2]
    replayed = enqueue(store, a, b, batch_key="stable-batch", observed_at=NOW - timedelta(days=1),
                       free_bytes=0, free_inodes=0)
    assert [e.entry_id for e in replayed] == [e.entry_id for e in entries]
    with pytest.raises(QueueAdmissionError, match="ordered request list"):
        enqueue(store, make_intent("different"), batch_key="stable-batch")
    assert enqueue(store, a)[0].entry_id == entries[0].entry_id
    with pytest.raises(QueueAdmissionError, match="different intent"):
        enqueue(store, make_intent("key-a", version="v3"), batch_key="new-batch")
    assert [e.entry_id for e in store.list_entries("vm-123")] == [e.entry_id for e in entries]

    # The second new entry cannot fit the aggregate campaign cap. The first must
    # not be committed when its sibling fails this finite batch admission.
    limited = SQLiteQueueStore(tmp_path / "limited.sqlite")
    with pytest.raises(QueueAdmissionError, match="cumulative campaign budget"):
        enqueue(limited, make_intent("k1", cost=30), make_intent("k2", cost=30),
                config=operator_config(campaign_cost_cap_usd=50))
    assert limited.list_entries("vm-123") == []


def test_old_pinned_capacity_sample_does_not_block_restart_or_fresh_enqueue(tmp_path):
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    stale_policy = operator_config(capacity_observed_at=NOW - timedelta(days=1))
    store.initialize(scope="vm-123", operator_config=stale_policy,
                     actor="service", now=NOW)
    entries = enqueue(store, make_intent("fresh"), config=stale_policy,
                      observed_at=NOW + timedelta(days=1), free_bytes=5000, free_inodes=500)
    assert entries[0].sequence == 1


def test_claim_persists_dispatch_identity_and_never_replays_expired_unknown(tmp_path):
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    entry = enqueue(store, make_intent("k"))[0]
    claim = store.claim_next(scope="vm-123", worker_id="worker-1", lease_seconds=1,
                             boot_id="boot-abc", observed_free_bytes=5000,
                             observed_free_inodes=500, capacity_evidence_ref="statfs:1",
                             capacity_observed_at=NOW, now=NOW)
    assert claim and claim.entry.state == QueueState.DISPATCHING
    assert claim.attempt.boot_id == "boot-abc"
    assert store.claim_next(scope="vm-123", worker_id="worker-2", lease_seconds=60,
                            boot_id="boot-abc", observed_free_bytes=5000,
                            observed_free_inodes=500, capacity_evidence_ref="statfs:2",
                            capacity_observed_at=NOW + timedelta(hours=1),
                            now=NOW + timedelta(hours=1)) is None
    blocked = store.record_dispatch(
        entry_id=entry.entry_id, expected_revision=claim.entry.revision,
        attempt_id=claim.attempt.attempt_id, dispatch_id=claim.attempt.dispatch_id,
        claim_token=claim.attempt.claim_token,
        external_outcome=ExternalOutcome.UNKNOWN, actor="worker-1", now=NOW + timedelta(seconds=2),
        evidence_ref="engine-timeout")
    assert blocked.state == QueueState.BLOCKED
    assert store.claim_next(scope="vm-123", worker_id="worker-2", lease_seconds=60,
                            boot_id="boot-abc", observed_free_bytes=5000,
                            observed_free_inodes=500, capacity_evidence_ref="statfs:3",
                            capacity_observed_at=NOW + timedelta(hours=2),
                            now=NOW + timedelta(hours=2)) is None
    with pytest.raises(QueueConflict, match="confirmed failed or verified interrupted"):
        store.retry(entry_id=entry.entry_id, expected_revision=blocked.revision,
                    actor="operator", now=NOW, evidence_ref="retry")


def test_retry_requires_confirmed_failure_and_honors_remaining_budget_and_backoff(tmp_path):
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    intent = make_intent("retry", attempts=2)
    entry = enqueue(store, intent, config=operator_config(retry_backoff_seconds=60))[0]
    first = store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                             boot_id="boot", observed_free_bytes=5000, observed_free_inodes=500,
                             capacity_evidence_ref="capacity-1", capacity_observed_at=NOW, now=NOW)
    terminal = store.record_dispatch(
        entry_id=entry.entry_id, expected_revision=first.entry.revision,
        attempt_id=first.attempt.attempt_id, dispatch_id=first.attempt.dispatch_id,
        claim_token=first.attempt.claim_token, external_outcome=ExternalOutcome.TERMINAL,
        actor="worker", now=NOW + timedelta(seconds=1), exit_code=2, evidence_ref="exit-2")
    failed = store.record_finalization(
        entry_id=entry.entry_id, expected_revision=terminal.revision,
        outputs_verified=False, mlflow_finalized=True, actor="worker",
        now=NOW + timedelta(seconds=2), evidence_ref="failed-output")
    assert failed.state == QueueState.BLOCKED
    assert failed.budget_reserved_usd == 10
    assert store.policy("vm-123")["state"] == QueuePolicyState.PAUSED.value
    for classification in (RetryClassification.UNKNOWN, RetryClassification.DETERMINISTIC,
                           RetryClassification.INTERRUPTED):
        with pytest.raises(QueueAdmissionError, match="recoverability classification"):
            store.retry(entry_id=entry.entry_id, expected_revision=failed.revision,
                        actor="operator", now=NOW + timedelta(seconds=3), evidence_ref="reviewed",
                        classification=classification)
        assert store.get_entry(entry.entry_id).revision == failed.revision
        assert len(store.list_attempts(entry.entry_id)) == 1
    waiting = store.retry(entry_id=entry.entry_id, expected_revision=failed.revision,
                          actor="operator", now=NOW + timedelta(seconds=3), evidence_ref="retry-approved",
                          classification=RetryClassification.TRANSIENT)
    assert waiting.next_eligible_at == NOW + timedelta(seconds=62)
    policy_revision = store.policy("vm-123")["revision"]
    store.set_policy(scope="vm-123", expected_policy_revision=policy_revision,
                     state=QueuePolicyState.RUNNING, actor="operator",
                     now=NOW + timedelta(seconds=4))
    assert store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                            boot_id="boot", observed_free_bytes=5000, observed_free_inodes=500,
                            capacity_evidence_ref="capacity-2",
                            capacity_observed_at=NOW + timedelta(seconds=60),
                            now=NOW + timedelta(seconds=60)) is None
    assert store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                            boot_id="boot", observed_free_bytes=5000, observed_free_inodes=500,
                            capacity_evidence_ref="capacity-early",
                            capacity_observed_at=NOW + timedelta(seconds=61),
                            now=NOW + timedelta(seconds=61)) is None
    second = store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                              boot_id="boot", observed_free_bytes=5000, observed_free_inodes=500,
                              capacity_evidence_ref="capacity-3",
                              capacity_observed_at=NOW + timedelta(seconds=62),
                              now=NOW + timedelta(seconds=62))
    assert second is not None
    assert second.attempt.attempt_id != first.attempt.attempt_id
    assert second.entry.container_exit_code is None
    assert second.entry.mlflow_finalized is None and second.entry.outputs_verified is None
    assert second.entry.cancellation_requested is False
    assert store.get_attempt(first.attempt.attempt_id).exit_code == 2


def test_start_deadline_is_anchored_to_first_persisted_start_intent(tmp_path):
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    entry = enqueue(store, make_intent("start-ack"))[0]
    claim = store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                             boot_id="boot", observed_free_bytes=5000, observed_free_inodes=500,
                             capacity_evidence_ref="capacity", capacity_observed_at=NOW, now=NOW)
    start_intent = store.record_dispatch(
        entry_id=entry.entry_id, expected_revision=claim.entry.revision,
        attempt_id=claim.attempt.attempt_id, dispatch_id=claim.attempt.dispatch_id,
        claim_token=claim.attempt.claim_token, external_outcome=ExternalOutcome.START_INTENT,
        actor="worker", now=NOW + timedelta(seconds=2), evidence_ref="before-start")
    attempt = store.get_attempt(claim.attempt.attempt_id)
    first_deadline = attempt.deadline_at
    assert first_deadline == NOW + timedelta(seconds=2 + 7200)
    started = store.record_dispatch(
        entry_id=entry.entry_id, expected_revision=start_intent.revision,
        attempt_id=claim.attempt.attempt_id, dispatch_id=claim.attempt.dispatch_id,
        claim_token=claim.attempt.claim_token, external_outcome=ExternalOutcome.STARTED,
        actor="worker", now=NOW + timedelta(minutes=5), evidence_ref="start-ack-late")
    attempt = store.get_attempt(claim.attempt.attempt_id)
    assert started.state == QueueState.ACTIVE
    assert attempt.started_at == NOW + timedelta(minutes=5)
    assert attempt.deadline_at == first_deadline


def test_verified_interruption_has_no_invented_exit_and_requires_explicit_retry(tmp_path):
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    entry = enqueue(store, make_intent("interrupted", attempts=2))[0]
    claim = store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                             boot_id="old-boot", observed_free_bytes=5000, observed_free_inodes=500,
                             capacity_evidence_ref="capacity", capacity_observed_at=NOW, now=NOW)
    started = store.record_dispatch(
        entry_id=entry.entry_id, expected_revision=claim.entry.revision,
        attempt_id=claim.attempt.attempt_id, dispatch_id=claim.attempt.dispatch_id,
        claim_token=claim.attempt.claim_token, external_outcome=ExternalOutcome.STARTED,
        actor="worker", now=NOW + timedelta(seconds=1))
    interrupted = store.mark_interrupted(
        entry_id=entry.entry_id, expected_revision=started.revision, actor="reconciler",
        now=NOW + timedelta(minutes=1), evidence_ref="new-boot-and-empty-gpu-proof")
    attempt = store.get_attempt(claim.attempt.attempt_id)
    assert attempt.state.value == "interrupted"
    assert attempt.external_outcome == ExternalOutcome.TERMINAL
    assert attempt.exit_code is None
    assert interrupted.container_exit_code is None
    assert interrupted.state == QueueState.BLOCKED
    assert store.policy("vm-123")["state"] == QueuePolicyState.PAUSED.value
    assert store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                            boot_id="new-boot", observed_free_bytes=5000, observed_free_inodes=500,
                            capacity_evidence_ref="fresh", capacity_observed_at=NOW + timedelta(minutes=1),
                            now=NOW + timedelta(minutes=1)) is None
    retry = store.retry(entry_id=entry.entry_id, expected_revision=interrupted.revision,
                        actor="operator", now=NOW + timedelta(minutes=2), evidence_ref="retry-approved",
                        classification=RetryClassification.INTERRUPTED)
    policy_revision = store.policy("vm-123")["revision"]
    store.set_policy(scope="vm-123", expected_policy_revision=policy_revision,
                     state=QueuePolicyState.RUNNING, actor="operator", now=NOW + timedelta(minutes=3))
    claim2 = store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                              boot_id="new-boot", observed_free_bytes=5000, observed_free_inodes=500,
                              capacity_evidence_ref="fresh-2",
                              capacity_observed_at=NOW + timedelta(minutes=3),
                              now=max(retry.next_eligible_at, NOW + timedelta(minutes=3)))
    assert claim2 is not None
    assert claim2.attempt.attempt_id != claim.attempt.attempt_id


def test_configured_continue_marks_failure_resolved_and_advances_fifo(tmp_path):
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    first, second_intent = make_intent("failure"), make_intent("next", version="v2")
    entries = enqueue(store, first, second_intent,
                      config=operator_config(failure_policy="continue"))
    claim = store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                             boot_id="boot", observed_free_bytes=5000, observed_free_inodes=500,
                             capacity_evidence_ref="capacity", capacity_observed_at=NOW, now=NOW)
    terminal = store.record_dispatch(
        entry_id=entries[0].entry_id, expected_revision=claim.entry.revision,
        attempt_id=claim.attempt.attempt_id, dispatch_id=claim.attempt.dispatch_id,
        claim_token=claim.attempt.claim_token, external_outcome=ExternalOutcome.TERMINAL,
        actor="worker", now=NOW + timedelta(seconds=1), exit_code=1, evidence_ref="exit-1")
    failed = store.record_finalization(
        entry_id=entries[0].entry_id, expected_revision=terminal.revision,
        outputs_verified=False, mlflow_finalized=True, actor="worker",
        now=NOW + timedelta(seconds=2), evidence_ref="failure-reviewed")
    assert failed.state == QueueState.FAILED
    assert store.policy("vm-123")["state"] == QueuePolicyState.RUNNING.value
    next_claim = store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                                  boot_id="boot", observed_free_bytes=5000, observed_free_inodes=500,
                                  capacity_evidence_ref="capacity-next",
                                  capacity_observed_at=NOW + timedelta(seconds=3),
                                  now=NOW + timedelta(seconds=3))
    assert next_claim.entry.entry_id == entries[1].entry_id


def test_continue_waits_for_tracking_finalization_before_advancing(tmp_path):
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    entries = enqueue(store, make_intent("tracking-failure"),
                      make_intent("after-tracking-failure", version="v2"),
                      config=operator_config(failure_policy="continue"))
    claim = store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                             boot_id="boot", observed_free_bytes=5000, observed_free_inodes=500,
                             capacity_evidence_ref="capacity", capacity_observed_at=NOW, now=NOW)
    terminal = store.record_dispatch(
        entry_id=entries[0].entry_id, expected_revision=claim.entry.revision,
        attempt_id=claim.attempt.attempt_id, dispatch_id=claim.attempt.dispatch_id,
        claim_token=claim.attempt.claim_token, external_outcome=ExternalOutcome.TERMINAL,
        actor="worker", now=NOW + timedelta(seconds=1), exit_code=1, evidence_ref="exit-1")
    incomplete = store.record_finalization(
        entry_id=entries[0].entry_id, expected_revision=terminal.revision,
        outputs_verified=False, mlflow_finalized=False, actor="worker",
        now=NOW + timedelta(seconds=2), evidence_ref="tracking-incomplete")
    assert incomplete.state == QueueState.BLOCKED
    assert "Tracking finalization is unverified" in incomplete.blocking_reason
    assert store.get_attempt(claim.attempt.attempt_id).state.value == "terminal"
    assert store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                            boot_id="boot", observed_free_bytes=5000, observed_free_inodes=500,
                            capacity_evidence_ref="capacity-blocked",
                            capacity_observed_at=NOW + timedelta(seconds=3),
                            now=NOW + timedelta(seconds=3)) is None
    confirmed = store.record_finalization(
        entry_id=entries[0].entry_id, expected_revision=incomplete.revision,
        outputs_verified=False, mlflow_finalized=True, actor="worker",
        now=NOW + timedelta(seconds=4), evidence_ref="tracking-finalized")
    assert confirmed.state == QueueState.FAILED
    next_claim = store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                                  boot_id="boot", observed_free_bytes=5000, observed_free_inodes=500,
                                  capacity_evidence_ref="capacity-next",
                                  capacity_observed_at=NOW + timedelta(seconds=5),
                                  now=NOW + timedelta(seconds=5))
    assert next_claim.entry.entry_id == entries[1].entry_id


def test_concurrent_workers_can_atomically_claim_only_one_fifo_head(tmp_path):
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    enqueue(store, make_intent("k1"), make_intent("k2", version="v2"))

    def claim(index):
        return store.claim_next(
            scope="vm-123", worker_id=f"worker-{index}", lease_seconds=60,
            boot_id="boot-abc", observed_free_bytes=5000, observed_free_inodes=500,
            capacity_evidence_ref=f"statfs:{index}", capacity_observed_at=NOW, now=NOW)

    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(claim, range(8)))
    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    assert winners[0].entry.sequence == 1


def test_waiting_cancel_races_claim_without_creating_cancelled_attempt(tmp_path):
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    entry = enqueue(store, make_intent("race"))[0]

    def claim():
        return store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                                boot_id="boot", observed_free_bytes=5000,
                                observed_free_inodes=500, capacity_evidence_ref="statfs",
                                capacity_observed_at=NOW, now=NOW)

    def cancel():
        try:
            return store.cancel_waiting(entry_id=entry.entry_id,
                                        expected_revision=entry.revision,
                                        actor="operator", now=NOW)
        except QueueConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        claim_future, cancel_future = pool.submit(claim), pool.submit(cancel)
        claimed, canceled = claim_future.result(), cancel_future.result()
    current = store.get_entry(entry.entry_id)
    if canceled:
        assert current.state == QueueState.CANCELED
        assert claimed is None
        assert store.list_attempts(entry.entry_id) == []
    else:
        assert claimed is not None
        assert current.state == QueueState.DISPATCHING
        assert len(store.list_attempts(entry.entry_id)) == 1


def test_rejects_unknown_persisted_schema_version(tmp_path):
    path = tmp_path / "queue.sqlite"
    store = SQLiteQueueStore(path)
    enqueue(store, make_intent("versioned"))
    with store._connect() as db:
        db.execute("UPDATE queue_scopes SET schema_version=999 WHERE scope='vm-123'")
    before = path.read_bytes()
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        tables_before = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    with pytest.raises(ValueError, match="unsupported queue schema"):
        SQLiteQueueStore(path)
    assert path.read_bytes() == before
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        assert db.execute("SELECT schema_version FROM queue_scopes").fetchone()[0] == 999
        assert db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall() == tables_before

    backup = tmp_path / "unknown-version-backup.sqlite"
    store.backup(str(backup))
    backup_before = backup.read_bytes()
    with pytest.raises(ValueError, match="unsupported queue schema"):
        SQLiteQueueStore(backup)
    assert backup.read_bytes() == backup_before
    with pytest.raises(ValueError, match="unsupported queue schema"):
        store.verify_restore(str(backup))


def test_pause_cas_capacity_freshness_and_durable_backup(tmp_path):
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    entry = enqueue(store, make_intent("k"))[0]
    paused_revision = store.set_policy(scope="vm-123", expected_policy_revision=1,
                                       state=QueuePolicyState.PAUSED, actor="operator", now=NOW)
    assert paused_revision == 2
    assert store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                            boot_id="boot", observed_free_bytes=5000, observed_free_inodes=500,
                            capacity_evidence_ref="statfs", capacity_observed_at=NOW, now=NOW) is None
    store.set_policy(scope="vm-123", expected_policy_revision=2,
                     state=QueuePolicyState.RUNNING, actor="operator", now=NOW)
    with pytest.raises(QueueAdmissionError, match="stale"):
        store.claim_next(scope="vm-123", worker_id="worker", lease_seconds=60,
                         boot_id="boot", observed_free_bytes=5000, observed_free_inodes=500,
                         capacity_evidence_ref="old-statfs", capacity_observed_at=NOW,
                         now=NOW + timedelta(hours=1))
    with pytest.raises(QueueConflict, match="revision"):
        store.cancel_waiting(entry_id=entry.entry_id, expected_revision=99,
                             actor="operator", now=NOW)
    backup = store.backup(str(tmp_path / "backup.sqlite"))
    restored = store.verify_restore(backup)
    assert restored.entries[0].entry_id == entry.entry_id
    assert restored.entries[0].state == QueueState.WAITING
    with pytest.raises(FileExistsError, match="new, non-symlink"):
        store.backup(backup)
    (tmp_path / "backup-link.sqlite").symlink_to(backup)
    with pytest.raises(FileExistsError, match="new, non-symlink"):
        store.backup(str(tmp_path / "backup-link.sqlite"))
    corrupted = tmp_path / "orphaned-backup.sqlite"
    shutil.copyfile(backup, corrupted)
    with sqlite3.connect(corrupted) as db:
        db.execute("DELETE FROM queue_entries WHERE entry_id=?", (entry.entry_id,))
        db.execute("DELETE FROM queue_events WHERE entry_id=?", (entry.entry_id,))
    with pytest.raises(ValueError, match="batch references a missing queue entry"):
        store.verify_restore(str(corrupted))


def test_uptime_exposure_is_durable_idempotent_and_pauses_after_recording_liability(tmp_path):
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    store.initialize(scope="vm-123", operator_config=operator_config(campaign_cost_cap_usd=10),
                     actor="service", now=NOW)
    later = NOW + timedelta(hours=20)
    total = store.account_until(scope="vm-123", expected_policy_revision=1,
                                observed_at=later, actor="meter", evidence_ref="billing-window-1")
    assert total == 20
    state = store.policy("vm-123")
    assert state["state"] == QueuePolicyState.PAUSED.value
    assert state["spent_usd"] == 20
    assert store.account_until(scope="vm-123", expected_policy_revision=state["revision"],
                               observed_at=later, actor="meter", evidence_ref="billing-window-1") == 20
