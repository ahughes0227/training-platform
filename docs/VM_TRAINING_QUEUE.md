# Compute Engine training queue operations

This guide describes the local single-VM queue interface and its current implementation boundary. The feature is not accepted for unattended production use: an operator must still supply verified VM/GPU, storage, cost, backup, and identity evidence, and the owning runtime/infrastructure roles must validate the actual Linux host. The reported 500 GB boot disk is provisioned capacity; it does not establish available bytes, inode headroom, throughput, or backup durability.

## Authority and operation

Run the CLI on the designated Compute Engine VM. It parses prepared YAML locally, validates the `QueueIntent` shape, and sends typed values over the configured Unix-domain socket. The privileged service never opens a path supplied by the caller. The service derives the operator or worker identity from Linux `SO_PEERCRED` and a protected UID allowlist. JSON fields cannot choose the actor. The socket is created with mode `0660` and the configured operator group; the worker UID and operator UIDs must be disjoint. No public listener is part of this interface.

The queue service and worker use the same protected configuration and durable store. The service owns admission and operator controls; the worker runs separately under the host supervisor and ticks at the explicitly configured interval. Training containers receive neither the queue database nor the container-engine socket. Normal queue execution creates a new container from the request's exact already-certified image digest; it does not build or push an image and does not submit a Vertex training job.

Use the command group after the CLI is installed on the VM:

```sh
defect train queue serve --config /etc/defect-platform/queue.yaml
defect train queue worker --config /etc/defect-platform/queue.yaml
defect train queue status
defect train queue list
defect train queue enqueue ./prepared-request.yaml
defect train queue enqueue ./prepared-batch/
defect train queue pause --reason 'operator maintenance'
defect train queue resume
defect train queue cancel ENTRY_ID
defect train queue cancel ENTRY_ID --active
defect train queue retry ENTRY_ID --revision 4 --classification transient --reason 'transient capacity issue resolved'
defect train queue skip ENTRY_ID --revision 4 --reason 'request withdrawn by operator'
sudo defect train queue resolve-interrupted ENTRY_ID --config /etc/defect-platform/queue.yaml --revision 4 --reason 'host process and GPU state independently inspected'
defect train queue backup /var/lib/defect-platform/backups/operator-check.sqlite
defect train queue restore-check /var/lib/defect-platform/backups/operator-check.sqlite
```

`enqueue` accepts one YAML request or a directory of YAML files; a YAML file may also contain a finite list. A submission accepts at most 100 requests. The client assigns a stable batch key from canonical request fingerprints. Every request has its own idempotency key. Repeating the same request is safe; reusing a key for changed content is an error. The service's controller additionally checks each runtime, dataset, VM profile, reviewed class catalog, semantic manifest, current readiness, storage reservation, and campaign authorization against protected operator configuration.

The `list` response returns queue entries in committed order plus attempts and the status view. Each entry includes its ID, run ID, sequence, state/revision, immutable intent fingerprint, budget and storage reservations, active attempt, blocker, and last observation timestamp. Each attempt carries its VM/boot, container/process, log/output, claim, and external dispatch identities. `status` reports the configured VM identity and current readiness window. Container IDs and blockers are reported from durable attempts/entries; no live Linux container query is implied by a status read. Readiness expires and must be refreshed by its evidence owner before delayed dispatch.

`resolve-interrupted` is deliberately a host-administrator command outside the Unix socket API. It requires root, a protected service config, the expected entry revision, and an operator reason. The controller must hold the worker/host lock and independently verify the exact VM/boot/GPU identity and absence of the prior container/process/workload before it changes a durable interrupted/unknown record. User-supplied text is audit context only; it is not recovery evidence. This command does not create a fake exit code or assert training success. Ordinary service operators cannot invoke it through the socket.

Waiting cancellation removes only a waiting entry. `cancel ENTRY_ID --active` records an intent against the exact active attempt; it does not report success until the worker observes termination. Retry requires the current revision and a reason, is limited by the profile's maximum attempts and remaining campaign authority, and preserves prior attempt history. Skip requires the current revision and an operator reason. A stale revision, unverified outcome, failed readiness gate, or conflicting active container returns an error and leaves the queue held for review.

## Prepared request shape

The protected config pins exact experiment records in `approved_experiments`
and binds each experiment to its reviewed cost/storage profile through
`experiment_profiles`. A requester cannot combine a large experiment with a
smaller approved profile or change the weights, dataset, or training settings
under an existing approval. Changed experiments require a new reviewed record.

This is a field guide, not a copy-and-run request. Replace every illustrative value with the exact approved immutable records and profile from the protected project catalogs. The service independently reloads those records and rejects substitutions. The profile's cost, runtime, capacity, and image fields are operator-approved values, not values the requester may use to grant authority.

```yaml
experiment:
  experiment_id: experiment-2026-09-29-a
  object_slug: panel
  dataset_version_id: panel-v17
  runtime_id: defect-trainer-2026-09
  catalog_sha256: <reviewed-catalog-sha256>
  model:
    backbone: <approved-backbone>
    weights_uri: gs://approved-bucket/weights/model.safetensors
    weights_sha256: <weights-sha256>
dataset:
  version_id: panel-v17
  object_slug: panel
  root_uri: gs://approved-bucket/datasets/panel/v17
  manifest_uri: gs://approved-bucket/datasets/panel/v17/manifest.json
  shard_uris:
    train: [gs://approved-bucket/datasets/panel/v17/train-000.tar]
  sample_counts: {train: 1200}
  sha256: <dataset-manifest-sha256>
  source_snapshot_uri: gs://approved-bucket/datasets/panel/v17/source.json
  semantic_manifest_uri: gs://approved-bucket/datasets/panel/v17/semantics.json
  semantic_sha256: <semantic-manifest-sha256>
runtime:
  runtime_id: defect-trainer-2026-09
  source_commit: <40-character-commit>
  image_tag: defect-trainer:approved
  image_digest: us-docker.pkg.dev/project/repository/trainer@sha256:<64-lowercase-hex>
  runtime_version: <approved-version>
  python_version: <validated-version>
  pytorch_version: <validated-version>
  cuda_version: <validated-version>
  validation:
    trainer: true
    container_gpu: true
    vertex_gpu: true
    gcs_read: true
    gcs_write: true
  certified: true
  certified_at: <timezone-aware-certification-time>
classes: [crack, dent]
job:
  profile_id: g2-approved-profile
  project_id: project
  zone: us-central1-a
  instance_id: <exact-compute-engine-instance-id>
  gpu_identity: <verified-gpu-identity>
  image_digest: us-docker.pkg.dev/project/repository/trainer@sha256:<same-digest-as-runtime>
  estimated_runtime_seconds: <measured-estimate>
  max_runtime_seconds: <approved-maximum>
  per_run_cost_usd: <approved-reservation>
  storage_peak_bytes: <measured-peak-plus-temporary-copies>
  storage_peak_inodes: <measured-peak>
  max_attempts: <explicit-bounded-attempt-count>
idempotency_key: panel-v17-experiment-2026-09-29-a
```

The exact schema is `QueueIntent` in `src/defect_platform/control/queue_contracts.py`; validation rejects unknown fields. Generate or validate a request using the repository's normal config/catalog tooling before enqueueing. Dataset versions and releases remain immutable. A new dataset or runtime identity gets its own request and key.

## Protected configuration and host setup

The service config is operator authority, not user input. The loader checks that the config is an absolute regular file with no symlink or group/world-writable path components and a trusted owner. Protect it and the credential file from the trainer, operator callers who should not change policy, and unrelated service users. Configure distinct UID sets for operators and the worker; assign the socket group only to the intended local operators. The full schema is `QueueServiceConfig` in `src/defect_platform/control/queue_admission.py`.

The config must state measured finite values for backlog size, campaign cost authorization and expiry, VM hourly cost, per-run maximum runtime/cost, disk bytes/inodes, reserved headroom, worker interval/lease, and backup policy. It also binds exact VM/project/zone/instance/boot/GPU identity, observed GPU/driver/toolkit/engine compatibility, approved immutable datasets/runtimes/profiles, output and MLflow endpoints, trainer UID/GID, container resource limits, isolated network, and credential path. Do not copy sample values from this document into production policy. Config readiness is evidence that must come from the owning infrastructure/runtime role and expires; setting a digest in YAML is not proof of compatibility.

Before starting the service or worker, the deployment owner must provision and validate the supported host supervisor, socket directory and group, protected storage path, Docker/compatible engine and GPU integration, trainer identity, mount paths, artifact credentials, and off-VM backup destination. This code does not install or configure those host services. The queue service and worker commands are intended to run as separate supervised processes so disconnecting SSH does not stop work while the VM remains up.

## Capacity, budget, and recovery

Retries require an authenticated operator's explicit recoverability classification and reason. Only `transient` permits retry of a confirmed failed attempt; `interrupted` permits retry after administrator-verified interruption. `deterministic` and `unknown` classifications are rejected. The classification is an operator judgment recorded with the original attempt, not a diagnosis inferred from an exit code. Attempts, backoff, duration, readiness, and remaining reserved budget remain enforced; unknown create/start outcomes cannot be retried. No automatic failure retry or checkpoint resume is implemented.

Measure the filesystem actually backing the reported 500 GB boot disk. Record usable capacity, free bytes, free inodes, current OS/container/cache use, mount identity, disk type, deletion behavior, and backup evidence. A 500 GB label does not mean 500 GB is available for training. The dispatch gate needs the next job's conservative peak dataset/weights staging, image pulls, scratch, checkpoint, output, and temporary-copy footprint plus the configured recovery/OS headroom. Unknown or insufficient footprint blocks launch. Because the queue, OS, container layers, and training share one boot filesystem, directory separation alone does not isolate capacity.

Campaign authorization covers cumulative queued-run reservations and VM uptime/idle exposure as configured. Queue admission and delayed dispatch both recheck the current authority. Budget exhaustion pauses/blocks future launches; it does not power off the VM or stop a running container. The VM may continue to incur charges while idle or blocked, and a stopped VM cannot run its queue worker.

The SQLite queue database and evidence live under the protected `storage_path`. `backup DESTINATION` is limited by the controller to its protected local backups directory and makes an application-consistent backup. The response indicates that an off-VM copy is still required; the local backup command does not itself prove remote replication. `restore-check SOURCE` validates a protected backup without replacing live state. Follow the approved off-VM backup procedure and retain its external evidence. Restore with in-flight attempts requires operator reconciliation against actual VM/container/process evidence before dispatch resumes.

On worker, container-engine, or VM interruption, the queue retains claims, attempts, and prior observations. An expired worker lease is not proof that a container stopped and does not authorize replay. The worker first reconciles the exact persisted container identity. If the outcome is unknown, it holds the queue and preserves reservations until the host administrator uses `resolve-interrupted` after the controller's independent absence/readiness checks. A reboot does not resume a training process or guarantee checkpoint recovery. Resume or retry only after checking VM/boot identity, container/process state, outputs, MLflow finalization, readiness expiry, and campaign budget. Never manually delete the database, container, logs, or attempt evidence to make the queue advance.

## Verification status and open acceptance

The local suite covers typed request loading, idempotency input shape, API role separation, forged actor rejection, bounded socket protocol handling, controller-factory command wiring, and interruptible worker polling. Socket protocol tests use socket pairs and an injected peer resolver. This macOS sandbox cannot verify Linux `SO_PEERCRED`, actual Unix socket ACL/group access, protected service startup, or real host identity. No live VM, GPU, certified image execution, paid job, cloud backup, billing, reboot recovery, or container ACL acceptance is claimed here.

All queue requirements in the planning proposal remain pending until their corresponding owner supplies evidence. In particular, verify finite queue admission under concurrent input; append requests while a real training container is active without changing it; fresh FIFO container IDs; actual GPU/driver/toolkit compatibility; artifact/trainer isolation; disk and inode pressure; cost accounting; offline backup and recovery; failed/unknown outcome handling; and closed-SSH continuation. The root integration owner must map evidence to the canonical requirements register and acceptance suite. Local green tests establish only the tested local paths, not end-to-end unattended readiness.
