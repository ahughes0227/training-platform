# Unattended Training Queue — Feature Plan, Requirements, and Risks

**Date:** September 29, 2026
**Status:** Revision 3 design adopted for local implementation by the user's September 29 authorization to GPT-6 Luna agents. Compute Engine execution uses a fresh container per job and a user-reported 500 GB boot disk. Local implementation and review are recorded in [the implementation evidence](TRAINING_QUEUE_IMPLEMENTATION.md); live acceptance and mitigation scores remain pending.
**Inspected source revision:** `024c6af5e9f4131b06297da5fbc7281fbd5dace7`
**Implementation owner:** Experiment Runner. Shared interfaces and deployment policy require primary integration review and explicit owner handoffs.

## 1. Intended outcome and decisions

The user prepares training requests, adds them to a durable queue on a **GCP Compute Engine VM**, and leaves. New requests can be appended while the current training container is running. The platform starts a **fresh container for the next job after the previous training finishes**, without restarting or modifying the current container and without keeping an SSH, CLI, or assistant session open.

**Confirmed:** Compute Engine is the training execution host; each job uses a fresh container; sequencing waits for the previous training to finish; the VM has a user-reported **500 GB boot disk**. Disk type, actual filesystem capacity/free bytes/inodes, existing usage, deletion settings, and backup configuration have not been inspected. FIFO and one active job on that VM are proposed policies. This revision replaces the first draft's Vertex job execution architecture. Vertex certification remains an existing repository requirement; it is a separate runtime-release gate, not the executor for this feature.

| Decision | Policy | Status |
| --- | --- | --- |
| Training backend | Existing authorized GCP Compute Engine VM; no Vertex training job submission in this execution path | Confirmed |
| Container lifecycle | A new container instance per training job from the request's exact certified immutable image digest | Confirmed |
| Timing | Start the next training only after the previous training finishes | Confirmed |
| Add while busy | Accept and durably append prepared requests without stopping, restarting, rebuilding, or modifying the active training container | Required consequence of the requested usage |
| Sequencing/resource scope | FIFO by committed sequence; one active training workload on the designated VM/GPU pool across object projects | Proposed; multi-VM/parallel scheduling requires a separate scope decision |
| Worker and storage | A supervised host queue worker, separate from training containers; queue state in a protected host directory on the existing 500 GB boot disk, outside container writable layers | Existing nominal capacity confirmed; storage layout and durability controls proposed |
| Submission interface | CLI on the VM, including through SSH, calls a local queue service with authenticated OS access; no public network API required | Proposed; remote submission is a separate interface decision |
| Existing running container | Treat existing GPU work as occupying the slot. Adopt it only with verified run/container identity and completion observation; otherwise hold and ask the operator to bind or resolve it | Proposed; never kill or replace it automatically |
| Immediate submission | Use the same VM resource lock; while busy, direct the user to enqueue rather than bypassing the queue | Proposed behavior change requiring integration review |
| Confirmed training failure | Pause and show the failed run until an authorized operator resumes or skips it | Recommended; failure-policy choice remains open |
| Unknown outcome | Hold until container/process reconciliation proves prior work stopped and resolves outstanding dispatch intent | Required safety constraint |
| Invalid or expired waiting request | Block and explain the failed gate; no silent dataset/image/configuration substitution or skipping | Proposed |
| Cancellation | Remove waiting work atomically; active cancellation targets the exact container/process group and waits for observed termination | Proposed |
| Spending | Per-run allocation plus explicit cumulative VM/queue authorization, including idle uptime, retries, storage, and other applicable costs | Proposed; numeric authorization must be supplied |
| VM power state | Execution requires the VM to be running. Queue state survives a tested restart; interrupted training is held for recovery. Automatic power-on, replacement, or idle shutdown needs an explicitly approved policy | Proposed; no promise of progress while the VM is off |
| Runtime approval | Preserve GPU-host validation and Vertex certification required by AGENTS.md; additionally verify this VM's GPU/driver/container runtime compatibility for the exact digest | Existing constraint plus proposed VM-specific gate |
| Promotion | Completion produces results/candidates; release promotion requires its separate approval | Existing platform constraint |

A fresh container is a new instance of an existing image, not a new image build. Each request pins its image independently; all entries may use one digest, but no entry is silently changed to whichever digest is cached or latest.

## 2. Current foundation and dependencies

At the inspected baseline, the repository provided SQLite/PostgreSQL run storage, request fingerprints, API/CLI submission, reviewed dataset semantics, image digest references, MLflow tracking, and a Vertex/Cloud Workflows execution path. The subsequent local VM queue implementation is documented in [the implementation evidence](TRAINING_QUEUE_IMPLEMENTATION.md); it has not been deployed or accepted on the designated VM.

- [The current controller](../src/defect_platform/control/controller.py) and [Vertex job configuration](../src/defect_platform/contracts.py) bind execution to Vertex. Add a typed Compute Engine/container job configuration and execution adapter; do not disguise VM work as a Vertex resource name.
- [The trainer entrypoint](../src/defect_platform/trainer/runner.py) reads a serialized request and executes training. Inspect and validate its existing request/output behavior in the intended VM container before deciding whether any trainer change is needed. Trainer modifications require an evidence-backed handoff to the Trainer Engineer.
- [InfrastructureCapabilities](../src/defect_platform/infrastructure_contract.py) currently validates project/region/service-account and artifact roots, but does not record VM instance ID/zone, boot identity, host driver, container engine/GPU toolkit, disk readiness, or existing workloads. Its VM extension requires primary integration review and operator evidence.
- Existing budget, authority, concurrent dispatch, cancellation, and orphan recovery gaps in [TP-078–TP-094](REQUIREMENTS.md#tp-078) apply to the VM backend as well. Calling a local container engine instead of a cloud job API does not remove uncertain-outcome races.
- [TP-160](REQUIREMENTS.md#tp-160) still requires observed readiness at delayed dispatch. A changed VM, driver, reboot, revoked digest, missing mount, or unavailable artifact service can invalidate a waiting request.
- Vertex-specific workflow bugs noted in the first draft remain separate existing issues. Fixing that backend is not a prerequisite for this VM queue unless it is explicitly retained as an additional supported executor.

**Deployment facts checked against official documentation:** Google's legacy container startup agent is deprecated; use a reviewed supported VM startup/supervision mechanism rather than building on `create-with-container`/konlet. See [Compute Engine startup-agent deprecation](https://docs.cloud.google.com/compute/docs/deprecations/container-startup-agent-on-compute). VM authorization depends on both the attached service account's IAM roles and access scopes; containers must not inherit unintended control authority. See [Compute Engine service accounts](https://docs.cloud.google.com/compute/docs/access/service-accounts). Host GPU driver installation and compatibility are a distinct readiness concern. See [Compute Engine GPU drivers](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu).

A stopped VM cannot execute its host worker. Queue persistence and training checkpoint recovery are different guarantees. VM stop/restart does not preserve a running training process; attached resources can continue to incur charges, and restart depends on available capacity. See [VM stop/restart behavior](https://docs.cloud.google.com/compute/docs/instances/stop-start-instance). Docker daemon access is a powerful host control capability, so the trainer must not receive that socket. See [Docker Engine security](https://docs.docker.com/engine/security/).

These are source/documentation observations, not live readiness or acceptance results for the user's VM. Earlier tests of temporary, removed code do not establish this feature's acceptance.

## 3. Proposed architecture and user operation

### Components

| Component | Responsibility and boundary |
| --- | --- |
| VM submission CLI/local service | Accept authorized prepared requests and expose queue status; usable while training is active |
| Durable queue/run store | Persist order, approved intent, claims, transitions, reservations, and evidence outside training containers |
| Host queue worker | A reviewed supervised service, for example systemd; consumes requests, controls existing-image container launches, observes completion, and periodically reconciles state |
| Container execution adapter | Create/start/inspect/stop exact container instances; bind deterministic run/attempt identity and labels to dispatch intent; never build/push or install dependencies |
| Fresh training container | Execute one approved request with assigned GPU, immutable/read-only inputs, per-run writable output/temp directories, and only required data/artifact credentials |
| GPU/container host configuration | Runtime/infrastructure-owned driver, toolkit, engine, mounts, resource limits, and startup configuration |
| GCS/MLflow and evidence stores | Preserve immutable dataset/runtime lineage, results, tracking/finalization outcomes, and recovery evidence |

For the declared **single VM**, a local transactional SQLite store in a protected host directory on the existing 500 GB boot disk is the default proposal, with tested commit durability, concurrency, integrity checks, and application-consistent backups. The worker and CLI access it through one authority boundary; users and trainers do not directly mutate its records. If the user later needs remote enqueue while the VM is off, multiple workers/VMs, or a separately hosted control service, revisit shared database storage and dispatch ownership before expanding the scope.

### Storage plan for the 500 GB boot disk

The plan uses the existing boot disk; it does not require purchasing or attaching another disk. The stated 500 GB is nominal provisioned capacity, **not 500 GB of available training space**. Verify actual filesystem capacity, units, free bytes/inodes, disk type, existing OS/container usage, and mount identity before accepting workload limits. Size alone does not establish storage throughput or sufficient dataset/checkpoint capacity.

| Storage area | Proposed treatment |
| --- | --- |
| Queue database, journal, configuration, and audit indexes | Protected host directory, such as `/var/lib/defect-platform/`; writable by the queue authority, inaccessible to trainers, retained when job containers are removed |
| OS, container engine, and image layers | Measure current/peak usage; reserve host headroom and limit image/cache growth; do not delete images used by active/approved jobs |
| Dataset and model-weight cache | Bound by a configured limit; validate immutable identities; reuse verified content where supported; do not stage every queued dataset automatically |
| Per-attempt outputs, checkpoints, scratch, and upload staging | Isolated paths with a conservative peak estimate including temporary copies; retain until verified finalization and the approved evidence policy allow cleanup |
| Logs and finished-container evidence | Bounded rotation/retention; export necessary evidence before eligible cleanup; no blind global container/disk pruning |
| Recovery copy | Application-consistent queue backup outside this VM/boot disk, with an approved destination, retention, access, restore procedure, and cost authorization |

**Dispatch capacity gate:** Observed free space minus outstanding storage reservations must cover the next job's conservative peak local footprint **plus reserved OS/queue recovery headroom**. Check free inodes as well. Include dataset/weight staging, image pulls, checkpoints, temporary copies, log growth, and uploads. Unknown job footprint or insufficient headroom blocks launch with an actionable reason. Recheck immediately before launch and monitor growth during training; a one-time check cannot prevent concurrent OS/container writes from consuming space.

Enqueue stores request metadata rather than requiring every dataset to be downloaded. Serial training can bound staging to the active job and an explicitly authorized bounded prefetch policy. Numeric headroom/cache/log limits remain decisions to set from measured usage and representative job sizes; this plan does not invent a fixed partition of the 500 GB.

Directories on one filesystem are organizational boundaries, not capacity isolation. Use supported quotas/reservations where suitable, and fail closed when growth cannot be bounded safely. Filling the boot filesystem can impair both training and the queue/OS, so the earlier QR-18 likelihood and residual target are revised below.

VM deletion can delete an attached boot disk according to its deletion rule; confirm and review preservation settings rather than assuming queue survival. Separate backups are required even when disk retention is enabled. No deletion-rule, disk-resize, VM, or backup operation is performed by this planning update. See [Compute Engine disk deletion settings](https://docs.cloud.google.com/compute/docs/disks/modify-persistent-disk) and [VM deletion behavior](https://docs.cloud.google.com/compute/docs/instances/deleting-instance).

The worker runs outside each training container and restarts independently of SSH. It starts on VM boot only after queue storage, container engine, GPU dependencies, and readiness checks are available. It combines completion observation with periodic reconciliation while the VM is up. No Cloud Run service or managed Cloud Workflow is required by this proposed single-VM execution design.

### Durable progression

1. **Prepare:** Reference already published immutable datasets, an already certified image digest, explicit experiment/VM job settings, and spending authorization. Image pulling or starting an already approved digest is distinct from building or certifying an image.
2. **Enqueue while busy:** Validate and atomically commit new entries and authorized budget reservations. Return stable IDs and positions without taking the active training job's lifecycle lock or changing its files/configuration. Single additions are independent transactions; a submitted finite batch has defined all-or-nothing admission.
3. **Schedule:** Verify the designated VM/GPU pool has no unresolved active work. Atomically claim the FIFO head and persist a dispatch intent protected by revisions and worker ownership.
4. **Launch a fresh container:** Recheck authorities, spending, disk/GPU/runtime readiness, and exact digest. Persist run/attempt/container identity before create/start; use unique per-attempt scratch/output paths and read-only approved inputs. Inspect existing matching containers after an uncertain create/start response before any retry.
5. **Observe:** Record engine state, process lifetime, exit code, logs, and output/finalization results. A worker exception, lost Docker connection, missed completion event, or missing container is not itself proof of successful or safely stopped training.
6. **Advance:** Release the slot after termination and GPU/resource ownership are reconciled. Apply the approved success/failure policy and launch the next fresh container. Training success requires coherent output/finalization evidence; no automatic promotion follows.
7. **Recover:** After worker/engine restart or VM reboot, reconcile queue intent against VM numeric instance ID, boot identity, container labels/state, processes, and persisted outputs. Hold interrupted/unknown work until the approved retry or checkpoint-resume policy authorizes a new attempt. An expired claim or absent container alone never authorizes duplicate execution.
8. **Retain and clean:** Preserve required exit/log/result evidence before removing finished containers. Bound scratch, logs, caches, and orphan retention; cleanup cannot delete queue state, immutable datasets, certified images still in use, or unfinalized outputs.

Exactly one active training workload is the proposed invariant on this VM. Direct/manual GPU jobs either obey the same admission boundary or are explicitly discovered and treated as blockers. A deterministic container name is a useful reconciliation key, not a proof that all side effects execute exactly once.

**VM-off behavior:** The local service cannot accept requests or execute training while the VM is stopped. Durable requests remain stored subject to the tested disk/backup policy. Processing resumes after the VM is started and recovery gates pass. Automatic VM power management and cross-VM failover are separate capabilities requiring explicit authorization.

### Queue records and states

Keep queue scheduling facts separate from trainer/artifact facts. Proposed versioned records include queue ID and VM/GPU scope, FIFO sequence, entry/run/attempt IDs, request fingerprint, exact dataset/image/semantic references, submitter and authorization revisions, budget reservation, dispatch intent/state revision, worker identity, GCP project/zone/numeric instance ID, boot identity, container ID/name/labels, observed process/exit state, log/output references, transition history, and last reconciliation time/result.

Queue states are **waiting, dispatching, active, blocked, canceling, completed, and canceled**. Queue policy records **running or paused**; interrupted/unknown training outcomes remain explicit. A reboot does not turn an interrupted run into success, and a new attempt does not overwrite an earlier attempt's evidence.

### Proposed user actions

The commands below describe the planned surface. The current implemented syntax and recovery procedures are maintained in [the operations guide](VM_TRAINING_QUEUE.md):

| Action | Proposed interface/result |
| --- | --- |
| Append while training runs | `defect train queue add REQUEST.yaml` or a prepared project directory; returns entry/run ID and position without affecting the running container |
| Add a finite batch | `defect train queue add-batch BATCH.yaml`; returns ordered IDs, estimate, and budget authorization |
| Inspect | Queue list/status show order, active dataset/run/container, VM readiness, blocked reason, costs, and last observation |
| Pause/resume | Stop new dispatch; revalidate before resuming. Active training continues unless explicitly canceled |
| Remove waiting work | Atomically cancel one queued entry without creating or stopping a training container |
| Cancel active work | Stop the exact owned container/process group; show canceling until termination is reconciled |
| Resolve failed/interrupted work | Explicit retry, skip, refreshed approval, or supported checkpoint resume with retained attempt history |
| Inspect results | Navigate run/config/dataset/digest, VM/container identity, logs, outputs, and MLflow |

Closing SSH after an acknowledged enqueue does not stop the worker. Queue messages explain the next required action. Notifications, if later requested/configured, cannot control scheduling.

## 4. Proposed new requirements

The IDs below are proposal IDs retained for traceability. They map in order to canonical **TP-161–TP-195**, now added to the [requirements register](REQUIREMENTS.md#tp-161). Adoption of the design and local implementation does not close their acceptance gates.

Every negative test must inspect independent before/after state for zero forbidden paid jobs or state changes. An exception alone is insufficient.

| ID | Proposed requirement | Required positive and negative acceptance | Owner |
| --- | --- | --- | --- |
| TQ-001 | The platform shall accept a finite ordered batch of prepared training requests, each binding an immutable dataset version and certified runtime digest | Admit multiple real versions/objects; reject unresolved datasets, tags, contradictory object/catalog references, and unavailable certification without paid effects | Experiment Runner |
| TQ-002 | Acknowledgment shall follow atomic persistence of queue intent, stable run IDs, admission evidence, and authorized budget reservations | Disconnect/restart immediately after commit and recover all accepted entries; interrupt before commit and observe no partial acknowledged batch | Experiment Runner |
| TQ-003 | Entries shall receive a durable FIFO sequence within an explicit VM/GPU resource scope | Concurrent admissions produce a stable unique order; timestamps, object filters, restart, and duplicate delivery cannot reorder it | Experiment Runner |
| TQ-004 | The scope shall permit at most one active training workload across all supported submission routes, including immediate submissions | Queue behind existing active work; race queue and immediate requests and independently verify no overlap or hidden bypass | Experiment Runner; primary integration review |
| TQ-005 | An entry shall not launch until all earlier applicable work is resolved and owned container/process termination and GPU slot release are reconciled | Advance after confirmed termination; running, queued, cancelling, timeout, missing observations, and monitor errors cannot release the slot | Experiment Runner |
| TQ-006 | Automatic progress shall be independent of SSH/CLI/agent sessions and shall recover safely from worker/engine restart while the VM is up | Complete a batch after closing clients; lose completion observations and restart the worker/engine; verify safe reconciliation within the accepted bound | Experiment Runner; infrastructure handoff |
| TQ-007 | Scheduling shall use atomic claims, protected revisions, and dispatch intents; stale workers shall not mutate accepted state or create competing workloads | Race workers and replay delayed events; fail stale claims before forbidden effects and preserve valid progress | Experiment Runner |
| TQ-008 | Repeated enqueue and dispatch attempts for the same effective intent shall resolve to the same entry/run/attempt/container, or an explicit held uncertain outcome | Retry identical requests through faults; changed intent under a reused key and uncertain external creation cannot create a second paid job | Experiment Runner |
| TQ-009 | Recovery shall reconcile queue, VM/boot, container, and process identities before replaying a side effect | Crash before/after each commit and external create; recover one accounted job or hold with evidence, never automatically launch again on lease expiry | Experiment Runner |
| TQ-010 | Enqueue and delayed dispatch shall independently validate authoritative approval, exact identities, integrity, budget, and current infrastructure capability evidence | Launch eligible stored requests; expire/revoke/mutate prerequisites while waiting and verify blocked state with zero new resources | Experiment Runner; authority owners provide evidence |
| TQ-011 | Queued training shall use trusted cost estimates, per-run limits, and an explicitly authorized cumulative queue/campaign cap covering retries, attributable running/idle VM time, and applicable storage/retained costs | Account for the finite batch and actual/reserved exposure; individually affordable jobs, concurrent requests, and retries cannot exceed the aggregate authorization | Experiment Runner; pricing/policy integration review |
| TQ-012 | Continuation on success, failure, cancellation, and blocked prerequisites shall follow a versioned visible policy | Exercise each approved policy; monitoring uncertainty and configuration changes cannot silently continue or skip the head | Experiment Runner |
| TQ-013 | Retry shall be bounded by classified recoverability, attempt count, backoff, duration, and remaining spending authority | Recover an eligible transient fault; deterministic failure, exhausted authorization, and unknown creation cannot loop or rebuild a runtime | Experiment Runner |
| TQ-014 | Queue pause shall prevent new dispatch and resume shall revalidate the next entry; the pause/claim race shall have a documented atomic outcome | Pause before a claim prevents launch; if the claim committed first, show active work and require explicit cancellation rather than claiming it stopped | Experiment Runner |
| TQ-015 | Cancellation of waiting entries shall be atomic and shall preserve the order and validity of remaining entries | Remove middle/head entries idempotently; race removal against claim and verify no canceled waiting entry creates resources | Experiment Runner |
| TQ-016 | Active cancellation and deadlines shall be reconciled against owned container/process and VM state before slot release | Request cancellation and observe terminal state; refused, delayed, or ambiguous cancellation cannot be displayed as stopped or permit overlap | Experiment Runner |
| TQ-017 | Scheduling, completion observations, inspection, pause/resume, retry, skip, and cancellation shall be independently authorized for the declared resource scope | Exercise permitted principals; submitter credentials cannot forge internal completion or control another unauthorized scope | Experiment Runner; IAM deployment owner |
| TQ-018 | The UI/CLI shall expose durable order, run identity, current activity, blocking reason, budget status, last observation, and required user action | Inspect a mixed queue without source-code knowledge; stale data, canceled work, and an unavailable observer cannot appear as current success or an empty healthy queue | Experiment Runner |
| TQ-019 | Every queue transition and recovery decision shall preserve actor, time, revisions, configuration identity, runtime digest, VM/boot/container/process identity, and log/evidence references | Trace a result and each failed action to its exact request; out-of-order completion observations or redaction cannot erase necessary lineage | Experiment Runner |
| TQ-020 | Queue limits and liveness objectives shall be configured and verified, including maximum backlog, start delay after eligibility, stalled-work threshold, and recovery bounds | Test the accepted limits under load; full queues, resource shortages, or worker/engine faults cannot create unbounded polling/retries or silently abandon entries | Experiment Runner; infrastructure capacity owner |
| TQ-021 | Queue persistence shall support reviewed schema migration, backup/restore, and safe mixed-version rollout | Restore accepted history and replay old/new supported records; older controllers cannot bypass reservations or lose queued intent | Experiment Runner; deployment integration review |
| TQ-022 | Ordinary queued experiments shall have zero image build/push, dependency-install, certification, dataset overwrite, or release-promotion effects | Observe permitted training end to end; instrument forbidden APIs/identities and attempt alternate paths to prove denial | Experiment Runner; primary integration review |
| TQ-023 | Training completion, output verification, MLflow finalization, and release eligibility shall remain explicit separate facts | Bind coherent outputs to the exact run and show finalization status; a zero container exit code, partial outputs, or registration errors cannot falsely report an accepted model or trigger promotion | Experiment Runner; output/registry owner handoffs |
| TQ-024 | Local and deployed evidence shall cover the complete queue behavior and fault model for the released revision | Run ordered multi-dataset work with disconnected clients and record source/image/configuration evidence; mocks or historical probes cannot satisfy live VM/GPU durability, permissions, or billing acceptance | Primary integration reviewer and owning roles |
| TQ-025 | Requests shall be accepted and appended while the designated VM is already training, without changing the active container or its request | Add requests during actual GPU training and concurrent submissions; verify original container ID, request, process, and outputs remain intact and new entries later execute in order | Experiment Runner |
| TQ-026 | Every admitted attempt shall run in a fresh container instance from its exact approved certified image digest, with isolated writable paths | Execute consecutive datasets in distinct container IDs; verify digest and per-attempt directories; mutable tags, reused containers, cross-run scratch, or runtime builds cannot satisfy admission | Experiment Runner; Runtime Engineer verifies packaging |
| TQ-027 | VM dispatch shall validate observed project/zone/instance/GPU/driver/toolkit/engine/disk readiness for the exact runtime digest in addition to existing certification | Observe the approved VM running the exact GPU image; changed drivers, incompatible GPUs, wrong instance, missing mounts, or failed GPU checks block launch without rebuilding | Runtime/infrastructure owners publish evidence; control enforces |
| TQ-028 | Queue intent shall reside outside trainer containers in protected host storage on the verified boot disk; a host supervisor shall start/recover the worker independently of SSH | Remove a finished trainer container, disconnect SSH, restart the worker, and reboot the VM; committed queue history remains readable and unresolved work is safely reconciled | Experiment Runner; runtime/deployment handoff |
| TQ-029 | VM stop/reboot, engine state loss, and instance replacement shall preserve explicit interrupted/unknown outcomes and shall not imply job completion or authorize replay | Interrupt actual training and reboot; restore the store and inspect VM/boot/container identities; no partial result is accepted and no duplicate workload starts without resolved authority | Experiment Runner; infrastructure owner |
| TQ-030 | Interrupted-job recovery shall declare whether training restarts, uses a verified supported checkpoint, or waits for operator action; every retry/resume has an attempt record and budget | Exercise the adopted supported path; checkpoint mismatch, unsupported resume, or lost evidence cannot silently skip data, overwrite prior results, or promise exact continuation | Experiment Runner; Trainer Engineer handoff if checkpoint support changes |
| TQ-031 | Running/idle VM exposure and queue-empty or blocked-queue behavior shall follow an explicit uptime/spending policy; power changes require separate authorization | Exhaust the authorized exposure and empty/block the queue; apply the approved stop/notify/hold behavior; no unauthorized shutdown or new VM creation occurs | Control/policy integration reviewer; infrastructure power-management owner |
| TQ-032 | Trainer containers shall not mutate queue state, access the engine socket, launch sibling workloads, or inherit unauthorized host/cloud control credentials | Execute permitted data reads/result writes; attempts to alter queue DB, invoke engine control, or use metadata credentials for forbidden operations are denied with independent effect checks | Runtime/deployment owner; primary security/integration review |
| TQ-033 | VM storage/GPU resource admission and cleanup shall protect queue commits and per-attempt evidence while bounding caches, logs, scratch, and orphan containers | Simulate disk/inode exhaustion, engine/GPU unavailability, and orphan processes; reject unsafe launch, preserve accepted queue state, and clean only verified eligible resources | Experiment Runner; Runtime Engineer/infrastructure handoff |
| TQ-034 | Storage admission on the user-reported 500 GB boot disk shall use measured usable/free capacity and inodes, conservative per-attempt peak footprint, outstanding reservations, and protected OS/queue headroom | Measure existing host/image use and run representative large jobs; unknown sizes, concurrent reservations, oversized staging, and log/cache growth cannot silently exhaust the host or corrupt accepted queue intent | Experiment Runner; runtime/infrastructure owner verifies filesystem limits |
| TQ-035 | Boot-disk deletion/retention policy and an independently stored application-consistent queue backup shall be reviewed and restore-tested before unattended acceptance | Restore backed-up queue/attempt records to an approved host and reconcile prior jobs; VM deletion, boot-disk loss, or deleting a finished container cannot be assumed to preserve the only authoritative queue copy | Infrastructure/backup owner; primary integration review |

### Existing requirements that need extension

| Existing requirement | Queue-specific extension |
| --- | --- |
| TP-023 | Define typed VM execution configuration, queue schema version, precedence, continuation-policy revision, and migration rules |
| TP-071–TP-072, TP-093 | Re-resolve authoritative runtime usability at delayed dispatch while preserving the exact approved digest |
| TP-078–TP-079 | Add per-operation queue authorization and repeat admission gates before delayed launch |
| TP-080–TP-082 | Include finite queued exposure and retries in trusted aggregate authorization and reservation accounting |
| TP-083–TP-086 | Atomically bind run/queue intent, FIFO sequence, state revision, resource claim, and idempotency |
| TP-087–TP-090, TP-094 | Add supervised VM queue wakeups, missed-observation reconciliation, VM/container interruption, explicit uncertainty, cancellation, and operator recovery; revise Workflows-only execution wording through canonical review |
| TP-092 | Define the boundary between finished compute and coherent verified training/finalization results |
| TP-122–TP-123 | Preserve queue failure ownership and uncertainty without converting them into proven external termination |
| TP-130 | Explicitly exclude runtime creation/certification from queue operation |
| TP-142 | Include queue reservations, policies, and dispatch history in recovery/restore acceptance |
| TP-160 | Specify VM/GPU/disk/boot capability evidence, long-wait refresh, and reapproval rules; stale evidence cannot authorize launch |

## 5. Risk scores and scores after mitigation

Use the same normalized method as [the existing assessment](MODULE_CHANGE_RISKS.md): **R = likelihood (L) × impact (I)**. Ratings are subjective prioritization estimates from 0 to 1, not calibrated probabilities or predicted financial losses. Larger values mean higher risk.

The **after-mitigation score is a conditional target**: target likelihood × the same impact, rounded half up to two decimals. It applies only after the linked controls are implemented, reviewed, and verified at the required local/deployed scope. Actual residual scores are unassessed; no risk reduction is claimed by this planning task. Close numerical differences are not decision-quality precision.

| Risk ID | Hazard | L before | I | R before | L after | Target R after |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| QR-01 | Retry or crash creates duplicate paid jobs | 0.70 | 0.90 | 0.63 | 0.20 | 0.18 |
| QR-02 | Queue and immediate submissions overlap GPU work | 0.70 | 0.90 | 0.63 | 0.20 | 0.18 |
| QR-03 | Queue stalls while the user is absent | 0.80 | 0.70 | 0.56 | 0.25 | 0.18 |
| QR-04 | Timeout/monitor failure falsely releases an active slot | 0.75 | 0.95 | 0.71 | 0.20 | 0.19 |
| QR-05 | Waiting request launches with stale approval/infrastructure | 0.80 | 0.90 | 0.72 | 0.25 | 0.23 |
| QR-06 | Wrong dataset, config, or image is trained; lineage is misassigned | 0.60 | 0.95 | 0.57 | 0.15 | 0.14 |
| QR-07 | Aggregate batch/retry spending exceeds authorization | 0.70 | 1.00 | 0.70 | 0.25 | 0.25 |
| QR-08 | Pause/cancel race launches unwanted work or leaves resources running | 0.70 | 0.90 | 0.63 | 0.20 | 0.18 |
| QR-09 | Forged events or unauthorized queue controls trigger paid work | 0.65 | 1.00 | 0.65 | 0.25 | 0.25 |
| QR-10 | Migration/restore or mixed controllers lose order/reservations | 0.65 | 0.85 | 0.55 | 0.25 | 0.21 |
| QR-11 | Blocked head, unavailable capacity, or unbounded backlog causes starvation | 0.55 | 0.60 | 0.33 | 0.20 | 0.12 |
| QR-12 | VM driver/toolkit/engine, mounts, zone/identity, or worker startup configuration prevents progress | 0.80 | 0.80 | 0.64 | 0.25 | 0.20 |
| QR-13 | Failure policy/retry cascade repeats invalid work unattended | 0.70 | 0.90 | 0.63 | 0.20 | 0.18 |
| QR-14 | Poor status/evidence hides stalled, unknown, or incorrectly attributed work | 0.70 | 0.70 | 0.49 | 0.20 | 0.14 |
| QR-15 | Queue recovery crosses roles or changes immutable artifacts | 0.60 | 0.95 | 0.57 | 0.20 | 0.19 |
| QR-16 | Compute success is confused with valid outputs/finalization or model approval | 0.65 | 0.75 | 0.49 | 0.20 | 0.15 |
| QR-17 | VM stop, reboot, maintenance, or preemption interrupts training | 0.70 | 0.90 | 0.63 | 0.25 | 0.23 |
| QR-18 | Shared boot-disk exhaustion/corruption loses queue evidence or impairs OS/worker recovery | 0.75 | 0.95 | 0.71 | 0.25 | 0.24 |
| QR-19 | Orphan GPU processes, resource leaks, or shared scratch contaminate/block the next job | 0.75 | 0.90 | 0.68 | 0.25 | 0.23 |
| QR-20 | Idle or blocked GPU VM keeps consuming the authorized budget | 0.80 | 1.00 | 0.80 | 0.25 | 0.25 |
| QR-21 | Trainer inherits host/engine/metadata permissions and bypasses queue authority | 0.65 | 1.00 | 0.65 | 0.20 | 0.20 |
| QR-22 | VM deletion or boot-disk loss removes the sole queue/history copy | 0.60 | 0.95 | 0.57 | 0.20 | 0.19 |

### Mitigations, proof, and remaining exposure

| Risk | Required mitigation | Verification needed before reassessment | Remaining exposure; requirement links |
| --- | --- | --- | --- |
| QR-01 | Persist immutable create/start intent; claim atomically; fence stale workers; inspect matching container labels/IDs before replay | Concurrent enqueue/dispatch; kill the worker before/after engine create/start and DB commits; count containers/processes and reservations independently | Engine and state DB do not share a transaction; ambiguity may require a hold. TQ-007–009, 029 |
| QR-02 | One VM/GPU reservation across direct/queued paths; observe existing GPU work; prohibit unaccounted launch capabilities | Race manual/immediate/queued launch and stale observations; verify GPU workload/process overlap, including a pre-existing container | Host administrators can bypass the worker; unmanaged activity holds the queue. TQ-003–005, 017, 025 |
| QR-03 | Host-supervised worker, boot ordering, completion observation plus periodic reconciliation, durable last-progress state | Disconnect SSH; drop an event; restart worker/engine and reboot VM; eligible work advances or enters explicit interrupted/blocked state within agreed bounds | The worker cannot run while the VM is off or storage is unavailable. TQ-006, 009, 020, 028–029 |
| QR-04 | Distinguish observed container/process termination from worker/engine errors; reconcile GPU ownership before release | Lose engine connection, timeout, omit container evidence, and delay stop; slot remains held until a genuine reconciled terminal outcome | Engine failure or incomplete observation intentionally reduces availability. TQ-005, 009, 016, 029 |
| QR-05 | Recheck authority, revocation, integrity, and observed capabilities at dispatch; require reapproval for material changes | Expire capabilities, revoke a runtime/catalog, or change identity after enqueue; independently verify zero paid launch | Long waits can block otherwise valid batches until approval is refreshed. TQ-010, 012 |
| QR-06 | Snapshot intent and dataset/runtime/semantic hashes; bind unique run/attempt/container/output/MLflow identities | Train distinct dataset versions in fresh containers; substitute/reorder classes or mount the wrong request; verify result lineage | Correct hashes do not prove human label ground truth. TQ-001, 010, 019, 023, 026 |
| QR-07 | Trusted full-cost estimates, cumulative authorization, bounded retries, and VM uptime/idle/retained-cost accounting | Race cumulative reservations; increase duration/shape or retry; audit VM exposure against the approved policy and billed usage | VM billing is continuous and not isolated per container; exact invoice caps are not guaranteed by a job timer. TQ-011, 013, 031 |
| QR-08 | Atomic pause/cancel/claim decisions; stop the exact owned process group/container and observe exit; retain reservation while unresolved | Race waiting removal/claim and active cancellation; delay/refuse stop; verify no new container after committed waiting cancellation | Work claimed before pause may already be starting and needs explicit cancellation. TQ-014–016 |
| QR-09 | Authorize OS/local-service callers separately from worker observations; protect DB/socket ownership and state revisions | Allowed/denied OS/service principal matrix; forge completion or access other entries; verify zero unauthorized launch/state changes | Root administrators and compromised permitted identities remain trust boundaries. TQ-017, 019, 032 |
| QR-10 | Versioned store/configuration migration; application-consistent backup; reconcile VM/container claims before restore or rollback | Restore with in-flight/unknown containers; test old/new supported records, disk remount, instance identity change, and rollback | Recovery depends on accepted RPO/RTO and available evidence. TQ-002, 007, 009, 021, 028–029 |
| QR-11 | Finite backlog, explicit blocked head, GPU/disk readiness, aging/stall signals, authorized skip; no implicit leapfrogging | Full queue, GPU busy/orphan work, disk pressure, and old blocked head produce bounded backlog and honest status | FIFO allows unresolved head entries to delay later work. TQ-003, 012, 018, 020, 033 |
| QR-12 | Validate the exact host GPU/driver/toolkit/engine, stable mounts, service startup order, VM identity, and GCS/MLflow access | Clean approved VM/container smoke and actual GPU training across a reboot; mismatch/missing mount blocks before launch | Driver updates and host configuration drift require fresh evidence. TQ-010, 020, 024, 027–028 |
| QR-13 | Version failure/continuation policy; pause by default pending user choice; retry only classified recoverable outcomes within cumulative bounds | Deterministic trainer failure, transient capacity failure, exhausted budget, and ambiguous submit cannot generate an unbounded cascade | Misclassification remains possible; unknowns hold for review. TQ-012–013 |
| QR-14 | Durable history with active/waiting/blocked/interrupted/unknown status; exact request/image/VM/container/log evidence | Walk through failed/stalled/rebooted cases; inject stale observations; verify retained attempt evidence and redaction | Host/cloud logs may be unavailable; report missing evidence explicitly. TQ-018–020, 029–030 |
| QR-15 | Enforce Experiment Runner capabilities and immutable references; hand failures to the owner; separate deployment/certification approval | Instrument build/push/install/certification and protected writes during normal and failed queue work; inspect actual IAM denial | Authorized owner work may still be needed to resolve a blocked entry. TQ-010, 017, 022 |
| QR-16 | Separate container exit, training outcome, output verification, MLflow finalization, and promotion | Zero exit with incomplete/corrupt outputs, registration failure, or no human approval cannot appear as an accepted/promoted model | Finalization outage can hold continuation even after compute has stopped. TQ-005, 012, 023 |
| QR-17 | Record VM/boot identity; classify interruption explicitly; preserve durable checkpoints/evidence if supported; resume/restart only under approved retry and budget policy | Interrupt/reboot the approved VM during training; restart worker; no false success or unauthorized repeat; verify recovery path | Checkpoint support is not assumed; the current attempt may need to restart. TQ-028–030 |
| QR-18 | Protected boot-disk queue directory; measured capacity and peak job footprint; reservations/quotas/headroom; bounded image/data/log caches; tested commit/backup controls | Grow images/logs/checkpoints, race storage reservations, exhaust inodes, and interrupt commits; deny unsafe launch and preserve queue/OS recovery at the accepted RPO | One filesystem shares a failure/capacity boundary; unknown footprints block dispatch. TQ-002, 021, 028, 033–035 |
| QR-19 | Fresh container and isolated paths per attempt; assigned GPU; observe owned process tree; bound caches/logs; cleanup only after evidence capture | Consecutive real GPU jobs; kill worker/container; inject orphan process and stale scratch; verify no cross-run state or premature next launch | Driver faults may require an authorized host repair or reboot. TQ-005, 026–027, 033 |
| QR-20 | Explicit uptime/idle budget and queue-empty/blocked behavior; alert or stop under approved power-management policy; avoid implicit auto-start | Leave queue empty/blocked and simulate budget exhaustion; verify adopted action and no unauthorized shutdown/start/replacement | Stopped-VM attached resources may still incur costs; budget evidence includes them. TQ-011, 031 |
| QR-21 | Keep daemon/queue volume outside trainer reach; deny privileged host access and unneeded metadata/control credentials; use least-permission artifact access | Trainer tries engine/socket, DB, metadata-derived control credentials, host mounts, and sibling creation; real forbidden effects remain zero | VM root and engine administrators remain trusted; identity isolation must be tested, not assumed. TQ-017, 022, 032 |
| QR-22 | Verify and review boot-disk deletion policy; keep application-consistent backups outside the VM/disk; test authorized restore and job reconciliation | Inspect actual disk settings; use an approved restore/loss simulation and independently verify recovered accepted intent/history | Retention alone does not cover disk loss; backup freshness sets the accepted RPO. TQ-021, 028–029, 035 |

The VM revision adds QR-17–21 and retargets the original hazards to container execution. Revision 3 adds QR-22 and increases QR-18 from 0.62 to 0.71 before mitigation, and from a 0.19 to a 0.24 conditional residual target, because queue/training/OS storage now share the known boot disk. No live risk reduction or disk readiness is claimed. The scores are planning estimates, not measurements from this VM. The highest priorities include idle/cumulative spending (QR-20/07), stale VM/runtime authority (QR-05), false termination (QR-04), and duplicate/overlapping workloads (QR-01–02). Progress without SSH depends on QR-03/12/17/18. Do not average scores into an overall pass metric.

## 6. Implementation sequence after adoption

The user has authorized local implementation. Live deployment, VM changes, paid training, certification, and power changes require their own authorization and evidence. Completion of a package is not full feature acceptance. TQ-001–TQ-035 are now mapped to canonical TP-161–TP-195; the implementation record tracks coverage and open gates.

| Stage | Planned work | Exit criterion and owner |
| --- | --- | --- |
| 1 — Adopt behavior and requirements | Confirm failure policy, designated VM/GPU identity, 500 GB boot-disk layout/free capacity/headroom, retention/backups, uptime/budget policy, finite queue limits, and numeric liveness/recovery targets; assign canonical IDs | Reviewed policy with no unresolved spending/dispatch choices; primary integration owner |
| 2 — Review VM contracts and dispatch design | Specify typed VM job/readiness records, states/claims/fencing, local OS authority, container identity/recovery, immutable paths, and store migrations | Enumerate crash/reboot windows and all launch paths; primary integration review plus Experiment Runner |
| 3 — Implement durable queue locally | Queue store/controller/local interface/CLI and execution adapter; pause/cancel/retry/inspection/budget accounting; supervised worker integration | Meaningful local adverse/concurrency/engine-integration checks pass; Experiment Runner with packaging/startup handoff |
| 4 — Validate host/container readiness | Exact certified digest on intended VM GPU/driver/toolkit; storage, mounts, identities, process observation, and reboot behavior | Owner-bound GPU/container and authority evidence; Runtime Engineer/infrastructure owner; certification remains separate |
| 5 — Package/deploy worker revision | Pin worker artifact/version and host service configuration; mount durable state; reviewed supervision, permissions, migrations, rollback, and logs | Runtime/deployment owner supplies reviewed artifacts/configuration; Experiment Runner does not build/push images |
| 6 — Verify actual unattended VM queue | Approved finite requests with explicit VM/verification budget; append while busy, fresh container IDs, real GPU sequencing, closed SSH, and interruption recovery | Positive/negative evidence plus VM/resource/billing reconciliation accepted; owning roles |
| 7 — Accept and document | Canonical requirements, operations, evidence, and actual residual-risk reassessment | Every applicable TQ requirement closed for the released VM/worker/image revision; primary integration reviewer |

### Mandatory acceptance scenarios

- Append distinguishable dataset versions while actual training runs on the designated VM. The active container, request, and process remain unchanged. Fresh subsequent container IDs execute FIFO after the prior training finishes.
- Disconnect SSH/CLI/assistant after acknowledgment; the supervised worker continues while the VM is running.
- Race submissions, immediate/manual paths, stale workers, engine observations, and queue controls; verify one active workload and one accounted container attempt per intent.
- Lose observation events and restart worker/engine; reconcile without duplicate execution. Reboot/stop during training; preserve the queue and explicit interruption, with the approved restart/resume policy rather than assumed checkpoint continuation.
- Start with an existing trainer container/GPU workload. Verified identity can bind it to observed completion; unbound activity holds dispatch and is not killed automatically.
- Revoke approval, change host/driver/image identity, remove a mount, exhaust GPU/disk/budget resources, and attempt unauthorized completion/control actions; no forbidden new workload follows.
- Exercise adopted failure/retry/skip, pause/resume, waiting removal, and active-stop policies with independent process/container/GPU inspection.
- Delete completed containers and restore a reviewed off-VM backup; queue state and attempt evidence remain intact. Verify boot-disk deletion settings. Exercise disk/inode exhaustion and competing image/log/checkpoint growth on the shared boot filesystem; admission protects queue/OS recovery headroom.
- Verify distinct request/output/temp paths, exact image digests, input integrity, GCS/MLflow authorization, and bounded cleanup across consecutive real GPU jobs.
- Container exit and validated outputs/finalization remain separate. No automatic model promotion, build/push/install/certification, immutable dataset overwrite, or unauthorized VM power change occurs.
- Verify actual trainer denial of queue DB/engine socket/host/control credential access; host startup/driver changes do not silently become approved runtime evidence.
- Test the adopted empty/blocked-queue uptime/spending policy; include applicable retained costs and confirm behavior when the VM is off.

Local simulations cannot close actual VM/GPU durability, permissions, driver compatibility, billing, or runtime acceptance. Paid verification requires its separately authorized budget and intended actor identities. No live deployment or runtime change is authorized by this planning update.

## 7. Files and boundaries for future work

**Experiment Runner:** Scheduling, durable queue state, config/submission, API/client/CLI integration, and control tests under the control boundary. Shared CLI wiring is a primary integration task. No trainer algorithm or dependency changes are part of this feature.

**Shared-interface handoff:** Any change to `contracts.py`, generated schemas, or deployment policy carries a reviewed compatibility matrix, migration plan, affected requirements, and explicit primary integration review.

**Runtime/deployment handoff:** Provide exact worker source/artifact/configuration IDs, designated VM/GPU profile, host supervisor/storage/mount/engine/toolkit/identity needs, and required evidence to the owning runtime/infrastructure role. Build/push, driver/toolkit installation, host service provisioning, and VM power operations remain owner-bound. Existing trainer certification stays bound to its original digest; VM execution adds compatibility evidence rather than removing Vertex certification.

**Canonical documentation after adoption:** Update [requirements.json](requirements/requirements.json) and its context, regenerate [REQUIREMENTS.md](REQUIREMENTS.md), and extend [operations](OPERATIONS.md), [start guide](START_HERE.md), and acceptance/evidence indexes. Preserve existing baseline/history; do not manually turn this draft into claimed acceptance.

The synced `sources/` directory and project-root `AGENTS.md` must remain untouched. Dataset versions and releases stay immutable. Failures preserve the failing command, exact configuration identity, image digest, VM/container/process identity, log reference, uncertainty, and owning layer.
