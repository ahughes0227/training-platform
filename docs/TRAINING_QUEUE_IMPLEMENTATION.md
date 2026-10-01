# Compute Engine Training Queue — Implementation and Evidence

Date: September 29, 2026.

The user authorized GPT-6 Luna agents to implement revision 3 of [the queue plan](TRAINING_QUEUE_PLAN.md). Earlier planning-only instructions no longer prohibit local implementation. This authorization does not authorize VM changes, image builds/pushes, paid jobs, certification, or deployment.

## Confirmed behavior

- GCP Compute Engine VM, user-reported 500 GB boot disk.
- A fresh container per job from its approved certified immutable image digest.
- Append requests while the current container trains; next training starts after previous training finishes.
- Runtime certification and immutable dataset/release constraints in AGENTS.md remain in force.

## Lanes and ownership

| Lane | Model | Files and responsibility |
| --- | --- | --- |
| Durable queue | GPT-6 Luna | control/queue_contracts.py, queue_store.py, storage/control tests |
| VM execution | GPT-6 Luna | control/vm_executor.py, queue_worker.py, executor/worker tests |
| Local interface | GPT-6 Luna | control/vm_queue_api.py, vm_queue_client.py, vm_queue_commands.py, interface tests |
| Primary integration | Primary agent | Admission/controller integration, shared CLI wiring, contract/security review, requirements/evidence and whole-suite review |

No lane edits the trainer, dependencies, root AGENTS.md, synced sources/, or another lane's files. Shared contract/deployment changes require primary review and an owning-role handoff.

## Decision and evidence boundaries

Failure continuation remains a pending user choice; implementation exposes pause/continue policy and treats unknown execution state as a hold in both modes. Production numeric budgets, storage headroom/footprints, backlog/timing bounds, GPU/driver/toolkit compatibility, actual free bytes/inodes, VM identity, credential isolation, and off-VM backup policy require operator configuration and verification. No current live readiness is inferred from nominal disk size or written policy.

Implementation progress and tests do not close the plan's live acceptance gates or demonstrate a reduction in risk scores. The requirements evidence index will distinguish implemented, local verification, and live acceptance pending.

## Integrated behavior

The local implementation provides typed queue requests; transactional SQLite FIFO, idempotency, reservations and claims; a host worker with container/boot/process reconciliation; an argv-only Docker adapter; and `defect train queue` commands through an authenticated local Unix socket. Existing immediate submission remains the Vertex backend; this queue is the sole implemented VM submission route. Manual or preexisting GPU work is discovered and blocks dispatch until resolved. There is no implicit adoption or termination of another workload.

Enqueue does not operate the engine or modify an active attempt. A fresh container consumes its exact protected approved experiment, dataset, certified digest, and reviewed job profile. Current approval/readiness is rechecked at launch. Protected configuration controls UIDs, credentials, isolated network, GPU identity, storage headroom, finite campaign authorization, and bounded duration/attempts/backoff. Requests cannot grant themselves those permissions.

Create/start intent is persisted before the corresponding side effect. Uncertain acknowledgments are reconciled against the original attempt and container; absent containers, expired leases, and VM reboots cannot authorize another launch. The worker observes an existing attempt before backup or spending maintenance, so a maintenance outage cannot suppress termination observation or deadline cancellation. Explicit root-only interrupted recovery requires independently observed absence and an empty GPU pool on the same numeric instance.

Cancellation retains its intent through terminal observations. A running container retains the slot until termination; an owned created-but-unstarted container is removed without replaying start or inventing an exit code. Each new retry clears only the entry's current-attempt facts while preserving prior attempt evidence. Operator retry requires a recoverability classification and reason: transient confirmed failure or independently verified interruption. Unknown and deterministic classifications are denied. No automatic failure retry or checkpoint resume is implemented.

Container exit, coherent verified outputs, MLflow finalization, and release eligibility are distinct facts. A zero exit alone cannot complete the queue entry. Verified success allows removal of the exact exited container and its per-attempt scratch; failed, canceled, and uncertain evidence is retained. No release is promoted. Recovery archives include application-consistent SQLite state, immutable requests, tracking journals, protected configuration and evidence; they exclude trainer credentials and model/scratch content and are verified before local backup retention cleanup.

The VM execution agent received an explicit Runtime Engineer handoff for the [systemd templates](../infra/systemd/README.md). Primary review integrated these with the control model and reviewed CLI wiring, protected configuration, credentials, filesystem roles, metadata-isolation evidence, cost/profile bindings, output validation and canonical requirements. Templates remain uninstalled. Trainer code, shared `contracts.py`, dependencies, certified images, synced `sources/`, and root `AGENTS.md` were not changed by this feature. Preexisting user documentation changes were preserved.

## Local verification on September 29, 2026

| Check | Observed result | Scope |
| --- | --- | --- |
| `.venv/bin/pytest -q --ignore=tests/integration/test_semantic_ray_http.py` | 177 passed; two dependency deprecation warnings | Final local source, including all 63 collected queue cases |
| `.venv/bin/pytest -q tests/integration/test_semantic_ray_http.py` | One passed; 17 dependency warnings | Separate earlier session run with approved loopback permission; trainer/Ray paths subsequently unchanged |
| Ruff over new queue/VM modules and corresponding tests | Passed | Changed control modules and test files |
| `git diff --check` | Passed | Tracked changes; no commit/push performed |
| `scripts/requirements_document.py --check` | Passed | Canonical 195 requirements and generated register consistency |

See [the local source/evidence receipt](evidence/training-queue-local-2026-09-29.json) for commands, source hashes and limits. The new engine, GCE identity, GCS and tracking tests use injected observations/runners. The queue suite did not launch Docker, create a paid job, connect to the user's VM, or establish Linux host readiness.

## Requirement coverage and remaining acceptance

TQ-001 through TQ-035 map in order to TP-161 through TP-195. **Full positive and negative acceptance remains pending for every row.** The local observations below establish tested paths, not full requirement satisfaction. Tests are in [storage](../tests/test_control_queue_store.py), [worker](../tests/test_control_queue_worker.py), [executor](../tests/test_control_vm_executor.py), [controller integration](../tests/test_control_queue_integration.py), [API](../tests/test_control_vm_queue_api.py), and [CLI](../tests/test_control_vm_queue_cli.py).

| TQ / TP | Implemented control and local observation | Remaining acceptance |
| --- | --- | --- |
| 001 / 161 | Typed immutable bindings and protected catalog/profile admission; self-asserted approval rejected | Real approved datasets and certification authority, denied effects |
| 002 / 162 | Atomic finite batch and durable IDs/reservations; failed admission has no partial entries | Power-loss/crash commit boundaries on the target filesystem |
| 003 / 163 | Transactional FIFO sequence and idempotent ordered batch | Concurrent real clients and reboot order preservation |
| 004 / 164 | Worker flock/CAS and GPU workload scan; orphan fixtures hold | Race all actual VM routes and independently measure GPU overlap |
| 005 / 165 | Uncertain/active/canceling slot holds; next fixture job follows verified finalization | Actual process termination and GPU release |
| 006 / 166 | Supervised polling commands and systemd templates; local lifecycle recovery | Closed SSH, actual service/engine restart and missed observations |
| 007 / 167 | Claim token, revisions, intent and host lock; concurrent claims tested | Stale-worker external-effect faults on Linux |
| 008 / 168 | Stable admission keys and deterministic attempt container; lost engine/tracking acknowledgments held | External create/start/MLflow fault campaign and zero duplicates |
| 009 / 169 | Exact instance/boot/container/process lineage; unknown never blindly replayed | Crash at every commit/side-effect boundary on target VM |
| 010 / 170 | Re-read protected approvals/semantic content and readiness before dispatch; revocation tested | Live revocation, integrity and service availability gates |
| 011 / 171 | Trusted profile reservation and durable cumulative uptime meter; nonfinite/self-priced policy rejected | Reviewed inclusive prices and actual billing comparison; see spending limits below |
| 012 / 172 | Default pause/configured continue; both policies and unfinalized tracking tested | Operator adopts failure policy and verifies it with real jobs |
| 013 / 173 | Explicit recoverability classification, count/backoff/duration/budget; deterministic/unknown denied | Validate operator classification procedure and fault taxonomy on real failures |
| 014 / 174 | Atomic pause/claim policy revision; resume validates waiting intent | Real pause/launch race, documented committed-claim behavior |
| 015 / 175 | Waiting cancellation CAS/idempotency; claim race tested | Concurrent real clients and independent zero launch effects |
| 016 / 176 | Running cancellation waits; unstarted cancellation cannot start/invent exit | Engine stop refusal, delayed death, timeout/process escape faults |
| 017 / 177 | UID role separation and fenced worker observation; forged principal denied | Actual SO_PEERCRED, socket/group/host permissions and scope tests |
| 018 / 178 | Status exposes queue/activity/blocker, budget, observation/heartbeat freshness | User review with real mixed/stalled queue and observer loss |
| 019 / 179 | Append-only transition/worker/finalization evidence with exact identities/logs | Full fault traceability, retention and redaction review |
| 020 / 180 | Configured finite backlog/timing limits and status staleness signals | Load/timing campaign; no measured deployed liveness claim |
| 021 / 181 | Schema-v1 guard before writes, application-consistent backup and read-only restore validation | First deployed restore, future migration implementation and mixed-version rollout |
| 022 / 182 | Fixed existing-digest execution path; no build/push/install/certify/promote API | Instrument actual credentials/APIs and independently prove forbidden effects denied |
| 023 / 183 | Output/bundle/hash/evaluation/GCS/MLflow gates; partial/tampered fixtures rejected | Certified trainer end-to-end output compatibility and finalization faults |
| 024 / 184 | Local evidence/register and complete mapping retained | All deployed positive/negative acceptance on the released revision |
| 025 / 185 | Enqueue independent of engine; append during active fixture lifecycle | Append during actual GPU training; unchanged container/request/process proof |
| 026 / 186 | Per-attempt ID/paths and exact cached digest; wrong digest/root/arbitrary argv denied | Distinct real containers and no cross-attempt contamination |
| 027 / 187 | Observed identity/GPU/driver/toolkit/engine compared against expiring approval | Runtime owner verifies exact image on designated VM/GPU |
| 028 / 188 | Protected host state, root worker/unprivileged API templates | Linux storage ACLs, container removal, service startup and boot persistence |
| 029 / 189 | Changed boot/instance fences; verified interruption has no fabricated success/exit | Real VM stop/reboot, engine state loss and replacement recovery |
| 030 / 190 | Explicit administrator recovery plus separately authorized new attempt | Supported restart-from-beginning procedure with live output/tracking reconciliation |
| 031 / 191 | Idle/running exposure persists; exhausted authority pauses new dispatch; no power operations | Owner adopts uptime/power/alert policy and monitors retained charges |
| 032 / 192 | Nonroot trainer, no engine/queue mounts, limited credentials, named isolated network required | In-container metadata/host/control denial and IAM effect tests |
| 033 / 193 | Byte/inode reservations, bounded Docker logs and exact verified cleanup | Quotas/cache/journal/evidence/retained-output bounds and orphan resource fault tests |
| 034 / 194 | Fresh measured statvfs bytes/inodes plus all reservations/OS headroom | Actual 500 GB boot filesystem, representative peak jobs and concurrent growth enforcement |
| 035 / 195 | Recovery archive/create-only off-VM transfer with hash verification (fixture) | GCS access/retention/cost, deletion rule review and independent restore/reconciliation |

## Outstanding operational decisions and limits

The [protected example config](../infra/systemd/queue-config.example.yaml) intentionally fails validation until real operator values replace every placeholder. Supply the existing VM's numeric identity/zone, boot/GPU IDs, driver/toolkit/engine evidence, certified digest compatibility, approved experiments/datasets/catalogs/profiles, actual disk bytes/inodes, conservative footprints/headroom, finite inclusive pricing/campaign window, trainer credential/network controls, UIDs/GIDs, and independent backup destination/retention/restore evidence. A nominal 500 GB disk does not authorize a storage estimate or establish free capacity.

Default failure behavior is pause. Continue is configurable only for a confirmed failed attempt after tracking finalization; uncertain outcomes always hold. Checkpoint continuation is unsupported; a verified interrupted attempt requires a separately authorized fresh attempt from the beginning. Future schema migrations have not been written for nonexistent schema versions: unsupported database versions fail closed without modification.

The spending meter records liability and pauses dispatch; it is not an invoice cap. The existing VM and retained disk/artifacts may keep accruing charges while idle, blocked or paused. In-flight cancellation, API calls and service delays can extend runtime beyond a planned duration. The owner must approve inclusive conservative prices, margin, monitoring and any shutdown/notification policy; this feature has no authorization to stop the VM. No new storage purchase is assumed. Finite reservations and admission checks alone cannot prevent unrelated host writes from exhausting the shared boot disk. Host quotas, cache/journal limits, retained outputs, failure evidence, indefinite campaign retention and alerts require deployed policy/enforcement review before unattended acceptance.

API/worker isolation uses host filesystem authority and Linux peer credentials; the worker's engine access is host-root control. The trainer network's metadata denial and credential least privilege must be independently tested, not inferred from a config flag. Backups must be restored outside the original VM; a successful injected upload does not prove actual GCS retention or disaster recovery. No service installation, cloud resource mutation, image build/push/certification, release promotion or paid training has been performed.

The plan's QR-01–QR-22 after-mitigation scores remain **conditional targets**. Local tests do not establish the actual residual scores. Runtime, infrastructure, backup and policy owners must supply the linked evidence before unattended deployment acceptance.
