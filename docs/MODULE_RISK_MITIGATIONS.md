# Module Risks — Mitigation Plan and Target Residual Scores

**Date:** September 28, 2026

**Status:** Proposed mitigations and conditional target scores. No mitigation is claimed implemented or verified by this document.

**Repository reviewed:** `b45ca11e56a53a07897aca52b1d26ba038fde72c`

## 1. Meaning of the mitigated score

This plan addresses all 20 risks in the [module-change assessment](MODULE_CHANGE_RISKS.md). It keeps the same definitions and preserves the original risk and detection ratings.

**Target residual risk = target likelihood × original impact**, rounded half up to two decimals.

The targets estimate the remaining risk **after the specified controls are implemented, reviewed, and verified at their required local/live scope**. They are engineering planning estimates, not measured probabilities or guaranteed outcomes. Successful tests permit reassessment; they do not automatically prove the forecast score.

The original impact is retained for every risk. Mitigation reduces the estimated likelihood of the hazardous outcome reaching an accepted downstream operation; it does not assume that a bad outcome would become harmless.

Detection is separate. A test that notices damage after a paid job or serving change has happened does not necessarily prevent that damage. This plan does **not** calculate residual risk by multiplying current risk by `1 − detection`.

**Current scores have not fallen.** The structured register preserves them and sets `actual_residual_risk` to `null`. Every mitigation remains `planned_unimplemented_unverified`. [module-change-risks.json](requirements/module-change-risks.json) contains controls, owners, verification obligations, target factors, and remaining risk.

## 2. Summary

| ID | Risk | Current R | Target residual R |
| --- | --- | ---: | ---: |
| [MR-01](#mr-01) | Contract or API incompatibility | 0.60 | 0.16 |
| [MR-02](#mr-02) | Silent changes to data or model meaning | 0.72 | 0.27 |
| [MR-03](#mr-03) | Invalid or substituted dataset consumed | 0.63 | 0.14 |
| [MR-04](#mr-04) | Incomplete training output accepted | 0.54 | 0.14 |
| [MR-05](#mr-05) | Runtime identity or certification becomes stale | 0.63 | 0.18 |
| [MR-06](#mr-06) | Module dependencies or packaging break | 0.48 | 0.12 |
| [MR-07](#mr-07) | Operational authority crosses module boundaries | 0.63 | 0.18 |
| [MR-08](#mr-08) | Retries create duplicate paid jobs | 0.54 | 0.18 |
| [MR-09](#mr-09) | Job state lies or cancellation leaves resources running | 0.54 | 0.18 |
| [MR-10](#mr-10) | Budget policy is bypassed across modules | 0.56 | 0.20 |
| [MR-11](#mr-11) | Lineage is lost or assigned to the wrong model | 0.49 | 0.14 |
| [MR-12](#mr-12) | Wrong or unapproved model reaches serving | 0.63 | 0.18 |
| [MR-13](#mr-13) | Telemetry cannot expose a regression | 0.48 | 0.15 |
| [MR-14](#mr-14) | Mixed versions or migration damage historical state | 0.56 | 0.20 |
| [MR-15](#mr-15) | Guided setup and navigation become confusing or broken | 0.35 | 0.10 |
| [MR-16](#mr-16) | Execution waiting returns to the agent | 0.40 | 0.12 |
| [MR-17](#mr-17) | Extra artifact transfers increase cost or latency | 0.30 | 0.18 |
| [MR-18](#mr-18) | Failures are attributed to the wrong owner | 0.36 | 0.12 |
| [MR-19](#mr-19) | Untrusted model artifact executes code or escapes its path | 0.50 | 0.15 |
| [MR-20](#mr-20) | Infrastructure capabilities are stale or only desired state | 0.56 | 0.20 |

## 3. Mitigation and verification for each risk

Positive checks establish that permitted work still succeeds. Negative checks establish that prohibited outcomes are prevented. All required checks below are planned. Negative evidence must include independent before/after state and zero forbidden effects; an exception or an empty log query alone is insufficient.

<a id="mr-01"></a>

### MR-01 — Contract or API incompatibility

**Owner:** Primary integration owner; contract CI.

**Current risk 0.60 → target residual risk 0.16**

Target likelihood 0.20 × retained impact 0.80.

**Mitigation**

1. Version the public input/output schemas and document supported producer/consumer pairs.
2. Make the shared contract package independent of module internals; require compatibility checks and serialized fixtures for every public interface.
3. Validate at each boundary and reject unsupported versions before side effects; use a reviewed migration for intentional breaking changes.

**Evidence required before reassessment**

- **Positive:** Every supported version pair reads valid fixtures and completes its intended handoff.
- **Negative:** Remove/rename required fields, change enums, and mix unsupported versions; the boundary rejects them before jobs or accepted writes.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** An omitted compatibility case or an incorrectly designed migration can still break consumers.

**Traceability:** [Original MR-01 assessment](MODULE_CHANGE_RISKS.md#mr-01) and [TP-016](REQUIREMENTS.md#tp-016), [TP-023](REQUIREMENTS.md#tp-023).

**Implementation/verification status:** planned; current detection remains 0.70.

<a id="mr-02"></a>

### MR-02 — Silent changes to data or model meaning

**Owner:** Dataset, trainer, and serving owners; primary integration review.

**Current risk 0.72 → target residual risk 0.27**

Target likelihood 0.30 × retained impact 0.90.

**Mitigation**

1. Bind ordered classes, label rules, group/split assignments, preprocessing, pooling, calibration, and selection policy to semantic fingerprints.
2. Use fixed reference fixtures across build, train, reload, and serve with declared numeric tolerances and actual DINOv3 architecture checks.
3. Gate integration on parity; intentional semantic changes receive a new version and explicit reviewed evidence rather than inheriting the old acceptance.

**Evidence required before reassessment**

- **Positive:** The same accepted fixture produces matching classes/splits/preprocessing and consistent selected model, predictions, and review flags across all paths.
- **Negative:** Reorder class indices, include register tokens, alter normalization/splits, or use test labels for calibration; semantic checks fail and the affected output cannot be accepted.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Reference fixtures cannot exhaust real data variation, stochastic execution, or all semantically wrong policies; this remains the highest target risk.

**Traceability:** [Original MR-02 assessment](MODULE_CHANGE_RISKS.md#mr-02) and [TP-030](REQUIREMENTS.md#tp-030), [TP-041](REQUIREMENTS.md#tp-041), [TP-054](REQUIREMENTS.md#tp-054).

**Implementation/verification status:** planned; current detection remains 0.45.

<a id="mr-03"></a>

### MR-03 — Invalid or substituted dataset consumed

**Owner:** Dataset publisher/verifier and admission service.

**Current risk 0.63 → target residual risk 0.14**

Target likelihood 0.15 × retained impact 0.90.

**Mitigation**

1. Bind human review to the exact source/preview fingerprint; revalidate on publication.
2. Publish create-only immutable artifacts with a final commit manifest containing full checksums and exact storage generations; protect accepted prefixes using IAM/retention policy.
3. Resolve accepted versions from the authoritative catalog and read the committed generations; verify object/classes/integrity before dispatch and consumption.

**Evidence required before reassessment**

- **Positive:** Training reads the exact committed dataset and source-bound decisions under its intended identity.
- **Negative:** Change a source after review, replace a shard during verification, swap an object/version, or attempt overwrite/delete; reject or deny with unchanged accepted generations and zero unauthorized jobs.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Incorrect human labels, overlooked near duplicates, compromised trusted administrators, or storage-policy defects remain possible.

**Traceability:** [Original MR-03 assessment](MODULE_CHANGE_RISKS.md#mr-03) and [TP-033](REQUIREMENTS.md#tp-033), [TP-044](REQUIREMENTS.md#tp-044), [TP-046](REQUIREMENTS.md#tp-046).

**Implementation/verification status:** planned; current detection remains 0.70.

<a id="mr-04"></a>

### MR-04 — Incomplete training output accepted

**Owner:** Trainer artifact exporter and control-plane finalizer.

**Current risk 0.54 → target residual risk 0.14**

Target likelihood 0.15 × retained impact 0.90.

**Mitigation**

1. Write outputs into a staged run prefix and produce a manifest binding model, preprocessing, calibration, metrics, lineage, and checksums.
2. Publish the protected completion marker only after full validation; consumers require the exact committed manifest.
3. Keep the run finalizing until outputs are verified; register an eligible candidate idempotently from the accepted result, not merely from Vertex process success.

**Evidence required before reassessment**

- **Positive:** A complete bundle reloads, evaluates, and registers one coherent candidate linked to the run.
- **Negative:** Interrupt each upload/finalization step, truncate artifacts, substitute metadata, or return job success early; no incomplete result becomes successful or release-eligible.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** A verifier defect or semantically incorrect but complete output can still pass; semantic and lineage gates are also required.

**Traceability:** [Original MR-04 assessment](MODULE_CHANGE_RISKS.md#mr-04) and [TP-061](REQUIREMENTS.md#tp-061), [TP-092](REQUIREMENTS.md#tp-092), [TP-099](REQUIREMENTS.md#tp-099).

**Implementation/verification status:** planned; current detection remains 0.30.

<a id="mr-05"></a>

### MR-05 — Runtime identity or certification becomes stale

**Owner:** Runtime Engineer, Runtime Certifier, and authoritative catalog owner.

**Current risk 0.63 → target residual risk 0.18**

Target likelihood 0.20 × retained impact 0.90.

**Mitigation**

1. Pin source, dependency lock, base image, candidate digest, software inventory, and supported hardware scope.
2. Separate building/publishing from existing-digest certification; authenticate immutable gate evidence in the server-side catalog with revocation policy.
3. Validate actual DINOv3 and the corrected fine-tuning path inside the exact GPU image and Vertex handshake; ordinary experiments resolve that record without builds or installs.

**Evidence required before reassessment**

- **Positive:** The current source's exact image passes required GPU/Vertex gates and is reused by ordinary experiments.
- **Negative:** Use an old/mutated/uncertified digest, revoked scope, or unsupported GPU count; reject before training. Intercept ordinary build/push/install/certification calls and verify none occur.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Hardware/driver drift, untested shapes, dependency vulnerabilities, and revocation propagation can still invalidate compatibility.

**Traceability:** [Original MR-05 assessment](MODULE_CHANGE_RISKS.md#mr-05) and [TP-067](REQUIREMENTS.md#tp-067), [TP-070](REQUIREMENTS.md#tp-070), [TP-073](REQUIREMENTS.md#tp-073).

**Implementation/verification status:** planned; current detection remains 0.55.

<a id="mr-06"></a>

### MR-06 — Module dependencies or packaging break

**Owner:** Runtime Engineer and module dependency owners.

**Current risk 0.48 → target residual risk 0.12**

Target likelihood 0.20 × retained impact 0.60.

**Mitigation**

1. Declare an acyclic dependency direction: shared contracts, module public interfaces, controller coordination, and scoped adapters.
2. Lock dependency sets and package resources; validate import boundaries so module internals are not imported across ownership.
3. Install clean environments and run public entry points/tests in the actual component images before release.

**Evidence required before reassessment**

- **Positive:** Each module runs with only its declared dependencies; the integrated fixture and container entry points work outside the checkout.
- **Negative:** Remove an undeclared dependency/template, introduce an import cycle/internal import, or rely on checkout-relative files; CI rejects the release.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Platform-specific packaging behavior and unrepresented optional configurations can still escape the clean-environment matrix.

**Traceability:** [Original MR-06 assessment](MODULE_CHANGE_RISKS.md#mr-06) and [TP-017](REQUIREMENTS.md#tp-017), [TP-065](REQUIREMENTS.md#tp-065), [TP-139](REQUIREMENTS.md#tp-139).

**Implementation/verification status:** planned; current detection remains 0.70.

<a id="mr-07"></a>

### MR-07 — Operational authority crosses module boundaries

**Owner:** Cloud policy administrator and control-plane/harness owners.

**Current risk 0.63 → target residual risk 0.18**

Target likelihood 0.20 × retained impact 0.90.

**Mitigation**

1. Treat client/assistant records as proposals; resolve certification, approval, prices, and accepted artifacts from authenticated protected services.
2. Use distinct least-privilege identities and constrained operational brokers; keep privileged cloud credentials out of assistant/module capabilities.
3. Enforce effects beneath hooks with filesystem/process/service/cloud boundaries; deny unknown roles and policy-check failures; verify direct and indirect paths.

**Evidence required before reassessment**

- **Positive:** Every role's allowed counterpart works with only its intended scoped identity and resources.
- **Negative:** Forge authority fields and attempt forbidden operations through CLI, SDK, subprocess, alternate endpoints, and impersonation; audit and after-state show no unauthorized effects or stronger credentials.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Policy misconfiguration, trusted-service compromise, administrator actions, and new bypass paths remain consequential.

**Traceability:** [Original MR-07 assessment](MODULE_CHANGE_RISKS.md#mr-07) and [TP-070](REQUIREMENTS.md#tp-070), [TP-093](REQUIREMENTS.md#tp-093), [TP-133](REQUIREMENTS.md#tp-133).

**Implementation/verification status:** planned; current detection remains 0.20.

<a id="mr-08"></a>

### MR-08 — Retries create duplicate paid jobs

**Owner:** Control service, durable state store, and dispatch broker.

**Current risk 0.54 → target residual risk 0.18**

Target likelihood 0.20 × retained impact 0.90.

**Mitigation**

1. Persist immutable intent/fingerprint and a transactional dispatch outbox with unique idempotency keys.
2. Serialize dispatch ownership with revision/fencing checks; bind cloud jobs to stable platform IDs and reconcile them before another creation attempt.
3. Use provider idempotency where actually supported; block redispatch of uncertain creation outcomes until authoritative reconciliation, with bounded recovery and accounting.

**Evidence required before reassessment**

- **Positive:** Sequential, concurrent, and interrupted retries resolve one effective run, job, reservation, and result.
- **Negative:** Crash before/after creation, lose the response, expire an ownership lease, and race requests; no blind second paid job is created and uncertain outcomes remain visible.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** External creation and local transactions are not automatically atomic; provider visibility delays and unresolved outcomes can delay recovery.

**Traceability:** [Original MR-08 assessment](MODULE_CHANGE_RISKS.md#mr-08) and [TP-083](REQUIREMENTS.md#tp-083), [TP-085](REQUIREMENTS.md#tp-085), [TP-090](REQUIREMENTS.md#tp-090).

**Implementation/verification status:** planned; current detection remains 0.30.

<a id="mr-09"></a>

### MR-09 — Job state lies or cancellation leaves resources running

**Owner:** Durable workflow, run store, and operations reconciler.

**Current risk 0.54 → target residual risk 0.18**

Target likelihood 0.20 × retained impact 0.90.

**Mitigation**

1. Record observed cloud state with authenticated events, revision checks, immutable job identity, and explicit finalization/uncertainty status.
2. On cancellation/deadline, request actual cancellation and reconcile until externally confirmed terminal; keep uncertain jobs accounted for.
3. Run an independent recovery reconciler and alert on orphaned/stopping jobs; reject contradictory or replayed transitions.

**Evidence required before reassessment**

- **Positive:** Observed job state, artifact readiness, and platform state agree; a canceled job is confirmed terminal.
- **Negative:** Drop/replay callbacks, stop the workflow, lose poll/cancel responses, and delay artifacts; no false running/success/stopped claim or unaccounted live job follows.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Cloud outages and delayed cancellation can prolong resource use; correct uncertainty handling does not guarantee instantaneous shutdown.

**Traceability:** [Original MR-09 assessment](MODULE_CHANGE_RISKS.md#mr-09) and [TP-086](REQUIREMENTS.md#tp-086), [TP-089](REQUIREMENTS.md#tp-089), [TP-123](REQUIREMENTS.md#tp-123).

**Implementation/verification status:** planned; current detection remains 0.25.

<a id="mr-10"></a>

### MR-10 — Budget policy is bypassed across modules

**Owner:** Budget admission service and platform operator.

**Current risk 0.56 → target residual risk 0.20**

Target likelihood 0.25 × retained impact 0.80.

**Mitigation**

1. Use operator-owned conservative price profiles covering preparation/scans, builds, GPU/CPU/disk, retries, transfers, storage, logs, and retained resources.
2. Reserve campaign and per-run estimated exposure atomically before dispatch; recheck changed shapes and account for uncertain jobs and concurrent work.
3. Enforce execution/scan/scale/duration limits, headroom, cleanup policy, and delayed billing reconciliation; reject unknown/unbounded cost.

**Evidence required before reassessment**

- **Positive:** Eligible work has a complete trusted estimate and reservation within the accepted campaign policy.
- **Negative:** Submit zero caller prices, omit a resource category, alter shapes after review, exhaust retry limits, or race reservations; paid dispatch is denied and reservations remain coherent.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Provider billing lag, price changes, retained storage, and delayed cancellation remain uncertain; an estimate/reservation is not an instantaneous hard bill cap.

**Traceability:** [Original MR-10 assessment](MODULE_CHANGE_RISKS.md#mr-10) and [TP-080](REQUIREMENTS.md#tp-080), [TP-081](REQUIREMENTS.md#tp-081), [TP-082](REQUIREMENTS.md#tp-082).

**Implementation/verification status:** planned; current detection remains 0.35.

<a id="mr-11"></a>

### MR-11 — Lineage is lost or assigned to the wrong model

**Owner:** Primary integration owner, result finalizer, and MLflow/release adapters.

**Current risk 0.49 → target residual risk 0.14**

Target likelihood 0.20 × retained impact 0.70.

**Mitigation**

1. Define one immutable lineage chain spanning object, dataset, weights, runtime, experiment/run, result digest, registry version, and release.
2. Generate metadata from accepted records instead of independently supplied copies; reconcile references and checksums at each handoff.
3. Reject missing/mismatched lineage before candidate eligibility, promotion, or serving; preserve an append-only evidence index.

**Evidence required before reassessment**

- **Positive:** An object/run/model/release can be traced end to end to the exact accepted artifacts.
- **Negative:** Swap each ID/hash, replay another object's result, or lose a metadata write; no wrongly attributed eligible candidate or release is produced.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** A shared verifier bug or compromised authoritative records can make internally consistent but false lineage.

**Traceability:** [Original MR-11 assessment](MODULE_CHANGE_RISKS.md#mr-11) and [TP-022](REQUIREMENTS.md#tp-022), [TP-048](REQUIREMENTS.md#tp-048), [TP-096](REQUIREMENTS.md#tp-096).

**Implementation/verification status:** planned; current detection remains 0.50.

<a id="mr-12"></a>

### MR-12 — Wrong or unapproved model reaches serving

**Owner:** Authenticated human approval service, release controller, and serving operator.

**Current risk 0.63 → target residual risk 0.18**

Target likelihood 0.20 × retained impact 0.90.

**Mitigation**

1. Bind explicit authenticated human approval to the exact staged release fingerprint and scope; candidate writers cannot change production aliases.
2. Pin immutable model artifact/version and serving digest; verify readiness and model identity before traffic cutover.
3. Serialize release transitions and persist recoverable intent; reconcile alias, deployment, traffic, and ledger, with verified rollback to a prior approved release.

**Evidence required before reassessment**

- **Positive:** An approved exact model serves healthy inference, and an approved rollback restores the expected prior model.
- **Negative:** Forge/replay approval, change aliases or artifacts, race promotions/rollback, and fail every cutover stage; no unapproved traffic or falsely completed release remains.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Control-service compromise, hidden health defects, and multi-system outages can still cause temporary inconsistency or disruption.

**Traceability:** [Original MR-12 assessment](MODULE_CHANGE_RISKS.md#mr-12) and [TP-104](REQUIREMENTS.md#tp-104), [TP-106](REQUIREMENTS.md#tp-106), [TP-112](REQUIREMENTS.md#tp-112).

**Implementation/verification status:** planned; current detection remains 0.25.

<a id="mr-13"></a>

### MR-13 — Telemetry cannot expose a regression

**Owner:** Shared telemetry library, module owners, and observability operator.

**Current risk 0.48 → target residual risk 0.15**

Target likelihood 0.25 × retained impact 0.60.

**Mitigation**

1. Require a versioned event schema and propagated object/run/job/dataset/runtime/release/trace context at every boundary.
2. Redact secrets/private payloads; emit structured stdout and configure bounded export/buffering with explicit delivery gaps.
3. Gate live integration on searchable Loki/collector evidence, correlated traces/metrics, independent failure signals, and context-isolation checks.

**Evidence required before reassessment**

- **Positive:** All required lifecycle and failure events are correctly correlated at the configured sink without secret canaries.
- **Negative:** Lose the collector, drop events, mix concurrent contexts, insert secret canaries, and exhaust buffers; detect the gaps/violations without false delivered status or unbounded resources.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Instrumentation omissions, sink outages, and retention gaps can still remove evidence; logs do not substitute for independent state checks.

**Traceability:** [Original MR-13 assessment](MODULE_CHANGE_RISKS.md#mr-13) and [TP-118](REQUIREMENTS.md#tp-118), [TP-119](REQUIREMENTS.md#tp-119), [TP-124](REQUIREMENTS.md#tp-124).

**Implementation/verification status:** planned; current detection remains 0.20.

<a id="mr-14"></a>

### MR-14 — Mixed versions or migration damage historical state

**Owner:** Contract integration owner and state-store/operator.

**Current risk 0.56 → target residual risk 0.20**

Target likelihood 0.25 × retained impact 0.80.

**Mitigation**

1. Maintain versioned schemas, compatibility rules, and explicit transformations; preserve immutable accepted records and provenance.
2. Use additive rollout before retiring old readers; bind active runs to their original supported contracts and prevent silent reinterpretation.
3. Back up state, dry-run historical/active-run replay in isolation, and verify restore/reconciliation before irreversible migration.

**Evidence required before reassessment**

- **Positive:** Supported old/new pairs and restored active runs preserve identity and behavior without duplicate work.
- **Negative:** Introduce unsupported versions, fail mid-migration, replay stale state, and resume active runs; no accepted history is overwritten or completed job resubmitted.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Undocumented historical records, rollback limitations, and uncommon active-run states can still complicate recovery.

**Traceability:** [Original MR-14 assessment](MODULE_CHANGE_RISKS.md#mr-14) and [TP-022](REQUIREMENTS.md#tp-022), [TP-023](REQUIREMENTS.md#tp-023), [TP-142](REQUIREMENTS.md#tp-142).

**Implementation/verification status:** planned; current detection remains 0.20.

<a id="mr-15"></a>

### MR-15 — Guided setup and navigation become confusing or broken

**Owner:** CLI/template/navigation owner and project reviewer.

**Current risk 0.35 → target residual risk 0.10**

Target likelihood 0.20 × retained impact 0.50.

**Mitigation**

1. Preserve one guided entry point and generate readable project/run/model indexes from authoritative records.
2. Version templates and validate configuration/link/command shapes; expose actionable failures without leaking module plumbing into routine use.
3. Exercise realistic briefs and human navigation tasks from setup through review, status, artifacts, and recovery.

**Evidence required before reassessment**

- **Positive:** A non-programmer can complete the documented flow and locate all expected evidence.
- **Negative:** Break a template/link, remove a required field, change a command, or interrupt setup; validation catches it and the user receives a truthful next action.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Ambiguous instructions and user expectations cannot be fully captured by automated fixture checks.

**Traceability:** [Original MR-15 assessment](MODULE_CHANGE_RISKS.md#mr-15) and [TP-007](REQUIREMENTS.md#tp-007), [TP-015](REQUIREMENTS.md#tp-015), [TP-024](REQUIREMENTS.md#tp-024).

**Implementation/verification status:** planned; current detection remains 0.60.

<a id="mr-16"></a>

### MR-16 — Execution waiting returns to the agent

**Owner:** Control-plane and GCP Workflows owners.

**Current risk 0.40 → target residual risk 0.12**

Target likelihood 0.15 × retained impact 0.80.

**Mitigation**

1. Persist setup/accepted intent and dispatch through runtime services with durable workflow/event handoffs.
2. Give the assistant proposal capabilities only; require no agent session, polling loop, or execution lease after admission.
3. Prove detached execution and recovery after CLI, assistant, and control-service interruption.

**Evidence required before reassessment**

- **Positive:** An admitted job runs/finalizes or reconciles while the assistant and CLI are closed.
- **Negative:** Terminate the agent and restart services mid-run; no repeated agent questions, token-consuming wait loop, or lost execution authority occurs.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Workflow/control-service availability can delay recovery, though it should not require assistant participation.

**Traceability:** [Original MR-16 assessment](MODULE_CHANGE_RISKS.md#mr-16) and [TP-003](REQUIREMENTS.md#tp-003), [TP-084](REQUIREMENTS.md#tp-084), [TP-087](REQUIREMENTS.md#tp-087).

**Implementation/verification status:** planned; current detection remains 0.45.

<a id="mr-17"></a>

### MR-17 — Extra artifact transfers increase cost or latency

**Owner:** Dataset/trainer/runtime owners and cost/performance operator.

**Current risk 0.30 → target residual risk 0.18**

Target likelihood 0.30 × retained impact 0.60.

**Mitigation**

1. Pass immutable references across modules instead of copying bulk artifacts; use bounded streaming, shard sizing, and checksum-keyed caching where allowed.
2. Avoid redundant full reads while preserving integrity; enforce transfer, memory, concurrency, timeout, and scale limits.
3. Benchmark representative workloads against accepted latency/resource/cost profiles and record regressions before production release.

**Evidence required before reassessment**

- **Positive:** Measured representative workloads meet the accepted profile with traceable transfer/resource usage.
- **Negative:** Increase artifact sizes/concurrency, force cache misses, or introduce duplicate downloads; enforce limits and fail the performance/cost gate.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Real input distributions, cache behavior, cloud contention, and variable network/storage costs keep this target uncertain.

**Traceability:** [Original MR-17 assessment](MODULE_CHANGE_RISKS.md#mr-17) and [TP-035](REQUIREMENTS.md#tp-035), [TP-115](REQUIREMENTS.md#tp-115), [TP-144](REQUIREMENTS.md#tp-144).

**Implementation/verification status:** planned; current detection remains 0.20.

<a id="mr-18"></a>

### MR-18 — Failures are attributed to the wrong owner

**Owner:** Failure taxonomy owner and each owning engineering role.

**Current risk 0.36 → target residual risk 0.12**

Target likelihood 0.20 × retained impact 0.60.

**Mitigation**

1. Use typed failure records containing stage, owner, retryability, original cause, artifact/job identity, and permitted next action.
2. Preserve causes through adapters; unknown errors remain explicitly unclassified and cannot authorize cross-role repair.
3. Gate retries/repair/build actions on classification plus actor capability; test failures through real module boundaries.

**Evidence required before reassessment**

- **Positive:** Each known fault reaches the correct owner with usable original evidence and a permitted response.
- **Negative:** Wrap/mislabel errors or present trainer faults as packaging/quota faults; no unauthorized edit, blind retry, or unrelated rebuild follows.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Novel failures and ambiguous external messages still need human diagnosis.

**Traceability:** [Original MR-18 assessment](MODULE_CHANGE_RISKS.md#mr-18) and [TP-076](REQUIREMENTS.md#tp-076), [TP-088](REQUIREMENTS.md#tp-088), [TP-122](REQUIREMENTS.md#tp-122).

**Implementation/verification status:** planned; current detection remains 0.65.

<a id="mr-19"></a>

### MR-19 — Untrusted model artifact executes code or escapes its path

**Owner:** Trainer/model-loader, runtime sandbox, and artifact policy owners.

**Current risk 0.50 → target residual risk 0.15**

Target likelihood 0.15 × retained impact 1.00.

**Mitigation**

1. Accept approved non-executable tensor artifacts with trusted provenance and integrity verification before deserialization; reject unsafe/unapproved formats and remote code.
2. Enforce contained download/extraction paths, source allowlists, byte/dimension/resource limits, and bundle schema.
3. Load/infer under least privilege with scoped credentials and constrained filesystem/network access; verify malicious canaries in an isolated environment.

**Evidence required before reassessment**

- **Positive:** Approved portable bundles load and infer with the intended resource/identity restrictions.
- **Negative:** Use substituted checkpoint content, unsafe serializer canaries, traversal paths, and hostile image data; reject before execution/escaped writes and inspect filesystem/credentials afterward.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** Parser vulnerabilities, dependency supply-chain compromise, and sandbox defects remain possible with severe consequences.

**Traceability:** [Original MR-19 assessment](MODULE_CHANGE_RISKS.md#mr-19) and [TP-028](REQUIREMENTS.md#tp-028), [TP-062](REQUIREMENTS.md#tp-062), [TP-136](REQUIREMENTS.md#tp-136).

**Implementation/verification status:** planned; current detection remains 0.20.

<a id="mr-20"></a>

### MR-20 — Infrastructure capabilities are stale or only desired state

**Owner:** Infrastructure operator and readiness/admission service.

**Current risk 0.56 → target residual risk 0.20**

Target likelihood 0.25 × retained impact 0.80.

**Mitigation**

1. Separate desired deployment configuration from observed capability output; record environment/resource revisions, identity, endpoints, policy scope, and observation time.
2. Validate OpenTofu/providers/plans and reconcile actual resources/drift; restrict trusted publication of capability records.
3. Recheck readiness/access and relevant quota/constraints at admission; stale/unknown facts fail closed or enter explicit uncertainty with bounded recovery.

**Evidence required before reassessment**

- **Positive:** The selected environment and execution identity match observed resources and can perform the intended bounded operation.
- **Negative:** Revoke access, alter endpoints/regions/resources, change quotas, or replay stale capabilities; revalidation detects drift before unauthorized work or records capacity uncertainty.
- **Evidence binding:** Record exact source/image/configuration/environment/principal, procedure, expected/observed outcome, artifact integrity, independent effects, and reviewer.

**Why risk remains:** GPU quota does not guarantee available capacity; cloud drift and availability can change after a successful preflight.

**Traceability:** [Original MR-20 assessment](MODULE_CHANGE_RISKS.md#mr-20) and [TP-018](REQUIREMENTS.md#tp-018), [TP-091](REQUIREMENTS.md#tp-091), [TP-140](REQUIREMENTS.md#tp-140).

**Implementation/verification status:** planned; current detection remains 0.20.

## 4. Implementation order and dependencies

### A. Establish authoritative contracts and accepted artifacts

Implement schema/compatibility boundaries (MR-01), semantic identity (MR-02), immutable dataset/output publication (MR-03/MR-04), trusted lineage (MR-11), and safe artifact loading (MR-19).

The controller must distinguish proposals from accepted evidence and desired infrastructure from observed capabilities. A typed record alone cannot establish those facts.

### B. Establish hard authority and runtime identity

Implement capability/IAM/service boundaries (MR-07), pinned packaging/certification (MR-05/MR-06), trusted budget reservations (MR-10), and current environment readiness (MR-20).

Agent hooks supplement these controls. Operational services must enforce them even when a caller uses another tool or editor. Live permission evidence requires actual intended principals.

### C. Establish durable execution and recovery

Implement dispatch reconciliation (MR-08), observed state/cancellation (MR-09), compatibility/migration recovery (MR-14), runtime-owned detached execution (MR-16), and scoped failure handling (MR-18).

An external creation call is not atomic with the local database by default. Uncertain creation must block blind redispatch. Cancellation remains uncertain until the external job is confirmed terminal.

### D. Establish approved deployment and usable operation

Implement authenticated exact-release approval, readiness/cutover/rollback (MR-12), complete delivered telemetry (MR-13), readable navigation (MR-15), and measured resource/performance profiles (MR-17).

Traceability and telemetry apply throughout the earlier phases as well; they cannot be deferred until after unobserved paid execution.

## 5. Gates for claiming mitigation effectiveness

For each risk:

1. The named owner implements every control and documents its enforcing boundary.
2. Public input/output contracts declare schema, meaning, authority, integrity, completion, freshness, failure, and retry behavior.
3. Positive counterpart checks pass, showing that the environment is not merely broken or inaccessible.
4. Negative/fault/concurrency/migration checks pass at the scopes the risk requires. Deliberately introduced faults are caught by the intended gate.
5. Independent state/audit checks show no unauthorized jobs, writes, approvals, traffic changes, escaped files/credentials, or false terminal claims.
6. Affected current-source/current-digest integration and live acceptance passes. Historical probes and mocks cannot substitute.
7. Remaining failure scenarios and any contradictory evidence are reviewed.
8. The owner reassesses likelihood, impact, and detection from the actual results. Record the accepted residual rating and evidence references separately from this target.

These scores do not replace the [full requirements](REQUIREMENTS.md) or its completion rule. A low target cannot excuse an unimplemented control or missing live acceptance.

## 6. Residual concerns requiring particular attention

- **Semantic correctness (target 0.27):** fingerprints detect identity changes, while representative evaluation and review are still needed to judge whether the meaning is correct.
- **Budgets (target 0.20):** reservation and execution limits reduce exposure; delayed billing, price changes, cancellation delays, and retained resources remain uncertain.
- **Mixed versions (target 0.20):** supported replay/migration coverage cannot assume every historical record or active state was represented.
- **Infrastructure readiness (target 0.20):** observed readiness can become stale, and quota does not guarantee immediate GPU capacity.
- **Severe security outcomes:** impact stays high for authority crossing (MR-07), wrong release (MR-12), and unsafe loading (MR-19), even with a lower target likelihood.

Targets must be reviewed against observed behavior. They should not be added or averaged into an overall incident probability, and score reductions should not be described as measured percentage reductions in failures.

## 7. Present limits and authorization

No application code, module boundary, IAM policy, runtime image, or cloud resource is changed by this documentation task. No fresh application test run, mutation campaign, or paid live acceptance is claimed.

The latest recorded local suite is 55 passing tests. Full current-source GPU training, deployed workflow recovery, live principal denial, MLflow, GKE serving, and Loki evidence remain incomplete.

Live verification needs resolved settings/endpoints/weights, an accepted operational profile, and an authorized campaign plan. The existing USD 5 test ceiling does not authorize unlimited implementation validation or deployment.

Related documents: [Risk assessment](MODULE_CHANGE_RISKS.md), [Requirements](REQUIREMENTS.md), [Testing status](TEST_STATUS.md), [Acceptance](ACCEPTANCE.md).
