# Module Separation — Risks and Regression Detection

**Date:** September 28, 2026

**Reviewed implementation:** `5a7817f5449162c1329fb2af348c99c8477f760c`

**Status:** Prospective assessment. The module refactor and its verification campaigns have not been executed by this assessment.

## 1. Decision being assessed

Make **datasets**, **training**, and **infrastructure** explicit modules with versioned input/output contracts in one repository. Keep the durable control plane responsible for admission, budgets, dispatch, retries, cancellation, reconciliation, and run state. Keep model approval/release/serving authority separate.

The assessed handoffs are:

- Dataset request → accepted immutable `DatasetVersion` plus validation/provenance evidence.
- Training request → verified `TrainingResult` containing model, evaluation, calibration, diagnostics, and lineage.
- Infrastructure request → observed `EnvironmentCapabilities` containing endpoints, identities, resource references, execution constraints, deployment revision, and timestamped readiness.

The product scope, model algorithm, and accepted data semantics are not intended to change. Introducing independent network services, separate repositories, or a new production environment would require an additional assessment because it adds deployment and distributed-system risks.

Several risks already exist. Explicit module handoffs can expose or enlarge them; a new typed record does not make an untrusted fact authoritative.

## 2. What the scores mean

### Risk score, R

`R = L × I`

- **L — likelihood:** relative likelihood of a regression during this migration and first integrated release.
- **I — impact:** relative consequence for model correctness, authority, paid execution, recovery, or user operation.
- **R:** 0 means negligible assessed risk; 1 means maximum assessed likelihood and impact.

L and I are subjective normalized engineering ratings. Their product is a prioritization index, **not a calibrated probability**, measured failure frequency, or financial-loss forecast.

### Detection score, D

D rates our **current evidence-backed ability to detect the specific regression before acceptance**, at the local or live scope it requires. Higher is better.

| D anchor | Interpretation |
| --- | --- |
| 0.00 | No credible detection mechanism |
| 0.20 | Mostly inspection, indirect symptoms, or limited mocks; key checks missing |
| 0.40 | Some targeted automated checks; substantial boundary/environment gaps |
| 0.60 | Several relevant integration/adverse checks; important gaps remain |
| 0.80 | Broad demonstrated detection at required scopes; small remaining gaps |
| 1.00 | Ideal complete detection within a declared bounded fault model |

No 1.00 score is claimed. Intermediate ratings reflect the inspected coverage and remaining gaps; they are reviewer estimates rather than mutation-test sensitivity measurements. Differences of a few hundredths should not drive decisions.

A low score can mean a harmful regression might remain unnoticed even though ordinary tests pass. A high score does not mean the regression is prevented. Prevention requires an enforcing control plus denial and zero-forbidden-effect evidence.

### Assessment evidence

The current code, selected tests, [requirements](REQUIREMENTS.md), [testing status](TEST_STATUS.md), and [acceptance record](ACCEPTANCE.md) were inspected. The latest recorded local suite has **55 passing tests**. It was not rerun for this scoring task, and no new mutation campaign or live cloud test was conducted.

The previous live A100/GCS probe covers an older image and a generic CUDA check. It does not establish current DINOv3 training or full platform detection. Live IAM denial, durable deployed workflows, cloud MLflow, GKE serving, and Loki delivery remain incomplete.

The structured assessment is [module-change-risks.json](requirements/module-change-risks.json).

## 3. Risk and detection scores

| ID | Regression risk | Risk R | Detection D |
| --- | --- | ---: | ---: |
| [MR-01](#mr-01) | Contract or API incompatibility | 0.60 | 0.70 |
| [MR-02](#mr-02) | Silent changes to data or model meaning | 0.72 | 0.45 |
| [MR-03](#mr-03) | Invalid or substituted dataset consumed | 0.63 | 0.70 |
| [MR-04](#mr-04) | Incomplete training output accepted | 0.54 | 0.30 |
| [MR-05](#mr-05) | Runtime identity or certification becomes stale | 0.63 | 0.55 |
| [MR-06](#mr-06) | Module dependencies or packaging break | 0.48 | 0.70 |
| [MR-07](#mr-07) | Operational authority crosses module boundaries | 0.63 | 0.20 |
| [MR-08](#mr-08) | Retries create duplicate paid jobs | 0.54 | 0.30 |
| [MR-09](#mr-09) | Job state lies or cancellation leaves resources running | 0.54 | 0.25 |
| [MR-10](#mr-10) | Budget policy is bypassed across modules | 0.56 | 0.35 |
| [MR-11](#mr-11) | Lineage is lost or assigned to the wrong model | 0.49 | 0.50 |
| [MR-12](#mr-12) | Wrong or unapproved model reaches serving | 0.63 | 0.25 |
| [MR-13](#mr-13) | Telemetry cannot expose a regression | 0.48 | 0.20 |
| [MR-14](#mr-14) | Mixed versions or migration damage historical state | 0.56 | 0.20 |
| [MR-15](#mr-15) | Guided setup and navigation become confusing or broken | 0.35 | 0.60 |
| [MR-16](#mr-16) | Execution waiting returns to the agent | 0.40 | 0.45 |
| [MR-17](#mr-17) | Extra artifact transfers increase cost or latency | 0.30 | 0.20 |
| [MR-18](#mr-18) | Failures are attributed to the wrong owner | 0.36 | 0.65 |
| [MR-19](#mr-19) | Untrusted model artifact executes code or escapes its path | 0.50 | 0.20 |
| [MR-20](#mr-20) | Infrastructure capabilities are stale or only desired state | 0.56 | 0.20 |

## 4. Individual assessments

Every required verification below is **planned**, not claimed to have passed. Existing support and missing coverage are separate.

<a id="mr-01"></a>

### MR-01 — Contract or API incompatibility

- **Origin:** Introduced/enlarged.
- **Regression:** A producer changes fields, required values, enums, or serialization and a consumer cannot read its output.
- **Scores:** L 0.75 × I 0.80 = **risk 0.60**; **current detection 0.70**.
- **Present detection evidence:** Strict Pydantic records and current CLI/controller integration tests catch several shape errors.
- **What can escape detection:** No versioned producer/consumer compatibility matrix or historical-contract replay.
- **Required verification:** Run every supported producer/consumer contract pair; reject unsupported versions before any job or write.
- **Traceability:** [TP-016](REQUIREMENTS.md#tp-016), [TP-023](REQUIREMENTS.md#tp-023). Inspected implementation/test: [tests/test_control.py](../tests/test_control.py).

<a id="mr-02"></a>

### MR-02 — Silent changes to data or model meaning

- **Origin:** Introduced/enlarged.
- **Regression:** Class order, label meaning, split policy, preprocessing, pooling, or metric selection changes while the record still validates.
- **Scores:** L 0.80 × I 0.90 = **risk 0.72**; **current detection 0.45**.
- **Present detection evidence:** Tests cover class matching, deterministic splits, DINOv3 pooling/gradients, and MCC/F1 selection.
- **What can escape detection:** No unified semantic fingerprint or complete trainer-to-serving comparison.
- **Required verification:** Use a fixed dataset/model bundle to compare labels, splits, preprocessing, selected checkpoint, predictions, and review flags; deliberately reorder classes and change preprocessing to prove rejection.
- **Traceability:** [TP-030](REQUIREMENTS.md#tp-030), [TP-041](REQUIREMENTS.md#tp-041), [TP-054](REQUIREMENTS.md#tp-054). Inspected implementation/test: [tests/test_dinov3_integration.py](../tests/test_dinov3_integration.py).

<a id="mr-03"></a>

### MR-03 — Invalid or substituted dataset consumed

- **Origin:** Existing, exposed at handoff.
- **Regression:** Training trusts a dataset reference whose shards, source evidence, object binding, or accepted labels changed.
- **Scores:** L 0.70 × I 0.90 = **risk 0.63**; **current detection 0.70**.
- **Present detection evidence:** Local checksum, commit-marker, class, source, manifest, and shard tampering cases are tested.
- **What can escape detection:** Live substitution, stale-review, and check-to-use races remain unverified.
- **Required verification:** Mutate each accepted artifact and swap object/version references; verify rejection and zero Vertex jobs, including a live GCS race case.
- **Traceability:** [TP-033](REQUIREMENTS.md#tp-033), [TP-044](REQUIREMENTS.md#tp-044), [TP-046](REQUIREMENTS.md#tp-046). Inspected implementation/test: [tests/test_dataset.py](../tests/test_dataset.py).

<a id="mr-04"></a>

### MR-04 — Incomplete training output accepted

- **Origin:** Existing, enlarged by new output contract.
- **Regression:** A partial upload or mismatched model/metrics/calibration bundle becomes a successful TrainingResult or registry candidate.
- **Scores:** L 0.60 × I 0.90 = **risk 0.54**; **current detection 0.30**.
- **Present detection evidence:** Local training/reload and candidate registration tests cover successful outputs.
- **What can escape detection:** Trainer uploads files individually; no complete verified output-commit protocol is established.
- **Required verification:** Interrupt uploads at each artifact, corrupt model/metrics metadata, and return premature job success; no accepted result, candidate eligibility, or release may follow.
- **Traceability:** [TP-061](REQUIREMENTS.md#tp-061), [TP-092](REQUIREMENTS.md#tp-092), [TP-099](REQUIREMENTS.md#tp-099). Inspected implementation/test: [src/defect_platform/trainer/runner.py](../src/defect_platform/trainer/runner.py).

<a id="mr-05"></a>

### MR-05 — Runtime identity or certification becomes stale

- **Origin:** Existing, enlarged by packaging changes.
- **Regression:** Moved code/dependencies run under certification for a different image or incompatible hardware.
- **Scores:** L 0.70 × I 0.90 = **risk 0.63**; **current detection 0.55**.
- **Present detection evidence:** Exact-digest gate tests and actual local DINOv3 regression tests exist; historical A100 probe covers its old digest.
- **What can escape detection:** Current fixed trainer image lacks live GPU certification; record authenticity and compatibility scope are incomplete.
- **Required verification:** Build the moved source once; validate actual DINOv3 in that exact image; reject mismatched/revoked hardware scope; intercept build/install APIs during ordinary submission.
- **Traceability:** [TP-067](REQUIREMENTS.md#tp-067), [TP-070](REQUIREMENTS.md#tp-070), [TP-073](REQUIREMENTS.md#tp-073). Inspected implementation/test: [tests/test_runtime_release.py](../tests/test_runtime_release.py).

<a id="mr-06"></a>

### MR-06 — Module dependencies or packaging break

- **Origin:** Introduced/enlarged.
- **Regression:** Imports, extras, entry points, templates, or installed package paths work in the checkout but fail in isolated images.
- **Scores:** L 0.80 × I 0.60 = **risk 0.48**; **current detection 0.70**.
- **Present detection evidence:** Existing local suite exercises imports and entry points; dependencies are locked.
- **What can escape detection:** No acceptance of new isolated module packaging, current image, or every component's clean environment.
- **Required verification:** Install each allowed module dependency set in a clean environment; run public entry points and full fixture integration; deny imports/capabilities outside the declared dependency direction.
- **Traceability:** [TP-017](REQUIREMENTS.md#tp-017), [TP-065](REQUIREMENTS.md#tp-065), [TP-139](REQUIREMENTS.md#tp-139). Inspected implementation/test: [pyproject.toml](../pyproject.toml).

<a id="mr-07"></a>

### MR-07 — Operational authority crosses module boundaries

- **Origin:** Existing, potentially disguised by typed contracts.
- **Regression:** An assistant/client/module supplies forged certification or approval, or gains submission/build/storage permissions it should not have.
- **Scores:** L 0.70 × I 0.90 = **risk 0.63**; **current detection 0.20**.
- **Present detection evidence:** Direct role-guard cases and certification field validation exist.
- **What can escape detection:** Client runtime overrides remain; regex checks are bypassable; actual service-account/harness denial is unverified.
- **Required verification:** Forge records and use direct, SDK, subprocess, and alternate endpoint paths under each identity; independently inspect jobs, writes, aliases, and grants for zero forbidden effects.
- **Traceability:** [TP-070](REQUIREMENTS.md#tp-070), [TP-093](REQUIREMENTS.md#tp-093), [TP-133](REQUIREMENTS.md#tp-133). Inspected implementation/test: [tests/test_role_guard.py](../tests/test_role_guard.py).

<a id="mr-08"></a>

### MR-08 — Retries create duplicate paid jobs

- **Origin:** Existing, enlarged by additional handoffs.
- **Regression:** Two module/controller requests or an uncertain acknowledgement dispatch the same intent twice.
- **Scores:** L 0.60 × I 0.90 = **risk 0.54**; **current detection 0.30**.
- **Present detection evidence:** Sequential idempotency, changed-key rejection, and fake Vertex lookup recovery are tested.
- **What can escape detection:** Concurrent/crash windows and distributed dispatch ownership are unverified.
- **Required verification:** Race identical requests and crash immediately before/after external creation; reconcile one run, one job, one reservation, and one result.
- **Traceability:** [TP-083](REQUIREMENTS.md#tp-083), [TP-085](REQUIREMENTS.md#tp-085), [TP-090](REQUIREMENTS.md#tp-090). Inspected implementation/test: [tests/test_control.py](../tests/test_control.py).

<a id="mr-09"></a>

### MR-09 — Job state lies or cancellation leaves resources running

- **Origin:** Existing, enlarged by state distribution.
- **Regression:** One module marks success/failure/cancellation while another has unfinished outputs or a still-running cloud job.
- **Scores:** L 0.60 × I 0.90 = **risk 0.54**; **current detection 0.25**.
- **Present detection evidence:** Some transition rejection and workflow callback tests exist; historical smoke cancellations were observed manually.
- **What can escape detection:** Workflow marks running on submission and records timeout/poll failure without canceling/reconciling Vertex.
- **Required verification:** Lose callbacks, expire a deadline, stop the workflow, and delay outputs; compare state to actual Vertex and artifact readiness until reconciled.
- **Traceability:** [TP-086](REQUIREMENTS.md#tp-086), [TP-089](REQUIREMENTS.md#tp-089), [TP-123](REQUIREMENTS.md#tp-123). Inspected implementation/test: [infra/workflow.yaml](../infra/workflow.yaml).

<a id="mr-10"></a>

### MR-10 — Budget policy is bypassed across modules

- **Origin:** Existing, enlarged by split cost ownership.
- **Regression:** Dataset preparation, scans, builds, retries, GPU work, or retained resources fall outside the admitted estimate/reservation.
- **Scores:** L 0.70 × I 0.80 = **risk 0.56**; **current detection 0.35**.
- **Present detection evidence:** Local configured per-run over-cap rejection is tested.
- **What can escape detection:** Price is caller-provided; aggregate trusted reservations and ancillary costs are not enforced.
- **Required verification:** Inject a zero rate, omitted cost, changed shape, and simultaneous reservations; verify no paid dispatch and reconcile campaign exposure.
- **Traceability:** [TP-080](REQUIREMENTS.md#tp-080), [TP-081](REQUIREMENTS.md#tp-081), [TP-082](REQUIREMENTS.md#tp-082). Inspected implementation/test: [src/defect_platform/control/controller.py](../src/defect_platform/control/controller.py).

<a id="mr-11"></a>

### MR-11 — Lineage is lost or assigned to the wrong model

- **Origin:** Introduced/enlarged.
- **Regression:** Dataset, weights, runtime, run, MLflow version, and release references diverge across handoffs.
- **Scores:** L 0.70 × I 0.70 = **risk 0.49**; **current detection 0.50**.
- **Present detection evidence:** Dataset provenance, controller snapshots, portable bundles, and local MLflow reuse provide partial checks.
- **What can escape detection:** No complete cross-service integrity-bound lineage reconciliation.
- **Required verification:** Trace one immutable identity chain end to end; deliberately swap each ID/digest and ensure registration/release refuses the mismatch.
- **Traceability:** [TP-022](REQUIREMENTS.md#tp-022), [TP-048](REQUIREMENTS.md#tp-048), [TP-096](REQUIREMENTS.md#tp-096). Inspected implementation/test: [tests/test_mlflow_registration.py](../tests/test_mlflow_registration.py).

<a id="mr-12"></a>

### MR-12 — Wrong or unapproved model reaches serving

- **Origin:** Existing, enlarged by deployment contracts.
- **Regression:** Infrastructure treats a deployment request as approval, or alias/deployment/traffic/ledger disagree.
- **Scores:** L 0.70 × I 0.90 = **risk 0.63**; **current detection 0.25**.
- **Present detection evidence:** Local missing-approver and selected promotion/rollback compensation tests exist.
- **What can escape detection:** Approval is text, serving resolves mutable champion, and live readiness/concurrent/crash release checks are absent.
- **Required verification:** Use authenticated approval bound to an exact release; race promotions, change aliases, fail each cutover stage, and verify approved model identity plus rollback.
- **Traceability:** [TP-104](REQUIREMENTS.md#tp-104), [TP-106](REQUIREMENTS.md#tp-106), [TP-112](REQUIREMENTS.md#tp-112). Inspected implementation/test: [tests/test_release_lifecycle.py](../tests/test_release_lifecycle.py).

<a id="mr-13"></a>

### MR-13 — Telemetry cannot expose a regression

- **Origin:** Introduced/enlarged.
- **Regression:** Module handoffs lose IDs, omit failures, leak context, or report delivery although evidence never reaches the sink.
- **Scores:** L 0.80 × I 0.60 = **risk 0.48**; **current detection 0.20**.
- **Present detection evidence:** Structured JSON/context code and old probe Cloud Logging observations exist.
- **What can escape detection:** No telemetry-specific suite or live Loki end-to-end acceptance.
- **Required verification:** Trace a run across all modules, inject concurrent context and collector failures, and query the real sink for correctly attributed events and explicit gaps.
- **Traceability:** [TP-118](REQUIREMENTS.md#tp-118), [TP-119](REQUIREMENTS.md#tp-119), [TP-124](REQUIREMENTS.md#tp-124). Inspected implementation/test: [src/defect_platform/telemetry.py](../src/defect_platform/telemetry.py).

<a id="mr-14"></a>

### MR-14 — Mixed versions or migration damage historical state

- **Origin:** Introduced/enlarged.
- **Regression:** Old records, active runs, or independently upgraded modules become unreadable or are reinterpreted incorrectly.
- **Scores:** L 0.70 × I 0.80 = **risk 0.56**; **current detection 0.20**.
- **Present detection evidence:** Strict present-day schemas exist.
- **What can escape detection:** No schema migration, mixed-version rollout, or historical replay campaign.
- **Required verification:** Replay historical records and simulate old/new producer/consumer combinations with active runs; reject unsupported transitions without rewriting accepted history.
- **Traceability:** [TP-022](REQUIREMENTS.md#tp-022), [TP-023](REQUIREMENTS.md#tp-023), [TP-142](REQUIREMENTS.md#tp-142). Inspected implementation/test: [src/defect_platform/contracts.py](../src/defect_platform/contracts.py).

<a id="mr-15"></a>

### MR-15 — Guided setup and navigation become confusing or broken

- **Origin:** Introduced/enlarged.
- **Regression:** Module separation leaks implementation details, changes command paths, or leaves missing project/run/artifact links.
- **Scores:** L 0.70 × I 0.50 = **risk 0.35**; **current detection 0.60**.
- **Present detection evidence:** A guided CLI fixture checks project creation and returned run ID; templates and docs are present.
- **What can escape detection:** Complete generated artifact indexes and actual non-programmer navigation are unverified.
- **Required verification:** Run the same plain-language brief through setup, review, submission, status, artifact discovery, and failure recovery; verify all generated links.
- **Traceability:** [TP-007](REQUIREMENTS.md#tp-007), [TP-015](REQUIREMENTS.md#tp-015), [TP-024](REQUIREMENTS.md#tp-024). Inspected implementation/test: [tests/test_guided_cli.py](../tests/test_guided_cli.py).

<a id="mr-16"></a>

### MR-16 — Execution waiting returns to the agent

- **Origin:** Introduced/enlarged.
- **Regression:** Modules call one another through an agent session rather than durable runtime handoffs.
- **Scores:** L 0.50 × I 0.80 = **risk 0.40**; **current detection 0.45**.
- **Present detection evidence:** Workflow template owns polling and local control tests use a workflow handoff.
- **What can escape detection:** No deployed agent-disconnect or durable resumption acceptance.
- **Required verification:** Close the CLI and agent after admission, interrupt the control service, and verify the workflow finishes/reconciles without assistant polling or repeated setup.
- **Traceability:** [TP-003](REQUIREMENTS.md#tp-003), [TP-084](REQUIREMENTS.md#tp-084), [TP-087](REQUIREMENTS.md#tp-087). Inspected implementation/test: [infra/workflow.yaml](../infra/workflow.yaml).

<a id="mr-17"></a>

### MR-17 — Extra artifact transfers increase cost or latency

- **Origin:** Introduced/enlarged.
- **Regression:** Module boundaries copy large data/models repeatedly or add unbounded validation/download work.
- **Scores:** L 0.50 × I 0.60 = **risk 0.30**; **current detection 0.20**.
- **Present detection evidence:** Shard policies and several timeout settings exist.
- **What can escape detection:** No representative load/transfer benchmark, accepted SLO profile, or complete usage telemetry.
- **Required verification:** Measure representative bytes, downloads, memory, stage duration, and cost against an accepted baseline; inject over-limit input and observe bounded rejection.
- **Traceability:** [TP-035](REQUIREMENTS.md#tp-035), [TP-115](REQUIREMENTS.md#tp-115), [TP-144](REQUIREMENTS.md#tp-144). Inspected implementation/test: [src/defect_platform/trainer/model.py](../src/defect_platform/trainer/model.py).

<a id="mr-18"></a>

### MR-18 — Failures are attributed to the wrong owner

- **Origin:** Introduced/enlarged.
- **Regression:** A wrapper removes the original failure context and triggers the wrong repair, retry, or rebuild.
- **Scores:** L 0.60 × I 0.60 = **risk 0.36**; **current detection 0.65**.
- **Present detection evidence:** Selected failure owner/retryability tests exist; the DINOv3 fix was assigned to trainer code.
- **What can escape detection:** Complete cross-module failure taxonomy and recovery authorization are unverified.
- **Required verification:** Inject trainer, data, image, permission, quota, registry, and serving failures through real boundaries; verify owner, preserved evidence, and permitted next action.
- **Traceability:** [TP-076](REQUIREMENTS.md#tp-076), [TP-088](REQUIREMENTS.md#tp-088), [TP-122](REQUIREMENTS.md#tp-122). Inspected implementation/test: [tests/test_trainer_runtime.py](../tests/test_trainer_runtime.py).

<a id="mr-19"></a>

### MR-19 — Untrusted model artifact executes code or escapes its path

- **Origin:** Existing, enlarged by artifact trust boundary.
- **Regression:** A typed TrainingResult points to malicious checkpoint content or unsafe download paths accepted by another module.
- **Scores:** L 0.50 × I 1.00 = **risk 0.50**; **current detection 0.20**.
- **Present detection evidence:** Weight checksums and no-hub-fallback tests establish partial integrity behavior.
- **What can escape detection:** Inference uses weights_only=False; complete provenance-before-deserialization and path containment are not established.
- **Required verification:** Use safe malicious canaries, substituted bundles, and traversal names in an isolated loader; verify rejection before deserialization/writes and unchanged filesystem/credentials.
- **Traceability:** [TP-028](REQUIREMENTS.md#tp-028), [TP-062](REQUIREMENTS.md#tp-062), [TP-136](REQUIREMENTS.md#tp-136). Inspected implementation/test: [src/defect_platform/trainer/inference.py](../src/defect_platform/trainer/inference.py).

<a id="mr-20"></a>

### MR-20 — Infrastructure capabilities are stale or only desired state

- **Origin:** Existing, enlarged by infrastructure output contract.
- **Regression:** Training trusts endpoints, identities, quota, region, network, or readiness declared in config but not true in the environment.
- **Scores:** L 0.70 × I 0.80 = **risk 0.56**; **current detection 0.20**.
- **Present detection evidence:** Infrastructure templates and historical quota/API observations exist.
- **What can escape detection:** OpenTofu provider/deployment validation and current live readiness are unverified; historical GPU success is not current capacity.
- **Required verification:** Compare capability output with observed resource identities/revisions; revoke access/change endpoints/quota and verify revalidation denies or reports uncertainty before paid execution.
- **Traceability:** [TP-018](REQUIREMENTS.md#tp-018), [TP-091](REQUIREMENTS.md#tp-091), [TP-140](REQUIREMENTS.md#tp-140). Inspected implementation/test: [infra/main.tf](../infra/main.tf).

## 5. Where to improve detection first

Use `U = R × (1 − D)` only as a rough index of risk that current checks may miss. It is not a probability of an incident or expected financial loss. Correlated risks must not be added or averaged into an alleged total platform failure probability.

| Priority | Risk | U index | Required focus |
| --- | --- | ---: | --- |
| 1 | [MR-07](#mr-07) | 0.504 | Trusted server-side authority and real capability/identity denial checks |
| 2 | [MR-12](#mr-12) | 0.473 | Authenticated exact-release approval, pinned serving identity, verified cutover/rollback |
| 3 | [MR-14](#mr-14) | 0.448 | Schema/version compatibility and historical/active-run replay |
| 4 | [MR-20](#mr-20) | 0.448 | Observed infrastructure output with freshness, identity, and readiness revalidation |
| 5 | [MR-09](#mr-09) | 0.405 | Real job-state reconciliation and externally confirmed cancellation |

Silent semantic drift (MR-02), duplicate jobs (MR-08), incomplete outputs (MR-04), and budget enforcement (MR-10) also require dedicated migration gates. A favorable ordering does not waive any acceptance requirement.

## 6. Detecting a change versus detecting a regression

| Change layer | How we detect it | Present limitation |
| --- | --- | --- |
| Tracked source/configuration | Git review, dependency-lock changes, ownership checks | A diff shows edits, not their behavioral consequences; ignored/private data and live resource changes need other controls |
| Contract shape | Versioned schemas, serialized fixtures, producer/consumer compatibility checks | Present strict records lack complete version compatibility/migration checks |
| Contract meaning | Class/preprocessing/split/selection fingerprints and fixed semantic fixtures | Meaning can change while types and field names remain valid |
| Artifact identity | Dataset, weights, image, model, and manifest checksums with authenticated provenance | A checksum supplied by an untrusted caller is insufficient authority |
| Infrastructure state | Observed deployment/resource revisions, access checks, live readiness, audit evidence | Desired configuration and historical readiness are not current capability |
| Operational behavior | End-to-end, adverse, concurrent, crash, and live acceptance checks | Current simulated cloud tests do not prove deployed boundaries |

Module inputs and outputs should be fingerprinted and versioned. The control plane must record the exact handoff versions consumed and produced. Compatibility checks should identify the consumers affected by a contract change; operational evidence must identify the artifact/environment version actually tested.

## 7. Verification strategy for the migration

1. **Capture the existing behavior.** Preserve representative dataset/model/configuration fixtures and historical serialized records. Establish expected class order, splits, preprocessing, selection, predictions, lineage, and navigation.
2. **Introduce contracts before moving implementation.** Define producer, consumer, authoritative lookup, schema version, integrity, completion marker, failure semantics, allowed effects, and freshness for each handoff.
3. **Move one boundary at a time.** Keep the durable controller's authority stable. Compare old and new paths on the same fixtures, with supported version combinations.
4. **Test whether the checks notice faults.** In isolated test environments deliberately reorder labels, omit artifacts, forge certification, drop/replay events, duplicate dispatch, and change endpoints. A detector only earns credit when the deliberately introduced fault is caught with the intended evidence.
5. **Verify prevention independently.** For prohibited actions capture before/after state, explicit rejection, and audit evidence. Confirm the allowed counterpart works. An exception after creating a job does not prove prevention.
6. **Exercise required live boundaries.** Refresh settings/quota and use the intended principals for GCS, Vertex, MLflow, telemetry, and serving. A module split cannot be accepted as fully verified through cloud mocks.
7. **Update scores from observed evidence.** Record which defects were introduced, caught, escaped, or inconclusive. Separate detection before dispatch, before release, and after impact. Keep previous scores as dated assessments.

The current USD 5 test authorization is not authorization for an unrestricted migration/deployment campaign. This assessment launches no cloud resources.

## 8. Acceptance criteria and scope of confidence

- All relevant existing requirements remain applicable; module separation does not retire them.
- Supported contract/version combinations must pass compatibility and meaning checks.
- Outputs become usable only after complete verified publication.
- Operational authority comes from authenticated protected services/stores, with real denial proof.
- Retries, cancellations, uncertain outcomes, and active-run migrations require crash/concurrency acceptance.
- Current source/runtime and required live services must pass the full lifecycle and prevention gates.
- Score improvements require recorded observations. More test files, passing mocks, or narrower scope alone do not establish stronger real-world detection.

The module structure is worth adopting because it provides clear locations for these controls and checks. The benefit depends on enforcing the contracts and observing failures at their real boundaries. Moving directories alone does not resolve the listed gaps.
