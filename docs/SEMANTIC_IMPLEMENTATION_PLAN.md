# Semantic Contracts Implementation Plan

Status: owner-approved local implementation integrated on September 28, 2026 using three parallel GPT-6 Luna agents. The combined 115-test suite passes, including actual local DINOv3-to-Ray HTTP. See [implementation evidence and outstanding live gates](SEMANTIC_TEST_STATUS.md). Existing cloud spending/acceptance constraints still apply; full platform completion is pending live acceptance.

## Goal

Implement the approved class catalog, semantic manifests, deterministic module handoffs, and adversarial tests across datasets, training, control, and serving. Text/image embeddings are representations or review suggestions; they do not establish class authority or ground truth.

## Shared contracts and compatibility

- Primary owns `semantics.py`, `contracts.py`, catalog store, shared schema/template decisions, serving integration, requirement/register updates, and final integration.
- `ClassCatalog` carries stable IDs, ordered labels, definitions, aliases, examples, version, and review metadata. Reject normalized label/alias collisions across different classes.
- A reviewed catalog needs explicit class definitions and review metadata. Metadata is not identity authentication. Production admission resolves the exact catalog from an operator-controlled catalog store.
- Existing string-only object definitions derive an explicitly marked `legacy` catalog with deterministic IDs; no reviewed definitions are invented. Local exploration remains available; paid admission requires a reviewed trusted catalog.
- Dataset manifests bind catalog, accepted mapping, split policy, source and split evidence. Include semantic identity in dataset version/provenance and final content verification.
- Model manifests inherit the exact dataset meaning and add preprocessing, training/selection/calibration rules, weights/runtime/experiment identity. Bundle and verify the manifest on reload and serving.
- Dataset content hashes and dataset semantic hashes must avoid a recursive checksum dependency: dataset semantic manifests omit their containing dataset content digest; model manifests may reference it.
- Normal experiments never build/push/certify images. Semantic validation must happen before operational side effects.

## Three parallel Luna workstreams

| Workstream | Owned files | Deliverable |
| --- | --- | --- |
| Dataset | `src/defect_platform/dataset/**`, `tests/test_dataset.py`, `tests/test_dataset_semantics.py` | Catalog-driven labels, semantic publication/versioning/integrity, tamper/collision/reorder tests |
| Trainer | `src/defect_platform/trainer/**`, trainer tests including actual DINOv3 | Manifest-bound preprocessing/policy, checkpoint export/reload, prediction class IDs, rejected tampering |
| Control | `src/defect_platform/control/**`, control/guided tests | Trusted catalog preflight, manifest handoff, no-side-effect rejection, runtime-owned execution |

Agents must not edit shared contracts or another lane. Report required changes to primary. No Git commits/pushes or paid cloud operations by subagents.

## Primary integration

1. Establish shared types/helpers and module interfaces before parallel edits.
2. Publish reviewed catalogs through a configured local/GCS store; validate payload fingerprint and object scope.
3. Configure catalog templates and guided/manual navigation, with explicit review/migration instructions.
4. Bind serving responses and release eligibility to verified model semantic identity. Preserve human release authority.
5. Add meaningful fixed-fixture and adversarial integration: actual dataset build → actual DINOv3 training → portable reload/evaluation/heatmap → Ray HTTP inference, with semantic identity throughout.
6. Validate rejection of class reorder, aliases, preprocessing/policy changes, source/manifest/artifact corruption, and untrusted/missing catalog before job dispatch.
7. Add an infrastructure capability contract for observed identity/endpoints/revisions/readiness; distinguish desired config from actual validated capability. Acceptance remains pending until live checks run.
8. Update requirements and MR-02 mitigation coverage; preserve original current/target risk ratings until observed results support reassessment.
9. Integrate, run required local suite, publish source/docs, and record exact local evidence and remaining live gates.

## Acceptance limits

No claim of perfect model accuracy, universal semantic proof, authenticated human identity from an arbitrary string, or full cloud acceptance. Existing GPU evidence remains tied to the old digest. Changed software requires a new image and relevant GPU certification before paid experiments. Runtime and IAM migration mitigations remain separately tracked where this increment does not implement them.

