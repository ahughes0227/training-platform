# Semantic implementation evidence

**Scope:** September 28, 2026 local implementation of the owner-approved semantic plan. Full live acceptance remains incomplete.

## Implemented

- Reviewed, versioned class catalogs; stable IDs, normalized aliases, explicit definitions and optional reference examples.
- Operator-controlled local/GCS catalog stores with create-only publication, version claims and exact fingerprint lookup.
- Dataset meaning/source/split manifests bound to immutable content. Consumers check individual split routing, counts, labels and the exact verified semantic bytes.
- Model manifests bind the dataset, complete experiment, preprocessing, pooling/selection/calibration policies, weights and runtime lineage. One transform implementation serves training/evaluation/inference.
- Portable model integrity, external bundle pins, weights-only checkpoint loading and contained artifact paths.
- Control rejects missing/untrusted/changed semantics before operational effects; workflow dispatch rechecks them and infrastructure freshness.
- Exact-version releases and prediction responses carrying checked stable class IDs and semantic/catalog fingerprints.
- Generated schemas, review CLI, plain-language migration guidance and separate catalog-store IAM templates.

## Verification record

The final combined suite passed **115 tests, zero failures, zero skips**, in **64.11 seconds**. It emitted 19 dependency warnings (Ray/Pydantic, FastAPI/Starlette and MLflow/SQLAlchemy). [The machine-readable observation](evidence/2026-09-28-semantic-contracts/local-acceptance.json) records every case, environment package versions, parent commit and hashes of code/tests/configuration/templates/schemas.

**Tested code snapshot:** `f70973533d1571ce43dab3a5973808b0014f5d15ee8607f1185a5501f040d11b`.

The suite includes actual CSV ingestion and image/shard publication → verified reviewed catalog and dataset semantics → actual installed DINOv3 architecture with generated weights → training/validation/test evaluation → original weights removed → verified portable reload → actual Ray workers and HTTP prediction with diagnostic heatmap. This proves that bounded local integration, not pretrained quality or a live GCP deployment.

Static checks also passed for the new semantic/store/infrastructure/trainer code, its tests, contract schema generation, all 160 requirement/procedure references, and whitespace. Legacy broad lint findings remain in unrelated/older control and CLI framework patterns. OpenTofu is unavailable on this host, so provider validation/application of the updated IAM templates was not performed.

## Integration defects found and repaired

1. The old verifier compared the union of shard URIs. Swapping train/test routes kept that union unchanged. It now checks per-split routes and record counts; an adversarial route-swap test establishes rejection.
2. The semantic loader verified one read then reread the file. It now returns the manifest parsed from the verified bytes; the mutation-between-reads test prevents an unchecked substitute result.
3. The builder writes named class labels in `.cls` members; generic WebDataset decoding expected numeric indices. The actual build-to-trainer acceptance found the incompatibility. The reader now parses supported metadata explicitly, avoiding generic pickle decoding.

## Positive and negative evidence boundaries

| Boundary | Positive local evidence | Attempted opposite / prevented effect |
| --- | --- | --- |
| Catalog | Reviewed immutable round-trip and alias/ID encoding | Colliding aliases, reused version with changed content, unreviewed publication and path escape rejected |
| Dataset | Source build, committed semantic reload and real trainer handoff | Semantic tamper, changed class meaning, swapped split routes and changed checked bytes rejected |
| Trainer | Actual DINOv3 fixture architecture, selected layers, training/evaluation, portable reload and heatmaps | Invalid labels, class reorder, policy/config/normalization mismatch and wrong Vertex semantic references rejected |
| Bundle | Original weights removable; exported weights reload under pinned identity | Altered files, rewritten integrity list against external pin, extra files, symlinks and path traversal rejected |
| Control | Reviewed store determines admitted classes and persisted lineage | Missing/untrusted/draft catalogs and mismatched references produce no run/workflow/tracker/Vertex effects in instrumented local tests |
| Infrastructure | Current matching declared acceptance record validates | Stale/unready/future records, wrong project/region/account/root/runtime and wrong record pin rejected |
| Serving/release | Exact model version and hashes rendered; trained HTTP response checked | Wrong class ID/name/semantic identity rejected; failed release proof produces no alias/ledger/deploy effects in instrumented local tests |

These are bounded tests of declared cases. They cannot prove arbitrary semantic correctness, correct real-world labels, or prevention against every actor. Example locations and reviewer strings are recorded metadata; they do not authenticate a reviewer or independently validate referenced images. Embedding similarity is not label authority.

## Outstanding live gates

No paid cloud operations were performed for this increment. The historical A100/GCS result belongs to the old source/image and does not certify these changes.

- New exact-digest image, GPU container check, Vertex handshake and end-to-end training with accepted pretrained DINOv3 weights.
- Authenticated MLflow registration and release, GKE/KubeRay inference/rollback, Loki delivery and live IAM allow/deny evidence.
- Observed infrastructure capability record, protected catalog-store permissions and supported mixed-version/migration acceptance.
- Independent operator approval identity, trusted runtime registry and authoritative budget pricing. This increment adds pins and capability checks; it does not replace all preexisting authority gaps.
- Real defect reference-set review, source-evidence validity and quality thresholds.

Original risk scores and conditional mitigation targets remain unchanged; local tests alone do not justify declaring the forecast residual risk achieved.
