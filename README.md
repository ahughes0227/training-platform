# Defect Training Platform

The [Compute Engine training queue](docs/VM_TRAINING_QUEUE.md) provides a local
queue service and supervised worker under `defect train queue`. Its
[implementation evidence](docs/TRAINING_QUEUE_IMPLEMENTATION.md) distinguishes
local validation from the remaining live VM acceptance gates.

Train a separate DINOv3 + MLP classifier for each type of inspected object. Each input is a crop of a defect that has already been found. The model names its defect class, reports confidence, and flags uncertain cases for review. It does not decide product disposition.

## Testing status

The latest local suite passed 177 tests, including 63 queue cases; the separate Ray HTTP integration test also passed. See [queue implementation evidence](docs/TRAINING_QUEUE_IMPLEMENTATION.md) for the remaining live VM gates. A published trainer image passed an A100 GPU and GCS handshake on Vertex AI. The updated trainer and semantic contracts need a new image and GPU validation. Full deployment acceptance, pretrained DINOv3 training, cloud MLflow, GKE serving, and Loki delivery remain incomplete. See [testing status](docs/TEST_STATUS.md) and [recorded acceptance evidence](docs/ACCEPTANCE.md).

## Start here

Read the [program feature guide](docs/FEATURES.md) for the implemented capabilities, configuration, commands, artifact locations, and current verification limits.

Read the [ChatGPT skill layer](docs/SKILLS.md) for the onboarding, user,
developer, and agent-only skills that expose those capabilities conversationally.

Read the [full system requirements](docs/REQUIREMENTS.md) for 195 requirements, their rationale and enforcement, and the positive and negative evidence needed for acceptance.

See the [semantic implementation plan](docs/SEMANTIC_IMPLEMENTATION_PLAN.md), [explicit module contracts](docs/MODULE_CONTRACTS.md), and [source-bound local evidence](docs/SEMANTIC_TEST_STATUS.md). Class meanings, IDs, dataset/model manifests and bundle hashes now travel across the modules; cloud submission requires a reviewed operator catalog and a current pinned infrastructure observation.

See the [module-change risk assessment](docs/MODULE_CHANGE_RISKS.md) and [mitigation plan](docs/MODULE_RISK_MITIGATIONS.md) for risks, regression-detection scores, and conditional residual risk targets for separating datasets, training, and infrastructure.

1. Read [the plain-language guide](docs/START_HERE.md).
2. Copy `.env.example` and `templates/object/` to a new `projects/<object-name>/` folder, use `defect object init`, or start with `defect train guided` after cloud settings are configured.
3. Define and approve class meanings with `defect object review-catalog OBJECT.yaml --reviewer YOUR_ID --approve` through the operator catalog store. Connect a CSV manifest or BigQuery table containing image locations and labels. The CLI previews unresolved labels before a dataset is published.
4. Configure a GCP project, region, storage, MLflow, a certified trainer image digest, and a per-run cost cap. `defect train start` returns a run ID immediately; `defect run status RUN_ID` displays progress and locations.
5. Review the candidate in MLflow. Promotion to Ray Serve requires an explicit command and approver identity.

The repository stores code, templates, and small readable project configs. Dataset shards, checkpoints, and heatmaps live in GCS. Run and model metadata live in the platform state store and MLflow. Nothing under `workspace/` is tracked by Git.

## Layout

| Location | What to find |
| --- | --- |
| `projects/<object>/` | Plain-language project description and editable config |
| `templates/object/` | Controlled shape for new object projects |
| `certifications/` | Reusable digest-specific trainer runtime records |
| `src/defect_platform/dataset/` | Label ingestion and immutable WebDataset building |
| `src/defect_platform/trainer/` | DINOv3 training, validation, and runtime certification |
| `src/defect_platform/control/` | Guided setup, job lifecycle, Vertex, and MLflow |
| `src/defect_platform/serve/` | Ray Serve prediction and approved release loading |
| `infra/` | OpenTofu and Kubernetes deployment templates |
| `docs/` | Architecture, operations, and acceptance records |

## Development

Requires Python 3.12 or 3.13 and `uv`. For the complete local acceptance suite, install the `dev`, `train`, `data`, `cloud`, and `serve` extras: `uv sync --extra dev --extra train --extra data --extra cloud --extra serve`. Run `uv run --extra dev --extra train --extra data --extra cloud --extra serve pytest`. The Ray integration test starts local workers and a loopback HTTP listener; cloud SDK tests use local instrumentation and require no GCP credentials. Install the `agent` extra when connecting guided setup to LiteLLM.

Cloud endpoints and credentials are intentionally unset in the template. Live deployment and acceptance remain blocked until they are supplied and verified.
