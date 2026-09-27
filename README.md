# Defect Training Platform

Train a separate DINOv3 + MLP classifier for each type of inspected object. Each input is a crop of a defect that has already been found. The model names its defect class, reports confidence, and flags uncertain cases for review. It does not decide product disposition.

## Start here

1. Read [the plain-language guide](docs/START_HERE.md).
2. Copy `.env.example` and `templates/object/` to a new `projects/<object-name>/` folder, use `defect object init`, or start with `defect train guided` after cloud settings are configured.
3. Connect a CSV manifest or BigQuery table containing image locations and labels. The CLI previews unresolved labels before a dataset is published.
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

Requires Python 3.12 or 3.13 and `uv`. Run `uv sync --extra dev --extra train --extra data` for local tests, then `uv run --extra dev --extra train --extra data pytest`. Install the `cloud`, `agent`, and `serve` extras for their corresponding environments.

Cloud endpoints and credentials are intentionally unset in the template. Live deployment and acceptance remain blocked until they are supplied and verified.
