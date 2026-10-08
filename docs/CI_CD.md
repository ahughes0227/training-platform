# CI and delivery

Two GitHub Actions workflows cover the repository. Neither one builds the trainer image, submits training, or promotes a model. Those stay with runtime certification on a GPU host (see [operations](OPERATIONS.md#runtime-release)) and with `defect release promote`.

## CI: `.github/workflows/ci.yml`

Runs on every pull request, every push to `main`, and on demand.

| Job | What it checks | Blocks merge when |
| --- | --- | --- |
| Lint | `ruff` on the Python files the change touches; `actionlint` (with shellcheck) on the workflows. A whole-repository ruff report is printed for information. | A touched file has a ruff finding, or a workflow is invalid |
| Tests (core) | The suite with only the `dev` extra, on Python 3.12 and 3.13. Cases that need an optional extra skip. | Any test fails |
| Tests (all extras) | The full suite with `train`, `data`, `cloud` and `serve` installed from `uv.lock`: DINOv3, WebDataset, Cleanlab, MLflow registry, FastAPI and a real Ray Serve HTTP round trip. Runs offline (`HF_HUB_OFFLINE=1`) on CPU. | Any test fails |
| Infrastructure | `tofu fmt -check` and `tofu validate` on `infra/` with no backend and no credentials | The configuration is unformatted or invalid |
| Service images | Builds the control and MLflow images without pushing | A Dockerfile no longer builds |

`main` still carries older ruff findings in files nobody has touched since, so the lint gate covers changed files only. Once the whole-repository report is clean, drop the diff filter in the Lint job.

To require these checks, add them under **Settings → Branches → Branch protection rules** for `main`.

## CD: `.github/workflows/cd.yml`

**On every push to `main`** that changes source, infrastructure or dependencies, it builds the `control`, `mlflow` and `serve` images from the digest-pinned base, pushes them to Artifact Registry tagged with the commit, and records each immutable digest in the run summary and as an artifact.

**Deploying is manual.** Run the workflow from the Actions tab with **apply** checked. It publishes images as above, then waits for approval on the `production` environment. After approval, it plans and applies `infra/` with the control and MLflow digests from that same run. The serving digest is reported for `defect release stage --serving-image-digest`, because serving changes only through an explicit release.

Until it is configured, the workflow lists in its summary what is missing and stops without failing.

### One-time setup

1. **Workload Identity Federation.** Create a pool and an OIDC provider for `token.actions.githubusercontent.com`, restricted to this repository (`attribute.repository == "ahughes0227/training-platform"`). Create a deployer service account and let the provider impersonate it (`roles/iam.workloadIdentityUser`). No service-account key is stored in GitHub.
2. **Deployer permissions.** Push access to the Artifact Registry repository, plus whatever `infra/` manages if you will deploy from CI: read/write on the state bucket, and the project roles OpenTofu needs to create the resources in `infra/main.tf`.
3. **Repository variables** (Settings → Secrets and variables → Actions → Variables):

   | Variable | Example |
   | --- | --- |
   | `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/123/locations/global/workloadIdentityPools/github/providers/training-platform` |
   | `GCP_DEPLOY_SERVICE_ACCOUNT` | `deployer@PROJECT.iam.gserviceaccount.com` |
   | `GCP_ARTIFACT_REPOSITORY` | `us-central1-docker.pkg.dev/PROJECT/defect-platform` |
   | `PYTHON_BASE_IMAGE` | `python:3.12-slim@sha256:<verified digest>`. A tag without a digest is refused. |
   | `TF_STATE_BUCKET` | An encrypted, versioned GCS bucket for OpenTofu state (deploy only) |

4. **Secrets** (deploy only):
   - `TF_VARS`: the contents of a filled `infra/terraform.tfvars`, without `control_image_digest` and `mlflow_image_digest`, which CD supplies.
   - `TF_DB_PASSWORD`: the Cloud SQL password, passed as `TF_VAR_db_password`.

   OpenTofu state can contain the database password, which is why the state bucket must be encrypted and access-restricted.
5. **Environment.** Create an environment named `production` with yourself as a required reviewer. The apply job cannot start without that approval.

### What CD does not do

- **Build or push the trainer image.** Each trainer image must pass GPU validation and Vertex certification, and experiments consume only certified digests.
- **Promote or roll back a model.** That is `defect release promote` and `defect release rollback`, with an approver identity.
- **Run training or touch datasets.**
