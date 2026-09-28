# Operations and deployment

## Configure the environment

Copy `.env.example` for local work; supply values through environment variables or your secrets manager in deployed services. `infra/terraform.tfvars.example` lists the inputs for OpenTofu. Bucket names, project and region, database password, LiteLLM model, GPU type/count, and all image digests are deployment inputs. Never commit credentials, resolved `.env` files, or state files.

OpenTofu creates private buckets, Artifact Registry, Cloud SQL, Cloud Run services, Workflows, a Cloud Tasks queue, GKE node pools, KubeRay operator, and dedicated service accounts. Images must be built first and supplied by immutable digest. Cloud SQL state must use an encrypted remote OpenTofu backend managed by the operator because Terraform state can contain the database password.

## Runtime release

1. Validate the trainer without Docker: `python -m defect_platform.trainer.validate`.
2. From a GPU host with Docker and cloud credentials, copy `templates/runtime-release.yaml`, fill its values, commit the source, then run `defect runtime certify CONFIG --output certifications/RUNTIME_ID.yaml`.
3. The command checks the source commit, runs local trainer validation, builds the image from a pinned base, validates that image with a local GPU, pushes it, resolves its registry digest, runs a short Vertex GPU/GCS handshake, and writes a `CertifiedRuntime` record only after all gates pass.

The trainer image should contain code and locked dependencies. It should not download mutable model weights during its build. Each experiment pins the DINOv3 checkpoint location and checksum separately. If any stage fails, stop at that stage and hand evidence to its owner; a training failure does not automatically rebuild the image.

### Bounded GCP image smoke test

`infra/cloudbuild/trainer-smoke.yaml` builds the trainer Dockerfile on Cloud Build's default machine and runs its synthetic CPU validation inside the image. It has a 20-minute overall timeout and does not push an image. It checks container packaging only; GPU and Vertex certification still require the runtime release flow above. Supply a verified PyTorch CUDA base image URI pinned to a digest and an approved Google Cloud credential path, then run from the repository root:

```sh
gcloud builds submit . \
  --project=PROJECT_ID \
  --region=us-central1 \
  --config=infra/cloudbuild/trainer-smoke.yaml \
  --substitutions=_TRAINING_BASE_IMAGE=PYTORCH_CUDA_IMAGE@sha256:DIGEST
```

The Cloud Build API must be enabled. Check the billing account and the current [Cloud Build pricing](https://cloud.google.com/build/pricing) before submission; the build timeout bounds build minutes but does not cap ancillary charges. Record the build ID, log URL, source commit, exact base digest, observed validation JSON, and actual billed cost in `docs/ACCEPTANCE.md`.

### Small real GPU container test on Vertex

Use `infra/cloudbuild/trainer-candidate.yaml` for the separate Runtime Engineer build/publish operation. Provide `_TRAINING_BASE_IMAGE` as a verified base digest and `_IMAGE_URI` as a private Artifact Registry candidate tag. It has a 20-minute timeout and performs a CPU packaging check before pushing the candidate. Resolve the resulting registry digest; never launch the GPU test with the tag.

Fill `infra/vertex/gpu-smoke.yaml` with that digest, an existing authorized service account, a unique run ID, and private GCS probe/result paths. Submit it with `gcloud ai custom-jobs create --project=PROJECT --region=REGION --display-name=RUN_ID --config=RESOLVED_CONFIG`. The configuration requests exactly one T4 on one `n1-standard-4` Spot worker, a 15-minute job timeout, disabled retries, and 120-second limits on each probe. GPU quota and actual Spot capacity are separate requirements. Verify the accepted job's scheduling settings, and cancel it if it remains queued beyond the test's approved waiting period.

The job forces GPU optimizer/checkpoint validation with `--require-gpu`, then runs the GPU arithmetic and GCS read/write handshake. A CPU fallback is a failure. Inspect both validation output and the handshake result, retaining the exact digest, source commit, GPU name/count, Python/PyTorch/CUDA versions, job status, logs, and budget estimate in `docs/ACCEPTANCE.md`. This test does not download DINOv3 weights or issue a full certification record. Ordinary experiments continue to use existing certified digests without calling either build configuration.

## Experiments and releases

The controller first stores an idempotent run and checks its cost cap. Workflows submits the Vertex CustomJob and monitors it independently of the CLI or agent. The trainer writes evaluation and model artifacts under the configured run root and logs to MLflow. Terminal state is recorded even if the user has closed the CLI.

Model promotion is a separate human action. Review dataset lineage, MCC, macro F1, class confusions, abstention behavior, and a sample of heatmaps. Stage a version with `defect release stage RUN_ID --model-version VERSION --serving-image-digest IMAGE@sha256:DIGEST`. Promote it with `defect release promote RELEASE_ID --approver NAME`; this records approver identity, moves the MLflow `champion` alias, and applies a RayService manifest. `defect release rollback OBJECT --approver NAME` restores the preceding approved model. `--no-deploy` is available for a registry-only change when the cluster is deliberately operated separately. Ray Serve loads only the champion model and never accepts a model URI from an inference request.

For a remote operator, set `DEFECT_CONTROL_SERVICE_URL` to the `control_url` output, `DEFECT_MLFLOW_TRACKING_URI` to `mlflow_url`, and `DEFECT_RELEASE_GCS_URI` to `release_ledger_uri`. Staging reads the run from the control API; each release revision is appended to GCS. The OpenTofu `release_operator_members` list grants MLflow invocation, append/read access to the artifact bucket, cluster discovery, and RayService write access only in the serving namespace. Configure a GKE context with the same Google user identity before promotion. The local SQLite release ledger remains available when `DEFECT_RELEASE_GCS_URI` is unset.

The RayService is initially an internal GKE service. Network ingress and callers must be configured for the intended environment; no public endpoint is created by default.

## Logging

All components emit JSON logs to stdout with object, dataset, run, Vertex job, model, stage, and failure identifiers. Set `DEFECT_OTLP_ENDPOINT` to an OpenTelemetry Collector logs endpoint when available. Configure its OTLP HTTP exporter for a Loki native OTLP endpoint. Keep high-cardinality run IDs as structured metadata instead of index labels.

## Coding-agent boundaries

The four `.github/agents/*.agent.md` files define role ownership. Their `PreToolUse` hooks are for the VS Code Local harness; test them in that harness before relying on them. Other harnesses may ignore those hook declarations. Remote operations are separately restricted by service-account IAM and protected release commands. A legitimate cross-layer repair starts with a short evidence handoff to the owning role.
