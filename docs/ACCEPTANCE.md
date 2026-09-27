# Acceptance record

This is the evidence checklist for the complete platform. Mark a row complete only after recording the command, configuration ID, artifact or job link, and observed result. Local successes do not certify a cloud deployment.

| Gate | Required evidence |
| --- | --- |
| Dataset | CSV and BigQuery ingestion; reviewed label exceptions; immutable GCS version; WebDataset can read every expected sample; split and duplicate report |
| Trainer | Local tiny forward/backward/optimizer/checkpoint/reload; real DINOv3 fixture training; MCC, macro F1, and abstention report |
| Container | Exact image starts on GPU host; CUDA, model, one optimizer step, checkpoint write/reload pass |
| Vertex runtime | Digest-specific handshake reports Python/PyTorch/CUDA, requested GPU count, and GCS read/write |
| Experiment | Repeated runs reuse a certified digest; no Docker build/push; budget denial and idempotent retry proven |
| Registry | MLflow run has dataset/runtime lineage, evaluation artifacts, and candidate model version |
| Serving | Approval gates promotion; Ray Serve returns class/confidence/review flag and optional heatmap; rollback restores previous release |
| Observability | Structured identifiers visible in Cloud Logging and, when configured, Loki; failure layer and job log reference retained |
| Agent boundaries | Role hooks block named prohibited actions in the selected VS Code harness; service-account IAM denies cross-role cloud operations |

Cloud acceptance cannot pass until the GCP project, region, quotas, service accounts, endpoints, and DINOv3 checkpoint access are supplied. Record any unmet gate here rather than calling the platform complete.

## Local verification, 2026-09-27

- Fixture images pass dataset preview/build, immutable shard verification, split and duplicate checks, trainer forward/backward/checkpoint reload, evaluation, calibration, and heatmap generation. The synthetic trainer validation also completed on CPU.
- The control API passes HTTP submission, idempotent retry, Vertex dispatch, and terminal callback tests. The Vertex request and certified digest reuse are tested with a fake cloud client. Normal experiment submission has no Docker call path.
- MLflow 3.16 accepted a raw model bundle as a candidate version in a local registry test. Ray Serve answered a local HTTP prediction with class, confidence, review flag, and a diagnostic heatmap using a fixture predictor. GCS release history and rollback passed with a fake storage client.
- The full local test suite and Python compile checks pass. Workflow YAML parsed. OpenTofu provider validation, GPU container validation, GCS publication, Vertex handshake/training, Cloud Run/SQL/Workflows, live MLflow registration, GKE Ray Serve, Loki delivery, and cloud IAM denial checks still require the configured cloud environment.

## Live evidence to record

For each live gate above, record the GCP project and region, configuration commit, exact image digest and DINOv3 weight checksum, dataset version, run ID, Vertex job link, MLflow model version, Loki query or Cloud Logging link, approver, RayService revision, observed inference response, and rollback result. Record actual cloud cost against the accepted estimate. Do not mark the platform accepted until all live gates have passed.

## GCP preflight, 2026-09-27

- Project selected in Google Cloud Console: `prefab-winter-256318` (My First Project). The signed-in user has the project Owner role.
- Before the user enabled billing, the Cloud Storage bucket page said, "You can use Cloud Storage after you enable billing." The console also said the free trial had ended. The billing detail page failed to load. A later bucket creation succeeded, so the earlier Storage restriction no longer blocked this test; the billing-account details remain unverified.
- The enabled-services page listed 27 APIs and did not include Vertex AI. A filtered quotas view yielded no Vertex AI rows; GPU quota remains unverified.
- The user authorized a maximum of USD 5 for testing. The platform's per-run cap does not cap always-on infrastructure costs.
- Live acceptance remains blocked by the disabled Vertex AI API, unverified GPU quota, missing project settings and DINOv3 weights, and absent certified container/runtime digests. Recheck each gate before incurring any job or infrastructure cost.

## GCP minimal storage smoke test, 2026-09-27

- Project: `prefab-winter-256318`. Created private Standard bucket `defect-platform-smoke-prefab-winter-256318-20260927` in `us-east1` using the Google Cloud Console. The bucket had uniform access, public access prevention, and seven-day soft delete.
- Uploaded one synthetic 75-byte text object, `defect-platform-gcp-smoke-20260927.txt`. The Console reported "1 file successfully uploaded" and listed it as 75 B, `text/plain`, and not public. Its local SHA-256 before upload was `cb591dc6a066e2c7df4de4fdf10566c774c79580676787bb12ca2c039fa5325f`.
- Opened the authenticated object read URL in the browser and observed the exact original three-line content. This verifies a small Console-mediated GCS write and read, not the platform's GCS publication code or a byte-level downloaded checksum.
- Deleted the object, then deleted the bucket. The Console showed the object deletion notice, no live objects, "Deleted 1 bucket", and no live buckets in the project list. Both deletions initially failed with a transient Console error and succeeded on retry. The seven-day soft-delete policy can retain recoverable data until expiry.
- No Vertex job, GPU, container build, MLflow server, Ray Serve cluster, or other paid compute was started. Actual billing charges were not available in the Console during this test; do not infer a measured cost or mark any full-platform live gate complete from this smoke test.
- Authenticated Cloud Shell activation was rejected by automatic approval review because it exposes credentials and enables broader cloud mutations. The application code was therefore not run against GCP. A separate authorized credential path and cloud configuration are required for the Vertex handshake and remaining acceptance gates.

## Trainer container test attempt, 2026-09-27

- On the local macOS ARM host, `.venv/bin/python -m defect_platform.trainer.validate` passed the synthetic optimizer and checkpoint reload check on CPU. `.venv/bin/python -m pytest -q tests/test_trainer_runtime.py tests/test_runtime_release.py tests/test_trainer_end_to_end.py` passed all 13 focused tests.
- `.venv/bin/python -m defect_platform.trainer.validate --require-gpu` exited with `GPU validation requested but CUDA is unavailable`, as expected on this host.
- No Docker, Podman, GPU container runtime, CUDA GPU, configured pinned PyTorch CUDA base image digest, or completed certification record was available. The image was not built or run; the Container and Vertex runtime gates above remain open. The repository's runtime release template still contains placeholders.
