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

## GCP trainer container preflight, 2026-09-27

- The GCP dashboard for `prefab-winter-256318` showed estimated charges of USD 0.00 for September 1–27 at the time of inspection. This is a delayed estimate, not an audited cost statement.
- The enabled API list contained Artifact Registry API but not Cloud Build API or Vertex AI API (27 services listed). The repository has no Git remote, and the runtime template has no resolved base image digest or registry location.
- Automatic approval review rejected activation of authenticated Cloud Shell again, citing credential exposure and broader cloud mutation authority. No Cloud Build build, image push, GPU job, or other GCP container execution was started, and no alternate execution path was used to bypass that decision.
- `infra/cloudbuild/trainer-smoke.yaml` is a prepared, unexecuted 20-minute CPU image build and validation with no registry push. It requires a verified, pinned PyTorch CUDA base image URI, enabled Cloud Build API, and an approved GCP execution path. Passing it would verify the image on GCP CPU infrastructure; the GPU container and Vertex runtime gates would remain open.

## GCP CPU trainer container smoke test passed, 2026-09-27

The user subsequently approved the authenticated Cloud Shell route, repository upload, Cloud Build API enablement, and bounded CPU container test under the existing USD 5 total limit. The earlier execution-path rejection and disabled Cloud Build API observations above are historical; Cloud Build was enabled and this test actually ran.

| Evidence | Observed value |
| --- | --- |
| Project / region | `prefab-winter-256318` / `us-central1` |
| Source commit | `cbf9dd19ba09df5a1c4236f620e83d20f73d7a94` |
| Uploaded Git archive SHA-256 | `783a0bb40cf09f63ddee5446285dc5fcaa62e6048e03995c025b34b6a3405ada`, matched by Cloud Shell `sha256sum` |
| Build configuration | `infra/cloudbuild/trainer-smoke.yaml`, overall timeout `1200s` |
| Base image | `pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755` |
| Build | [8ce8af21-0e2d-419c-ab26-1eb168fb1df0](https://console.cloud.google.com/cloud-build/builds;region=us-central1/8ce8af21-0e2d-419c-ab26-1eb168fb1df0?project=prefab-winter-256318), `SUCCESS` |
| Start / finish UTC | `2026-09-27T23:42:15.078344943Z` / `2026-09-27T23:48:13.284218Z` |
| Console durations | Overall `00:05:58`; image build `00:05:36`; validation `00:00:18` |
| Built local image ID | `6c75341e42ff`, tagged `defect-trainer-smoke:local`; no registry image was pushed |
| Runtime packages observed in build log | `torch==2.14.0`, `torchvision==0.29.0`, `transformers==5.17.0` |
| Validation command | `docker run --rm --entrypoint python defect-trainer-smoke:local -m defect_platform.trainer.validate` |

Submitted from the verified source snapshot using:

```sh
gcloud builds submit . \
  --project=prefab-winter-256318 \
  --region=us-central1 \
  --config=infra/cloudbuild/trainer-smoke.yaml \
  --substitutions=_TRAINING_BASE_IMAGE=pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755 \
  --async
```

Both build steps finished successfully. The validation printed:

```json
{
  "success": true,
  "device": "cpu",
  "loss": 0.670745849609375,
  "checkpoint": "/tmp/defect-trainer-check-dk7c3b59/synthetic-validation.pt",
  "cuda_name": null,
  "gpu_required": false
}
```

The validation uses the synthetic backbone and proves container startup, an optimizer step, and checkpoint write/reload. Its checkpoint was temporary inside the removed test container. It does not prove DINOv3 weight loading, CUDA execution, GPU compatibility, Vertex handshake, real dataset training, MLflow, or serving. No runtime certification record was issued; the Container and Vertex runtime gates remain open. In particular, the CUDA base and newly installed locked PyTorch must still be tested together on a GPU.

At the published default `e2-standard-2` rate of USD 0.006/minute, the observed duration implies approximately USD 0.0358 for build compute before free-tier credits. The full 20-minute cap would have bounded build compute to USD 0.12. Storage, logs, and network are additional; these are estimates, not measured billing charges. The dashboard still displayed estimated USD 0.00, which can lag usage. [Cloud Build pricing](https://cloud.google.com/build/pricing).

Logs remain in the linked build record. Downloading a local raw-log copy timed out, so no downloaded-log checksum is claimed. The temporary source object was staged at `gs://prefab-winter-256318_cloudbuild/source/1790552530.0085-6e2042d5470c45fdbc2e972459644cb0.tgz`; its bucket has a seven-day (`604800s`) soft-delete policy. `gcloud storage rm` reported `Completed 1/1`, and the subsequent live bucket listing was empty. The source can remain recoverable until soft-delete expiry; the empty staging bucket and build logs are retained. No registry image, GPU job, or serving deployment was created.
