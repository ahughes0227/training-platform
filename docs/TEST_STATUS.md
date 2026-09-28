# What has actually been tested?

**The platform is not yet fully tested on live infrastructure.** The cloud container check passed, while the complete deployed training-to-serving workflow remains unverified.

## Status as of September 28, 2026

| Area | Local evidence | Live infrastructure evidence |
| --- | --- | --- |
| Labels, duplicates, splits, immutable WebDataset files | Passing tests, including label exceptions and tampering checks | GCS read/write handshake passed; platform dataset publication and live BigQuery ingestion remain untested |
| DINOv3 training, reload, evaluation, heatmaps | Actual DINOv3 architecture with generated weights now passes, including selected-layer training and portable reload | A100 ran a generic synthetic MLP; pretrained DINOv3 training remains untested |
| Container and Vertex | Runtime failure and handshake contracts tested | Build/publish and one A100 CUDA/GCS handshake passed for the recorded digest |
| Guided setup and agent questions | CLI behavior tested | Live LiteLLM setup remains untested |
| Durable execution, retries, budget limits, digest reuse | Controller/API tests pass using simulated cloud clients | Deployed Cloud Run, Cloud SQL, and Workflows flow remains untested |
| MLflow experiments and candidate registration | Local registry test passes | Configured cloud MLflow service, authentication, lineage, and registration remain untested |
| Ray Serve, approval, promotion, rollback | Local prediction and release tests; earlier local Ray HTTP fixture check recorded | GKE/KubeRay deployment, trained-model inference, promotion, and rollback remain untested |
| Logs and telemetry | Structured logging implemented | GPU probe identifiers/stages appeared in Cloud Logging; Loki delivery remains untested |
| Agent permissions | Role guard tests pass | Actual service-account denial tests and selected agent harness enforcement remain untested |
| Infrastructure templates | Ten YAML files parse | OpenTofu provider validation and deployment acceptance remain unperformed |

The full local suite currently passes **55 tests with no skips**. Local tests do not establish the untested live rows above.

## Training-code fix and container follow-up

The audit found a real trainer compatibility bug: the installed DINOv3 implementation places its transformer layers at `model.layer`. The factory now recognizes that layout. Four regression tests exercise the real DINOv3 implementation without using a substitute model or accessing the model hub.

The successful A100 image was built before this fix. Its evidence remains valid for the original digest. The release sequence for corrected code is:

1. Trainer Engineer: reproduce the failure, fix trainer code, and pass regression tests — completed locally.
2. Runtime Engineer: build/publish the committed corrected source as a new immutable digest — pending.
3. Runtime validation: run the new image on a GPU and exercise the corrected DINOv3 path — pending.
4. Runtime Certifier: complete the ordered certification gates and record the digest — pending.
5. Experiment Runner: reuse that certified digest for ordinary experiments, without building an image — pending live verification.

No new container build, paid job, or production infrastructure was started by this audit.

## What is needed for full live acceptance?

- An approved pretrained DINOv3 artifact location and checksum, plus a labeled defect dataset.
- Resolved deployment settings and service endpoints, or a reviewed plan to provision them.
- A deployment cost estimate and cleanup plan within the authorized spending limit. The existing USD 5 approval remains the limit; unmeasured billing has not been treated as zero.
- Execution and recorded results for every untested live row above, including failure/retry, permission denial, promotion, inference, and rollback.

The configured endpoint example remains blank; the deployment example and certification request contain placeholders. The audit has not established that any independently deployed services are available. The earlier GCP quota observations are historical and must be refreshed before another paid GPU operation.

See [the full acceptance record](ACCEPTANCE.md) and [the successful GPU job evidence](evidence/2026-09-27-gpu-container/README.md).
