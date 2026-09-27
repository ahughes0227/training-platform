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
