# Training platform engineering rules

## Architecture

Training code is validated locally. Runtime images are validated on a GPU host, then certified against Vertex AI. Experiments consume certified immutable image digests. A normal experiment never builds or pushes an image. Dataset versions and releases are immutable.

Diagnose across boundaries; modify only within the active role's boundary. A failure in one layer never triggers automatic edits in another. Preserve the failing command, config identifier, image digest, job name, and log reference.

## Ownership

| Role | Owns | Must not do |
| --- | --- | --- |
| Trainer Engineer | `src/defect_platform/trainer/`, trainer tests | Build/push images, submit jobs, certify runtime |
| Runtime Engineer | Docker/runtime packaging, GPU container checks | Change algorithms, submit training, certify runtime |
| Runtime Certifier | Existing digest validation and certification | Edit trainer or Dockerfile, rebuild images |
| Experiment Runner | `src/defect_platform/control/`, configs and submission | Build/push images, change trainer or dependencies |
| Analysis | `src/defect_platform/analysis/`, analysis tests | Submit runs, build datasets, change label meaning, stage or promote releases |

Cross-layer changes need a handoff describing evidence and the owning layer. `src/defect_platform/contracts.py` and deployment policy are shared interfaces and require primary-agent integration review.

Never edit the ChatGPT project's synced `sources/` directory or its project-root `AGENTS.md`.
