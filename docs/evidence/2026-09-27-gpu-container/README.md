# GPU container test: passed

The trainer container ran successfully on one NVIDIA A100 GPU in GCP. It trained a small synthetic MLP, saved and reloaded its checkpoint, performed CUDA arithmetic, and wrote/read GCS data with a matching checksum. Vertex reported success and stopped the worker.

## Open the evidence

- [Successful GCP job](https://console.cloud.google.com/agent-platform/locations/us-central1/training/5919343314829574144?project=prefab-winter-256318)
- [Cloud logs](https://console.cloud.google.com/logs/query?project=prefab-winter-256318); filter `resource.labels.job_id="5919343314829574144"`.
- [Container publication build](https://console.cloud.google.com/cloud-build/builds;region=us-central1/ea99c24f-6583-4486-b661-439437f1688f?project=prefab-winter-256318)
- [GPU and storage result](result.json)
- [Training and checkpoint result](validation.json)
- [Job summary](job-summary.json)
- [Full acceptance history](../../ACCEPTANCE.md)

These small JSON documents are exact values transcribed from observed Cloud Shell responses and application logs, rather than downloaded raw response files. The result was independently read from GCS at:

`gs://prefab-winter-256318_cloudbuild/gpu-smoke/defect-gpu-smoke-20260927-cbf9dd1-a100/result.json`

The image source was commit `cbf9dd19ba09df5a1c4236f620e83d20f73d7a94`. Its immutable URI is:

`us-central1-docker.pkg.dev/prefab-winter-256318/defect-gpu-smoke-20260927/trainer@sha256:54b8a0ffd2a8bb0cb444c6f61069f33c7fca4ce98ce88abd734eba0855c17c27`

## Test configuration

The accepted job used one `a2-highgpu-1g`, one `NVIDIA_TESLA_A100`, 100 GB Standard boot disk, Spot scheduling, `600s` timeout, disabled retries, and the existing compute service account. It ran:

```sh
set -euo pipefail
timeout 120s /opt/venv/bin/python -m defect_platform.trainer.validate --require-gpu
timeout 120s /opt/venv/bin/python -m defect_platform.trainer.runtime_probe
```

The starting template is `infra/vertex/gpu-smoke.yaml`. For the observed A100 configuration, change its machine type to `a2-highgpu-1g`, accelerator type to `NVIDIA_TESLA_A100`, and runtime timeout to `600s`; resolve the digest, service account, and unique run/result paths before submission. Check current quota, prices, and budget before repeating the test. The independent Cloud Shell controller cancelled stalled earlier T4 attempts and bounded this attempt to 20 minutes before RUNNING or 30 minutes total.

## Raw responses retained in Cloud Shell

`~/defect-gpu-evidence-20260927.tgz` contains the result, final A100 response, resolved job configuration, watchdog log, 30 recent log entries, and final responses confirming the two T4 attempts were cancelled. Its observed Cloud Shell SHA-256 is `e612e1d96d7315f1cd340057e41ca9bed37429dc0d9d08f8ae7f08c7a5ac0eeb`. A local download has not been verified, so no local archive checksum match is claimed. Cloud Shell files are retained there; they should not be treated as a durable substitute for the linked cloud job and result.

## Remaining checks

This synthetic GPU result does not establish pretrained DINOv3 training, T4 compatibility, full ordered runtime certification, MLflow registration, Loki delivery, or Ray Serve deployment. Those remain in the acceptance checklist. Actual billing charges remain unverified; see the acceptance history for bounded cost estimates and retained-image storage.
