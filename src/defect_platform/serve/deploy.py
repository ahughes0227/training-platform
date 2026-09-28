"""Render and apply a versioned KubeRay RayService for an approved model."""

from __future__ import annotations

import subprocess

import yaml

from defect_platform.contracts import ModelRelease


def rayservice_manifest(
    release: ModelRelease,
    *,
    mlflow_tracking_uri: str,
    kubernetes_service_account: str = "defect-serving",
    namespace: str = "defect-serving",
    ray_version: str = "2.58.0",
    gpu_count: float = 1,
    catalog_root: str | None = None,
) -> dict:
    if release.state != "promoted" or not release.approved_by or release.approved_at is None:
        raise ValueError("only an approved promoted release may be deployed")
    if "@sha256:" not in release.serving_image_digest:
        raise ValueError("serving image must be addressed by digest")
    if not all((release.catalog_sha256, release.model_semantic_sha256, release.bundle_sha256)):
        raise ValueError("release requires pinned catalog, model semantics, and bundle integrity")
    if not catalog_root:
        raise ValueError("serving deployment requires an operator class catalog root")
    if gpu_count <= 0 or not float(gpu_count).is_integer():
        raise ValueError("serving requires a positive whole GPU allocation")

    env = [
        {"name": "DEFECT_MLFLOW_TRACKING_URI", "value": mlflow_tracking_uri},
        {"name": "DEFECT_MODEL_NAME", "value": release.model_name},
        {"name": "DEFECT_MODEL_VERSION", "value": release.model_version},
        {"name": "DEFECT_MODEL_SEMANTIC_SHA256", "value": release.model_semantic_sha256},
        {"name": "DEFECT_CATALOG_SHA256", "value": release.catalog_sha256},
        {"name": "DEFECT_BUNDLE_SHA256", "value": release.bundle_sha256},
        {"name": "DEFECT_CLASS_CATALOG_ROOT", "value": catalog_root},
        {"name": "DEFECT_RELEASE_ID", "value": release.release_id},
        {"name": "DEFECT_SERVE_GPUS", "value": str(gpu_count)},
        {"name": "DEFECT_DEVICE", "value": "cuda"},
    ]
    image = release.serving_image_digest
    serve_config = {
        "applications": [
            {
                "name": release.object_slug,
                "import_path": "defect_platform.serve.ray_entrypoint:app",
                "route_prefix": "/",
                "deployments": [
                    {
                        "name": "DefectClassifier",
                        "num_replicas": 1,
                        "ray_actor_options": {"num_gpus": gpu_count},
                    }
                ],
            }
        ]
    }
    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayService",
        "metadata": {
            "name": f"defect-{release.object_slug}",
            "namespace": namespace,
            "labels": {"app": "defect-classifier", "object": release.object_slug},
            "annotations": {"defect.platform/release-id": release.release_id},
        },
        "spec": {
            "serveConfigV2": yaml.safe_dump(serve_config, sort_keys=False),
            "rayClusterConfig": {
                "rayVersion": ray_version,
                "enableInTreeAutoscaling": True,
                "headGroupSpec": {
                    "rayStartParams": {"dashboard-host": "0.0.0.0"},
                    "template": {
                        "spec": {
                            "serviceAccountName": kubernetes_service_account,
                            "containers": [
                                {
                                    "name": "ray-head",
                                    "image": image,
                                    "env": env,
                                    "resources": {"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "2", "memory": "4Gi"}},
                                }
                            ],
                        }
                    },
                },
                "workerGroupSpecs": [
                    {
                        "groupName": "gpu-workers",
                        "replicas": 1,
                        "minReplicas": 1,
                        "maxReplicas": 2,
                        "rayStartParams": {},
                        "template": {
                            "spec": {
                                "serviceAccountName": kubernetes_service_account,
                                "nodeSelector": {"workload": "ray-serve"},
                                "containers": [
                                    {
                                        "name": "ray-worker",
                                        "image": image,
                                        "env": env,
                                        "resources": {
                                            "requests": {"cpu": "2", "memory": "6Gi", "nvidia.com/gpu": str(int(gpu_count))},
                                            "limits": {"cpu": "4", "memory": "12Gi", "nvidia.com/gpu": str(int(gpu_count))},
                                        },
                                    }
                                ],
                            }
                        },
                    }
                ],
            },
        },
    }


def apply_rayservice(manifest: dict, *, kubectl: str = "kubectl", context: str | None = None) -> None:
    """Apply the reviewed manifest through the caller's configured GKE context."""
    command = [kubectl]
    if context:
        command += ["--context", context]
    command += ["apply", "--server-side", "-f", "-"]
    subprocess.run(
        command,
        input=yaml.safe_dump(manifest, sort_keys=False),
        text=True,
        check=True,
        capture_output=True,
    )
