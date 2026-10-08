import json

import pytest

from defect_platform.trainer.metrics import calibrate_abstention, evaluate_predictions
from defect_platform.trainer.runtime import (
    RuntimeValidationError,
    certify_runtime,
    classify_failure,
    lookup_certified_runtime,
    vertex_gpu_handshake,
)
from defect_platform.trainer.weights import artifact_sha256, verify_artifact_sha256


def test_selection_metrics_prefer_mcc_then_macro_f1():
    result = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 1], ["good", "bad"])
    assert result.mcc == pytest.approx(1 / 3**0.5)
    assert result.macro_f1 == pytest.approx((2 / 3 + 0.8) / 2)
    assert result.selection_key == (result.mcc, result.macro_f1)


def test_abstention_threshold_is_calibrated_from_validation():
    result = calibrate_abstention([[.99, .01], [.61, .39], [.42, .58]], [0, 1, 1], .25)
    assert result["confidence_threshold"] == pytest.approx(.99)
    assert result["accepted_error_rate"] == 0
    assert result["review_rate"] == pytest.approx(2 / 3)


def test_unreachable_review_target_abstains_on_everything_and_reports_it():
    # No cutoff can hold the accepted error rate at 0 here: the most confident
    # prediction is already wrong.
    result = calibrate_abstention([[.9, .1], [.8, .2], [.6, .4]], [1, 1, 0], 0.0)
    assert result["target_met"] is False
    assert result["review_rate"] == 1
    assert result["accepted_count"] == 0
    # Strictly above any probability, so serving reviews every prediction.
    assert result["confidence_threshold"] > 1
    assert json.loads(json.dumps(result)) == result


def test_reachable_review_target_reports_accepted_counts():
    result = calibrate_abstention([[.99, .01], [.61, .39], [.42, .58]], [0, 1, 1], .25)
    assert result["target_met"] is True
    assert result["accepted_count"] == 1


def test_certification_runs_all_gates_and_binds_exact_digest():
    called = []
    checks = [lambda name=name: called.append(name) or True for name in
              ("trainer", "container", "vertex", "read", "write")]
    runtime = certify_runtime(
        runtime_id="gpu-v1", source_commit="abc", image_tag="trainer:abc",
        image_digest="us-docker.pkg.dev/project/train@sha256:" + "a" * 64,
        runtime_version="1", python_version="3.11", pytorch_version="2.5", cuda_version="12",
        trainer_check=checks[0], container_gpu_check=checks[1], vertex_gpu_check=checks[2],
        gcs_read_check=checks[3], gcs_write_check=checks[4],
    )
    assert called == ["trainer", "container", "vertex", "read", "write"]
    assert lookup_certified_runtime([runtime], runtime.image_digest) == runtime
    with pytest.raises(LookupError):
        lookup_certified_runtime([runtime], runtime.image_digest.replace("a" * 64, "b" * 64))


def test_certification_stops_at_failed_stage():
    with pytest.raises(RuntimeValidationError, match="container_gpu"):
        certify_runtime(
            runtime_id="r", source_commit="abc", image_tag="t",
            image_digest="x@y@sha256:" + "b" * 64,
            runtime_version="1", python_version="3.11", pytorch_version="2", cuda_version="12",
            trainer_check=lambda: True, container_gpu_check=lambda: False,
            vertex_gpu_check=lambda: True, gcs_read_check=lambda: True, gcs_write_check=lambda: True,
        )


def test_handshake_requires_requested_gpu_count_and_gcs_roundtrip():
    from defect_platform.contracts import VertexJobConfig
    config = VertexJobConfig(project="p", region="r", machine_type="g2", accelerator_type="nvidia-l4",
                             accelerator_count=2, service_account="sa", staging_uri="gs://b/staging",
                             max_run_hours=1, estimated_hourly_usd=1, max_run_cost_usd=2)
    image = "us-docker.pkg.dev/p/train@sha256:" + "c" * 64
    with pytest.raises(RuntimeValidationError, match="fewer GPUs"):
        vertex_gpu_handshake(config, image_uri=image, staging_bucket_uri="gs://b/probe",
                             submit_and_wait=lambda payload: {"gpu_count": 1, "gcs_read": True,
                                 "gcs_write": True, "image_digest": "sha256:" + "c" * 64,
                                 "gpu_name": "NVIDIA L4", "gpu_tensor_operation": True,
                                 "python_version": "3.11", "pytorch_version": "2.5",
                                 "cuda_version": "12.4"})


def test_failure_classifier_names_owner_and_retryability():
    assert classify_failure("gcs", "permission denied").owner == "cloud_configuration"
    assert classify_failure("vertex", "temporarily unavailable").retryable is True
    assert classify_failure("webdataset", "label missing").code == "DATA_INVALID"


def test_weight_artifact_checksum_is_stable_and_enforced(tmp_path):
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"hidden_size": 384}')
    (model_dir / "model.safetensors").write_bytes(b"pinned test weights")
    digest = artifact_sha256(model_dir)
    assert verify_artifact_sha256(model_dir, digest) == digest
    (model_dir / "model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_artifact_sha256(model_dir, digest)


def test_model_factory_refuses_unpinned_remote_fallback():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from defect_platform.trainer.model import build_model
    with pytest.raises(ValueError, match="remote hub fallback is disabled"):
        build_model(2, weights_uri=None, weights_sha256="a" * 64)


def test_mlflow_logging_authenticates_and_resumes_controller_run(tmp_path, monkeypatch):
    import sys
    from contextlib import contextmanager
    from types import SimpleNamespace

    import defect_platform.mlflow_auth as auth
    from defect_platform.contracts import ExperimentConfig, ModelSpec
    from defect_platform.trainer.training import _log_mlflow

    events = []

    class RunContext:
        def __enter__(self):
            events.append("run-enter")
            return SimpleNamespace(info=SimpleNamespace(run_id="controller-run-17"))
        def __exit__(self, *_args):
            events.append("run-exit")

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: events.append(("tracking-uri", uri)),
        start_run=lambda **kwargs: events.append(("start-run", kwargs)) or RunContext(),
        log_params=lambda params: events.append("params"),
        log_metrics=lambda metrics, step=None: events.append(("metrics", step)),
        log_artifacts=lambda path, artifact_path: events.append(("artifacts", artifact_path)),
        log_artifact=lambda path: events.append("artifact"),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    @contextmanager
    def token(uri):
        events.append(("auth-enter", uri))
        try:
            yield
        finally:
            events.append("auth-exit")

    monkeypatch.setattr(auth, "mlflow_tracking_auth", token)
    config = ExperimentConfig(experiment_id="e", object_slug="fixture", dataset_version_id="v1",
                              runtime_id="r", model=ModelSpec(weights_uri="gs://bucket/model",
                                                               weights_sha256="a" * 64))
    model_dir = tmp_path / "model"; model_dir.mkdir()
    (model_dir / "model.pt").write_bytes(b"checkpoint")
    result = _log_mlflow("https://mlflow.run.app", config,
                         {"validation": {"mcc": .5, "macro_f1": .6},
                          "test": {"mcc": .4, "macro_f1": .5},
                          "history": [{"epoch": 1, "validation_mcc": .3,
                                       "validation_macro_f1": .4}]},
                         model_dir / "model.pt", mlflow_run_id="controller-run-17")
    assert result == "controller-run-17"
    # Validation metrics are logged for the run and per epoch; test metrics are not.
    assert ("metrics", None) in events and ("metrics", 1) in events
    assert ("start-run", {"run_id": "controller-run-17"}) in events
    assert events.index(("auth-enter", "https://mlflow.run.app")) < events.index("run-enter")
    assert events[-1] == "auth-exit"


def test_vertex_handshake_includes_result_uri_and_requires_gpu_proof():
    from defect_platform.contracts import VertexJobConfig
    config = VertexJobConfig(project="p", region="r", machine_type="g2", accelerator_type="nvidia-l4",
                             accelerator_count=1, service_account="sa", staging_uri="gs://b/staging",
                             max_run_hours=1, estimated_hourly_usd=1, max_run_cost_usd=2)
    image = "us-docker.pkg.dev/p/train@sha256:" + "d" * 64
    captured = {}
    result = {"gpu_count": 1, "gpu_name": "NVIDIA L4", "gpu_tensor_operation": True,
              "python_version": "3.11", "pytorch_version": "2.5", "cuda_version": "12.4",
              "gcs_read": True, "gcs_write": True, "image_digest": "sha256:" + "d" * 64}
    result_uri = "gs://b/certification/run-123/result.json"
    outcome = vertex_gpu_handshake(config, image_uri=image, staging_bucket_uri="gs://b/probe",
                                   result_uri=result_uri,
                                   submit_and_wait=lambda payload: captured.update(payload) or result)
    assert outcome == result
    assert captured["environment"]["DEFECT_PLATFORM_GCS_RESULT_URI"] == result_uri
    result["gpu_tensor_operation"] = False
    with pytest.raises(RuntimeValidationError, match="GPU tensor operation"):
        vertex_gpu_handshake(config, image_uri=image, staging_bucket_uri="gs://b/probe",
                             submit_and_wait=lambda payload: result)
