"""Small end-to-end train/reload/heatmap check using synthetic WebDataset shards."""

import base64
import importlib.util
import io
import json

import pytest

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("torch", "webdataset", "PIL", "torchvision")),
    reason="trainer optional dependencies are not installed",
)


def test_webdataset_train_checkpoint_reload_and_heatmap(tmp_path, monkeypatch):
    import torch
    import webdataset as wds
    from PIL import Image

    from defect_platform.contracts import ExperimentConfig, ModelSpec
    from defect_platform.semantics import SemanticManifest, legacy_catalog, read_manifest
    from defect_platform.trainer import inference, model, training

    class TinyBackbone(torch.nn.Sequential):
        def save_pretrained(self, path):
            from pathlib import Path
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "fixture.json").write_text("{}")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = TinyBackbone(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 12))
            self.head = torch.nn.Sequential(torch.nn.Linear(12, 2))

        def forward_features(self, pixel_values):
            return self.backbone(pixel_values)


        def forward(self, pixel_values):
            return self.head(self.forward_features(pixel_values))

    monkeypatch.setattr(training, "build_model", lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(model, "build_model", lambda *args, **kwargs: TinyModel())
    shards = {}
    for split, count in (("train", 8), ("validation", 4), ("test", 4)):
        tar_path = tmp_path / f"{split}.tar"
        with wds.TarWriter(str(tar_path)) as sink:
            for index in range(count):
                label = index % 2
                image = Image.new("RGB", (24, 24), (255, 20, 20) if label == 0 else (20, 20, 255))
                stream = io.BytesIO(); image.save(stream, format="JPEG")
                sink.write({"__key__": f"{split}-{index}", "jpg": stream.getvalue(),
                            "json": json.dumps({"label": str(label)}).encode()})
        shards[split] = [str(tar_path)]
    config = ExperimentConfig(
        experiment_id="tiny-fixture", object_slug="fixture", dataset_version_id="v1",
        runtime_id="local", model=ModelSpec(weights_uri=str(tmp_path), weights_sha256="0" * 64,
                                             image_size=16, hidden_dim=8),
        epochs=1, batch_size=4,
    )
    semantics = SemanticManifest(kind="dataset", object_slug="fixture", catalog=legacy_catalog("fixture", ["scratch", "dent"]), dataset_version_id="v1")
    report = training.train_experiment(config, tmp_path, tmp_path / "output", ["scratch", "dent"], shards,
                                       semantic_manifest=semantics, dataset_sha256="b" * 64)
    model_dir = tmp_path / "output" / "model"
    bundle = inference.load_inference_bundle(model_dir)
    prediction = inference.predict_crop(bundle, Image.new("RGB", (24, 24), (255, 20, 20)), explain=True)
    assert report["checkpoint"].endswith("model/model.pt")
    assert report["test"]["confusion"]
    assert prediction["class"] in {"scratch", "dent"}
    assert prediction["review"] is False
    assert prediction["heatmap_kind"] == "predicted_class_patch_attribution_diagnostic"
    assert prediction["heatmap"]

    # Exercise the actual serving request adapter with the trained and reloaded
    # bundle. Ray itself is an optional deployment dependency and not needed for
    # this local prediction service contract check.
    from defect_platform.serve.api import PredictionService, PredictRequest
    request_image = Image.new("RGB", (24, 24), (255, 20, 20))
    encoded = io.BytesIO(); request_image.save(encoded, format="PNG")
    model_semantics = read_manifest(model_dir / "semantics.json")
    service = PredictionService(lambda image, explain: inference.predict_crop(bundle, image, explain),
                                "fixture-model", "1", catalog=model_semantics.catalog,
                                semantic_sha256=bundle.semantic_sha256)
    response = service.predict(PredictRequest(image_base64=base64.b64encode(encoded.getvalue()).decode(),
                                              explain=True))
    assert response.class_name in {"scratch", "dent"}
    assert response.model_name == "fixture-model"
    assert response.heatmap_png_base64

    configured = config.model_copy(update={
        "optimizer": "sgd", "loss": "focal", "focal_gamma": 1.5,
        "horizontal_flip_probability": 0.0, "class_weights": {"scratch": 1.2, "dent": 0.8},
    })
    configured_report = training.train_experiment(
        configured, tmp_path, tmp_path / "output-configured", ["scratch", "dent"], shards,
        semantic_manifest=semantics, dataset_sha256="b" * 64)
    assert configured_report["training_config"]["optimizer"] == "sgd"
    assert configured_report["training_config"]["loss"] == "focal"
    assert configured_report["training_config"]["class_weights"] == {"scratch": 1.2, "dent": 0.8}
    invalid_weights = configured.model_copy(update={"class_weights": {"scratch": 1.0}})
    with pytest.raises(ValueError, match="class weights must match the catalog labels"):
        training.train_experiment(invalid_weights, tmp_path, tmp_path / "invalid", ["scratch", "dent"], shards,
                                  semantic_manifest=semantics, dataset_sha256="b" * 64)
