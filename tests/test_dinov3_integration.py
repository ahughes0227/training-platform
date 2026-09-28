"""Exercise the installed DINOv3 implementation without hub access or pretrained weights."""

import io
import json
import shutil

import pytest


@pytest.fixture
def tiny_dinov3(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from defect_platform.trainer.weights import artifact_sha256

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    torch.manual_seed(17)
    config = transformers.AutoConfig.for_model(
        "dinov3_vit", hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, num_register_tokens=2, patch_size=16, image_size=32,
    )
    backbone = transformers.AutoModel.from_config(config)
    weights = tmp_path / "generated-weights"
    backbone.save_pretrained(weights)
    return weights, artifact_sha256(weights)


@pytest.mark.parametrize("unfreeze", [0, 1])
def test_real_dinov3_pooling_and_selected_layer_gradients(tiny_dinov3, unfreeze):
    import torch
    from defect_platform.trainer.model import build_model

    weights, digest = tiny_dinov3
    model = build_model(2, hidden_dim=8, dropout=0, unfreeze_last_n=unfreeze,
                        weights_uri=str(weights), weights_sha256=digest).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model.backbone(pixel_values=images).last_hidden_state
        # Exclude CLS and both register tokens: only four spatial patches remain.
        assert output.shape == (2, 7, 32)
        assert torch.allclose(model.forward_features(images), output[:, 3:].mean(dim=1))
    logits = model(images)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    for name, parameter in model.backbone.named_parameters():
        selected = bool(unfreeze and name.startswith("model.layer.1."))
        assert parameter.requires_grad == selected, name
        assert (parameter.grad is not None) == selected, name
    assert model.head[-1].weight.grad is not None


def test_real_dinov3_refuses_unfreeze_past_depth(tiny_dinov3):
    from defect_platform.trainer.model import build_model

    weights, digest = tiny_dinov3
    with pytest.raises(ValueError, match="exceeds backbone transformer depth"):
        build_model(2, unfreeze_last_n=3, weights_uri=str(weights), weights_sha256=digest)


def test_real_dinov3_webdataset_training_portable_reload_and_heatmap(tiny_dinov3, tmp_path):
    wds = pytest.importorskip("webdataset")
    from PIL import Image
    from defect_platform.contracts import ExperimentConfig, ModelSpec
    from defect_platform.trainer.inference import load_inference_bundle, predict_crop
    from defect_platform.trainer.training import train_experiment

    weights, digest = tiny_dinov3
    shards = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.tar"
        with wds.TarWriter(str(path)) as writer:
            for index in range(4):
                label = index % 2
                image = Image.new("RGB", (32, 32), "red" if label == 0 else "blue")
                encoded = io.BytesIO()
                image.save(encoded, format="PNG")
                writer.write({"__key__": f"{split}-{index}", "png": encoded.getvalue(),
                              "json": json.dumps({"label": label}).encode()})
        shards[split] = [str(path)]
    config = ExperimentConfig(
        experiment_id="actual-dinov3-fixture", object_slug="fixture", dataset_version_id="v1",
        runtime_id="local", epochs=1, batch_size=2,
        model=ModelSpec(weights_uri=str(weights), weights_sha256=digest, image_size=32,
                        hidden_dim=8, dropout=0, unfreeze_last_n=1),
    )
    report = train_experiment(config, tmp_path, tmp_path / "output", ["scratch", "dent"], shards)
    # Reload must use the exported backbone, even when the original weights are gone.
    shutil.rmtree(weights)
    bundle = load_inference_bundle(tmp_path / "output" / "model")
    crop = Image.new("RGB", (32, 32), "red")
    ordinary = predict_crop(bundle, crop)
    explained = predict_crop(bundle, crop, explain=True)
    assert ordinary["class"] == explained["class"]
    assert ordinary["confidence"] == pytest.approx(explained["confidence"])
    assert len(explained["heatmap"]) == 2
    assert all(len(row) == 2 for row in explained["heatmap"])
    assert 0 <= explained["confidence"] <= 1
    assert report["validation"]["confusion"] and report["test"]["confusion"]
