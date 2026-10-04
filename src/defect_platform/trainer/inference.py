"""Checkpoint loading and single-crop inference for Ray Serve."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import ExperimentConfig
from ..semantics import assert_classes_match, assert_training_compatible, read_manifest


@dataclass
class InferenceBundle:
    model: Any
    classes: list[str]
    image_size: int
    confidence_threshold: float
    device: str
    class_ids: list[str]
    catalog_sha256: str
    semantic_sha256: str
    preprocessing: Any


def load_inference_bundle(
    checkpoint_path: str | Path,
    device: str = "cpu",
    *,
    expected_semantic_sha256: str | None = None,
    expected_catalog_sha256: str | None = None,
    expected_bundle_sha256: str | None = None,
) -> InferenceBundle:
    """Load a training checkpoint, including its class order and preprocessing."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("inference requires PyTorch") from exc
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_symlink():
        raise ValueError("model checkpoint path cannot be a symbolic link")
    if checkpoint_path.parent.is_symlink():
        raise ValueError("model bundle directory cannot be a symbolic link")
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.pt"
    model_dir = checkpoint_path.parent.resolve()
    from .weights import verify_bundle_integrity

    verify_bundle_integrity(model_dir, expected_bundle_sha256)
    semantics_path = model_dir / "semantics.json"
    manifest = read_manifest(semantics_path, expected_semantic_sha256)
    if manifest.kind != "model" or (
        expected_catalog_sha256 is not None and manifest.catalog.sha256 != expected_catalog_sha256
    ):
        raise ValueError("model semantic identity does not match serving expectations")
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    if checkpoint.get("semantic_sha256") != manifest.sha256:
        raise ValueError("checkpoint semantic fingerprint does not match model manifest")
    preprocessing = manifest.preprocessing
    if manifest.training_policy is None or preprocessing is None:
        raise ValueError("model semantic manifest is missing training policy or preprocessing")
    policy = manifest.training_policy.model_dump()
    checkpoint_policy = checkpoint.get("training_config")
    if not isinstance(checkpoint_policy, dict) or any(
        checkpoint_policy.get(key) != policy[key]
        for key in (
            "optimizer",
            "loss",
            "focal_gamma",
            "horizontal_flip_probability",
            "class_weights",
            "seed",
            "max_review_error_rate",
        )
    ):
        raise ValueError("checkpoint training policy does not match model semantic manifest")
    metadata = checkpoint.get("model")
    if not isinstance(metadata, dict) or not checkpoint.get("classes"):
        raise ValueError("checkpoint is missing model configuration or class names")
    assert_classes_match(manifest, list(checkpoint["classes"]))
    experiment = ExperimentConfig.model_validate(checkpoint.get("experiment"))
    assert_training_compatible(manifest, experiment, list(checkpoint["classes"]))
    if metadata.get("image_size") != preprocessing.image_size:
        raise ValueError("checkpoint preprocessing does not match model semantic manifest")
    for field in ("backbone", "image_size", "hidden_dim", "dropout", "unfreeze_last_n"):
        if metadata.get(field) != getattr(experiment.model, field):
            raise ValueError("checkpoint model configuration does not match semantic experiment")
    from .model import build_model

    weights_uri = metadata.get("weights_uri")
    if weights_uri and not str(weights_uri).startswith("gs://"):
        candidate = Path(weights_uri)
        if candidate.is_absolute():
            raise ValueError("portable model weights path must be relative to the model bundle")
        if ".." in candidate.parts:
            raise ValueError("portable model weights path cannot traverse outside the model bundle")
        weights_uri = str((model_dir / candidate).resolve())
        if not Path(weights_uri).is_relative_to(model_dir):
            raise ValueError("portable model weights path escapes the model bundle")
    model = build_model(
        len(checkpoint["classes"]),
        metadata.get("hidden_dim", 256),
        metadata.get("dropout", 0.2),
        metadata.get("unfreeze_last_n", 0),
        metadata.get("backbone", "facebook/dinov3-vits16-pretrain-lvd1689m"),
        weights_uri,
        metadata.get("weights_sha256"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return InferenceBundle(
        model,
        list(checkpoint["classes"]),
        preprocessing.image_size,
        float(checkpoint.get("confidence_threshold", 0.0)),
        str(device),
        [item.class_id for item in manifest.catalog.classes],
        manifest.catalog.sha256,
        manifest.sha256,
        preprocessing,
    )


def _preprocess(image: Any, preprocessing):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("crop inference requires torch, torchvision and Pillow") from exc
    if isinstance(image, (bytes, bytearray)):
        import io

        image = Image.open(io.BytesIO(image))
    if not isinstance(image, Image.Image):
        raise TypeError("image must be PIL.Image or encoded image bytes")
    from .preprocessing import image_transform

    transform = image_transform(preprocessing)
    transformed: Any = transform(image.convert("RGB"))
    return transformed.unsqueeze(0)


def predict_crop(bundle: InferenceBundle, image: Any, explain: bool = False) -> dict[str, Any]:
    """Predict one crop and optionally return a diagnostic patch attribution map.

    The heatmap is a gradient attribution over pooled DINOv3 patch embeddings;
    it is an inspection aid and is not a segmentation or causal explanation.
    """
    import torch

    tensor = _preprocess(image, bundle.preprocessing).to(bundle.device)
    if explain:
        bundle.model.zero_grad(set_to_none=True)
        tensor.requires_grad_(True)
        logits = bundle.model(tensor)
        predicted = int(logits.argmax(dim=1).item())
        logits[0, predicted].backward()
        # Use the input gradient as a spatial diagnostic at DINOv3 patch scale.
        input_gradient = tensor.grad.detach().abs().mean(dim=1)[0]
        side = max(1, bundle.image_size // 16)
        pooled = torch.nn.functional.adaptive_avg_pool2d(input_gradient[None, None], (side, side))[
            0, 0
        ]
        heat = pooled.cpu().numpy()
        maximum = float(heat.max())
        if maximum:
            heat = heat / maximum
    else:
        with torch.inference_mode():
            logits = bundle.model(tensor)
            predicted = int(logits.argmax(dim=1).item())
        heat = None
    probs = torch.softmax(logits.detach(), dim=1)[0].cpu().tolist()
    confidence = float(probs[predicted])
    result: dict[str, Any] = {
        "class": bundle.classes[predicted],
        "class_index": predicted,
        "class_id": bundle.class_ids[predicted],
        "catalog_sha256": bundle.catalog_sha256,
        "semantic_sha256": bundle.semantic_sha256,
        "confidence": confidence,
        "review": confidence < bundle.confidence_threshold,
        "probabilities": {name: float(probs[i]) for i, name in enumerate(bundle.classes)},
    }
    if heat is not None:
        result["heatmap"] = heat.tolist()
        result["heatmap_kind"] = "predicted_class_patch_attribution_diagnostic"
    return result
