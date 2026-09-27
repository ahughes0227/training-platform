"""Checkpoint loading and single-crop inference for Ray Serve."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class InferenceBundle:
    model: Any
    classes: list[str]
    image_size: int
    confidence_threshold: float
    device: str


def load_inference_bundle(checkpoint_path: str | Path, device: str = "cpu") -> InferenceBundle:
    """Load a training checkpoint, including its class order and preprocessing."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("inference requires PyTorch") from exc
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.pt"
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    metadata = checkpoint.get("model")
    if not isinstance(metadata, dict) or not checkpoint.get("classes"):
        raise ValueError("checkpoint is missing model configuration or class names")
    from .model import build_model
    weights_uri = metadata.get("weights_uri")
    if weights_uri and not str(weights_uri).startswith("gs://") and not Path(weights_uri).is_absolute():
        weights_uri = str(checkpoint_path.parent / weights_uri)
    model = build_model(
        len(checkpoint["classes"]), metadata.get("hidden_dim", 256),
        metadata.get("dropout", 0.2), metadata.get("unfreeze_last_n", 0),
        metadata.get("backbone", "facebook/dinov3-vits16-pretrain-lvd1689m"),
        weights_uri, metadata.get("weights_sha256"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return InferenceBundle(model, list(checkpoint["classes"]), metadata.get("image_size", 256),
                           float(checkpoint.get("confidence_threshold", 0.0)), str(device))


def _preprocess(image: Any, image_size: int):
    try:
        import torchvision.transforms as T
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("crop inference requires torch, torchvision and Pillow") from exc
    if isinstance(image, (bytes, bytearray)):
        import io
        image = Image.open(io.BytesIO(image))
    if not isinstance(image, Image.Image):
        raise TypeError("image must be PIL.Image or encoded image bytes")
    transform = T.Compose([
        T.Resize((image_size, image_size)), T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def predict_crop(bundle: InferenceBundle, image: Any, explain: bool = False) -> dict[str, Any]:
    """Predict one crop and optionally return a diagnostic patch attribution map.

    The heatmap is a gradient attribution over pooled DINOv3 patch embeddings;
    it is an inspection aid and is not a segmentation or causal explanation.
    """
    import torch
    tensor = _preprocess(image, bundle.image_size).to(bundle.device)
    if explain:
        bundle.model.zero_grad(set_to_none=True)
        tensor.requires_grad_(True)
        logits = bundle.model(tensor)
        predicted = int(logits.argmax(dim=1).item())
        logits[0, predicted].backward()
        # Use the input gradient as a spatial diagnostic at DINOv3 patch scale.
        input_gradient = tensor.grad.detach().abs().mean(dim=1)[0]
        side = max(1, bundle.image_size // 16)
        pooled = torch.nn.functional.adaptive_avg_pool2d(input_gradient[None, None], (side, side))[0, 0]
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
        "class": bundle.classes[predicted], "class_index": predicted,
        "confidence": confidence, "review": confidence < bundle.confidence_threshold,
        "probabilities": {name: float(probs[i]) for i, name in enumerate(bundle.classes)},
    }
    if heat is not None:
        result["heatmap"] = heat.tolist()
        result["heatmap_kind"] = "predicted_class_patch_attribution_diagnostic"
    return result
