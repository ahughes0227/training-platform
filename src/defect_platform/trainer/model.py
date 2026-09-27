"""DINOv3 patch-token pooling with an MLP classifier head."""

from __future__ import annotations

from .weights import verify_artifact_sha256


def build_model(num_classes: int, hidden_dim: int = 256, dropout: float = 0.2,
                unfreeze_last_n: int = 0, backbone: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
                weights_uri: str | None = None, weights_sha256: str | None = None):
    """Create a DINOv3 ViT-S/16 feature extractor plus MLP head.

    Model packages and weights are loaded lazily. ``weights_uri`` may be a local
    HF directory or a GCS URI, copied to a temporary directory by the GCS SDK.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    try:
        from torch import nn
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("training requires torch and transformers with DINOv3 support") from exc

    if not weights_uri:
        raise ValueError("a pinned local or GCS weights_uri is required; remote hub fallback is disabled")
    if not weights_sha256:
        raise ValueError("a pinned weights_sha256 is required")
    source = weights_uri
    if source.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise RuntimeError("GCS model weights require google-cloud-storage") from exc
        import tempfile
        from pathlib import Path
        bucket_name, _, key = source[5:].partition("/")
        temp = tempfile.mkdtemp(prefix="dinov3-weights-")
        client = storage.Client()
        blobs = list(client.list_blobs(bucket_name, prefix=key.rstrip("/") + "/"))
        if not blobs:
            raise FileNotFoundError(f"no model weight files at {source}")
        for blob in blobs:
            relative = blob.name[len(key.rstrip("/")) + 1:]
            path = Path(temp, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(path))
        source = temp
    verify_artifact_sha256(source, weights_sha256)
    backbone_model = AutoModel.from_pretrained(source, trust_remote_code=False)
    layers = getattr(getattr(backbone_model, "encoder", None), "layer", None)
    if layers is None:
        layers = getattr(getattr(backbone_model, "layers", None), "layers", None)
    if unfreeze_last_n < 0 or (layers is not None and unfreeze_last_n > len(layers)):
        raise ValueError("unfreeze_last_n exceeds backbone transformer depth")
    for parameter in backbone_model.parameters():
        parameter.requires_grad = False
    if unfreeze_last_n:
        if layers is None:
            raise ValueError("backbone implementation did not expose transformer layers")
        for layer in list(layers)[-unfreeze_last_n:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    feature_dim = int(getattr(backbone_model.config, "hidden_size", 384))

    class DINOv3MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone_model
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
            )

        def forward_features(self, pixel_values):
            output = self.backbone(pixel_values=pixel_values)
            tokens = output.last_hidden_state
            # DINOv3 places register tokens between CLS and image patches. The
            # config is authoritative (ViT-S/16 LVD-1689M uses four registers).
            prefix_tokens = 1 + int(getattr(self.backbone.config, "num_register_tokens", 0))
            patch_tokens = tokens[:, prefix_tokens:, :]
            if patch_tokens.shape[1] == 0:
                raise RuntimeError("DINOv3 output contains no spatial patch tokens")
            return patch_tokens.mean(dim=1)

        def forward(self, pixel_values):
            return self.head(self.forward_features(pixel_values))

    return DINOv3MLP()
