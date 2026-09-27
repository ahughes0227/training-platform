"""Ray Serve HTTP interface for approved defect-type classifiers."""

import base64
import io
import os
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field


MAX_IMAGE_BYTES = 8 * 1024 * 1024


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_base64: str
    explain: bool = False


class PredictResponse(BaseModel):
    class_name: str
    confidence: float = Field(ge=0, le=1)
    review_required: bool
    model_name: str
    model_version: str
    heatmap_png_base64: str | None = None


def decode_image(encoded: str):
    from PIL import Image

    if len(encoded) > (MAX_IMAGE_BYTES * 4 // 3) + 16:
        raise ValueError("image exceeds the 8 MiB request limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds the 8 MiB request limit")
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError("image_base64 is not a supported image") from exc
    return image


def encode_heatmap(heatmap: Any, base_image: Any) -> str | None:
    if heatmap is None:
        return None
    if isinstance(heatmap, str):
        return heatmap
    if isinstance(heatmap, bytes):
        return base64.b64encode(heatmap).decode("ascii")
    if isinstance(heatmap, list):
        from PIL import Image

        height = len(heatmap)
        width = len(heatmap[0]) if height else 0
        if not height or not width or any(len(row) != width for row in heatmap):
            raise ValueError("invalid heatmap dimensions")
        pixels = [max(0, min(255, round(float(value) * 255)))
                  for row in heatmap for value in row]
        low_res = Image.new("L", (width, height))
        low_res.putdata(pixels)
        activity = low_res.resize(base_image.size, Image.Resampling.BILINEAR)
        overlay = Image.new("RGBA", base_image.size, (255, 0, 0, 0))
        overlay.putalpha(activity.point(lambda value: int(value * 0.65)))
        image = Image.alpha_composite(base_image.convert("RGBA"), overlay)
    else:
        image = heatmap
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


class PredictionService:
    """Pure request handler; Ray only wraps this class at deployment time."""

    def __init__(
        self,
        inference: Callable[[Any, bool], dict[str, Any]],
        model_name: str,
        model_version: str,
    ):
        self.inference = inference
        self.model_name = model_name
        self.model_version = model_version

    def predict(self, request: PredictRequest) -> PredictResponse:
        image = decode_image(request.image_base64)
        result = self.inference(image, request.explain)
        return PredictResponse(
            class_name=result.get("class_name", result.get("class")),
            confidence=float(result["confidence"]),
            review_required=bool(result.get("review_required", result.get("review"))),
            model_name=self.model_name,
            model_version=self.model_version,
            heatmap_png_base64=encode_heatmap(result.get("heatmap"), image) if request.explain else None,
        )


def _load_approved_service() -> PredictionService:
    """Resolve MLflow champion only; never load an arbitrary request-supplied model."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from defect_platform.trainer.inference import load_inference_bundle, predict_crop
    from defect_platform.mlflow_auth import mlflow_tracking_auth

    tracking_uri = os.environ["DEFECT_MLFLOW_TRACKING_URI"]
    model_name = os.environ["DEFECT_MODEL_NAME"]
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow_tracking_auth(tracking_uri):
        version = MlflowClient().get_model_version_by_alias(model_name, "champion")
        artifact_dir = mlflow.artifacts.download_artifacts(artifact_uri=version.source)
    bundle = load_inference_bundle(artifact_dir, device=os.getenv("DEFECT_DEVICE", "cpu"))

    def infer(image: Any, explain: bool) -> dict[str, Any]:
        return predict_crop(bundle, image, explain=explain)

    return PredictionService(infer, model_name, str(version.version))


def build_ray_app():
    """Ray deployment factory; import Ray only in the serving image."""
    from fastapi import Body, FastAPI, HTTPException
    from ray import serve
    from defect_platform.telemetry import bind_context, configure_logging

    app = FastAPI(title="Defect classifier", docs_url=None, redoc_url=None)

    @serve.deployment(ray_actor_options={"num_gpus": float(os.getenv("DEFECT_SERVE_GPUS", "0"))})
    @serve.ingress(app)
    class DefectClassifier:
        def __init__(self):
            configure_logging()
            bind_context(object_slug=os.getenv("DEFECT_MODEL_NAME", "").removeprefix("defect-"),
                         release_id=os.getenv("DEFECT_RELEASE_ID", ""))
            self.service = _load_approved_service()

        @app.get("/health")
        def health(self):
            return {
                "status": "ready",
                "model_name": self.service.model_name,
                "model_version": self.service.model_version,
            }

        @app.post("/predict", response_model=PredictResponse)
        def predict(self, request: PredictRequest = Body(...)):
            try:
                return self.service.predict(request)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DefectClassifier.bind()
