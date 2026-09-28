"""Canonical image preprocessing for trainer and inference."""

from __future__ import annotations

from ..semantics import PreprocessingSpec


def image_transform(spec: PreprocessingSpec, *, training: bool = False,
                    horizontal_flip_probability: float = 0.5):
    try:
        import torchvision.transforms as T
    except ImportError as exc:
        raise RuntimeError("image preprocessing requires torchvision") from exc
    interpolation = getattr(T.InterpolationMode, spec.interpolation.upper())
    transforms = [T.Resize((spec.image_size, spec.image_size), interpolation=interpolation)]
    if training:
        transforms.append(T.RandomHorizontalFlip(p=horizontal_flip_probability))
    transforms.extend((T.ToTensor(), T.Normalize(spec.normalization_mean, spec.normalization_std)))
    return T.Compose(transforms)
