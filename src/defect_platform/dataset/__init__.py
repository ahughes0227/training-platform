"""Dataset ingestion, review, versioning, and WebDataset construction."""

from .builder import build_dataset
from .duplicates import DuplicateReport, find_duplicates
from .labels import LabelException, PreviewResult, preview_dataset
from .sources import LabelRow, read_label_source
from .verification import load_dataset_semantics, verify_dataset_version

__all__ = [
    "DuplicateReport",
    "LabelException",
    "LabelRow",
    "PreviewResult",
    "build_dataset",
    "find_duplicates",
    "load_dataset_semantics",
    "preview_dataset",
    "read_label_source",
    "verify_dataset_version",
]
