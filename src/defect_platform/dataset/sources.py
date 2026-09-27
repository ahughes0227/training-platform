"""CSV and BigQuery adapters for image-label manifests."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from defect_platform.contracts import LabelSource, SourceKind


@dataclass(frozen=True, slots=True)
class LabelRow:
    image_uri: str
    raw_label: str
    sample_id: str | None = None
    group_id: str | None = None
    source: str = ""
    row_number: int | None = None


def read_label_source(source: LabelSource, *, bq_client: Any | None = None) -> list[LabelRow]:
    """Read labels from a CSV path/URI or fully qualified BigQuery table.

    CSV rows are streamed into memory after strict header and required-value checks.
    Google libraries are imported only when their corresponding source is selected.
    """
    if source.kind == SourceKind.CSV:
        return _read_csv(source)
    if source.kind == SourceKind.BIGQUERY:
        return _read_bigquery(source, bq_client=bq_client)
    raise ValueError(f"Unsupported label source kind: {source.kind}")


def _read_csv(source: LabelSource) -> list[LabelRow]:
    location = source.location
    parsed = urlparse(location)
    if parsed.scheme == "gs":
        try:
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover - cloud extra
            raise RuntimeError("Reading gs:// CSV files requires the cloud extra") from exc
        data = storage.Client().bucket(parsed.netloc).blob(parsed.path.lstrip("/")).download_as_bytes()
        stream = io.StringIO(data.decode("utf-8-sig", errors="strict"), newline="")
        base = source.image_root or location.rsplit("/", 1)[0] + "/"
    else:
        path = Path(location).expanduser()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return _parse_csv(source, handle, base=source.image_root or str(path.parent))
    return _parse_csv(source, stream, base=base)


def _parse_csv(source: LabelSource, handle: Any, *, base: str) -> list[LabelRow]:
    reader = csv.DictReader(handle)
    if not reader.fieldnames:
        raise ValueError(f"CSV source {source.location!r} is empty or has no header")
    required = {source.image_uri_column, source.label_column}
    missing = required - set(reader.fieldnames)
    for optional in (source.sample_id_column, source.group_id_column):
        if optional:
            # Optional identifiers are allowed to be absent; the request still works.
            continue
    if missing:
        raise ValueError(f"CSV {source.location!r} lacks columns: {', '.join(sorted(missing))}")
    result: list[LabelRow] = []
    for row_number, row in enumerate(reader, start=2):
        image = _required_cell(row.get(source.image_uri_column), "image URI", source, row_number)
        label = _required_cell(row.get(source.label_column), "label", source, row_number)
        image = _resolve_image_uri(image, base)
        result.append(
            LabelRow(
                image_uri=image,
                raw_label=label,
                sample_id=_optional_cell(row.get(source.sample_id_column)) if source.sample_id_column else None,
                group_id=_optional_cell(row.get(source.group_id_column)) if source.group_id_column else None,
                source=source.location,
                row_number=row_number,
            )
        )
    if not result:
        raise ValueError(f"CSV source {source.location!r} has a header but no data rows")
    return result


def _read_bigquery(source: LabelSource, *, bq_client: Any | None) -> list[LabelRow]:
    if not _is_fully_qualified_table(source.location):
        raise ValueError("BigQuery location must be project.dataset.table")
    if bq_client is None:
        try:
            from google.cloud import bigquery
        except ImportError as exc:  # pragma: no cover - cloud extra
            raise RuntimeError("BigQuery sources require the cloud extra") from exc
        bq_client = bigquery.Client()
    table = source.location.replace("`", "")
    columns = [source.image_uri_column, source.label_column]
    if source.sample_id_column:
        columns.append(source.sample_id_column)
    if source.group_id_column:
        columns.append(source.group_id_column)
    # Identifiers are configured by the trusted project template; quote each field
    # and table to avoid accidental SQL syntax injection.
    sql = "SELECT " + ", ".join(f"`{name.replace('`', '')}`" for name in columns)
    sql += f" FROM `{table}`"
    rows = bq_client.query(sql).result()
    result: list[LabelRow] = []
    for row_number, row in enumerate(rows, start=1):
        mapping = dict(row.items()) if hasattr(row, "items") else dict(row)
        image = _required_cell(mapping.get(source.image_uri_column), "image URI", source, row_number)
        label = _required_cell(mapping.get(source.label_column), "label", source, row_number)
        result.append(
            LabelRow(
                image_uri=_resolve_image_uri(image, source.image_root or ""),
                raw_label=label,
                sample_id=_optional_cell(mapping.get(source.sample_id_column)) if source.sample_id_column else None,
                group_id=_optional_cell(mapping.get(source.group_id_column)) if source.group_id_column else None,
                source=f"bigquery:{table}",
                row_number=row_number,
            )
        )
    if not result:
        raise ValueError(f"BigQuery source {table!r} returned no rows")
    return result


def _is_fully_qualified_table(value: str) -> bool:
    parts = value.replace("`", "").split(".")
    return (
        len(parts) == 3
        and bool(re.fullmatch(r"[A-Za-z0-9_-]+", parts[0]))
        and bool(re.fullmatch(r"[A-Za-z0-9_]+", parts[1]))
        and bool(re.fullmatch(r"[A-Za-z0-9_$]+", parts[2]))
    )


def _resolve_image_uri(image: str, base: str) -> str:
    if urlparse(image).scheme in {"gs", "https", "http", "file"} or Path(image).is_absolute():
        return image
    if not base:
        return image
    if base.startswith("gs://"):
        return base.rstrip("/") + "/" + image.lstrip("/")
    return str((Path(base).expanduser() / image).resolve())


def _required_cell(value: Any, field: str, source: LabelSource, row_number: int) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"Missing {field} in {source.location!r} row {row_number}")
    return str(value).strip()


def _optional_cell(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip()
