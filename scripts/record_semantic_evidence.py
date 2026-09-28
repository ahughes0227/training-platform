#!/usr/bin/env python3
"""Record a successful local JUnit observation bound to the tested code files."""
import argparse
import hashlib
import json
import platform
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_manifest():
    paths = []
    for directory in ("src", "tests", "infra", "schemas", "templates", "scripts"):
        paths.extend(path for path in (ROOT / directory).rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and path.suffix != ".pyc")
    paths.extend(ROOT / name for name in ("pyproject.toml", ".env.example"))
    entries = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(paths)}
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return entries, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command", required=True)
    args = parser.parse_args()
    root = ET.parse(args.junit).getroot()
    cases = [{"class": case.get("classname"), "name": case.get("name"),
              "seconds": float(case.get("time", "0"))}
             for case in root.iter("testcase")]
    if not cases or any(list(root.iter(tag)) for tag in ("failure", "error", "skipped")):
        raise SystemExit("Only an executed all-passing, no-skip observation can be recorded here.")
    files, digest = source_manifest()
    result = {
        "schema_version": 1, "observed_at": datetime.now(UTC).isoformat(),
        "scope": "local_fixture_real_dinov3_and_ray_http",
        "command": args.command, "source_code_sha256": digest, "source_files": files,
        "parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "passed": len(cases), "failed": 0, "skipped": 0, "cases": cases,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in ("torch", "transformers", "webdataset", "ray", "mlflow", "pydantic")}},
        "limits": [
            "Generated DINOv3 weights and fixture images do not prove pretrained defect quality.",
            "Ray HTTP runs on local CPU processes; no new GPU/GCP certification is implied.",
            "Cloud adapters and authority rejection cases use explicit local instrumentation.",
            "Infrastructure capability records are operator-pinned declarations of observed acceptance; provider evidence is not independently probed by this loader.",
            "Live Vertex/MLflow/GKE/Loki and IAM positive/negative acceptance remain pending.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Recorded {len(cases)} passing cases against code snapshot {digest}.")


if __name__ == "__main__":
    main()
