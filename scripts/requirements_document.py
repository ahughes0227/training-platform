#!/usr/bin/env python3
"""Render and validate the requirements document using only the standard library."""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "docs/requirements/requirements.json"
CONTEXT = ROOT / "docs/requirements/context.md"
OUTPUT = ROOT / "docs/REQUIREMENTS.md"


def validate(data: dict, context: str) -> None:
    records = data["requirements"]
    ids = [row["id"] for row in records]
    count = data.get("requirement_count", 150)
    expected = [f"TP-{number:03d}" for number in range(1, count + 1)]
    if ids != expected:
        raise ValueError(f"Register must contain ordered unique IDs TP-001 through TP-{count:03d}")
    sections = {item["key"]: item for item in data["sections"]}
    catalog = {item["id"]: item for item in data["evidence_catalog"]}
    if len(sections) != len(data["sections"]) or len(catalog) != len(data["evidence_catalog"]):
        raise ValueError("Duplicate section or evidence ID")
    test_names = {
        node.name
        for path in (ROOT / "tests").rglob("test_*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }
    for row in records:
        for field in ("title", "requirement", "what", "why", "how", "positive", "negative", "current", "owner"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{row['id']}: missing {field}")
        if row["section"] not in sections:
            raise ValueError(f"{row['id']}: unknown section")
        if row["positive_procedure_id"] != f"P-{row['id']}" or row["negative_procedure_id"] != f"N-{row['id']}":
            raise ValueError(f"{row['id']}: invalid procedure identity")
        if row["full_verification_status"] not in {"pending", "verified"}:
            raise ValueError(f"{row['id']}: unknown status")
        if not row.get("required_scopes"):
            raise ValueError(f"{row['id']}: missing scopes")
        if set(row["evidence_refs"]) - catalog.keys():
            raise ValueError(f"{row['id']}: unknown evidence reference")
        for check in row.get("observed_checks", []):
            if check["direction"] not in {"positive", "negative"}:
                raise ValueError(f"{row['id']}: invalid observed-check direction")
            target = (ROOT / check["path"]).resolve()
            if not target.is_relative_to(ROOT) or not target.exists():
                raise ValueError(f"{row['id']}: invalid observed-check path")
            if check.get("test"):
                names = {node.name for node in ast.walk(ast.parse(target.read_text(encoding="utf-8")))
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
                if check["test"] not in names:
                    raise ValueError(f"{row['id']}: observed test not found in cited file")
        if "shall" not in row["requirement"].lower():
            raise ValueError(f"{row['id']}: normative wording missing")
        # A documentation renderer cannot certify evidence or close a requirement.
        if row["full_verification_status"] == "verified":
            raise ValueError("Verified status requires a separate reviewed evidence validator, not this renderer")
    for item in catalog.values():
        for relative in item["paths"]:
            target = (ROOT / relative).resolve()
            if not target.is_relative_to(ROOT) or not target.exists():
                raise ValueError(f"{item['id']}: missing or unsafe path {relative}")
        for test in item["tests"]:
            if test not in test_names:
                raise ValueError(f"{item['id']}: cited test does not exist: {test}")
    for marker in ("<!-- EVIDENCE_CATALOG -->", "<!-- REQUIREMENTS_REGISTER -->"):
        if context.count(marker) != 1:
            raise ValueError(f"Expected one marker: {marker}")


def render(data: dict, context: str) -> str:
    evidence = []
    for item in data["evidence_catalog"]:
        links = ", ".join(
            f"[{path}](../{path})" for path in item["paths"]
        )
        tests = ", ".join(f"`{name}`" for name in item["tests"]) or "No named test claim."
        evidence.append(
            f"### {item['id']} — {item['title']}\n\n"
            f"- **Scope:** `{item['scope']}`.\n"
            f"- **Sources:** {links}.\n"
            f"- **Selected tests:** {tests}\n"
            f"- **Observed support:** {item['observed']}\n"
            f"- **Limits:** {item['limits']}\n"
        )
    index = ["| Section | IDs | Count |", "| --- | --- | --- |"]
    body = []
    for section in data["sections"]:
        rows = [row for row in data["requirements"] if row["section"] == section["key"]]
        index.append(f"| [{section['title']}](#requirements-{section['key']}) | {section['range']} | {len(rows)} |")
        body.append(f'<a id="requirements-{section["key"]}"></a>\n\n'
                    f"### {section['title']}\n\n")
        for row in rows:
            refs = ", ".join(
                f"[{key}](#{key.lower()})" for key in row["evidence_refs"]
            ) or "Requirements register and acceptance plan."
            scopes = ", ".join(f"`{scope}`" for scope in row["required_scopes"])
            observations = "".join(
                f"- **Observed {check['direction']} support ({check['scope']}):** "
                f"[{check['test'] or check['path']}](../{check['path']}) — {check['claim']} "
                f"**Limit:** {check['limits']}\n"
                for check in row.get("observed_checks", [])
            )
            body.append(
                f'<a id="{row["id"].lower()}"></a>\n\n'
                f"#### {row['id']} — {row['title']}\n\n"
                f"- **Requirement:** {row['requirement']}\n"
                f"- **What:** {row['what']}\n"
                f"- **Why:** {row['why']}\n"
                f"- **How:** {row['how']}\n"
                f"- **Enforcement owner:** {row['owner']}.\n"
                f"- **Positive evidence — `{row['positive_procedure_id']}` (required):** {row['positive']}\n"
                f"- **Negative evidence — `{row['negative_procedure_id']}` (required):** {row['negative']}\n"
                f"- **Current evidence:** {row['current']} References: {refs}\n"
                f"{observations}"
                f"- **Required scope:** {scopes}. **Full verification:** pending.\n\n"
            )
    result = context.replace("<!-- EVIDENCE_CATALOG -->", "\n".join(evidence))
    result = result.replace("<!-- REQUIREMENTS_REGISTER -->", "\n".join(index) + "\n\n" + "".join(body))
    if result.count("\n```") % 2:
        raise ValueError("Unbalanced fenced code blocks")
    # Explicit anchors avoid relying on punctuation-specific renderer slug rules.
    result = re.sub(r"(?m)^### (E-[A-Z]+) —", lambda match: f'<a id="{match[1].lower()}"></a>\n\n### {match[1]} —', result)
    for target in re.findall(r"\]\(([^)]+)\)", result):
        if target.startswith(("https://", "http://")):
            continue
        path, _, anchor = target.partition("#")
        if not path:
            if f'id="{anchor}"' not in result:
                raise ValueError(f"Missing anchor: {anchor}")
        elif not (OUTPUT.parent / path).exists():
            raise ValueError(f"Missing document link: {target}")
    return result.rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the readable document")
    mode.add_argument("--check", action="store_true", help="validate records, references, and generated view")
    args = parser.parse_args()
    data = json.loads(REGISTER.read_text(encoding="utf-8"))
    context = CONTEXT.read_text(encoding="utf-8")
    validate(data, context)
    rendered = render(data, context)
    if args.write:
        OUTPUT.write_text(rendered, encoding="utf-8")
    elif not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
        raise SystemExit("Requirements view is stale; run with --write")
    count = len(data["requirements"])
    print(f"Validated {count} requirements, {count} positive and {count} negative procedures, "
          f"{len(data['evidence_catalog'])} evidence entries, and document references.")


if __name__ == "__main__":
    main()
