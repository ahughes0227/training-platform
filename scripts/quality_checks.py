"""Run optional security, dependency, and dead-code checks.

The repository deliberately does not install analysis tools at runtime.  A
missing tool is reported as ``unavailable`` and does not make an offline run
fail.  Use ``--require-tools`` in CI after installing the selected tools.

The command emits one compact JSON document so an agent can consume a small
finding projection instead of replaying each tool's human-oriented output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAX_DIAGNOSTICS = 80
MAX_DIAGNOSTIC_CHARS = 240


@dataclass(frozen=True)
class CheckSpec:
    name: str
    executable: str
    module: str
    arguments: tuple[str, ...]
    network: bool = False
    parser: Callable[[str], list[dict[str, Any]]] | None = None


def _parse_bandit(output: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return []
    results = payload.get("results", []) if isinstance(payload, dict) else []
    findings = []
    for result in results:
        if not isinstance(result, dict):
            continue
        findings.append(
            {
                "file": result.get("filename"),
                "line": result.get("line_number"),
                "rule": result.get("test_id"),
                "severity": result.get("issue_severity"),
                "confidence": result.get("issue_confidence"),
                "message": result.get("issue_text"),
            }
        )
    return findings


def _parse_pip_audit(output: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return []
    dependencies = payload.get("dependencies", []) if isinstance(payload, dict) else payload
    if not isinstance(dependencies, list):
        return []
    findings = []
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            continue
        for vulnerability in dependency.get("vulns", []) or []:
            if not isinstance(vulnerability, dict):
                continue
            findings.append(
                {
                    "package": dependency.get("name"),
                    "version": dependency.get("version"),
                    "id": vulnerability.get("id"),
                    "fix_versions": vulnerability.get("fix_versions", []),
                    "description": vulnerability.get("description"),
                }
            )
    return findings


_LINE_DIAGNOSTIC = re.compile(r"^(?P<file>[^:\n]+):(?P<line>\d+):\s*(?P<message>.+)$")


def _parse_line_diagnostics(output: str) -> list[dict[str, Any]]:
    findings = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _LINE_DIAGNOSTIC.match(line)
        if match:
            findings.append(
                {
                    "file": match.group("file"),
                    "line": int(match.group("line")),
                    "message": match.group("message"),
                }
            )
        else:
            findings.append({"message": line})
    return findings


CHECKS = {
    "bandit": CheckSpec(
        name="bandit",
        executable="bandit",
        module="bandit",
        arguments=("-r", "src", "scripts", "-f", "json", "-q"),
        parser=_parse_bandit,
    ),
    "pip-audit": CheckSpec(
        name="pip-audit",
        executable="pip-audit",
        module="pip_audit",
        # --local avoids resolving the environment's requirements.  The
        # vulnerability database can still require network access, therefore
        # this check is disabled unless --allow-network is explicit.
        arguments=("--local", "--format", "json"),
        network=True,
        parser=_parse_pip_audit,
    ),
    "vulture": CheckSpec(
        name="vulture",
        executable="vulture",
        module="vulture",
        arguments=("src", "scripts", "--min-confidence", "80"),
        parser=_parse_line_diagnostics,
    ),
    "deptry": CheckSpec(
        name="deptry",
        executable="deptry",
        module="deptry",
        arguments=(".",),
        parser=_parse_line_diagnostics,
    ),
}


def _tool_command(spec: CheckSpec) -> list[str] | None:
    executable = shutil.which(spec.executable)
    if executable:
        return [executable, *spec.arguments]
    if importlib.util.find_spec(spec.module) is not None:
        return [sys.executable, "-m", spec.module, *spec.arguments]
    return None


def _compact_diagnostics(stdout: str, stderr: str) -> list[str]:
    diagnostics: list[str] = []
    for stream in (stdout, stderr):
        for line in stream.splitlines():
            value = line.strip()
            if value:
                diagnostics.append(value[:MAX_DIAGNOSTIC_CHARS])
            if len(diagnostics) >= MAX_DIAGNOSTICS:
                return diagnostics
    return diagnostics


def _run_check(spec: CheckSpec, *, allow_network: bool, timeout: int) -> dict[str, Any]:
    if spec.network and not allow_network:
        return {
            "name": spec.name,
            "status": "skipped",
            "reason": "network disabled; pass --allow-network to query the advisory database",
        }

    command = _tool_command(spec)
    if command is None:
        return {
            "name": spec.name,
            "status": "unavailable",
            "reason": f"install {spec.executable} before running this check",
        }

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"name": spec.name, "status": "error", "reason": f"timed out after {timeout}s"}
    except OSError as exc:
        return {"name": spec.name, "status": "error", "reason": str(exc)}

    findings = spec.parser(completed.stdout) if spec.parser else []
    if not findings and completed.stderr and spec.parser:
        # Some versions of text-oriented tools write findings to stderr.
        # Keep the primary JSON stream clean while still recognizing them.
        findings = spec.parser(completed.stderr)
    result: dict[str, Any] = {
        "name": spec.name,
        "status": "findings" if findings else ("passed" if completed.returncode == 0 else "error"),
        "exit_code": completed.returncode,
        "finding_count": len(findings),
    }
    if findings:
        result["findings"] = findings[:MAX_DIAGNOSTICS]
    if completed.returncode != 0 and not findings:
        result["diagnostics"] = _compact_diagnostics(completed.stdout, completed.stderr)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=tuple(CHECKS),
        action="append",
        metavar="CHECK",
        help="run only this check; repeat for multiple checks (default: all)",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="allow pip-audit to query its vulnerability advisory database",
    )
    parser.add_argument(
        "--require-tools",
        action="store_true",
        help="return exit code 2 when a selected tool is unavailable",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return exit code 1 when a check reports findings or an error",
    )
    parser.add_argument("--timeout", type=int, default=120, help="per-check timeout in seconds")
    parser.add_argument("--output", type=Path, help="also write the JSON result to this path")
    parser.add_argument("--pretty", action="store_true", help="indent the JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    selected = args.only or list(CHECKS)
    checks = [
        _run_check(CHECKS[name], allow_network=args.allow_network, timeout=args.timeout)
        for name in selected
    ]
    result = {
        "schema_version": "quality-checks/v1",
        "repository": str(ROOT),
        "network_allowed": bool(args.allow_network),
        "checks": checks,
        "summary": {
            "findings": sum(item.get("finding_count", 0) for item in checks),
            "passed": sum(item["status"] == "passed" for item in checks),
            "skipped": sum(item["status"] == "skipped" for item in checks),
            "unavailable": sum(item["status"] == "unavailable" for item in checks),
            "errors": sum(item["status"] == "error" for item in checks),
        },
    }
    indent = 2 if args.pretty else None
    separators = None if indent else (",", ":")
    rendered = json.dumps(result, sort_keys=True, indent=indent, separators=separators)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if args.require_tools and result["summary"]["unavailable"]:
        return 2
    if args.strict and (result["summary"]["findings"] or result["summary"]["errors"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
