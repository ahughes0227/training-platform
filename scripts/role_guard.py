"""VS Code Local PreToolUse boundary check for named engineering agents.

This hook is a convenience guard. IAM and protected release commands enforce
remote actions even when a different editor or agent harness is used.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import PurePosixPath
from typing import Any

OWNED_PREFIXES = {
    "trainer": ("src/defect_platform/trainer/", "tests/test_trainer"),
    "runtime": ("infra/docker/", "Dockerfile", "requirements-runtime", "tests/test_runtime"),
    "certifier": ("certifications/", "tests/test_certification"),
    "experiment": ("src/defect_platform/control/", "projects/", "tests/test_control"),
    "analysis": ("src/defect_platform/analysis/", "tests/test_analysis"),
}

# Every CLI path that submits a paid run, not only `defect run submit`.
SUBMIT = r"\bdefect\s+(?:run\s+submit|train\s+(?:start|guided|queue))\b"
BUILD_IMAGE = r"\bdocker\s+(?:image\s+)?build\b"
PUSH_IMAGE = r"\bdocker\s+(?:image\s+)?push\b"
BUILD_OR_PUSH = r"\bdocker\s+(?:image\s+)?(?:build|push)\b"
CERTIFY = r"\bdefect\s+runtime\s+certify\b"
VERTEX_CREATE = r"\bgcloud\s+ai\s+custom-jobs\s+create\b"
CLOUD_BUILD = r"\bgcloud\s+builds\s+submit\b"

DENIED_COMMANDS = {
    "trainer": [BUILD_OR_PUSH, VERTEX_CREATE, SUBMIT, CERTIFY],
    "runtime": [PUSH_IMAGE, VERTEX_CREATE, SUBMIT, CERTIFY],
    "certifier": [BUILD_IMAGE, SUBMIT],
    "experiment": [BUILD_OR_PUSH, CLOUD_BUILD, CERTIFY],
    "analysis": [BUILD_OR_PUSH, CLOUD_BUILD, VERTEX_CREATE, SUBMIT,
                 CERTIFY, r"\bdefect\s+dataset\s+build\b",
                 r"\bdefect\s+release\s+(?:stage|promote|rollback)\b"],
}


def _normalize(path: str) -> str:
    marker = "training-platform/"
    normalized = path.replace("\\", "/")
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    return str(PurePosixPath(normalized)).removeprefix("./")


# Mutating tools an agent reaches a boundary through, by harness. Claude Code
# names its write tools' argument file_path and exposes platform actions as
# mcp__<server>__<tool>, neither of which the editor key names cover.
MUTATING_TOOL_WORDS = ("edit", "write", "create_file", "apply_patch", "delete", "notebookedit")
DENIED_TOOLS = {
    "trainer": ("launch_run", "create_project"),
    "runtime": ("launch_run", "create_project"),
    "certifier": ("launch_run", "create_project"),
    "experiment": (),
    "analysis": ("launch_run", "create_project"),
}


def _tool_action(tool_name: str) -> str:
    """The bare action of an MCP tool name, e.g. mcp__training-platform__launch_run."""
    return tool_name.rsplit("__", 1)[-1] if tool_name.startswith("mcp__") else tool_name


def _file_paths(tool_input: dict[str, Any]) -> list[str]:
    paths = []
    for key in ("filePath", "file_path", "path", "target", "uri", "notebook_path"):
        value = tool_input.get(key)
        if isinstance(value, str) and not value.startswith(("http:", "https:")):
            paths.append(_normalize(value))
    patch = tool_input.get("patch", tool_input.get("input", ""))
    if isinstance(patch, str):
        paths.extend(_normalize(path) for path in re.findall(
            r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", patch, re.MULTILINE
        ))
    return paths


def policy_decision(role: str, payload: dict[str, Any]) -> str | None:
    if role not in OWNED_PREFIXES:
        return "unknown role"
    tool_name = str(payload.get("tool_name", "")).lower()
    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return "unexpected tool input"
    command = tool_input.get("command", tool_input.get("cmd", ""))
    if isinstance(command, str):
        for pattern in DENIED_COMMANDS[role]:
            if re.search(pattern, command, re.IGNORECASE):
                return f"{role} role cannot run this cross-layer command"
    action = _tool_action(tool_name)
    if action in DENIED_TOOLS[role]:
        return f"{role} role cannot use the {action} platform tool"
    if any(word in tool_name for word in MUTATING_TOOL_WORDS):
        for path in _file_paths(tool_input):
            if path.startswith(("../", "/")):
                return "file write escapes the repository"
            if not any(path.startswith(prefix) for prefix in OWNED_PREFIXES[role]):
                return f"{role} role does not own {path}"
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("role argument required", file=sys.stderr)
        return 2
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        print("invalid hook JSON", file=sys.stderr)
        return 2
    reason = policy_decision(sys.argv[1], payload)
    if reason:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
