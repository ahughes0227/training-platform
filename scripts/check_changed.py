#!/usr/bin/env python3
"""Run fast, changed-file validation for local edits.

The command intentionally emits one short result per check.  It is suitable
for a pre-commit hook or for a Luna agent that needs high-signal feedback
without resending a full test log.  Use ``--base`` to compare against a
branch, tag, or commit; untracked files are included automatically.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _changed_files(base: str) -> list[Path]:
    names = set(_git("diff", "--name-only", "--diff-filter=ACMR", base, "--"))
    names.update(_git("ls-files", "--others", "--exclude-standard"))
    paths = []
    for name in sorted(names):
        path = (ROOT / name).resolve()
        if path.is_file() and (path == ROOT or ROOT in path.parents):
            paths.append(path.relative_to(ROOT))
    return paths


def _run(
    label: str,
    command: list[str],
    *,
    allow_missing: bool,
    env: dict[str, str] | None = None,
) -> bool:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            env=env,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        if allow_missing:
            print(f"SKIP {label}: executable unavailable")
            return True
        print(f"FAIL {label}: executable unavailable", file=sys.stderr)
        return False
    if result.returncode:
        print(f"FAIL {label} (exit {result.returncode})", file=sys.stderr)
        diagnostics = [
            line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()
        ]
        for line in diagnostics[:6]:
            print(f"  {line[:240]}", file=sys.stderr)
        if len(diagnostics) > 6:
            print(f"  ... {len(diagnostics) - 6} more diagnostic lines omitted", file=sys.stderr)
        return False
    print(f"PASS {label}")
    return True


def _run_module(
    label: str,
    module: str,
    arguments: list[str],
    *,
    allow_missing: bool,
) -> bool:
    if importlib.util.find_spec(module) is None:
        if allow_missing:
            print(f"SKIP {label}: {module} is not installed")
            return True
        print(f"FAIL {label}: {module} is not installed", file=sys.stderr)
        return False
    return _run(
        label,
        [sys.executable, "-m", module, *arguments],
        allow_missing=allow_missing,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="HEAD",
        help="Git ref used as the comparison base (default: HEAD).",
    )
    parser.add_argument(
        "--allow-missing-tools",
        action="store_true",
        help="Skip optional tools that are not installed in the active environment.",
    )
    parser.add_argument(
        "--skip-pyright",
        action="store_true",
        help="Skip type checking; useful when reviewing non-Python changes.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        changed = _changed_files(args.base)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else "unknown git error"
        print(f"FAIL changed-file discovery: {detail}", file=sys.stderr)
        return 2

    python_files = [path for path in changed if path.suffix == ".py"]
    print(f"Changed files: {len(changed)}; Python files: {len(python_files)}")
    checks: list[bool] = []

    checks.append(_run("git diff --check", ["git", "diff", "--check"], allow_missing=False))
    if not python_files:
        print("No changed Python files; source checks skipped.")
        return 0 if all(checks) else 1

    paths = [str(path) for path in python_files]
    checks.append(
        _run_module(
            "ruff check (changed Python files)",
            "ruff",
            ["check", "--select", "E9,F", *paths],
            allow_missing=args.allow_missing_tools,
        )
    )
    checks.append(
        _run_module(
            "ruff format --check (changed Python files)",
            "ruff",
            ["format", "--check", *paths],
            allow_missing=args.allow_missing_tools,
        )
    )
    checks.append(
        _run(
            "compileall (changed Python files)",
            [sys.executable, "-m", "compileall", "-q", *paths],
            allow_missing=False,
        )
    )
    if not args.skip_pyright:
        checks.append(
            _run_module(
                "pyright (changed Python files)",
                "pyright",
                paths,
                allow_missing=args.allow_missing_tools,
            )
        )

    contract_files = {
        Path("scripts/export_contracts.py"),
        Path("src/defect_platform/contracts.py"),
        Path("src/defect_platform/infrastructure_contract.py"),
    }
    if any(path in contract_files or path.parts[:1] == ("schemas",) for path in changed):
        checks.append(
            _run(
                "generated contract schemas",
                [sys.executable, "scripts/export_contracts.py", "--check"],
                allow_missing=False,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        [str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
                    ).rstrip(os.pathsep),
                },
            )
        )

    passed = sum(checks)
    print(f"Checks: {passed}/{len(checks)} passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
