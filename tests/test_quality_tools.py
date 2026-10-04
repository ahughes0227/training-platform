from __future__ import annotations

import json

from scripts import quality_checks


def test_bandit_parser_keeps_machine_readable_fields() -> None:
    findings = quality_checks._parse_bandit(
        json.dumps(
            {
                "results": [
                    {
                        "filename": "src/example.py",
                        "line_number": 7,
                        "test_id": "B101",
                        "issue_severity": "LOW",
                        "issue_confidence": "HIGH",
                        "issue_text": "assert used",
                    }
                ]
            }
        )
    )

    assert findings == [
        {
            "file": "src/example.py",
            "line": 7,
            "rule": "B101",
            "severity": "LOW",
            "confidence": "HIGH",
            "message": "assert used",
        }
    ]


def test_pip_audit_parser_flattens_each_vulnerability() -> None:
    findings = quality_checks._parse_pip_audit(
        json.dumps(
            {
                "dependencies": [
                    {
                        "name": "example",
                        "version": "1.0",
                        "vulns": [
                            {"id": "PYSEC-1", "fix_versions": ["1.1"], "description": "upgrade"},
                            {"id": "CVE-2", "fix_versions": [], "description": "patch"},
                        ],
                    }
                ]
            }
        )
    )

    assert [finding["id"] for finding in findings] == ["PYSEC-1", "CVE-2"]
    assert all(finding["package"] == "example" for finding in findings)


def test_line_parser_preserves_unstructured_tool_diagnostics() -> None:
    findings = quality_checks._parse_line_diagnostics(
        "src/example.py:12: unused function\nsummary: 1 issue\n"
    )

    assert findings == [
        {"file": "src/example.py", "line": 12, "message": "unused function"},
        {"message": "summary: 1 issue"},
    ]


def test_quality_main_writes_compact_result_and_strict_status(tmp_path, monkeypatch) -> None:
    def fake_run_check(spec, *, allow_network, timeout):
        assert allow_network is False
        assert timeout == 3
        return {"name": spec.name, "status": "passed", "finding_count": 0, "exit_code": 0}

    monkeypatch.setattr(quality_checks, "_run_check", fake_run_check)
    output = tmp_path / "quality.json"

    assert (
        quality_checks.main(
            ["--only", "vulture", "--strict", "--timeout", "3", "--output", str(output)]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "quality-checks/v1"
    assert payload["summary"] == {
        "errors": 0,
        "findings": 0,
        "passed": 1,
        "skipped": 0,
        "unavailable": 0,
    }


def test_quality_main_require_tools_returns_two_for_unavailable(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        quality_checks,
        "_run_check",
        lambda spec, *, allow_network, timeout: {"name": spec.name, "status": "unavailable"},
    )

    assert quality_checks.main(["--only", "bandit", "--require-tools"]) == 2
    assert '"unavailable":1' in capsys.readouterr().out
