"""The full goal loop on the CPU demo: real dataset build, DINOv3 training and reports."""

from __future__ import annotations

import json

import pytest

for module in ("torch", "transformers", "webdataset", "PIL"):
    pytest.importorskip(module)


def test_demo_goal_is_met_and_unreachable_demo_explains_why(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from defect_platform.cli import app

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cli = CliRunner()

    met = cli.invoke(app, ["goal", "demo", str(tmp_path / "demo")])
    assert met.exit_code == 0, met.output
    assert "model delivered" in met.output
    scorecard = json.loads((tmp_path / "demo" / "goals" / "demo" / "scorecard.json").read_text())
    assert scorecard["outcome"] == "model_delivered"
    assert (tmp_path / "demo" / "goals" / "demo" / "deliverable" / "model" / "model.pt").exists()

    again = cli.invoke(app, ["goal", "demo", str(tmp_path / "demo")])
    assert again.exit_code == 0, again.output
    assert "earlier demo attempt was moved" in again.output
    assert "model delivered" in again.output

    unmet = cli.invoke(app, ["goal", "demo", str(tmp_path / "hard"), "--unreachable"])
    assert unmet.exit_code == 0, unmet.output
    assert "goal not met" in unmet.output
    report = (tmp_path / "hard" / "goals" / "demo-unreachable" / "REPORT.md").read_text()
    assert "What would most likely reach the goal" in report

    status = cli.invoke(app, ["goal", "status", str(tmp_path / "hard" / "goals" / "demo-unreachable")])
    assert status.exit_code == 0
    assert "goal not met" in status.output
