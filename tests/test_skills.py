from datetime import UTC, datetime

from defect_platform.skills import (
    FilesystemProjectReader,
    ProjectState,
    SkillNamespace,
    check_local_readiness,
    generate_project_ref,
    render_project_table,
    skill_definitions,
    welcome_text,
)


def test_skill_catalog_has_normal_names_and_namespaces():
    skills = skill_definitions()
    assert len(skills) >= 30
    assert all(1 <= len(skill.name.split()) <= 3 for skill in skills)
    assert {skill.namespace for skill in skills} == set(SkillNamespace)
    assert "Welcome to the DINOv3 Training Platform." in welcome_text()


def test_project_ref_is_readable_and_stable_for_seed():
    now = datetime(2026, 10, 1, 12, 30, tzinfo=UTC)
    first = generate_project_ref(now=now, seed="demo")
    second = generate_project_ref(now=now, seed="demo")
    assert first.project_id == second.project_id
    assert first.display_name.endswith(" 20261001-123000Z")
    assert "-" in first.display_name
    assert len(first.project_id.rsplit("-", 1)[-1]) == 8


def test_project_reader_derives_states(tmp_path):
    root = tmp_path / "projects"
    draft = root / "draft"
    draft.mkdir(parents=True)
    reader = FilesystemProjectReader(root)
    assert reader.list_projects()[0].state is ProjectState.DRAFT

    (draft / "object.yaml").write_text(
        "slug: widget\ndisplay_name: Widget\ndescription: Crops\nclasses: [a, b]\n"
    )
    assert reader.list_projects()[0].state is ProjectState.DATA_INGESTION

    (draft / "dataset-version.yaml").write_text("version_id: v1\n")
    assert reader.list_projects()[0].state is ProjectState.DATASET_READY

    (draft / "experiment.yaml").write_text("experiment_id: e1\n")
    assert reader.list_projects()[0].state is ProjectState.EXPERIMENT_READY
    table = render_project_table(reader.list_projects())
    assert "Project" in table and "experiment_ready" in table


def test_readiness_reports_unconfigured_cloud_without_blocking():
    report = check_local_readiness(root=".")
    checks = {check.check_id: check for check in report.checks}
    assert checks["project_root"].status == "ready"
    assert checks["DEFECT_CONTROL_SERVICE_URL"].status == "unconfigured"
    assert report.status in {"ready", "blocked"}
