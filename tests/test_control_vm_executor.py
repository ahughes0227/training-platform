import json
import subprocess
from types import SimpleNamespace

import pytest

from defect_platform.control.vm_executor import (
    BindMount,
    ContainerSpec,
    ExecutorError,
    HostIdentity,
    VMExecutor,
)

DIGEST = "us-docker.pkg.dev/project/repo/trainer@sha256:" + "a" * 64
RUN = "run-123"
ATTEMPT = "attempt-456"
COMMAND = ("python", "-m", "defect_platform.trainer.runner", "--request",
           "/defect/input/request.json", "--output-dir", "/defect/output")


class FakeRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run(self, argv, timeout):
        self.calls.append((tuple(argv), timeout))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def cp(code=0, out="", err=""):
    return subprocess.CompletedProcess([], code, out, err)


def inspect(state="created", cid="b" * 64, image=DIGEST, labels=None):
    return cp(out=json.dumps([{"Id": cid, "State": {"Status": state, "ExitCode": 0},
                              "Config": {"Image": image, "Labels": labels or {
                                  "defect-platform.managed": "true",
                                  "defect-platform.run-id": RUN,
                                  "defect-platform.attempt-id": ATTEMPT}}}]))


def spec(tmp_path):
    inputs = tmp_path / "request.json"
    inputs.write_text("{}")
    output, scratch = tmp_path / "output", tmp_path / "scratch"
    output.mkdir(); scratch.mkdir()
    return ContainerSpec(
        run_id=RUN, attempt_id=ATTEMPT, image_digest=DIGEST, command=COMMAND,
        output_dir=output, scratch_dir=scratch,
        inputs=(BindMount(inputs, "/defect/input/request.json", True),),
        uid_gid="10001:10001", gpu_devices=("0",), network_mode="defect-trainer-isolated",
        memory_bytes=8_000_000_000, pids_limit=512, timeout_seconds=3600,
        metadata_isolation_verified=True)


def test_create_uses_full_registry_digest_and_argv_without_shell(tmp_path):
    runner = FakeRunner([
        cp(1, err="Error: No such object"),
        cp(out=json.dumps([{"RepoDigests": [DIGEST]}])),
        cp(out="b" * 64),
        inspect(),
    ])
    identity = HostIdentity("project", "zone-a", "123", "boot-a")
    executor = VMExecutor(runner=runner, expected_identity=identity, identity_provider=lambda: identity)
    result = executor.create(spec(tmp_path))
    create_call = next(args for args, _ in runner.calls if args[1] == "create")
    assert result.observation.state == "created"
    assert result.container_id == "b" * 64
    assert "--privileged" not in create_call
    assert "--cap-drop" in create_call and "ALL" in create_call
    assert create_call[create_call.index("--mount") + 1].endswith("dst=/defect/input/request.json,ro")
    assert runner.calls[0][0][0] == "docker"


def test_uncertain_create_reconciles_deterministic_container_without_retry(tmp_path):
    runner = FakeRunner([
        cp(1, err="No such object"),
        cp(out=json.dumps([{"RepoDigests": [DIGEST]}])),
        cp(1, err="connection reset"),
        inspect(state="created"),
    ])
    identity = HostIdentity("project", "zone-a", "123", "boot-a")
    result = VMExecutor(runner=runner, expected_identity=identity,
                        identity_provider=lambda: identity).create(spec(tmp_path))
    assert result.observation.state == "created"
    assert sum(1 for args, _ in runner.calls if len(args) > 1 and args[1] == "create") == 1


@pytest.mark.parametrize("uid", ["0:0", "0:10001", "10001:0", "root:root"])
def test_container_spec_rejects_root_or_non_numeric_identity(tmp_path, uid):
    base = spec(tmp_path)
    with pytest.raises(ValueError, match="uid:gid"):
        ContainerSpec(**{**base.__dict__, "uid_gid": uid})


def test_container_spec_rejects_arbitrary_command(tmp_path):
    base = spec(tmp_path)
    with pytest.raises(ValueError, match="fixed defect trainer"):
        ContainerSpec(**{**base.__dict__, "command": ("sh", "-c", "echo nope")})


def test_container_spec_requires_named_isolated_network(tmp_path):
    base = spec(tmp_path)
    with pytest.raises(ValueError, match="named, isolated"):
        ContainerSpec(**{**base.__dict__, "network_mode": "bridge"})


def test_disk_admission_fails_closed_for_insufficient_peak_space(tmp_path):
    executor = VMExecutor(runner=FakeRunner([]))
    with pytest.raises(ExecutorError, match="insufficient free disk"):
        executor.require_capacity(tmp_path, peak_bytes=10**30, reserved_bytes=1,
                                  required_inodes=0, reserved_inodes=1)


def test_vm_boot_identity_mismatch_fences_side_effects(tmp_path):
    expected = HostIdentity("project", "zone-a", "123", "boot-a")
    current = HostIdentity("project", "zone-a", "123", "boot-b")
    executor = VMExecutor(runner=FakeRunner([]), expected_identity=expected,
                          identity_provider=lambda: current)
    with pytest.raises(ExecutorError, match="identity changed"):
        executor.create(spec(tmp_path))


def test_cancel_keeps_unknown_slot_held_until_exit_is_observed():
    cid = "b" * 64
    runner = FakeRunner([inspect(state="running", cid=cid), cp(out=cid), inspect(state="running", cid=cid)])
    identity = HostIdentity("project", "zone-a", "123", "boot-a")
    result = VMExecutor(runner=runner, expected_identity=identity,
                        identity_provider=lambda: identity).cancel(cid, run_id=RUN,
                                                                  attempt_id=ATTEMPT)
    assert result.observation.state == "running"
    assert "not yet observed" in result.detail


def test_cancel_created_container_removes_exact_instance_without_start():
    cid = "b" * 64
    runner = FakeRunner([inspect(state="created", cid=cid), cp(out=cid),
                         cp(1, err="No such object")])
    identity = HostIdentity("project", "zone-a", "123", "boot-a")
    result = VMExecutor(runner=runner, expected_identity=identity,
                        identity_provider=lambda: identity).cancel(cid, run_id=RUN, attempt_id=ATTEMPT)
    assert result.observation.state == "absent" and result.observation.exit_code is None
    mutations = [args for args, _ in runner.calls if args[1] != "inspect"]
    assert mutations == [("docker", "rm", "--", cid)]


def test_start_does_not_issue_engine_start_for_unknown_container():
    runner = FakeRunner([cp(1, err="daemon unavailable")])
    identity = HostIdentity("project", "zone-a", "123", "boot-a")
    result = VMExecutor(runner=runner, expected_identity=identity,
                        identity_provider=lambda: identity).start("b" * 64,
                                                                  run_id=RUN, attempt_id=ATTEMPT)
    assert result.observation.state == "unknown"
    assert not any(args[1] == "start" for args, _ in runner.calls)


def test_wrong_local_repo_digest_never_creates_container(tmp_path):
    runner = FakeRunner([
        cp(1, err="No such object"),
        cp(out=json.dumps([{"RepoDigests": ["other/repo@sha256:" + "c" * 64]}])),
    ])
    identity = HostIdentity("project", "zone-a", "123", "boot-a")
    with pytest.raises(ExecutorError, match="exact digest"):
        VMExecutor(runner=runner, expected_identity=identity,
                   identity_provider=lambda: identity).create(spec(tmp_path))
    assert not any(args[1] == "create" for args, _ in runner.calls)


def test_readiness_compares_selected_gpu_driver_engine_toolkit_and_digest():
    runner = FakeRunner([
        cp(out="GPU-123, 550.54.15\nGPU-456, 550.54.15\n"),
        cp(out="28.0.1"),
        cp(out="NVIDIA Container Runtime version 1.17.8\n"),
    ])
    readiness = SimpleNamespace(gpu_identity="GPU-123", driver_version="550.54.15",
                                engine_version="28.0.1", toolkit_version="1.17.8",
                                compatible_digests=[DIGEST])
    observed = VMExecutor(runner=runner).assert_readiness(
        readiness, image_digest=DIGEST, gpu_devices=("0",))
    assert observed["gpu_identity"] == "GPU-123"
    assert observed["driver_version"] == readiness.driver_version
    assert observed["engine_version"] == readiness.engine_version
    assert "1.17.8" in observed["toolkit_version"]


def test_readiness_fails_closed_when_gpu_driver_changes():
    runner = FakeRunner([cp(out="GPU-123, 555.1\n")])
    readiness = SimpleNamespace(gpu_identity="GPU-123", driver_version="550.54.15",
                                engine_version="28.0.1", toolkit_version="1.17.8",
                                compatible_digests=[DIGEST])
    with pytest.raises(ExecutorError, match="driver version"):
        VMExecutor(runner=runner).assert_readiness(
            readiness, image_digest=DIGEST, gpu_devices=("0",))
