"""Restricted local container-engine adapter for queue-owned VM executions.

This module never builds, pushes, or pulls images. It only creates a fresh
container from an already-present, exact immutable digest.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class ExecutorError(RuntimeError):
    """An engine operation failed; callers must preserve its evidence."""

    def __init__(self, message: str, *, argv: Sequence[str] = (), detail: str = ""):
        super().__init__(message)
        self.argv = tuple(argv)
        self.detail = detail


class CommandRunner(Protocol):
    def run(self, argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]: ...


class SubprocessRunner:
    def run(self, argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(list(argv), text=True, capture_output=True, timeout=timeout,
                              check=False, shell=False)


@dataclass(frozen=True)
class BindMount:
    source: Path
    target: str
    readonly: bool = True

    def __post_init__(self) -> None:
        if not self.source.is_absolute() or not self.target.startswith("/"):
            raise ValueError("mount source and target must be absolute")
        if "," in str(self.source) or "," in self.target:
            raise ValueError("mount paths cannot contain commas")


@dataclass(frozen=True)
class ContainerSpec:
    run_id: str
    attempt_id: str
    image_digest: str
    command: tuple[str, ...]
    output_dir: Path
    scratch_dir: Path
    inputs: tuple[BindMount, ...]
    uid_gid: str
    gpu_devices: tuple[str, ...]
    network_mode: str
    memory_bytes: int
    pids_limit: int
    timeout_seconds: int
    environment: tuple[tuple[str, str], ...] = ()
    credential_mounts: tuple[BindMount, ...] = ()
    labels: tuple[tuple[str, str], ...] = ()
    # Explicit opt-in; ordinary Docker isolation does not prove metadata or IAM isolation.
    metadata_isolation_verified: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[a-f0-9]{64}", self.image_digest):
            raise ValueError("image_digest must be a repository-qualified immutable sha256 digest")
        for name, value in (("run_id", self.run_id), ("attempt_id", self.attempt_id)):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
                raise ValueError(f"invalid {name}")
        if not self.command or any("\x00" in part for part in self.command):
            raise ValueError("container command must be a non-empty argv")
        if self.command != ("python", "-m", "defect_platform.trainer.runner", "--request",
                            "/defect/input/request.json", "--output-dir", "/defect/output"):
            raise ValueError("container command must invoke the fixed defect trainer entrypoint")
        if len(self.inputs) != 1 or self.inputs[0].target != "/defect/input/request.json" or not self.inputs[0].readonly:
            raise ValueError("only the immutable readonly request file may be mounted as input")
        for mount in (*self.inputs, *self.credential_mounts):
            if mount.source.is_symlink() or not mount.source.is_file():
                raise ValueError("trainer input and credential mounts must be existing regular files")
        for mount in self.credential_mounts:
            if not mount.readonly or not (mount.target == "/defect/credentials.json"
                                           or mount.target.startswith("/defect/credentials/")):
                raise ValueError("credentials must use readonly mounts under /defect/credentials/")
            source = mount.source.resolve()
            if source == Path("/") or any(source == root or root in source.parents for root in (
                    Path("/run"), Path("/var/run"), Path("/var/lib/defect-platform"),
                    Path("/proc"), Path("/sys"), Path("/dev"))):
                raise ValueError("credential mount source overlaps a protected host control path")
        if not re.fullmatch(r"([1-9][0-9]*):([1-9][0-9]*)", self.uid_gid):
            raise ValueError("a non-root numeric uid:gid is required")
        if not self.gpu_devices or any(not re.fullmatch(r"[0-9]+", x) for x in self.gpu_devices):
            raise ValueError("explicit GPU device indices are required")
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", self.network_mode)
                or self.network_mode in {"host", "container", "none", "bridge"}):
            raise ValueError("a named, isolated trainer network must be explicit")
        if self.memory_bytes <= 0 or self.pids_limit <= 0 or self.timeout_seconds <= 0:
            raise ValueError("memory, pid, and timeout limits must be positive")
        if not self.metadata_isolation_verified:
            raise ValueError("metadata and cloud credential isolation must be verified before launch")
        if not self.output_dir.is_absolute() or not self.scratch_dir.is_absolute():
            raise ValueError("attempt output and scratch directories must be absolute")
        if (self.output_dir == self.scratch_dir or self.output_dir in self.scratch_dir.parents
                or self.scratch_dir in self.output_dir.parents):
            raise ValueError("output and scratch directories must be distinct siblings")
        if self.output_dir.is_symlink() or self.scratch_dir.is_symlink() or not self.output_dir.is_dir() or not self.scratch_dir.is_dir():
            raise ValueError("attempt output and scratch mounts must be existing real directories")
        names = [key for key, _ in self.environment]
        if len(names) != len(set(names)) or any(not re.fullmatch(r"[A-Z_][A-Z0-9_]*", n) for n in names):
            raise ValueError("environment names must be unique uppercase identifiers")
        if set(names) - {"GOOGLE_APPLICATION_CREDENTIALS", "HF_HOME", "TORCH_HOME"}:
            raise ValueError("container environment contains a non-allowlisted variable")
        if any("\x00" in value for _, value in self.environment):
            raise ValueError("environment values cannot contain NUL")


@dataclass(frozen=True)
class ExecutionObservation:
    state: str  # absent, created, running, exited, unknown
    container_id: str | None = None
    exit_code: int | None = None
    image: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    process_id: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    container_id: str | None
    observation: ExecutionObservation
    command: tuple[str, ...]
    detail: str = ""


@dataclass(frozen=True)
class DiskCapacity:
    path: Path
    free_bytes: int
    free_inodes: int
    capacity_bytes: int


@dataclass(frozen=True)
class HostIdentity:
    project: str
    zone: str
    instance_id: str
    boot_id: str


def compute_engine_identity(*, metadata_get=None,
                            boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id")) -> HostIdentity:
    """Read the current GCE numeric instance identity and Linux boot identity.

    Metadata reads use the required Metadata-Flavor header and short timeout;
    missing/unavailable identity raises instead of falling back to hostnames.
    """
    if metadata_get is None:
        def metadata_get(path: str) -> str:
            request = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/" + path,
                headers={"Metadata-Flavor": "Google"})
            with urllib.request.urlopen(request, timeout=2) as response:
                if response.headers.get("Metadata-Flavor") != "Google":
                    raise ExecutorError("GCE metadata response did not identify its source")
                return response.read().decode("utf-8").strip()
    try:
        project = metadata_get("project/project-id")
        zone = metadata_get("instance/zone").rsplit("/", 1)[-1]
        instance_id = metadata_get("instance/id")
        boot_id = boot_id_path.read_text().strip()
    except Exception as exc:
        raise ExecutorError("Compute Engine instance or boot identity is unavailable", detail=str(exc)) from exc
    if not all((project, zone, instance_id, boot_id)):
        raise ExecutorError("Compute Engine instance or boot identity is incomplete")
    return HostIdentity(project, zone, instance_id, boot_id)


def read_gce_host_identity() -> HostIdentity:
    """Default identity provider used by the supervised Compute Engine worker."""
    return compute_engine_identity()


class VMExecutor:
    """Local engine operations, keyed by the persisted attempt/container identity."""

    def __init__(self, *, runner: CommandRunner | None = None, engine: str = "docker",
                 timeout_seconds: int = 30, expected_identity: HostIdentity | None = None,
                 identity_provider=None, max_log_bytes: int = 8 * 1024 * 1024):
        if max_log_bytes <= 0:
            raise ValueError("max_log_bytes must be positive")
        self.runner = runner or SubprocessRunner()
        self.engine = engine
        self.timeout_seconds = timeout_seconds
        self.expected_identity = expected_identity
        self.identity_provider = identity_provider
        self.max_log_bytes = max_log_bytes
        if not os.path.isabs(engine) and shutil.which(engine) is None and runner is None:
            raise ValueError(f"container engine not found: {engine}")

    @staticmethod
    def container_name(run_id: str, attempt_id: str) -> str:
        # Must exactly match QueueStore's persisted deterministic name.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", attempt_id):
            raise ValueError("run and attempt IDs cannot form a safe deterministic container name")
        return f"defect-{run_id}-{attempt_id}"

    @staticmethod
    def attempt_directories(root: Path, run_id: str, attempt_id: str) -> tuple[Path, Path]:
        base = root.resolve() / run_id / attempt_id
        output, scratch = base / "output", base / "scratch"
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        try:
            scratch.mkdir(mode=0o700)
        except Exception:
            # The directory is newly allocated and contains no evidence yet.
            output.rmdir()
            output.parent.rmdir()
            raise
        os.chmod(output, 0o700)
        os.chmod(scratch, 0o700)
        return output, scratch

    def disk_capacity(self, path: Path) -> DiskCapacity:
        stat = os.statvfs(path)
        return DiskCapacity(path, stat.f_bavail * stat.f_frsize,
                            stat.f_favail, stat.f_blocks * stat.f_frsize)

    def require_capacity(self, path: Path, *, peak_bytes: int, reserved_bytes: int,
                         required_inodes: int, reserved_inodes: int) -> DiskCapacity:
        if min(peak_bytes, reserved_bytes, required_inodes, reserved_inodes) < 0:
            raise ValueError("disk reservations cannot be negative")
        cap = self.disk_capacity(path)
        if cap.free_bytes < peak_bytes + reserved_bytes:
            raise ExecutorError("insufficient free disk capacity for peak footprint and headroom")
        if cap.free_inodes < required_inodes + reserved_inodes:
            raise ExecutorError("insufficient free inodes for peak footprint and headroom")
        return cap

    def _run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            return self.runner.run(argv, self.timeout_seconds)
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise ExecutorError("container engine command outcome is uncertain", argv=argv,
                                detail=str(exc)) from exc

    def scan_gpu_workloads(self) -> list[dict[str, str]]:
        """Observe running engine GPU containers and host GPU compute processes.

        Unmappable GPU PIDs are emitted as unknown blockers. An engine or NVIDIA
        observer failure is an error, never interpreted as an empty GPU pool.
        """
        ps_argv = [self.engine, "ps", "-q"]
        ps = self._run(ps_argv)
        if ps.returncode:
            raise ExecutorError("cannot enumerate running containers", argv=ps_argv,
                                detail=(ps.stderr or ps.stdout).strip())
        found: dict[str, dict[str, str]] = {}
        for cid in filter(None, ps.stdout.splitlines()):
            obs = self._inspect(cid)
            if obs.state == "unknown":
                raise ExecutorError("cannot inspect running container", argv=[self.engine, "inspect", cid],
                                    detail=obs.error or "unknown inspect result")
            if obs.state != "running":
                continue
            inspect = self._run([self.engine, "inspect", cid])
            if inspect.returncode:
                raise ExecutorError("cannot inspect GPU configuration", argv=[self.engine, "inspect", cid],
                                    detail=(inspect.stderr or inspect.stdout).strip())
            try:
                data = json.loads(inspect.stdout)[0]
                requests = data.get("HostConfig", {}).get("DeviceRequests") or []
                uses_gpu = any("gpu" in [c.lower() for c in req.get("Capabilities", [[]])[0]]
                               for req in requests)
                if uses_gpu:
                    full_id = data.get("Id") or cid
                    found[full_id] = {"container_id": full_id, "kind": "container-gpu"}
            except (ValueError, IndexError, TypeError, AttributeError) as exc:
                raise ExecutorError("invalid container GPU observation", detail=str(exc)) from exc
        nvidia_argv = ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"]
        nvidia = self._run(nvidia_argv)
        if nvidia.returncode:
            raise ExecutorError("cannot observe host GPU compute processes", argv=nvidia_argv,
                                detail=(nvidia.stderr or nvidia.stdout).strip())
        for line in filter(None, nvidia.stdout.splitlines()):
            pid_text = line.split(",", 1)[0].strip()
            try:
                cgroup = Path(f"/proc/{int(pid_text)}/cgroup").read_text()
            except (ValueError, OSError):
                found[f"pid:{pid_text}"] = {"container_id": "", "kind": "unmapped-gpu-process"}
                continue
            match = re.search(r"(?:docker-|/)([a-f0-9]{64})(?:\.scope|/|$)", cgroup)
            if match and match.group(1) in found:
                continue
            # Short IDs may appear in cgroups; resolve against enumerated full IDs.
            possible = [cid for cid in found if match and cid.startswith(match.group(1))]
            if len(possible) == 1:
                continue
            found[f"pid:{pid_text}"] = {"container_id": "", "kind": "unmapped-gpu-process"}
        return list(found.values())

    @staticmethod
    def _labels(spec: ContainerSpec) -> dict[str, str]:
        labels = dict(spec.labels)
        labels.update({"defect-platform.managed": "true", "defect-platform.run-id": spec.run_id,
                       "defect-platform.attempt-id": spec.attempt_id})
        return labels

    def _inspect(self, identifier: str, *, name: str | None = None) -> ExecutionObservation:
        argv = [self.engine, "inspect", identifier]
        result = self._run(argv)
        if result.returncode:
            # A nonzero response means absence only when engine explicitly said so.
            if "no such object" in (result.stderr or "").lower() or "not found" in (result.stderr or "").lower():
                return ExecutionObservation("absent")
            return ExecutionObservation("unknown", error=(result.stderr or result.stdout).strip())
        try:
            data = json.loads(result.stdout)[0]
            state = data.get("State", {})
            status = state.get("Status", "unknown")
            labels = data.get("Config", {}).get("Labels") or {}
            pid = state.get("Pid")
            return ExecutionObservation(
                {"created": "created", "running": "running", "exited": "exited",
                 "dead": "unknown", "paused": "unknown", "restarting": "unknown"}.get(status, "unknown"),
                data.get("Id"), state.get("ExitCode") if status == "exited" else None,
                data.get("Config", {}).get("Image"), labels,
                None if status in {"created", "running", "exited"} else f"unexpected engine state: {status}",
                str(pid) if status == "running" and isinstance(pid, int) and pid > 0 else None)
        except (ValueError, IndexError, TypeError, AttributeError) as exc:
            return ExecutionObservation("unknown", error=f"invalid inspect response: {exc}")

    def inspect(self, run_id: str, attempt_id: str, container_id: str | None = None) -> ExecutionObservation:
        ident = container_id or self.container_name(run_id, attempt_id)
        obs = self._inspect(ident)
        if obs.state != "absent" and obs.state != "unknown":
            labels = obs.labels
            if labels.get("defect-platform.managed") != "true" or labels.get("defect-platform.run-id") != run_id or labels.get("defect-platform.attempt-id") != attempt_id:
                return ExecutionObservation("unknown", obs.container_id, obs.exit_code, obs.image,
                                            labels, "container identity labels do not match persisted attempt")
        return obs

    def assert_host_identity(self) -> HostIdentity:
        if self.expected_identity is None or self.identity_provider is None:
            raise ExecutorError("VM/boot identity fencing is not configured")
        current = self.identity_provider()
        if current != self.expected_identity:
            raise ExecutorError("VM or boot identity changed; execution is fenced")
        return current

    def _observe(self, argv: Sequence[str]) -> str:
        result = self._run(argv)
        if result.returncode:
            raise ExecutorError("VM runtime readiness command failed", argv=argv,
                                detail=(result.stderr or result.stdout).strip())
        return result.stdout.strip()

    def assert_readiness(self, readiness, *, image_digest: str | None = None,
                         gpu_devices: Sequence[str] | None = None) -> dict[str, str]:
        """Compare observed GPU, driver, container engine, and toolkit with reviewed evidence."""
        if image_digest is not None and image_digest not in readiness.compatible_digests:
            raise ExecutorError("exact certified digest is not approved for this VM")
        gpu_argv = ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"]
        gpu_rows = self._observe(gpu_argv).splitlines()
        parsed: list[tuple[str, str]] = []
        for line in gpu_rows:
            values = [part.strip() for part in line.split(",", 1)]
            if len(values) != 2 or not values[0] or not values[1]:
                raise ExecutorError("GPU readiness response is malformed", argv=gpu_argv, detail=line)
            parsed.append((values[0], values[1]))
        expected_gpu_identity = ",".join(part.strip() for part in readiness.gpu_identity.split(","))
        device_indices = tuple(gpu_devices) if gpu_devices is not None else tuple(
            str(index) for index in range(len(expected_gpu_identity.split(","))))
        selected = []
        for device in device_indices:
            if not re.fullmatch(r"[0-9]+", device) or int(device) >= len(parsed):
                raise ExecutorError("configured GPU index is not observed on this VM", argv=gpu_argv,
                                    detail=f"gpu index={device}; observed={len(parsed)}")
            selected.append(parsed[int(device)])
        observed_gpu_identity = ",".join(uuid for uuid, _ in selected)
        observed_drivers = {version for _, version in selected}
        if observed_gpu_identity != expected_gpu_identity:
            raise ExecutorError("selected GPU UUIDs differ from approved VM readiness", argv=gpu_argv,
                                detail=f"expected={expected_gpu_identity}; observed={observed_gpu_identity}")
        if observed_drivers != {readiness.driver_version}:
            raise ExecutorError("selected GPU driver version differs from approved readiness", argv=gpu_argv,
                                detail=f"expected={readiness.driver_version}; observed={sorted(observed_drivers)}")
        engine_argv = [self.engine, "version", "--format", "{{.Server.Version}}"]
        observed_engine = self._observe(engine_argv)
        if observed_engine != readiness.engine_version:
            raise ExecutorError("container engine version differs from approved readiness", argv=engine_argv,
                                detail=f"expected={readiness.engine_version}; observed={observed_engine}")
        toolkit_argv = ["nvidia-container-runtime", "--version"]
        toolkit_output = self._observe(toolkit_argv)
        first_line = toolkit_output.splitlines()[0] if toolkit_output.splitlines() else ""
        observed_toolkit_version = first_line.split()[-1] if first_line.split() else ""
        if observed_toolkit_version != readiness.toolkit_version:
            raise ExecutorError("NVIDIA container toolkit differs from approved readiness",
                                argv=toolkit_argv,
                                detail=f"expected={readiness.toolkit_version}; observed={first_line[:1000]}")
        return {"gpu_identity": observed_gpu_identity, "driver_version": next(iter(observed_drivers)),
                "engine_version": observed_engine, "toolkit_version": observed_toolkit_version,
                "image_digest": image_digest or ";".join(readiness.compatible_digests)}

    def create(self, spec: ContainerSpec) -> ExecutionResult:
        self.assert_host_identity()
        name = self.container_name(spec.run_id, spec.attempt_id)
        before = self.inspect(spec.run_id, spec.attempt_id)
        if before.state != "absent":
            return ExecutionResult(before.container_id, before, (), "create suppressed: matching identity already exists or is uncertain")
        # Verify the exact digest is already present; never trigger a pull/build.
        image_argv = [self.engine, "image", "inspect", spec.image_digest]
        image_result = self._run(image_argv)
        if image_result.returncode:
            raise ExecutorError("exact certified image digest is unavailable locally", argv=image_argv,
                                detail=(image_result.stderr or image_result.stdout).strip())
        try:
            image_info = json.loads(image_result.stdout)[0]
            if spec.image_digest not in (image_info.get("RepoDigests") or []):
                raise ExecutorError("local image RepoDigests do not contain the approved exact digest",
                                    argv=image_argv, detail=str(image_info.get("RepoDigests")))
        except (ValueError, IndexError, TypeError) as exc:
            raise ExecutorError("invalid exact image inspection response", argv=image_argv,
                                detail=str(exc)) from exc
        argv = [self.engine, "create", "--name", name, "--restart", "no", "--user", spec.uid_gid,
                "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
                "--pids-limit", str(spec.pids_limit), "--memory", str(spec.memory_bytes),
                "--network", spec.network_mode, "--gpus", "device=" + ",".join(spec.gpu_devices),
                "--log-driver", "local", "--log-opt", "max-size=10m",
                "--log-opt", "max-file=3",
                "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=1g"]
        for key, value in self._labels(spec).items():
            argv += ["--label", f"{key}={value}"]
        for key, value in spec.environment:
            argv += ["--env", f"{key}={value}"]
        argv += ["--mount", f"type=bind,src={spec.inputs[0].source},dst=/defect/input/request.json,ro",
                 "--mount", f"type=bind,src={spec.output_dir},dst=/defect/output,rw",
                 "--mount", f"type=bind,src={spec.scratch_dir},dst=/defect/scratch,rw"]
        for mount in spec.credential_mounts:
            mode = "ro" if mount.readonly else "rw"
            argv += ["--mount", f"type=bind,src={mount.source},dst={mount.target},{mode}"]
        argv += [spec.image_digest, *spec.command]
        result = self._run(argv)
        if result.returncode == 0:
            cid = result.stdout.strip()
            if not re.fullmatch(r"[a-f0-9]{12,64}", cid):
                return ExecutionResult(None, ExecutionObservation("unknown", error="engine returned invalid container id"), tuple(argv), result.stdout)
            return ExecutionResult(cid, self.inspect(spec.run_id, spec.attempt_id, cid), tuple(argv), result.stderr.strip())
        # Create may have committed even when its response was lost. Inspect deterministic name before any retry.
        observed = self.inspect(spec.run_id, spec.attempt_id)
        if observed.state not in {"absent", "unknown"}:
            return ExecutionResult(observed.container_id, observed, tuple(argv), "create response failed; matching container reconciled")
        if observed.state == "unknown":
            return ExecutionResult(None, observed, tuple(argv), "create response failed; outcome remains unknown")
        raise ExecutorError("container create failed", argv=argv, detail=(result.stderr or result.stdout).strip())

    def start(self, container_id: str, *, run_id: str, attempt_id: str) -> ExecutionResult:
        self.assert_host_identity()
        obs = self.inspect(run_id, attempt_id, container_id)
        if obs.state != "created":
            return ExecutionResult(obs.container_id, obs, (), f"start suppressed for observed state {obs.state}")
        argv = [self.engine, "start", container_id]
        result = self._run(argv)
        observed = self.inspect(run_id, attempt_id, container_id)
        if observed.state == "running" or observed.state in {"exited", "unknown"}:
            return ExecutionResult(observed.container_id, observed, tuple(argv), (result.stderr or "").strip())
        if result.returncode:
            raise ExecutorError("container start failed or remains unresolved", argv=argv,
                                detail=(result.stderr or result.stdout).strip())
        return ExecutionResult(observed.container_id, observed, tuple(argv), "start response did not yield an observed running state")

    def cancel(self, container_id: str, *, run_id: str, attempt_id: str,
               grace_seconds: int = 30) -> ExecutionResult:
        if grace_seconds < 0:
            raise ValueError("grace_seconds cannot be negative")
        self.assert_host_identity()
        before = self.inspect(run_id, attempt_id, container_id)
        if before.state in {"absent", "exited"}:
            return ExecutionResult(before.container_id, before, (), "already terminal or absent; reconcile process ownership separately")
        if before.state == "created":
            argv = [self.engine, "rm", "--", container_id]
            result = self._run(argv)
            after = self.inspect(run_id, attempt_id, container_id)
            return ExecutionResult(after.container_id, after, tuple(argv),
                                   (result.stderr or "unstarted container cancellation observed").strip())
        if before.state != "running":
            return ExecutionResult(before.container_id, before, (), "cancel held because container state is uncertain")
        argv = [self.engine, "stop", "--time", str(grace_seconds), container_id]
        result = self._run(argv)
        after = self.inspect(run_id, attempt_id, container_id)
        if after.state in {"exited", "absent"}:
            return ExecutionResult(after.container_id, after, tuple(argv), (result.stderr or "").strip())
        return ExecutionResult(after.container_id, after, tuple(argv), "termination is not yet observed; reservation remains held")

    def logs(self, container_id: str) -> str:
        argv = [self.engine, "logs", "--timestamps", "--tail", "1000", container_id]
        result = self._run(argv)
        if result.returncode:
            raise ExecutorError("container log retrieval failed", argv=argv,
                                detail=(result.stderr or result.stdout).strip())
        value = result.stdout + result.stderr
        encoded = value.encode("utf-8", errors="replace")
        if len(encoded) > self.max_log_bytes:
            encoded = b"[older log output truncated by configured byte limit]\n" + encoded[-self.max_log_bytes:]
        return encoded.decode("utf-8", errors="replace")

    def remove_verified_attempt(self, run_id: str, attempt_id: str, container_id: str,
                                image_digest: str) -> ExecutionResult:
        """Remove only this exact exited queue-owned container, without volumes or prune."""
        before = self.inspect(run_id, attempt_id, container_id)
        if before.state == "absent":
            return ExecutionResult(container_id, before, (), "container was already removed")
        if before.state != "exited":
            return ExecutionResult(before.container_id, before, (),
                                  "container removal held until exact owned container is exited")
        if before.image != image_digest:
            raise ExecutorError("refusing removal because observed image digest differs",
                                detail=f"expected={image_digest}; observed={before.image}")
        argv = [self.engine, "rm", "--", container_id]
        result = self._run(argv)
        after = self.inspect(run_id, attempt_id, container_id)
        if after.state == "absent":
            return ExecutionResult(container_id, after, tuple(argv), (result.stderr or "").strip())
        return ExecutionResult(after.container_id, after, tuple(argv),
                              "container removal is unresolved; preserve success and cleanup evidence")
