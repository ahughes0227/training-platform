from __future__ import annotations

import json
import socket
import threading

import pytest

from defect_platform.control.vm_queue_api import (
    Peer,
    Principal,
    UnixQueueServer,
    VMQueueAPI,
)


class Controller:
    def __init__(self):
        self.calls = []

    def enqueue(self, intent, **kwargs): self.calls.append(("enqueue", intent, kwargs)); return {"entry_id": "e1"}
    def enqueue_batch(self, intents, **kwargs): self.calls.append(("enqueue_batch", intents, kwargs)); return {"entries": []}
    def list_queue(self): return {"entries": [], "current_vm": "vm-a", "current_container": None}
    def status(self): return {"ready": False, "blocker": "disk readiness unverified", "last_observation": None}
    def pause(self, **kwargs): self.calls.append(("pause", kwargs)); return {"policy_state": "paused"}
    def resume(self, **kwargs): self.calls.append(("resume", kwargs)); return {"policy_state": "running"}
    def cancel_waiting(self, entry_id, **kwargs): self.calls.append(("cancel_waiting", entry_id, kwargs)); return {"entry_id": entry_id}
    def request_active_cancel(self, entry_id, **kwargs): self.calls.append(("cancel_active", entry_id, kwargs)); return {"entry_id": entry_id}
    def retry(self, entry_id, **kwargs): self.calls.append(("retry", entry_id, kwargs)); return {"entry_id": entry_id}
    def skip(self, entry_id, **kwargs): self.calls.append(("skip", entry_id, kwargs)); return {"entry_id": entry_id}
    def record_observation(self, observation, **kwargs): self.calls.append(("observe", observation, kwargs)); return {"entry_id": "e1"}
    def backup(self, destination): self.calls.append(("backup", destination)); return {"path": str(destination)}
    def verify_restore(self, source): self.calls.append(("restore_check", source)); return {"valid": True}


def test_peer_role_controls_actions_and_body_actor_is_ignored():
    controller = Controller()
    api = VMQueueAPI(controller, lambda peer: Principal("uid-501", "operator") if peer.uid == 501 else None)
    assert api.dispatch({"op": "pause", "reason": "maintenance", "actor": "forged"}, Peer(2, 501, 20))["ok"]
    assert controller.calls[-1] == ("pause", {"reason": "maintenance", "actor": "uid-501"})
    with pytest.raises(PermissionError):
        api.dispatch({"op": "observe", "observation": {}}, Peer(2, 501, 20))
    with pytest.raises(PermissionError):
        api.dispatch({"op": "resume", "actor": "worker"}, Peer(2, 502, 20))
    assert len(controller.calls) == 1


def test_worker_observations_cannot_call_operator_actions():
    controller = Controller()
    api = VMQueueAPI(controller, lambda _peer: Principal("worker-service", "worker"))
    api.dispatch({"op": "observe", "observation": {"entry_id": "e1", "state": "active"}}, Peer(2, 1, 1))
    assert controller.calls[0][0] == "observe"
    with pytest.raises(PermissionError):
        api.dispatch({"op": "skip", "entry_id": "e1", "expected_revision": 1, "reason": "x"}, Peer(2, 1, 1))
    assert len(controller.calls) == 1


def test_backup_and_restore_check_paths_are_explicit_and_absolute():
    controller = Controller()
    api = VMQueueAPI(controller, lambda _peer: Principal("operator", "operator"))
    with pytest.raises(ValueError, match="absolute"):
        api.dispatch({"op": "backup", "destination": "relative.sqlite"}, Peer(1, 1, 1))
    assert api.dispatch({"op": "backup", "destination": "/var/lib/queue/backups/q.sqlite"}, Peer(1, 1, 1))["result"]
    assert api.dispatch({"op": "restore_check", "source": "/var/lib/queue/backups/q.sqlite"}, Peer(1, 1, 1))["result"] == {"valid": True}


def test_retry_requires_operator_reason_and_revision():
    controller = Controller()
    api = VMQueueAPI(controller, lambda _peer: Principal("operator", "operator"))
    with pytest.raises(ValueError, match="reason"):
        api.dispatch({"op": "retry", "entry_id": "e1", "expected_revision": 4}, Peer(1, 1, 1))
    with pytest.raises(ValueError, match="classification"):
        api.dispatch({"op": "retry", "entry_id": "e1", "expected_revision": 4,
                      "reason": "reviewed"}, Peer(1, 1, 1))
    assert not controller.calls
    api.dispatch({"op": "retry", "entry_id": "e1", "expected_revision": 4,
                  "reason": "transient capacity issue resolved", "classification": "transient"}, Peer(1, 1, 1))
    assert controller.calls[-1] == ("retry", "e1", {
        "expected_revision": 4, "reason": "transient capacity issue resolved", "actor": "operator",
        "classification": "transient"})


def test_server_protocol_handles_bounded_socketpair_request():
    controller = Controller()
    api = VMQueueAPI(controller, lambda peer: Principal(f"uid-{peer.uid}", "operator") if peer.uid == 501 else None)
    server = UnixQueueServer(api, "/unused", peer_reader=lambda _conn: Peer(42, 501, 20))
    client, server_side = socket.socketpair()
    thread = threading.Thread(target=server._serve_connection, args=(server_side,), daemon=True)
    thread.start()
    client.sendall(b'{"op":"status"}\n')
    assert json.loads(client.recv(4096))["result"]["blocker"] == "disk readiness unverified"
    thread.join(timeout=2)
    client.close()


def test_unix_service_rejects_untrusted_peer_without_controller_effect():
    api = VMQueueAPI(Controller(), lambda _peer: None)
    server = UnixQueueServer(api, "/unused", peer_reader=lambda _conn: Peer(42, 999, 20))
    client, server_side = socket.socketpair()
    thread = threading.Thread(target=server._serve_connection, args=(server_side,), daemon=True)
    thread.start()
    client.sendall(b'{"op":"pause","reason":"forged"}\n')
    response = json.loads(client.recv(4096))
    assert response["error"]["code"] == "forbidden"
    thread.join(timeout=2)
    client.close()


def test_server_refuses_to_unlink_non_socket(tmp_path):
    path = tmp_path / "queue.sock"
    path.write_text("keep", encoding="utf-8")
    server = UnixQueueServer(VMQueueAPI(Controller(), lambda _peer: None), path)
    with pytest.raises(RuntimeError, match="non-socket"):
        server.serve_forever()
    assert path.read_text(encoding="utf-8") == "keep"
