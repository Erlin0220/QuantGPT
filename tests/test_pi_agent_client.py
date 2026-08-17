from __future__ import annotations

import io
import json

from quantgpt import pi_agent_client


class _RecordingStringIO(io.StringIO):
    def close(self) -> None:
        self.flush()


class _FakeProcess:
    def __init__(self, stdout_lines: list[dict]):
        self.stdin = _RecordingStringIO()
        self.stdout = io.StringIO("".join(json.dumps(line) + "\n" for line in stdout_lines))
        self.stderr = io.StringIO("")
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        del timeout
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


def test_run_pi_prompt_uses_rpc_model_and_max_thinking(monkeypatch, tmp_path):
    final = '{"skill_candidates": []}'
    process = _FakeProcess(
        [
            {"type": "response", "id": "req_1", "success": True, "data": {"accepted": True}},
            {
                "type": "agent_end",
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": final}]},
                ],
            },
            {
                "type": "response",
                "id": "req_2",
                "success": True,
                "data": {
                    "messages": [
                        {"role": "assistant", "content": [{"type": "text", "text": final}]},
                    ]
                },
            },
        ]
    )
    seen: dict = {}

    def fake_popen(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return process

    monkeypatch.setattr(pi_agent_client.shutil, "which", lambda _command: "pi")
    monkeypatch.setattr(pi_agent_client.subprocess, "Popen", fake_popen)

    result = pi_agent_client.run_pi_prompt(
        "return json",
        workspace=tmp_path,
        model="opencode-go/deepseek-v4-flash",
        thinking="max",
        timeout_seconds=30,
    )

    assert result == final
    assert "--mode" in seen["command"]
    assert "rpc" in seen["command"]
    assert "--model" in seen["command"]
    assert "opencode-go/deepseek-v4-flash" in seen["command"]
    assert "--thinking" in seen["command"]
    assert "max" in seen["command"]
    assert "--no-tools" in seen["command"]
    assert "--no-session" in seen["command"]
    sent = process.stdin.getvalue().splitlines()
    assert json.loads(sent[0]) == {"type": "prompt", "message": "return json", "id": "req_1"}
    assert json.loads(sent[1]) == {"type": "get_messages", "id": "req_2"}


def test_pi_thinking_legacy_enabled_maps_to_max():
    assert pi_agent_client is not None
    from quantgpt import wq_local_candidate_generator as generator

    assert generator._pi_thinking_level("enabled", "max") == "max"
    assert generator._pi_thinking_level("disabled", "max") == "off"
