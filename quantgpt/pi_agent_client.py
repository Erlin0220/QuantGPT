"""Minimal synchronous Pi Agent RPC client used by QuantGPT research."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


class PiAgentError(RuntimeError):
    """Raised when Pi cannot produce a final assistant response."""


def _unwrap(value: Any) -> Any:
    if isinstance(value, dict):
        if "data" in value:
            return value["data"]
        if "result" in value:
            return value["result"]
    return value


def _assistant_text(value: Any) -> str:
    root = _unwrap(value)
    messages = root if isinstance(root, list) else root.get("messages") if isinstance(root, dict) else None
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        text = "\n\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and str(part.get("text") or "").strip()
        ).strip()
        if text:
            return text
    return ""


def _streaming_text(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        if event.get("type") != "message_update":
            continue
        update = event.get("assistantMessageEvent")
        if not isinstance(update, dict) or update.get("type") != "text_delta":
            continue
        delta = update.get("delta")
        if isinstance(delta, str):
            parts.append(delta)
    return "".join(parts).strip()


def _provider_error(value: Any) -> str:
    root = _unwrap(value)
    if isinstance(root, list):
        for item in reversed(root):
            error = _provider_error(item)
            if error:
                return error
        return ""
    if isinstance(root, dict) and isinstance(root.get("messages"), list):
        return _provider_error(root["messages"])
    if not isinstance(root, dict):
        return ""
    message = root.get("message") if isinstance(root.get("message"), dict) else root
    if not isinstance(message, dict):
        return ""
    error = message.get("errorMessage") or message.get("error")
    return str(error or "").strip()


def _launch_command(command: str, args: list[str]) -> list[str]:
    if os.name != "nt":
        return [command, *args]
    comspec = os.environ.get("ComSpec") or os.environ.get("COMSPEC") or "cmd.exe"
    return [comspec, "/d", "/s", "/c", command, *args]


def run_pi_prompt(
    prompt: str,
    *,
    workspace: str | Path,
    model: str = "opencode-go/deepseek-v4-flash",
    thinking: str = "max",
    timeout_seconds: float = 120.0,
    command: str | None = None,
) -> str:
    """Run one ephemeral Pi Agent prompt over Pi's JSON-line RPC mode."""
    pi_command = str(command or os.environ.get("WQ_PI_COMMAND") or "pi").strip()
    if not pi_command:
        raise PiAgentError("Pi command is empty")
    if os.path.sep not in pi_command and (os.altsep is None or os.altsep not in pi_command):
        resolved = shutil.which(pi_command)
        if resolved:
            pi_command = resolved

    args = [
        "--mode",
        "rpc",
        "--model",
        str(model).strip(),
        "--thinking",
        str(thinking).strip(),
        "--no-tools",
        "--no-session",
        "--no-context-files",
        "--no-skills",
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        process = subprocess.Popen(
            _launch_command(pi_command, args),
            cwd=str(Path(workspace).resolve()),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
    except OSError as exc:
        raise PiAgentError(f"unable to start Pi Agent: {exc}") from exc

    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    stdout_queue: queue.Queue[str | None] = queue.Queue()
    stderr_parts: list[str] = []

    def read_stdout() -> None:
        try:
            for line in process.stdout:
                stdout_queue.put(line)
        finally:
            stdout_queue.put(None)

    def read_stderr() -> None:
        for line in process.stderr:
            stderr_parts.append(line)
            if len(stderr_parts) > 80:
                del stderr_parts[:20]

    stdout_thread = threading.Thread(target=read_stdout, name="quantgpt-pi-stdout", daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, name="quantgpt-pi-stderr", daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    deadline = time.monotonic() + max(10.0, float(timeout_seconds))
    events: list[dict[str, Any]] = []
    request_counter = 0

    def send(payload: dict[str, Any]) -> str:
        nonlocal request_counter
        request_counter += 1
        request_id = f"req_{request_counter}"
        process.stdin.write(json.dumps({**payload, "id": request_id}, ensure_ascii=False) + "\n")
        process.stdin.flush()
        return request_id

    def next_message() -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PiAgentError("Pi Agent RPC timed out waiting for completion")
        try:
            line = stdout_queue.get(timeout=remaining)
        except queue.Empty as exc:
            raise PiAgentError("Pi Agent RPC timed out waiting for completion") from exc
        if line is None:
            stderr = "".join(stderr_parts).strip()
            suffix = f": {stderr}" if stderr else ""
            raise PiAgentError(f"Pi Agent RPC exited before completion{suffix}")
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PiAgentError(f"Pi Agent RPC emitted malformed JSON: {line.strip()}") from exc
        if not isinstance(message, dict):
            raise PiAgentError("Pi Agent RPC emitted a non-object message")
        return message

    agent_end: dict[str, Any] | None = None
    messages_payload: Any = None
    try:
        prompt_request_id = send({"type": "prompt", "message": prompt})
        while agent_end is None:
            message = next_message()
            if message.get("type") == "response":
                if message.get("id") == prompt_request_id and (message.get("success") is False or message.get("error")):
                    raise PiAgentError(f"Pi Agent prompt failed: {message.get('error') or message}")
                continue
            events.append(message)
            if message.get("type") == "agent_end":
                agent_end = message

        messages_request_id = send({"type": "get_messages"})
        while messages_payload is None:
            message = next_message()
            if message.get("type") != "response":
                events.append(message)
                continue
            if message.get("id") != messages_request_id:
                continue
            if message.get("success") is False or message.get("error"):
                raise PiAgentError(f"Pi Agent get_messages failed: {message.get('error') or message}")
            messages_payload = message.get("data", message.get("result", message))

        final = _assistant_text(agent_end) or _assistant_text(messages_payload) or _streaming_text(events)
        if final:
            return final
        provider_error = _provider_error(agent_end) or _provider_error(messages_payload) or _provider_error(events)
        if provider_error:
            raise PiAgentError(f"Pi Agent returned an error: {provider_error}")
        raise PiAgentError("Pi Agent did not return a final assistant response")
    finally:
        try:
            process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
