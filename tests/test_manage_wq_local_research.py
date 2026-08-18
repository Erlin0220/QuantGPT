from __future__ import annotations

from unittest.mock import MagicMock

import psutil

from scripts import manage_wq_local_research as manager


def test_access_denied_pid_file_is_treated_as_authoritative_running_process(monkeypatch):
    process = MagicMock(pid=1234)
    process.cmdline.side_effect = psutil.AccessDenied(pid=1234)

    monkeypatch.setattr(manager, "_read_pid", lambda: 1234)
    monkeypatch.setattr(manager.psutil, "Process", lambda _pid: process)
    monkeypatch.setattr(manager.psutil, "process_iter", lambda _attrs: ())

    assert manager._running_process() is process


def test_process_needs_restart_when_execution_path_source_is_newer(monkeypatch, tmp_path):
    source = tmp_path / "daemon.py"
    source.write_text("# changed\n", encoding="utf-8")
    process = MagicMock()
    process.create_time.return_value = source.stat().st_mtime - 10

    monkeypatch.setattr(manager, "_RELOAD_PATHS", (source,))

    assert manager._process_needs_restart(process) is True


def test_start_restarts_stale_daemon_before_launch(monkeypatch):
    stale = MagicMock(pid=1234)
    fresh = MagicMock(pid=5678)
    running_calls = iter([stale, None, fresh])
    popen = MagicMock()
    popen.poll.return_value = None

    monkeypatch.setattr(manager, "_running_process", lambda: next(running_calls))
    monkeypatch.setattr(manager, "_process_needs_restart", lambda _process: True)
    monkeypatch.setattr(manager, "stop", MagicMock())
    monkeypatch.setattr(manager, "_launcher_python", lambda: manager.ROOT / ".venv" / "Scripts" / "pythonw.exe")
    monkeypatch.setattr(manager.subprocess, "Popen", MagicMock(return_value=popen))
    monkeypatch.setattr(manager, "_write_pid", MagicMock())
    monkeypatch.setattr(manager.time, "sleep", lambda _seconds: None)

    manager.start(timeout=1.0)

    manager.stop.assert_called_once()
    manager.subprocess.Popen.assert_called_once()
