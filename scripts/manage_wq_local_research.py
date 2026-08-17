"""Lifecycle manager for the local Pi Agent WorldQuant research daemon."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
PID_FILE = ROOT / ".wq-local-research.pid"
LAUNCHER = ROOT / "scripts" / "run_wq_local_research_background.pyw"


def _read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text(encoding="ascii").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def _write_pid(pid: int) -> None:
    temporary = PID_FILE.with_suffix(".pid.tmp")
    temporary.write_text(str(pid), encoding="ascii")
    temporary.replace(PID_FILE)


def _remove_pid_file() -> None:
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def _is_research_process(process: psutil.Process) -> bool:
    try:
        command = " ".join(process.cmdline()).lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    return str(ROOT).lower() in command and "run_wq_local_research_background.pyw" in command


def _running_process() -> psutil.Process | None:
    pid = _read_pid()
    if pid:
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            process = None
        if process is not None and _is_research_process(process):
            return process
    for process in psutil.process_iter(["pid"]):
        if _is_research_process(process):
            _write_pid(process.pid)
            return process
    _remove_pid_file()
    return None


def status() -> bool:
    process = _running_process()
    if process is None:
        print("WQ local research stopped")
        return False
    print(f"WQ local research running: pid={process.pid}")
    return True


def _launcher_python() -> Path:
    executable = Path(sys.executable)
    if os.name == "nt":
        pythonw = executable.with_name("pythonw.exe")
        if pythonw.is_file():
            return pythonw
    return executable


def start(timeout: float = 10.0) -> None:
    if status():
        return
    command = [str(_launcher_python()), str(LAUNCHER)]
    if os.name == "nt":
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
        )
    else:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    _write_pid(process.pid)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _remove_pid_file()
            raise RuntimeError(f"WQ local research exited during startup with code {process.returncode}")
        running = _running_process()
        if running is not None:
            print(f"WQ local research started: pid={running.pid}")
            return
        time.sleep(0.1)
    stop()
    raise RuntimeError("WQ local research did not stay alive during startup")


def stop(timeout: float = 15.0) -> None:
    process = _running_process()
    if process is None:
        print("WQ local research already stopped")
        return
    targets: dict[int, psutil.Process] = {process.pid: process}
    try:
        for child in process.children(recursive=True):
            targets[child.pid] = child
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    for target in targets.values():
        try:
            target.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    gone, alive = psutil.wait_procs(list(targets.values()), timeout=timeout)
    del gone
    for target in alive:
        try:
            target.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if alive:
        _, alive = psutil.wait_procs(alive, timeout=5)
    if alive:
        raise RuntimeError(f"Unable to stop WQ local research processes: {[p.pid for p in alive]}")
    _remove_pid_file()
    print("WQ local research stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "restart", "status"))
    args = parser.parse_args()
    if args.action == "start":
        start()
    elif args.action == "stop":
        stop()
    elif args.action == "restart":
        stop()
        start()
    else:
        raise SystemExit(0 if status() else 1)


if __name__ == "__main__":
    main()
