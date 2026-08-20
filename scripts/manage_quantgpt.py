"""Cross-platform lifecycle manager for the local QuantGPT HTTP/MCP server."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import psutil

ROOT = Path(__file__).resolve().parent.parent
PID_FILE = ROOT / ".quantgpt.pid"
LAUNCHER = ROOT / "scripts" / "run_quantgpt_background.pyw"
PORT = int(os.environ.get("QUANTGPT_PORT", "8003"))
HEALTH_URL = f"http://127.0.0.1:{PORT}/api/v1/health"


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


def _is_quantgpt_process(process: psutil.Process) -> bool:
    try:
        command = " ".join(process.cmdline()).lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    root = str(ROOT).lower()
    return root in command and ("quantgpt" in command or "run_quantgpt_background.pyw" in command)


def _listener_processes() -> list[psutil.Process]:
    processes: list[psutil.Process] = []
    seen: set[int] = set()
    try:
        connections = psutil.net_connections(kind="tcp")
    except (psutil.AccessDenied, OSError):
        return processes
    for connection in connections:
        if not connection.pid or connection.status != psutil.CONN_LISTEN:
            continue
        if not connection.laddr or connection.laddr.port != PORT:
            continue
        if connection.pid in seen:
            continue
        try:
            process = psutil.Process(connection.pid)
        except psutil.NoSuchProcess:
            continue
        if _is_quantgpt_process(process):
            seen.add(connection.pid)
            processes.append(process)
    return processes


def _health_ok(timeout: float = 1.0) -> bool:
    try:
        with urlopen(HEALTH_URL, timeout=timeout) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def _running_processes() -> list[psutil.Process]:
    processes = _listener_processes()
    pid = _read_pid()
    if pid and all(process.pid != pid for process in processes):
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            process = None
        if process is not None and _is_quantgpt_process(process):
            processes.append(process)
    return processes


def _wait_for_exit(processes: list[psutil.Process], timeout: float) -> list[psutil.Process]:
    deadline = time.monotonic() + timeout
    alive = processes
    while alive and time.monotonic() < deadline:
        alive = [process for process in alive if psutil.pid_exists(process.pid)]
        if alive:
            time.sleep(0.1)
    return alive


def status() -> bool:
    processes = _running_processes()
    listeners = _listener_processes()
    healthy = _health_ok()
    if listeners and healthy:
        listener_pid = listeners[0].pid
        _write_pid(listener_pid)
        print(f"QuantGPT running: pid={listener_pid}, port={PORT}, health=ok")
        return True
    if processes:
        print(f"QuantGPT process exists but health check failed: pids={[p.pid for p in processes]}")
    elif healthy:
        print(f"Port {PORT} responds, but listener is not a validated QuantGPT process")
    else:
        print("QuantGPT stopped")
        _remove_pid_file()
    return False


def stop(timeout: float = 15.0) -> None:
    processes = _running_processes()
    if not processes:
        _remove_pid_file()
        print("QuantGPT already stopped")
        return

    targets: dict[int, psutil.Process] = {}
    for process in processes:
        targets[process.pid] = process
        try:
            for child in process.children(recursive=True):
                targets[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    for process in targets.values():
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    alive = _wait_for_exit(list(targets.values()), timeout)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if alive:
        alive = _wait_for_exit(alive, 5)
    if alive:
        raise RuntimeError(f"Unable to stop QuantGPT processes: {[process.pid for process in alive]}")

    _remove_pid_file()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _health_ok():
        time.sleep(0.2)
    if _health_ok():
        raise RuntimeError(f"QuantGPT still responds on port {PORT} after stop")
    print("QuantGPT stopped")


def _launcher_python() -> Path:
    executable = Path(sys.executable)
    if os.name == "nt":
        pythonw = executable.with_name("pythonw.exe")
        if pythonw.is_file():
            return pythonw
    return executable


def start(timeout: float = 30.0) -> None:
    if status():
        return
    if _running_processes():
        stop()
    if _health_ok():
        raise RuntimeError(f"Port {PORT} is already occupied by another service")

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    process = subprocess.Popen(
        [str(_launcher_python()), str(LAUNCHER)],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=creationflags,
    )
    _write_pid(process.pid)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _remove_pid_file()
            raise RuntimeError(f"QuantGPT exited during startup with code {process.returncode}")
        if _health_ok():
            listeners = _listener_processes()
            if not listeners:
                raise RuntimeError("Health endpoint is ready, but listener process validation failed")
            _write_pid(listeners[0].pid)
            print(f"QuantGPT started: pid={listeners[0].pid}, port={PORT}, health=ok")
            return
        time.sleep(0.25)

    stop()
    raise RuntimeError(f"QuantGPT did not become healthy within {timeout:.0f}s")


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
