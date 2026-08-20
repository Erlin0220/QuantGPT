"""Cross-platform process lock enforcing one QuantGPT HTTP server instance."""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO


class ServerInstanceLock:
    def __init__(self, path: Path):
        self.path = path
        self._handle: BinaryIO | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(
                "Another QuantGPT server process is already active; HTTP/MCP mode supports exactly one process"
            ) from exc
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


def acquire_server_instance_lock(path: Path | None = None) -> ServerInstanceLock:
    default_path = Path(__file__).resolve().parent.parent / ".quantgpt.server.lock"
    lock = ServerInstanceLock(path or Path(os.environ.get("QUANTGPT_INSTANCE_LOCK_FILE", str(default_path))))
    lock.acquire()
    return lock
