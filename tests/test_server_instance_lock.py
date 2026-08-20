from pathlib import Path

import pytest

from quantgpt.server_instance_lock import ServerInstanceLock


def test_server_instance_lock_is_exclusive_and_reusable(tmp_path: Path):
    path = tmp_path / "server.lock"
    first = ServerInstanceLock(path)
    second = ServerInstanceLock(path)

    first.acquire()
    with pytest.raises(RuntimeError, match="exactly one process"):
        second.acquire()
    first.release()

    second.acquire()
    second.release()
