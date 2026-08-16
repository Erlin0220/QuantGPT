from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv


@contextmanager
def _single_instance_lock(path: Path) -> Iterator[bool]:
    """Hold an OS-level advisory lock for the lifetime of the research daemon."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                yield False
                return
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                yield False
                return
        acquired = True
        yield True
    finally:
        if acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    lock_file = root / ".wq-local-research.lock"
    with _single_instance_lock(lock_file) as acquired:
        if not acquired:
            return

        load_dotenv(root / ".env", override=False)
        log_dir = root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        pid_file = root / ".wq-local-research.pid"
        pid_file.write_text(str(os.getpid()), encoding="ascii")
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        with (
            (log_dir / "wq-local-research.stdout.log").open("a", encoding="utf-8", buffering=1) as stdout,
            (log_dir / "wq-local-research.stderr.log").open("a", encoding="utf-8", buffering=1) as stderr,
        ):
            sys.stdout = stdout
            sys.stderr = stderr
            os.chdir(root)
            logging.basicConfig(
                level=logging.INFO,
                stream=stdout,
                format="%(asctime)s %(levelname)s %(name)s %(message)s",
            )
            print(f"\n[{datetime.now().isoformat(timespec='seconds')}] WQ local research launcher start", file=stderr)
            try:
                from quantgpt.wq_local_research_daemon import run_local_research_daemon

                run_local_research_daemon(
                    account=os.environ.get("WQ_LOCAL_RESEARCH_ACCOUNT", "primary"),
                    target_simulations=int(os.environ.get("WQ_LOCAL_RESEARCH_TARGET_SIMULATIONS", "100")),
                    budget_minutes=int(os.environ.get("WQ_LOCAL_RESEARCH_BUDGET_MINUTES", "50")),
                    batch_size=int(os.environ.get("WQ_LOCAL_RESEARCH_BATCH_SIZE", "4")),
                    max_simulations_per_batch=int(os.environ.get("WQ_LOCAL_RESEARCH_SIMULATIONS_PER_BATCH", "4")),
                    reasoning_policy=os.environ.get("WQ_LOCAL_RESEARCH_REASONING_POLICY", "adaptive"),
                )
            finally:
                try:
                    if pid_file.read_text(encoding="ascii").strip() == str(os.getpid()):
                        pid_file.unlink()
                except (FileNotFoundError, OSError):
                    pass


if __name__ == "__main__":
    main()
