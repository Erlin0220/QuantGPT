from __future__ import annotations

import multiprocessing as mp
import os
import runpy
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # pythonw.exe starts without a console. Redirect only in the real server
    # process. Spawned ProcessPool workers import this file as __mp_main__ and
    # must not start another HTTP server or reopen the server log streams.
    with (
        (log_dir / "quantgpt.stdout.log").open("a", encoding="utf-8", buffering=1) as stdout,
        (log_dir / "quantgpt.stderr.log").open("a", encoding="utf-8", buffering=1) as stderr,
    ):
        sys.stdout = stdout
        sys.stderr = stderr

        os.chdir(root)
        sys.argv = [
            "quantgpt",
            "--transport",
            "http",
            "--host",
            "0.0.0.0",
            "--port",
            "8003",
        ]
        print(
            f"\n[{datetime.now().isoformat(timespec='seconds')}] QuantGPT background launcher start",
            file=stderr,
        )
        runpy.run_module("quantgpt", run_name="__main__")


if __name__ == "__main__":
    mp.freeze_support()
    main()
