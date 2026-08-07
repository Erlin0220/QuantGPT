"""Regression tests for bounded MCP factor-value result pages."""

import gzip
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from quantgpt import mcp_server


class TestFactorValuesPagination(unittest.IsolatedAsyncioTestCase):
    async def test_completed_artifact_is_read_in_bounded_pages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            artifact_root = project_root / "reports" / "mcp_factor_values"
            artifact_root.mkdir(parents=True)
            artifact = artifact_root / "factor-task.jsonl.gz"
            with gzip.open(artifact, "wt", encoding="utf-8") as output:
                for day in range(1, 7):
                    output.write(
                        json.dumps(
                            {
                                "date": f"2026-01-{day:02d}",
                                "values": {"000001.SZ": float(day)},
                                "count": 1,
                            }
                        )
                    )
                    output.write("\n")

            snapshot = {
                "task_id": "factor-task",
                "task_type": "compute_factor_values",
                "status": "completed",
                "result": {
                    "trading_days": 6,
                    "result_storage": {"path": "reports/mcp_factor_values/factor-task.jsonl.gz"},
                },
            }
            with (
                patch.object(mcp_server, "_PROJECT_ROOT", project_root),
                patch.object(mcp_server, "_FACTOR_VALUES_DIR", artifact_root),
                patch.object(
                    mcp_server,
                    "get_mcp_task_snapshot",
                    new=AsyncMock(return_value=snapshot),
                ),
            ):
                payload = json.loads(await mcp_server.get_factor_values_page("factor-task", page=2, page_size=2))

        self.assertEqual(payload["total_pages"], 3)
        self.assertTrue(payload["has_more"])
        self.assertEqual(
            [row["date"] for row in payload["data"]],
            [
                "2026-01-03",
                "2026-01-04",
            ],
        )

    def test_cleanup_enforces_ttl_count_and_total_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir)
            now = time.time()
            files = []
            for index in range(4):
                path = artifact_root / f"{index}.jsonl.gz"
                path.write_bytes(b"x" * 10)
                os.utime(path, (now - index * 10, now - index * 10))
                files.append(path)
            expired = artifact_root / "expired.jsonl.gz"
            expired.write_bytes(b"x")
            os.utime(expired, (now - 1000, now - 1000))

            with (
                patch.object(mcp_server, "_FACTOR_VALUES_DIR", artifact_root),
                patch.object(mcp_server, "_MAX_FACTOR_VALUE_ARTIFACTS", 3),
                patch.object(mcp_server, "_MAX_FACTOR_VALUE_TOTAL_BYTES", 20),
                patch.object(mcp_server, "_FACTOR_VALUE_ARTIFACT_TTL_SECONDS", 100),
            ):
                mcp_server._cleanup_factor_value_artifacts()

            self.assertEqual([path.exists() for path in files], [True, True, False, False])
            self.assertFalse(expired.exists())


if __name__ == "__main__":
    unittest.main()
