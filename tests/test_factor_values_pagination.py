"""Regression tests for bounded MCP factor-value result pages."""

import gzip
import json
import tempfile
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


if __name__ == "__main__":
    unittest.main()
