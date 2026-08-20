"""Regression tests for non-blocking MCP task submission."""

import asyncio
import json
import os
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

import quantgpt.mcp_server as mcp_server
import quantgpt.mcp_task_helper as task_helper


class TestWQMCPAsyncSurface(unittest.IsolatedAsyncioTestCase):
    async def _assert_enqueued(self, name, call):
        with (
            patch.dict(
                os.environ,
                {"WQ_BRAIN_EMAIL": "test@example.com", "WQ_BRAIN_PASSWORD": "pw"},
                clear=False,
            ),
            patch.object(
                mcp_server,
                "start_mcp_task",
                new=AsyncMock(return_value=f"{name}-task"),
            ) as start_task,
            patch.object(mcp_server, "start_mcp_background_task") as start_background,
        ):
            started = time.perf_counter()
            payload = json.loads(await call())
            elapsed = time.perf_counter() - started

        self.assertEqual(payload["task_id"], f"{name}-task")
        self.assertTrue(payload["request_id"])
        self.assertTrue(payload["async"])
        self.assertEqual(payload["poll_with"], "get_task_status")
        self.assertEqual(
            payload["legacy_poll"],
            {
                "tool": "wq_brain_check_alphas",
                "alpha_ids": [f"task:{name}-task"],
            },
        )
        self.assertLess(elapsed, 0.5)
        self.assertEqual(start_task.await_count, 1)
        self.assertEqual(start_background.call_count, 1)
        self.assertEqual(start_task.await_args.kwargs["request_id"], payload["request_id"])
        self.assertEqual(start_task.await_args.args[2]["request_id"], payload["request_id"])
        self.assertEqual(start_background.call_args.args[2]["request_id"], payload["request_id"])

    async def test_proxy_request_id_is_inherited_by_async_task(self):
        context = Mock()
        context.request_context.request.headers = {"x-request-id": "proxy-request-123"}
        with (
            patch.dict(
                os.environ,
                {"WQ_BRAIN_EMAIL": "test@example.com", "WQ_BRAIN_PASSWORD": "pw"},
                clear=False,
            ),
            patch.object(mcp_server.mcp, "get_context", return_value=context),
            patch.object(mcp_server, "start_mcp_task", new=AsyncMock(return_value="proxy-task")) as start_task,
            patch.object(mcp_server, "start_mcp_background_task"),
        ):
            payload = json.loads(await mcp_server.wq_brain_submit("rank(close)", "proxy-id-test"))

        self.assertEqual(payload["request_id"], "proxy-request-123")
        self.assertEqual(start_task.await_args.kwargs["request_id"], "proxy-request-123")
        self.assertEqual(start_task.await_args.args[2]["request_id"], "proxy-request-123")

    async def test_all_long_wq_tools_enqueue_and_return_immediately(self):
        cases = [
            ("single", lambda: mcp_server.wq_brain_submit("rank(close/open)", "smoke")),
            ("batch", lambda: mcp_server.wq_brain_batch_submit("rank(close/open)", "smoke")),
            ("research", lambda: mcp_server.wq_brain_research(["rank(close/open)"])),
            (
                "autonomous",
                lambda: mcp_server.wq_brain_autonomous_research(
                    skill_candidates=[{
                        "expression": "rank(close/open)",
                        "hypothesis": "relative close/open strength should predict next-day cross-sectional returns",
                        "skill_chain": ["wq-alpha-hypothesis", "wq-alpha-review", "wq-robustness-validation", "wq-candidate-evidence"],
                        "review_decision": "RUN",
                        "robustness_plan": {"mode": "skill_defined", "checks": [{"universe": "TOP1000", "purpose": "smoke-test universe sensitivity"}]},
                        "candidate_evidence_policy": {"mode": "calibrated_evidence_hierarchy"},
                    }]
                ),
            ),
            ("submitids", lambda: mcp_server.wq_brain_submit_by_ids(["abc123"])),
            ("finalize", lambda: mcp_server.wq_brain_finalize_submissions(["abc123"])),
        ]
        for name, call in cases:
            with self.subTest(name=name):
                await self._assert_enqueued(name, call)

    def test_research_source_run_stamp_reaches_nested_persistence_payloads(self):
        result = {
            "results": [{"alpha_id": "a1", "research_meta": {"family": "price_volume"}}],
            "candidates": [{"alpha_id": "a1"}],
            "generations": [{"results": [{"alpha_id": "a2"}], "failed": [{"expression": "rank(x)"}]}],
            "best": {"alpha_id": "a1"},
        }

        stamped = mcp_server._stamp_wq_research_source_run(result, "round-123")

        self.assertEqual(stamped["source_run_id"], "round-123")
        self.assertEqual(stamped["results"][0]["research_meta"]["source_run_id"], "round-123")
        self.assertEqual(stamped["candidates"][0]["research_meta"]["source_run_id"], "round-123")
        self.assertEqual(stamped["generations"][0]["results"][0]["research_meta"]["source_run_id"], "round-123")
        self.assertEqual(stamped["generations"][0]["failed"][0]["research_meta"]["source_run_id"], "round-123")
        self.assertEqual(stamped["best"]["research_meta"]["source_run_id"], "round-123")

    async def test_autonomous_research_requires_devspace_skill_candidates_by_default(self):
        with (
            patch.dict(
                os.environ,
                {"WQ_BRAIN_EMAIL": "test@example.com", "WQ_BRAIN_PASSWORD": "pw"},
                clear=False,
            ),
            patch.object(mcp_server, "start_mcp_task", new=AsyncMock()) as start_task,
        ):
            payload = json.loads(await mcp_server.wq_brain_autonomous_research())

        self.assertEqual(payload["status"], "skill_generation_required")
        self.assertEqual(payload["required_skill_chain"], ["wq-alpha-hypothesis", "wq-alpha-review", "wq-robustness-validation", "wq-candidate-evidence"])
        self.assertIn("wq-economic-hypothesis", payload["next_step"])
        start_task.assert_not_awaited()

    async def test_auto_submit_is_rejected_before_task_enqueue(self):
        with (
            patch.dict(
                os.environ,
                {"WQ_BRAIN_EMAIL": "test@example.com", "WQ_BRAIN_PASSWORD": "pw"},
                clear=False,
            ),
            patch.object(mcp_server, "start_mcp_task", new=AsyncMock()) as start_task,
        ):
            single = json.loads(await mcp_server.wq_brain_submit("rank(close)", "smoke", auto_submit=True))
            batch = json.loads(await mcp_server.wq_brain_batch_submit("rank(close)", "smoke", auto_submit=True))

        self.assertEqual(single["reason"], "auto_submit_disabled_use_candidate_pipeline")
        self.assertEqual(batch["reason"], "auto_submit_disabled_use_candidate_pipeline")
        start_task.assert_not_awaited()

    async def test_research_reuses_existing_singleflight_task(self):
        existing = {
            "task_id": "research-active",
            "status": "researching",
            "task_type": "wq_research",
        }
        with (
            patch.dict(
                os.environ,
                {"WQ_BRAIN_EMAIL": "test@example.com", "WQ_BRAIN_PASSWORD": "pw"},
                clear=False,
            ),
            patch.object(
                mcp_server,
                "start_mcp_task",
                new=AsyncMock(side_effect=task_helper.MCPTaskAlreadyRunningError(existing)),
            ),
            patch.object(mcp_server, "start_mcp_background_task") as start_background,
        ):
            payload = json.loads(await mcp_server.wq_brain_research(["rank(close/open)"]))

        self.assertEqual(payload["task_id"], "research-active")
        self.assertEqual(payload["status"], "researching")
        self.assertTrue(payload["reused_existing"])
        self.assertFalse(payload["new_task"])
        self.assertEqual(payload["poll_with"], "get_task_status")
        start_background.assert_not_called()

    async def test_legacy_check_alphas_path_reads_task_without_wq_request(self):
        snapshot = {
            "task_id": "abc123",
            "status": "simulating",
            "task_type": "wq_brain_submit",
        }
        with patch.object(
            mcp_server,
            "get_mcp_task_snapshot",
            new=AsyncMock(return_value=snapshot),
        ) as get_snapshot:
            payload = json.loads(await mcp_server.wq_brain_check_alphas(["task:abc123"]))

        self.assertEqual(payload["task"]["task_id"], "abc123")
        self.assertFalse(payload["task"]["done"])
        self.assertEqual(payload["task"]["poll_after_seconds"], 10)
        get_snapshot.assert_awaited_once_with("abc123")


class TestLocalMCPAsyncSurface(unittest.IsolatedAsyncioTestCase):
    async def _assert_enqueued(self, name, call):
        with (
            patch.object(
                mcp_server,
                "start_mcp_task",
                new=AsyncMock(return_value=f"{name}-task"),
            ) as start_task,
            patch.object(mcp_server, "start_mcp_background_task") as start_background,
        ):
            started = time.perf_counter()
            payload = json.loads(await call())
            elapsed = time.perf_counter() - started

        self.assertEqual(payload["task_id"], f"{name}-task")
        self.assertTrue(payload["request_id"])
        self.assertTrue(payload["async"])
        self.assertEqual(payload["poll_with"], "get_task_status")
        self.assertNotIn("legacy_poll", payload)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(start_task.await_count, 1)
        self.assertEqual(start_background.call_count, 1)
        self.assertEqual(start_task.await_args.kwargs["request_id"], payload["request_id"])
        self.assertEqual(start_task.await_args.args[2]["request_id"], payload["request_id"])
        self.assertEqual(start_background.call_args.args[2]["request_id"], payload["request_id"])

    async def test_all_long_local_tools_enqueue_and_return_immediately(self):
        cases = [
            ("backtest", lambda: mcp_server.run_backtest("rank(close)")),
            ("score", lambda: mcp_server.score_factor("rank(close)")),
            ("anti", lambda: mcp_server.run_anti_overfit("rank(close)")),
            ("rolling", lambda: mcp_server.run_rolling_validation("rank(close)")),
            ("values", lambda: mcp_server.compute_factor_values("rank(close)")),
        ]
        for name, call in cases:
            with self.subTest(name=name):
                await self._assert_enqueued(name, call)

    def test_wq_worker_propagates_request_id_to_client_and_service(self):
        request_id = "req-worker-chain"
        params = {
            "request_id": request_id,
            "expression": "rank(close)",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 0,
            "neutralization": "SUBINDUSTRY",
            "truncation": 0.08,
            "auto_submit": False,
            "tag": "request-id-test",
        }
        client = Mock()
        client.authenticate.return_value = True

        with (
            patch("quantgpt.wq_brain_client.get_client", return_value=client) as get_client,
            patch.object(
                mcp_server,
                "run_single_simulation",
                return_value={"ok": True, "request_id": request_id},
            ) as run_single,
        ):
            result = mcp_server._run_wq_single_mcp_task("task-request-id", params)

        get_client.assert_called_once_with("primary", request_id=request_id)
        self.assertEqual(run_single.call_args.kwargs["request_id"], request_id)
        self.assertEqual(result["request_id"], request_id)
        client.close.assert_called_once_with()

    def test_local_backtest_wait_honors_cooperative_cancellation(self):
        future = Mock()
        executor = Mock()
        executor.submit_cpu_work.return_value = future
        params = {
            "expression": "rank(close)",
            "n_groups": 5,
            "holding_period": 5,
            "neutralize_industry": True,
            "neutralize_cap": True,
        }
        with (
            patch.object(mcp_server, "_load_local_factor_data", return_value=(object(), ["000001.SZ"], None)),
            patch.object(mcp_server, "get_executor", return_value=executor),
            patch.object(mcp_server, "is_mcp_task_cancelled", return_value=True),
            patch.object(mcp_server, "update_mcp_task"),
        ):
            result, _stocks, error = mcp_server._run_local_backtest_process("task", params)

        self.assertIsNone(result)
        assert error is not None
        self.assertTrue(error["cancelled"])
        future.cancel.assert_called_once_with()


class TestMCPBackgroundLifecycle(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.persist_async = patch.object(task_helper, "persist_task_to_db_async", new=AsyncMock())
        self.persist_sync = patch.object(task_helper, "persist_task_to_db")
        self.persist_async.start()
        self.persist_sync.start()
        self.task_ids = []

    async def asyncTearDown(self):
        self.persist_async.stop()
        self.persist_sync.stop()
        with task_helper.tasks_lock:
            for task_id in self.task_ids:
                task_helper.tasks.pop(task_id, None)

    async def _new_task(self, task_type="smoke"):
        task_id = await task_helper.start_mcp_task(task_type, "x", {}, status="pending")
        self.task_ids.append(task_id)
        return task_id

    async def test_background_worker_persists_final_result(self):
        task_id = await self._new_task()

        def worker(tid):
            task_helper.update_mcp_task(tid, status="simulating", progress=50)
            time.sleep(0.02)
            return {"ok": True, "value": 42}

        task_helper.start_mcp_background_task(task_id, worker, expression="x")

        deadline = time.time() + 2
        snapshot = None
        while time.time() < deadline:
            snapshot = await task_helper.get_mcp_task_snapshot(task_id)
            if snapshot and snapshot.get("status") == "completed":
                break
            await asyncio.sleep(0.01)

        assert snapshot is not None
        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["progress"], 100)
        self.assertEqual(snapshot["result"]["value"], 42)

    async def test_failed_worker_is_persisted_as_failed(self):
        task_id = await self._new_task("smoke-failure")

        def worker(_tid):
            return {"ok": False, "error": "expected failure"}

        task_helper.start_mcp_background_task(task_id, worker)

        deadline = time.time() + 2
        snapshot = None
        while time.time() < deadline:
            snapshot = await task_helper.get_mcp_task_snapshot(task_id)
            if snapshot and snapshot.get("status") == "failed":
                break
            await asyncio.sleep(0.01)

        assert snapshot is not None
        self.assertEqual(snapshot["status"], "failed")
        self.assertEqual(snapshot["error"], "expected failure")

    async def test_request_id_and_http_error_contract_survive_background_lifecycle(self):
        request_id = "req-502-chain"
        task_id = await task_helper.start_mcp_task(
            "wq_brain_submit",
            "rank(close)",
            {},
            status="pending",
            request_id=request_id,
        )
        self.task_ids.append(task_id)

        def worker(_tid):
            return {
                "ok": False,
                "error": "HTTP 502: bad gateway",
                "request_id": request_id,
                "layer": "wq_http",
                "kind": "http_error",
                "http_status": 502,
                "retryable": True,
            }

        task_helper.start_mcp_background_task(task_id, worker)

        deadline = time.time() + 2
        snapshot = None
        while time.time() < deadline:
            snapshot = await task_helper.get_mcp_task_snapshot(task_id)
            if snapshot and snapshot.get("status") == "failed":
                break
            await asyncio.sleep(0.01)

        assert snapshot is not None
        self.assertEqual(snapshot["request_id"], request_id)
        self.assertEqual(snapshot["params"]["request_id"], request_id)
        self.assertEqual(snapshot["result"]["request_id"], request_id)
        self.assertEqual(snapshot["layer"], "wq_http")
        self.assertEqual(snapshot["kind"], "http_error")
        self.assertEqual(snapshot["http_status"], 502)
        self.assertTrue(snapshot["retryable"])

    async def test_task_limit_is_enforced_atomically(self):
        first_task_id = await self._new_task("capacity")
        with (
            patch.object(task_helper, "MAX_ACTIVE_TASKS", 1),
            self.assertRaises(task_helper.MCPTaskCapacityError),
        ):
            await task_helper.start_mcp_task("capacity", "x", {}, status="pending")
        self.assertIn(first_task_id, task_helper.tasks)

    async def test_singleflight_blocks_second_active_task(self):
        with patch.object(
            task_helper,
            "_get_persisted_active_mcp_task",
            new=AsyncMock(return_value=None),
        ) as get_persisted:
            first_task_id = await task_helper.start_mcp_task(
                "wq_research",
                "rank(close)",
                {},
                status="pending",
                singleflight=True,
            )
            self.task_ids.append(first_task_id)
            with self.assertRaises(task_helper.MCPTaskAlreadyRunningError) as ctx:
                await task_helper.start_mcp_task(
                    "wq_research",
                    "rank(open)",
                    {},
                    status="pending",
                    singleflight=True,
                )

        self.assertEqual(ctx.exception.task["task_id"], first_task_id)
        self.assertEqual(get_persisted.await_count, 1)
        with task_helper.tasks_lock:
            params = task_helper.tasks[first_task_id]["params"]
        self.assertTrue(params["_singleflight"])
        self.assertEqual(params["_singleflight_stale_seconds"], task_helper.MCP_SINGLEFLIGHT_STALE_SECONDS)
        self.assertIn("_heartbeat_at", params)

    async def test_singleflight_releases_recent_task_from_previous_process(self):
        persisted = {
            "task_id": "recent-research",
            "status": "researching",
            "task_type": "wq_research",
            "params": {
                "_singleflight": True,
                "_mcp_instance_id": "old-process",
                "_singleflight_stale_seconds": 900,
                "_heartbeat_at": time.time() - 30,
            },
        }
        with (
            patch.object(
                task_helper,
                "_get_persisted_active_mcp_task",
                new=AsyncMock(return_value=persisted),
            ),
            patch.object(
                task_helper,
                "_fail_persisted_mcp_task",
                new=AsyncMock(),
            ) as fail_persisted,
        ):
            active = await task_helper.get_active_mcp_task("wq_research")

        self.assertIsNone(active)
        fail_persisted.assert_awaited_once()
        self.assertEqual(fail_persisted.await_args.args[0], "recent-research")
        self.assertIn("previous QuantGPT process", fail_persisted.await_args.args[1])
        self.assertTrue(task_helper.is_mcp_singleflight_lease_fresh(persisted))

    async def test_singleflight_releases_orphaned_task_after_restart(self):
        persisted = {
            "task_id": "old-research",
            "status": "researching",
            "task_type": "wq_research",
            "params": {
                "_singleflight": True,
                "_mcp_instance_id": "old-process",
                "_singleflight_stale_seconds": 900,
                "_heartbeat_at": time.time() - 1000,
            },
        }
        with (
            patch.object(
                task_helper,
                "_get_persisted_active_mcp_task",
                new=AsyncMock(return_value=persisted),
            ),
            patch.object(
                task_helper,
                "_fail_persisted_mcp_task",
                new=AsyncMock(),
            ) as fail_persisted,
        ):
            active = await task_helper.get_active_mcp_task("wq_research")

        self.assertIsNone(active)
        fail_persisted.assert_awaited_once()
        self.assertEqual(fail_persisted.await_args.args[0], "old-research")

    async def test_cancelled_background_worker_stays_cancelled(self):
        task_id = await self._new_task("smoke-cancel")

        def worker(tid):
            for _ in range(100):
                if task_helper.is_mcp_task_cancelled(tid):
                    return {"ok": False, "cancelled": True, "error": "cancelled"}
                time.sleep(0.005)
            return {"ok": True}

        task_helper.start_mcp_background_task(task_id, worker, expression="x")
        await asyncio.sleep(0.02)
        cancelled = await task_helper.cancel_mcp_task(task_id)
        await asyncio.sleep(0.03)
        snapshot = await task_helper.get_mcp_task_snapshot(task_id)

        assert cancelled is not None
        assert snapshot is not None
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(snapshot["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
