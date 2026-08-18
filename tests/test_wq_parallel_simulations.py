"""Regression tests for adaptive parallel WorldQuant simulations."""

from __future__ import annotations

import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from quantgpt.wq_brain_service import (
    _reset_adaptive_concurrency_hint,
    run_batch_simulation,
    run_single_simulation,
)
from quantgpt.wq_research_agent import run_research_batch


class _SharedState:
    def __init__(self, throttle_above: int | None = None):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.throttle_above = throttle_above


class _ParallelFakeClient:
    def __init__(self, shared: _SharedState):
        self.shared = shared

    def fork_authenticated(self):
        return _ParallelFakeClient(self.shared)

    def close(self):
        pass

    def list_operator_names(self):
        return {"rank", "ts_delta"}

    def simulate(self, expression, progress_callback=None, **_kwargs):
        with self.shared.lock:
            self.shared.active += 1
            self.shared.max_active = max(self.shared.max_active, self.shared.active)
            active = self.shared.active

        try:
            if self.shared.throttle_above is not None and active > self.shared.throttle_above:
                if progress_callback:
                    progress_callback(0, "并发限制，等待 30s")
            time.sleep(0.03)
            return {
                "ok": True,
                "alpha_id": f"alpha-{expression}",
                "is": {
                    "sharpe": 1.5,
                    "fitness": 1.2,
                    "returns": 0.1,
                    "turnover": 0.2,
                    "checks": [],
                },
                "oos": {},
                "settings": {},
                "simulation_id": f"sim-{expression}",
            }
        finally:
            with self.shared.lock:
                self.shared.active -= 1


class TestAdaptiveParallelResearch(unittest.TestCase):
    def setUp(self):
        _reset_adaptive_concurrency_hint()

    def test_research_runs_multiple_simulations_concurrently(self):
        shared = _SharedState()
        expressions = [f"rank(-ts_delta(close, {window}))" for window in range(2, 8)]

        with patch.dict(
            os.environ,
            {"WQ_SIM_CONCURRENCY": "3", "WQ_SIM_CONCURRENCY_MAX": "3"},
            clear=False,
        ):
            result = run_research_batch(
                _ParallelFakeClient(shared),
                expressions,
                skip_existing=False,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"]["simulated"], len(expressions))
        self.assertGreaterEqual(shared.max_active, 2)
        self.assertLessEqual(shared.max_active, 3)
        self.assertEqual(result["summary"]["concurrency"]["initial"], 3)
        self.assertEqual(result["summary"]["concurrency"]["peak"], 3)

    def test_default_concurrency_ceiling_stays_at_three(self):
        shared = _SharedState()
        expressions = [f"rank(-ts_delta(close, {window}))" for window in range(2, 10)]

        with patch.dict(os.environ, {}, clear=False):
            for key in ("WQ_SIM_CONCURRENCY", "WQ_SIM_CONCURRENCY_MAX", "WQ_SIM_CONCURRENCY_GROWTH_WAVES"):
                os.environ.pop(key, None)
            result = run_research_batch(
                _ParallelFakeClient(shared),
                expressions,
                skip_existing=False,
            )

        concurrency = result["summary"]["concurrency"]
        self.assertEqual(concurrency["initial"], 3)
        self.assertEqual(concurrency["max"], 3)
        self.assertEqual(concurrency["peak"], 3)
        self.assertLessEqual(shared.max_active, 3)

    def test_research_reduces_concurrency_after_brain_throttle_signal(self):
        shared = _SharedState(throttle_above=1)
        expressions = [f"rank(-ts_delta(close, {window}))" for window in range(2, 8)]

        with patch.dict(
            os.environ,
            {
                "WQ_SIM_CONCURRENCY": "3",
                "WQ_SIM_CONCURRENCY_MAX": "4",
                "WQ_SIM_CONCURRENCY_GROWTH_WAVES": "2",
            },
            clear=False,
        ):
            result = run_research_batch(
                _ParallelFakeClient(shared),
                expressions,
                skip_existing=False,
            )

        concurrency = result["summary"]["concurrency"]
        self.assertGreaterEqual(concurrency["throttle_events"], 1)
        self.assertLess(concurrency["final"], concurrency["initial"])


    def test_next_task_reuses_learned_backoff(self):
        throttled = _SharedState(throttle_above=1)
        expressions = [f"rank(-ts_delta(close, {window}))" for window in range(2, 8)]

        with patch.dict(
            os.environ,
            {"WQ_SIM_CONCURRENCY": "3", "WQ_SIM_CONCURRENCY_MAX": "4", "WQ_SIM_CONCURRENCY_GROWTH_WAVES": "2"},
            clear=False,
        ):
            first = run_research_batch(_ParallelFakeClient(throttled), expressions, skip_existing=False)
            learned = first["summary"]["concurrency"]["final"]
            second = run_research_batch(_ParallelFakeClient(_SharedState()), expressions[:3], skip_existing=False)

        self.assertLess(learned, 3)
        self.assertEqual(second["summary"]["concurrency"]["initial"], learned)


class TestAdaptiveParallelSweep(unittest.TestCase):
    def setUp(self):
        _reset_adaptive_concurrency_hint()

    def test_parameter_sweep_uses_same_parallel_runner(self):
        shared = _SharedState()

        with patch.dict(
            os.environ,
            {"WQ_SIM_CONCURRENCY": "3", "WQ_SIM_CONCURRENCY_MAX": "3"},
            clear=False,
        ):
            result = run_batch_simulation(
                _ParallelFakeClient(shared),
                "rank(close)",
                regions=["USA"],
                delays=[1],
                universes=["TOP3000", "TOP1000", "TOP500"],
                neutralizations=["SUBINDUSTRY", "MARKET"],
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["total_combinations"], 6)
        self.assertGreaterEqual(shared.max_active, 2)
        self.assertLessEqual(shared.max_active, 3)
        self.assertEqual(result["concurrency"]["initial"], 3)


class TestDeprecatedAutoSubmit(unittest.TestCase):
    def setUp(self):
        _reset_adaptive_concurrency_hint()

    def test_single_simulation_never_bypasses_candidate_pipeline(self):
        result = run_single_simulation(
            _ParallelFakeClient(_SharedState()),
            "rank(close)",
            auto_submit=True,
            submission_guard=lambda _alpha_id: self.fail("submission guard must not be called"),
        )

        self.assertTrue(result["ok"])
        self.assertFalse(result["submitted"])
        self.assertEqual(
            result["submission_blocked"]["reason"],
            "auto_submit_disabled_use_candidate_pipeline",
        )

    def test_batch_simulation_never_bypasses_candidate_pipeline(self):
        result = run_batch_simulation(
            _ParallelFakeClient(_SharedState()),
            "rank(close)",
            regions=["USA"],
            delays=[1],
            universes=["TOP3000"],
            neutralizations=["SUBINDUSTRY"],
            auto_submit=True,
            submission_guard=lambda _alpha_id: self.fail("submission guard must not be called"),
        )

        entry = next(iter(result["sub_results"].values()))
        self.assertFalse(entry["submitted"])
        self.assertEqual(
            entry["submission_blocked"]["reason"],
            "auto_submit_disabled_use_candidate_pipeline",
        )


class TestGlobalSimulationGate(unittest.TestCase):
    def setUp(self):
        _reset_adaptive_concurrency_hint()

    def test_single_simulations_share_the_global_gate(self):
        shared = _SharedState()
        client = _ParallelFakeClient(shared)

        with ThreadPoolExecutor(max_workers=7) as pool:
            futures = [pool.submit(run_single_simulation, client, f"rank(close + {index})") for index in range(7)]
            results = [future.result() for future in futures]

        self.assertTrue(all(result["ok"] for result in results))
        self.assertLessEqual(shared.max_active, 4)


if __name__ == "__main__":
    unittest.main()
