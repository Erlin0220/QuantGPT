"""Regression tests for WorldQuant simulation recovery and cancellation."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import quantgpt.wq_brain_client as wq_client_module
from quantgpt.wq_brain_client import WQBrainClient


class _Response:
    def __init__(self, status_code=200, data=None, headers=None, text=""):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._data


class _RecoveringSession:
    def __init__(self):
        self.alpha_list_calls = 0

    def post(self, url, json=None, timeout=None):
        assert timeout is not None
        return _Response(201, headers={"Location": "/simulations/sim-recover-1"})

    def get(self, url, params=None, timeout=None):
        if "users/self/alphas" in url:
            self.alpha_list_calls += 1
            if self.alpha_list_calls == 1:
                return _Response(200, {"results": []})
            return _Response(
                200,
                {
                    "results": [
                        {
                            "id": "alpha-recovered-1",
                            "dateCreated": datetime.now(timezone.utc).isoformat(),
                            "regular": {"code": "rank(close)"},
                            "settings": {
                                "region": "USA",
                                "universe": "TOP3000",
                                "delay": 1,
                                "decay": 0,
                                "neutralization": "SUBINDUSTRY",
                                "truncation": 0.08,
                            },
                            "is": {"sharpe": 1.9, "fitness": 1.1},
                            "oos": {},
                        }
                    ]
                },
            )
        return _Response(
            200,
            {
                "id": "sim-recover-1",
                "status": "IN_PROGRESS",
                "progress": 0.95,
            },
        )

    def close(self):
        pass


class TestWQSimulationRecovery(unittest.TestCase):
    def test_simulation_recovers_alpha_when_poll_endpoint_lags(self):
        with (
            patch.object(wq_client_module, "_RECOVERY_LOOKUP_EVERY", 1),
            patch.object(wq_client_module, "_POLL_MAX_ATTEMPTS", 2),
        ):
            client = WQBrainClient("user@example.com", "pw")
            client._session = _RecoveringSession()

            result = client.simulate("rank(close)")

        self.assertTrue(result["ok"])
        self.assertEqual(result["alpha_id"], "alpha-recovered-1")
        self.assertEqual(result["simulation_id"], "sim-recover-1")
        self.assertTrue(result["recovered_from_platform"])

    def test_simulation_can_be_cancelled_before_remote_submit(self):
        client = WQBrainClient("user@example.com", "pw")
        result = client.simulate("rank(close)", cancel_callback=lambda: True)

        self.assertFalse(result["ok"])
        self.assertTrue(result["cancelled"])
        self.assertIn("cancelled", result["error"].lower())

    def test_fork_authenticated_uses_independent_session_and_copies_state(self):
        client = WQBrainClient("user@example.com", "pw")
        session = client._get_session()
        session.headers["X-Test-Header"] = "copied"
        session.cookies.set("sessionid", "abc123")
        client._operators_cache = [{"name": "rank"}]

        child = client.fork_authenticated()
        child_session = child._get_session()

        self.assertIsNot(child_session, session)
        self.assertEqual(child_session.headers["X-Test-Header"], "copied")
        self.assertEqual(child_session.cookies.get("sessionid"), "abc123")
        self.assertEqual(child._operators_cache, [{"name": "rank"}])

        child._operators_cache.append({"name": "ts_delta"})
        self.assertEqual(client._operators_cache, [{"name": "rank"}])

        child.close()
        client.close()


if __name__ == "__main__":
    unittest.main()
