"""Tests for wq_brain_client.py and routes/wq_brain.py."""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from quantgpt.wq_brain_client import WQBrainClient, configured_accounts, get_client, is_configured
from quantgpt.wq_brain_service import run_account_status, run_list_alphas, run_submit_by_ids


class TestIsConfigured:
    def test_not_configured_when_empty(self):
        with patch.dict(os.environ, {"WQ_BRAIN_EMAIL": "", "WQ_BRAIN_PASSWORD": ""}, clear=False):
            assert is_configured() is False

    def test_not_configured_when_missing(self):
        env = os.environ.copy()
        env.pop("WQ_BRAIN_EMAIL", None)
        env.pop("WQ_BRAIN_PASSWORD", None)
        with patch.dict(os.environ, env, clear=True):
            assert is_configured() is False

    def test_configured_when_both_set(self):
        with patch.dict(os.environ, {"WQ_BRAIN_EMAIL": "a@b.com", "WQ_BRAIN_PASSWORD": "pw"}, clear=False):
            assert is_configured() is True

    def test_configured_specific_account(self):
        with patch.dict(os.environ, {
            "WQ_BRAIN_EMAIL": "a@b.com", "WQ_BRAIN_PASSWORD": "pw",
            "WQ_BRAIN_ALT_EMAIL": "", "WQ_BRAIN_ALT_PASSWORD": "",
        }, clear=False):
            assert is_configured("primary") is True
            assert is_configured("alt") is False

    def test_configured_alt_account(self):
        with patch.dict(os.environ, {
            "WQ_BRAIN_EMAIL": "", "WQ_BRAIN_PASSWORD": "",
            "WQ_BRAIN_ALT_EMAIL": "alt@b.com", "WQ_BRAIN_ALT_PASSWORD": "pw2",
        }, clear=False):
            assert is_configured("primary") is False
            assert is_configured("alt") is True
            assert is_configured() is True

    def test_configured_accounts(self):
        with patch.dict(os.environ, {
            "WQ_BRAIN_EMAIL": "a@b.com", "WQ_BRAIN_PASSWORD": "pw",
            "WQ_BRAIN_ALT_EMAIL": "alt@b.com", "WQ_BRAIN_ALT_PASSWORD": "pw2",
        }, clear=False):
            accts = configured_accounts()
            assert "primary" in accts
            assert "alt" in accts

    def test_get_client_primary(self):
        with patch.dict(os.environ, {
            "WQ_BRAIN_EMAIL": "main@b.com", "WQ_BRAIN_PASSWORD": "pw1",
            "WQ_BRAIN_ALT_EMAIL": "alt@b.com", "WQ_BRAIN_ALT_PASSWORD": "pw2",
        }, clear=False):
            c = get_client("primary")
            assert c.email == "main@b.com"
            c2 = get_client("alt")
            assert c2.email == "alt@b.com"


class TestWQBrainClient:
    def test_init_from_params(self):
        c = WQBrainClient(email="test@test.com", password="pass")
        assert c.email == "test@test.com"
        assert c.password == "pass"

    def test_init_from_env(self):
        with patch.dict(os.environ, {"WQ_BRAIN_EMAIL": "env@test.com", "WQ_BRAIN_PASSWORD": "envpw"}):
            c = WQBrainClient()
            assert c.email == "env@test.com"
            assert c.password == "envpw"

    def test_close_without_session(self):
        c = WQBrainClient(email="a", password="b")
        c.close()

    def test_authenticate_success(self):
        c = WQBrainClient(email="a@b.com", password="pw")
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"permissions": ["SUBMIT"]}
        mock_session.post.return_value = mock_resp
        c._session = mock_session
        assert c.authenticate() is True

    def test_authenticate_failure(self):
        c = WQBrainClient(email="a@b.com", password="wrong")
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_session.post.return_value = mock_resp
        c._session = mock_session
        assert c.authenticate() is False

    def test_authenticate_biometric(self):
        c = WQBrainClient(email="a@b.com", password="pw")
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"inquiry": "biometric_required"}
        mock_session.post.return_value = mock_resp
        c._session = mock_session
        assert c.authenticate() is False

    def test_submit_connection_timeout_is_unknown_and_never_blindly_retried(self):
        c = WQBrainClient(email="a@b.com", password="pw")
        mock_session = MagicMock()
        mock_session.post.side_effect = requests.Timeout("socket timed out")
        c._session = mock_session

        result = c.submit_alpha("alpha-timeout")

        assert result["ok"] is False
        assert result["platform_status"] == "UNKNOWN"
        assert result["submission_uncertain"] is True
        assert mock_session.post.call_count == 1

    def test_submit_poll_timeout_with_unknown_platform_status_is_not_marked_active(self):
        c = WQBrainClient(email="a@b.com", password="pw")
        mock_session = MagicMock()
        submit_response = MagicMock(status_code=201, text="accepted")
        mock_session.post.return_value = submit_response
        c._session = mock_session
        c._poll_alpha_submission = MagicMock(return_value={
            "status_code": 200,
            "ok": False,
            "detail": "poll timed out",
            "platform_status": "TIMEOUT",
        })
        c._fetch_alpha = MagicMock(return_value={})

        result = c.submit_alpha("alpha-unknown")

        assert result["ok"] is False
        assert result["platform_status"] == "UNKNOWN"
        assert result["submission_uncertain"] is True

    def test_account_metadata_endpoints(self):
        c = WQBrainClient(email="a@b.com", password="pw")
        mock_session = MagicMock()
        competition_resp = MagicMock(status_code=200)
        competition_resp.json.return_value = {"results": [{"id": "challenge"}]}
        summary_resp = MagicMock(status_code=200)
        summary_resp.json.return_value = {"active": 1, "unsubmitted": 2, "decommissioned": 0}
        mock_session.get.side_effect = [competition_resp, summary_resp]
        c._session = mock_session

        assert c.get_user_competitions("U1")["results"][0]["id"] == "challenge"
        assert c.get_user_alpha_summary()["active"] == 1
        assert mock_session.get.call_args_list[0].args[0].endswith("/users/U1/competitions")
        assert mock_session.get.call_args_list[1].args[0].endswith("/users/self/alphas/summary")


class TestSubmitByIdsService:
    def test_uncertain_remote_result_maps_to_fail_closed_submit_unknown(self):
        client = MagicMock()
        client.submit_alpha.return_value = {
            "ok": False,
            "platform_status": "UNKNOWN",
            "submission_uncertain": True,
            "detail": "network outcome unknown",
        }

        result = run_submit_by_ids(
            client,
            ["alpha-unknown"],
            submission_guard=lambda _alpha_id: {"allowed": True},
        )

        assert result["active"] == 0
        assert result["timeout"] == 1
        assert result["results"]["alpha-unknown"]["final_status"] == "SUBMIT_UNKNOWN"
        assert result["results"]["alpha-unknown"]["submission_uncertain"] is True

    def test_explicit_403_platform_check_failure_is_terminal_other_fail(self):
        client = MagicMock()
        client.submit_alpha.return_value = {
            "ok": False,
            "status_code": 403,
            "detail": '{"is":{"checks":[{"name":"LOW_SUB_UNIVERSE_SHARPE","result":"FAIL","limit":0.55,"value":0.5},{"name":"SELF_CORRELATION","result":"PENDING"}]}}',
        }

        result = run_submit_by_ids(
            client,
            ["alpha-platform-fail"],
            submission_guard=lambda _alpha_id: {"allowed": True},
        )

        entry = result["results"]["alpha-platform-fail"]
        assert entry["final_status"] == "OTHER_FAIL"
        assert entry["confirmed_not_submitted"] is True
        assert entry["platform_check_failure"] == "LOW_SUB_UNIVERSE_SHARPE"
        assert result["timeout"] == 0


class TestListAlphasService:
    def test_status_filter_is_sent_to_platform_before_pagination(self):
        client = MagicMock()
        session = MagicMock()
        client._get_session.return_value = session

        def get_alphas(_url, *, params, timeout):
            response = MagicMock(status_code=200)
            if params.get("status") == "ACTIVE":
                response.json.return_value = {
                    "results": [{
                        "id": "active-1",
                        "status": "ACTIVE",
                        "regular": {"code": "rank(close)"},
                        "settings": {"neutralization": "INDUSTRY"},
                        "is": {"fitness": 1.2, "sharpe": 1.8},
                    }],
                }
            else:
                response.json.return_value = {
                    "results": [{
                        "id": "recent-unsubmitted",
                        "status": "UNSUBMITTED",
                        "regular": {"code": "rank(volume)"},
                        "settings": {"neutralization": "INDUSTRY"},
                        "is": {"fitness": 0.5, "sharpe": 0.7},
                    }],
                }
            return response

        session.get.side_effect = get_alphas

        result = run_list_alphas(client, limit=100, offset=0, status_filter="active")

        assert result["ok"] is True
        assert result["total"] == 1
        assert result["alphas"][0]["alpha_id"] == "active-1"
        assert session.get.call_args.kwargs["params"]["status"] == "ACTIVE"


class TestAccountStatusService:
    def test_maps_points_levels_and_alpha_counts(self):
        client = MagicMock()
        client.get_user_info.return_value = {
            "id": "U1",
            "geniusLevel": "BRONZE",
            "level": "NONE",
            "onboarding": None,
        }
        client.get_user_competitions.return_value = {
            "results": [{
                "id": "challenge",
                "leaderboard": {"score": 1234.0, "level": "BRONZE"},
                "progress": {"level": "SILVER", "score": {"remaining": 1766.0}},
            }],
        }
        client.get_user_alpha_summary.return_value = {
            "active": 4,
            "unsubmitted": 12,
            "decommissioned": 1,
        }

        result = run_account_status(client)

        assert result["ok"] is True
        assert result["points"] == 1234
        assert result["points_source"] == "challenge.leaderboard.score"
        assert result["genius_level"] == "BRONZE"
        assert result["next_genius_level"] == "SILVER"
        assert result["points_remaining"] == 8766
        assert result["alpha_counts"] == {
            "total": 17,
            "submitted": 5,
            "active": 4,
            "unsubmitted": 12,
            "decommissioned": 1,
        }
        assert result["goal_reached"] is False

    def test_does_not_treat_next_level_as_current_level(self):
        client = MagicMock()
        client.get_user_info.return_value = {"id": "U1", "geniusLevel": None, "level": "NONE"}
        client.get_user_competitions.return_value = {
            "results": [{
                "id": "challenge",
                "leaderboard": {"score": 0.0, "level": None},
                "progress": {"level": "BRONZE", "score": {"remaining": 1000.0}},
            }],
        }
        client.get_user_alpha_summary.return_value = {
            "active": 1,
            "unsubmitted": 32,
            "decommissioned": 0,
        }

        result = run_account_status(client)

        assert result["points"] == 0
        assert result["genius_level"] == "NONE"
        assert result["next_genius_level"] == "BRONZE"
        assert result["gold_reached"] is False
        assert result["consultant_status"] == "NOT_CONSULTANT"

    def test_marks_leaderboard_lag_when_active_alphas_are_not_counted(self):
        client = MagicMock()
        client.get_user_info.return_value = {"id": "U1", "geniusLevel": None, "level": "NONE"}
        client.get_user_competitions.return_value = {
            "results": [{
                "id": "challenge",
                "status": "ACCEPTED",
                "signUpDate": "2026-04-11T12:04:42-04:00",
                "submissions": False,
                "leaderboard": {"rank": 129114, "score": 0.0, "alphas": 0, "level": None},
                "progress": {"level": "BRONZE", "score": {"remaining": 1000.0}},
            }],
        }
        client.get_user_alpha_summary.return_value = {
            "active": 3,
            "unsubmitted": 272,
            "decommissioned": 0,
        }

        result = run_account_status(client)

        assert result["points"] == 0
        assert result["points_status"] == "LEADERBOARD_LAGGING"
        assert result["leaderboard"]["rank"] == 129114
        assert result["leaderboard"]["alpha_count"] == 0
        assert result["leaderboard"]["active_alpha_gap"] == 3
        assert result["challenge"]["status"] == "ACCEPTED"
        assert result["challenge"]["submissions"] is False

    @patch("quantgpt.wq_brain_service.run_list_alphas")
    def test_missing_platform_alpha_counts_never_claims_points_current(self, mock_list_alphas):
        client = MagicMock()
        client.get_user_info.return_value = {"id": "U1", "geniusLevel": "BRONZE", "level": "BRONZE"}
        client.get_user_competitions.return_value = {
            "results": [{"id": "challenge", "leaderboard": {"score": 3949.0, "alphas": 5, "level": "BRONZE"}}],
        }
        client.get_user_alpha_summary.return_value = {}
        mock_list_alphas.return_value = {"ok": False, "error": "platform unavailable"}

        result = run_account_status(client)

        assert result["points"] == 3949
        assert result["alpha_counts"]["active"] is None
        assert result["leaderboard"]["active_alpha_gap"] is None
        assert result["points_status"] == "SYNC_UNKNOWN"

    @patch("quantgpt.wq_brain_service.run_list_alphas")
    def test_falls_back_to_paginated_alpha_counts(self, mock_list_alphas):
        client = MagicMock()
        client.get_user_info.return_value = {"id": "U1", "geniusLevel": None, "level": "NONE"}
        client.get_user_competitions.return_value = {
            "results": [{"id": "challenge", "leaderboard": {"score": 0.0, "level": None}}],
        }
        client.get_user_alpha_summary.return_value = {}
        mock_list_alphas.return_value = {
            "ok": True,
            "alphas": [
                {"status": "ACTIVE"},
                {"status": "UNSUBMITTED"},
                {"status": "UNSUBMITTED"},
                {"status": "DECOMMISSIONED"},
            ],
        }

        result = run_account_status(client)

        assert result["alpha_counts"] == {
            "total": 4,
            "submitted": 2,
            "active": 1,
            "unsubmitted": 2,
            "decommissioned": 1,
        }
        mock_list_alphas.assert_called_once_with(client, limit=100, offset=0)


class TestWQBrainStatusEndpoint:
    @pytest.mark.asyncio
    async def test_status_returns_config(self, client):
        resp = await client.get("/api/v1/wq-brain/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "configured" in data
        assert "thresholds" in data
        assert data["thresholds"]["sharpe"] == 1.25


class TestWQBrainSubmitEndpoint:
    @pytest.mark.asyncio
    async def test_submit_returns_503_when_not_configured(self, client):
        with patch.dict(os.environ, {"WQ_BRAIN_EMAIL": "", "WQ_BRAIN_PASSWORD": ""}, clear=False):
            resp = await client.post("/api/v1/wq-brain/submit", json={
                "expression": "rank(close)",
                "tag": "test-agent",
            })
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_submit_creates_task(self, client):
        with (
            patch.dict(os.environ, {"WQ_BRAIN_EMAIL": "a@b.com", "WQ_BRAIN_PASSWORD": "pw"}, clear=False),
            patch("quantgpt.routes.wq_brain._run_wq_brain_task"),
        ):
            resp = await client.post("/api/v1/wq-brain/submit", json={
                "expression": "rank(close)",
                "tag": "test-agent",
            })
            assert resp.status_code == 202
            data = resp.json()
            assert "task_id" in data
            assert data["status"] == "pending"


class TestSubmittedAlphasEndpoint:
    @pytest.mark.asyncio
    async def test_list_returns_empty(self, client):
        resp = await client.get("/api/v1/wq-brain/submitted-alphas")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["alphas"] == []


class TestSubmitAlphaEndpoint:
    @pytest.mark.asyncio
    async def test_submit_alpha_task_not_found(self, client):
        with patch.dict(os.environ, {"WQ_BRAIN_EMAIL": "a@b.com", "WQ_BRAIN_PASSWORD": "pw"}, clear=False):
            resp = await client.post("/api/v1/wq-brain/nonexistent/submit-alpha")
            assert resp.status_code == 404
