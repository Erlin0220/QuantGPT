"""Regression tests for remote MCP authentication and rate limiting."""

import os
import unittest
from unittest.mock import AsyncMock, patch

from starlette.requests import Request
from starlette.responses import Response

from quantgpt import api_server


def _request(*, client_ip="203.0.113.10", headers=None):
    raw_headers = [(name.lower().encode("ascii"), value.encode("ascii")) for name, value in (headers or {}).items()]
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "https",
            "path": "/mcp/",
            "raw_path": b"/mcp/",
            "query_string": b"",
            "headers": raw_headers,
            "client": (client_ip, 12345),
            "server": ("quantgpt.example", 443),
        }
    )


class TestMCPSecurityMiddleware(unittest.IsolatedAsyncioTestCase):
    async def test_remote_access_requires_configured_key(self):
        call_next = AsyncMock(return_value=Response(status_code=200))
        with (
            patch.dict(os.environ, {"QUANTGPT_MCP_API_KEY": ""}, clear=False),
            patch.object(api_server.task_store, "check_rate_limit", return_value=True),
        ):
            response = await api_server._mcp_security_and_path_rewrite(_request(), call_next)
        self.assertEqual(response.status_code, 503)
        call_next.assert_not_awaited()

    async def test_valid_bearer_key_is_accepted(self):
        call_next = AsyncMock(return_value=Response(status_code=200))
        with (
            patch.dict(os.environ, {"QUANTGPT_MCP_API_KEY": "secret-value"}, clear=False),
            patch.object(api_server.task_store, "check_rate_limit", return_value=True),
        ):
            response = await api_server._mcp_security_and_path_rewrite(
                _request(headers={"Authorization": "Bearer secret-value"}),
                call_next,
            )
        self.assertEqual(response.status_code, 200)
        call_next.assert_awaited_once()

    async def test_rate_limit_is_enforced_before_dispatch(self):
        call_next = AsyncMock(return_value=Response(status_code=200))
        with patch.object(api_server.task_store, "check_rate_limit", return_value=False):
            response = await api_server._mcp_security_and_path_rewrite(_request(client_ip="127.0.0.1"), call_next)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "60")
        call_next.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
