"""Regression tests for bounded WQ BRAIN HTTP calls."""

import unittest
from unittest.mock import MagicMock

from quantgpt.wq_brain_client import WQBrainClient


class TestWQHTTPTimeouts(unittest.TestCase):
    def test_authentication_has_connect_and_read_timeout(self):
        client = WQBrainClient(email="a@b.com", password="pw")
        session = MagicMock()
        response = MagicMock(status_code=200)
        response.json.return_value = {}
        session.post.return_value = response
        client._session = session

        self.assertTrue(client.authenticate(_max_retries=1))
        timeout = session.post.call_args.kwargs.get("timeout")
        self.assertIsInstance(timeout, tuple)
        self.assertEqual(len(timeout), 2)
        self.assertGreater(timeout[0], 0)
        self.assertGreater(timeout[1], 0)


if __name__ == "__main__":
    unittest.main()
