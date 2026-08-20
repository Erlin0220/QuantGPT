"""Tests for account-visible WorldQuant BRAIN data catalogs."""

from quantgpt.wq_brain_client import WQBrainClient


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self._payload


class _CatalogSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if url.endswith("/data-fields"):
            offset = int((params or {}).get("offset", 0))
            if offset == 0:
                return _Response({"count": 3, "results": [{"id": "f1", "type": "MATRIX"}, {"id": "f2", "type": "MATRIX"}]})
            return _Response({"count": 3, "results": [{"id": "f3", "type": "MATRIX"}]})
        if url.endswith("/data-sets"):
            return _Response({"count": 1, "results": [{"id": "fundamental6", "name": "Fundamentals"}]})
        return _Response({}, status_code=404)


class _DatasetPaginationSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        self.calls.append((url, params))
        if not url.endswith("/data-sets"):
            return _Response({}, status_code=404)
        if int(params.get("limit", 0)) > 50:
            return _Response({"detail": "pagination limit too high"}, status_code=400)
        offset = int(params.get("offset", 0))
        if offset == 0:
            return _Response({"count": 3, "results": [{"id": "pv1"}, {"id": "fundamental6"}]})
        return _Response({"count": 3, "results": [{"id": "analyst4"}]})


def test_list_data_fields_paginates_and_caches():
    client = WQBrainClient(email="x", password="y")
    session = _CatalogSession()
    client._session = session

    first = client.list_data_fields(limit=3)
    second = client.list_data_fields(limit=3)

    assert [item["id"] for item in first] == ["f1", "f2", "f3"]
    assert second == first
    assert len([call for call in session.calls if call[0].endswith("/data-fields")]) == 2
    assert session.calls[0][1]["instrumentType"] == "EQUITY"
    assert session.calls[0][1]["region"] == "USA"
    assert session.calls[0][1]["universe"] == "TOP3000"


def test_list_data_fields_passes_dataset_and_search_filters():
    client = WQBrainClient(email="x", password="y")
    session = _CatalogSession()
    client._session = session

    client.list_data_fields(dataset_id="fundamental6", search="cashflow", limit=1)

    params = session.calls[0][1]
    assert params["dataset.id"] == "fundamental6"
    assert params["search"] == "cashflow"


def test_list_datasets_uses_brain_data_sets_endpoint_and_caches():
    client = WQBrainClient(email="x", password="y")
    session = _CatalogSession()
    client._session = session

    first = client.list_datasets(limit=10)
    second = client.list_datasets(limit=10)

    assert first == [{"id": "fundamental6", "name": "Fundamentals"}]
    assert second == first
    assert len([call for call in session.calls if call[0].endswith("/data-sets")]) == 1


def test_list_datasets_paginates_with_brain_safe_page_size():
    client = WQBrainClient(email="x", password="y")
    session = _DatasetPaginationSession()
    client._session = session

    result = client.list_datasets(limit=100)

    assert [item["id"] for item in result] == ["pv1", "fundamental6", "analyst4"]
    dataset_calls = [call for call in session.calls if call[0].endswith("/data-sets")]
    assert len(dataset_calls) == 2
    assert all(call[1]["limit"] <= 50 for call in dataset_calls)
    assert [call[1]["offset"] for call in dataset_calls] == [0, 2]
