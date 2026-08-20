import pytest

from quantgpt import wq_official_knowledge_seed as seed


def test_official_worldquant_seed_is_source_grounded_and_reproducible():
    source_keys = [item["source_key"] for item in seed.OFFICIAL_WORLDQUANT_SOURCES]
    assert len(source_keys) == len(set(source_keys))
    assert len(source_keys) >= 20
    assert "worldquant:learn2quant" in source_keys
    assert "worldquant:alpha-examples-104" in source_keys
    assert "worldquant:iqc-guidelines-2026" in source_keys
    assert "worldquant:learn2quant-lesson-10" in source_keys
    assert "worldquant:learn2quant-lesson-11-coda" in source_keys

    known = set(source_keys)
    assert len(seed.OFFICIAL_WORLDQUANT_CARDS) >= 9
    for card in seed.OFFICIAL_WORLDQUANT_CARDS:
        evidence = card.get("evidence") or []
        assert len({item["source_key"] for item in evidence}) >= 2
        assert all(item["source_key"] in known for item in evidence)
        assert card.get("status") == "active"
        assert card.get("hypothesis")


def test_dated_brain_snapshots_do_not_pretend_to_be_live():
    snapshots = {
        item["source_key"]: item for item in seed.OFFICIAL_WORLDQUANT_SOURCES
        if "snapshot-20260810" in item["source_key"]
    }
    assert len(snapshots) == 3
    for item in snapshots.values():
        assert item["access_scope"] == "authenticated_snapshot"
        assert item["metadata"]["captured_date"] == "2026-08-10"
        content = item["content"].lower()
        assert "snapshot" in content
        assert "current" in content or "live" in content


@pytest.mark.asyncio
async def test_seed_calls_existing_knowledge_upserts(monkeypatch):
    seen_sources = []
    seen_cards = []

    async def fake_source(**payload):
        seen_sources.append(payload["source_key"])
        return payload

    async def fake_card(payload):
        seen_cards.append(payload["concept"])
        return payload

    monkeypatch.setattr(seed, "upsert_knowledge_source", fake_source)
    monkeypatch.setattr(seed, "upsert_knowledge_card", fake_card)

    result = await seed.seed_official_worldquant_knowledge()

    assert result == {
        "sources": len(seed.OFFICIAL_WORLDQUANT_SOURCES),
        "cards": len(seed.OFFICIAL_WORLDQUANT_CARDS),
    }
    assert len(seen_sources) == result["sources"]
    assert len(seen_cards) == result["cards"]
