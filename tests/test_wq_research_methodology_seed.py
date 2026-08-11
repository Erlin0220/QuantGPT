import pytest

from quantgpt import wq_research_methodology_seed as seed


def test_research_methodology_seed_is_multi_source_and_reproducible():
    source_keys = [item["source_key"] for item in seed.RESEARCH_METHODOLOGY_SOURCES]
    assert len(source_keys) == len(set(source_keys))
    assert len(source_keys) >= 6
    assert "textbook:garnett-bayesian-optimization-2023" in source_keys
    assert "textbook:lattimore-szepesvari-bandit-algorithms-2020" in source_keys
    assert "textbook:slivkins-introduction-multi-armed-bandits-2019" in source_keys
    assert "textbook:powell-ryzhov-optimal-learning-2012" in source_keys
    assert "textbook:santner-williams-notz-computer-experiments-2018" in source_keys

    known = set(source_keys)
    assert len(seed.RESEARCH_METHODOLOGY_CARDS) >= 4
    for card in seed.RESEARCH_METHODOLOGY_CARDS:
        evidence = card.get("evidence") or []
        assert len({item["source_key"] for item in evidence}) >= 2
        assert all(item["source_key"] in known for item in evidence)
        assert card.get("status") == "active"
        assert card.get("hypothesis")
        assert card.get("family") == "research_allocation"


def test_methodology_sources_do_not_claim_worldquant_rules():
    combined = " ".join(
        str(value)
        for item in seed.RESEARCH_METHODOLOGY_SOURCES
        for value in (item.get("title"), item.get("content"))
    ).lower()
    assert "submission threshold" not in combined
    assert "official worldquant rule" not in combined


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

    result = await seed.seed_research_methodology_knowledge()

    assert result == {
        "sources": len(seed.RESEARCH_METHODOLOGY_SOURCES),
        "cards": len(seed.RESEARCH_METHODOLOGY_CARDS),
    }
    assert len(seen_sources) == result["sources"]
    assert len(seen_cards) == result["cards"]
