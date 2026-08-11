import pytest

from quantgpt import wq_economic_hypothesis_seed as seed


def test_economic_hypothesis_seed_is_multi_source_and_mechanism_first():
    source_keys = [item["source_key"] for item in seed.ECONOMIC_HYPOTHESIS_SOURCES]
    assert len(source_keys) == len(set(source_keys))
    assert "textbook:tulchinsky-finding-alphas-2e-2019" in source_keys
    assert "chapter:finding-alphas-data-alpha-design-2019" in source_keys
    assert "chapter:finding-alphas-triple-axis-plan-2019" in source_keys
    assert "textbook:ilmanen-expected-returns-2011" in source_keys
    assert "author:aqr-ilmanen-expected-returns-framework-2011" in source_keys

    known = set(source_keys)
    assert all(item["source_type"] in {"textbook", "arxiv", "ssrn", "worldquant", "paper", "other"} for item in seed.ECONOMIC_HYPOTHESIS_SOURCES)
    assert len(seed.ECONOMIC_HYPOTHESIS_CARDS) >= 5
    concepts = {card["concept"] for card in seed.ECONOMIC_HYPOTHESIS_CARDS}
    assert "economic_mechanism_before_expression" in concepts
    assert "three_input_expected_return_triangulation" in concepts
    assert "structured_semantic_expansion_before_parameter_sweep" in concepts
    for card in seed.ECONOMIC_HYPOTHESIS_CARDS:
        evidence = card.get("evidence") or []
        assert len({item["source_key"] for item in evidence}) >= 2
        assert all(item["source_key"] in known for item in evidence)
        assert card.get("family") == "economic_hypothesis"
        assert card.get("status") == "active"
        assert card.get("hypothesis")


def test_economic_hypothesis_sources_do_not_claim_brain_rules_or_guarantees():
    combined = " ".join(
        str(value)
        for item in seed.ECONOMIC_HYPOTHESIS_SOURCES
        for value in (item.get("title"), item.get("content"))
    ).lower()
    assert "official brain threshold" not in combined
    assert "guaranteed alpha" not in combined
    assert "guarantee an alpha" not in combined


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

    result = await seed.seed_economic_hypothesis_knowledge()

    assert result == {
        "sources": len(seed.ECONOMIC_HYPOTHESIS_SOURCES),
        "cards": len(seed.ECONOMIC_HYPOTHESIS_CARDS),
    }
    assert len(seen_sources) == result["sources"]
    assert len(seen_cards) == result["cards"]
