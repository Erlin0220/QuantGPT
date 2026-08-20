import pytest

from quantgpt import wq_failure_diversity_seed as seed


def test_failure_diversity_cards_are_multi_source_and_partitioned():
    assert len(seed.FAILURE_DIVERSITY_CARDS) >= 5
    families = {card["family"] for card in seed.FAILURE_DIVERSITY_CARDS}
    assert {"failure_diagnosis", "diversification"} <= families
    for card in seed.FAILURE_DIVERSITY_CARDS:
        evidence = card.get("evidence") or []
        assert len({item["source_key"] for item in evidence}) >= 2
        assert card.get("status") == "active"
        assert card.get("hypothesis")
        assert card.get("mechanism")


def test_failure_cards_do_not_turn_metric_labels_into_causal_rules():
    text = " ".join(
        str(value)
        for card in seed.FAILURE_DIVERSITY_CARDS
        if card["family"] == "failure_diagnosis"
        for value in (card.get("hypothesis"), card.get("mechanism"), card.get("failure_modes"))
    ).lower()
    assert "observation" in text or "symptom" in text
    assert "counterfactual" in text or "discriminating" in text


def test_diversity_cards_require_information_level_independence():
    text = " ".join(
        str(value)
        for card in seed.FAILURE_DIVERSITY_CARDS
        if card["family"] == "diversification"
        for value in (card.get("hypothesis"), card.get("mechanism"), card.get("mutation_strategies"))
    ).lower()
    assert "information" in text
    assert "economic mechanism" in text
    assert "noise" in text


@pytest.mark.asyncio
async def test_seed_calls_existing_knowledge_upsert(monkeypatch):
    seen = []

    async def fake_card(payload):
        seen.append(payload["concept"])
        return payload

    monkeypatch.setattr(seed, "upsert_knowledge_card", fake_card)
    result = await seed.seed_failure_diversity_knowledge()

    assert result == {"cards": len(seed.FAILURE_DIVERSITY_CARDS)}
    assert len(seen) == result["cards"]
