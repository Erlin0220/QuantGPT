import pytest

from quantgpt import wq_validation_evidence_seed as seed


def test_validation_evidence_seed_is_multi_source_and_reproducible():
    source_keys = [item["source_key"] for item in seed.VALIDATION_EVIDENCE_SOURCES]
    assert len(source_keys) == len(set(source_keys))
    assert len(source_keys) >= 6
    assert "ssrn:bailey-borwein-lopez-zhu-pbo-2015" in source_keys
    assert "ssrn:bailey-lopez-deflated-sharpe-2014" in source_keys
    assert "ssrn:bailey-lopez-probabilistic-sharpe-2012" in source_keys
    assert "textbook:murphy-probabilistic-machine-learning-2022" in source_keys
    assert "paper:arrieta-ibarra-calibration-metrics-2022" in source_keys
    assert "worldquant:learn2quant-validation-evidence" in source_keys

    known = set(source_keys)
    concepts = {card["concept"] for card in seed.VALIDATION_EVIDENCE_CARDS}
    assert {
        "hypothesis_targeted_robustness_design",
        "lineage_aware_multiple_testing_evidence",
        "calibrated_active_probability_requires_resolved_outcomes",
        "candidate_evidence_hierarchy_separates_rules_from_preferences",
    } <= concepts
    for card in seed.VALIDATION_EVIDENCE_CARDS:
        evidence = card.get("evidence") or []
        assert len({item["source_key"] for item in evidence}) >= 2
        assert all(item["source_key"] in known for item in evidence)
        assert card.get("status") == "active"
        assert card.get("family") in {"robustness_validation", "candidate_evidence"}


def test_candidate_probability_card_rejects_magic_score_semantics():
    card = next(
        item
        for item in seed.VALIDATION_EVIDENCE_CARDS
        if item["concept"] == "calibrated_active_probability_requires_resolved_outcomes"
    )
    joined = " ".join(
        [card["hypothesis"], *card["mechanism"], *card["failure_modes"], *card["mutation_strategies"]]
    ).lower()
    assert "hand-weighted" in joined or "weighted score" in joined
    assert "probability" in joined
    assert "resolved" in joined


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

    result = await seed.seed_validation_evidence_knowledge()

    assert result == {
        "sources": len(seed.VALIDATION_EVIDENCE_SOURCES),
        "cards": len(seed.VALIDATION_EVIDENCE_CARDS),
    }
    assert len(seen_sources) == result["sources"]
    assert len(seen_cards) == result["cards"]
