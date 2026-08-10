"""Tests for multi-source Alpha knowledge and WQ research integration."""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quantgpt.models import Base
from quantgpt.wq_autonomous_research import build_knowledge_seed_plan
from quantgpt.wq_knowledge import (
    cross_distill_sources,
    load_knowledge_guidance,
    search_alpha_knowledge,
    upsert_knowledge_card,
    upsert_knowledge_source,
)
from quantgpt.wq_research_memory import record_research_trials


@pytest_asyncio.fixture
async def knowledge_db(monkeypatch):
    import quantgpt.db as db

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(db, "_engine", engine)
    monkeypatch.setattr(db, "_session_factory", factory)
    yield
    await engine.dispose()


async def _seed_sources() -> None:
    await upsert_knowledge_source(
        source_type="textbook",
        source_key="textbook:test",
        title="Test Textbook",
        content="Short-term price pressure can reverse; turnover and implementation costs matter.",
        access_scope="licensed_excerpt",
    )
    await upsert_knowledge_source(
        source_type="arxiv",
        source_key="arxiv:1234.5678",
        title="Test Paper",
        content="We study short-horizon reversal and report evidence consistent with temporary price pressure.",
        access_scope="abstract",
    )


@pytest.mark.asyncio
async def test_single_source_card_is_draft_but_corroborated_card_is_active(knowledge_db):
    await _seed_sources()
    single = await upsert_knowledge_card(
        {
            "concept": "short_term_reversal",
            "family": "momentum_reversal",
            "hypothesis": "Very short-horizon returns may reverse.",
            "evidence": [{"source_key": "textbook:test", "support": "positive"}],
            "operators": ["rank", "ts_delta"],
            "expression_templates": ["-rank(ts_delta(close, 5))"],
            "confidence": 0.8,
        }
    )
    assert single["status"] == "draft"
    assert single["source_count"] == 1

    corroborated = await upsert_knowledge_card(
        {
            "card_key": single["card_key"],
            "concept": "short_term_reversal",
            "family": "momentum_reversal",
            "hypothesis": "Very short-horizon returns may reverse.",
            "evidence": [
                {"source_key": "textbook:test", "support": "positive"},
                {"source_key": "arxiv:1234.5678", "support": "positive"},
            ],
            "operators": ["rank", "ts_delta"],
            "expression_templates": ["-rank(ts_delta(close, 5))"],
            "failure_modes": ["high turnover"],
            "mutation_strategies": ["increase lookback"],
            "confidence": 0.8,
        }
    )
    assert corroborated["status"] == "active"
    assert corroborated["source_count"] == 2

    search = await search_alpha_knowledge(
        query="reversal turnover",
        families=["momentum_reversal"],
        limit=5,
    )
    assert search["count"] == 1
    assert search["cards"][0]["card_key"] == corroborated["card_key"]


@pytest.mark.asyncio
async def test_missing_evidence_sources_cannot_activate_card(knowledge_db):
    card = await upsert_knowledge_card(
        {
            "concept": "unverified_claim",
            "family": "other",
            "hypothesis": "An unverified hypothesis.",
            "evidence": [
                {"source_key": "missing:a", "support": "positive"},
                {"source_key": "missing:b", "support": "positive"},
            ],
            "confidence": 0.99,
        }
    )
    assert card["status"] == "draft"
    assert set(card["missing_source_keys"]) == {"missing:a", "missing:b"}


@pytest.mark.asyncio
async def test_cross_distillation_uses_only_stored_sources_and_activates_multisource_card(knowledge_db, monkeypatch):
    await _seed_sources()

    def fake_llm(system_prompt: str, user_prompt: str, **_kwargs) -> str:
        assert "Use ONLY the supplied sources" in system_prompt
        assert "SOURCE_KEY=textbook:test" in user_prompt
        assert "SOURCE_KEY=arxiv:1234.5678" in user_prompt
        return json.dumps(
            {
                "concept": "short_term_reversal",
                "family": "momentum_reversal",
                "hypothesis": "Temporary price pressure may create short-horizon reversal.",
                "mechanism": ["temporary price pressure"],
                "scope": {"asset": ["equity"], "horizon": ["short"]},
                "evidence": [
                    {"source_key": "textbook:test", "support": "positive", "claim": "cost-aware reversal"},
                    {"source_key": "arxiv:1234.5678", "support": "positive", "claim": "short-horizon reversal"},
                ],
                "operators": ["rank", "ts_delta"],
                "expression_templates": ["-rank(ts_delta(close, 5))"],
                "failure_modes": ["high turnover"],
                "mutation_strategies": ["increase lookback"],
                "confidence": 0.78,
            }
        )

    import quantgpt.iteration as iteration

    monkeypatch.setattr(iteration, "_call_llm", fake_llm)
    result = await cross_distill_sources(
        ["textbook:test", "arxiv:1234.5678"],
        concept="short_term_reversal",
        family="momentum_reversal",
    )
    assert result["stored"]["status"] == "active"
    assert result["stored"]["source_count"] == 2


@pytest.mark.asyncio
async def test_real_trials_empirically_calibrate_knowledge_guidance(knowledge_db):
    await upsert_knowledge_source(source_type="textbook", source_key="textbook:a", title="Textbook A")
    await upsert_knowledge_source(source_type="arxiv", source_key="arxiv:b", title="Paper B")
    card = await upsert_knowledge_card(
        {
            "concept": "reversal",
            "family": "momentum_reversal",
            "hypothesis": "Short-term reversal.",
            "evidence": [
                {"source_key": "textbook:a", "support": "positive"},
                {"source_key": "arxiv:b", "support": "positive"},
            ],
            "operators": ["rank", "ts_delta"],
            "expression_templates": ["-rank(ts_delta(close, 5))"],
            "confidence": 0.7,
        }
    )
    for index, status in enumerate(("candidate", "candidate", "rejected"), start=1):
        expression = f"-rank(ts_delta(close, {index + 2}))"
        await record_research_trials(
            "primary",
            {
                "settings": {"region": "USA", "universe": "TOP3000", "delay": 1},
                "results": [
                    {
                        "alpha_id": f"alpha-{index}",
                        "expression": expression,
                        "is_metrics": {"sharpe": 1.5, "fitness": 1.1, "returns": 0.05, "turnover": 0.2},
                        "research_meta": {
                            "family": "momentum_reversal",
                            "knowledge_card_ids": [card["id"]],
                        },
                    }
                ],
                "candidates": ([{"alpha_id": f"alpha-{index}", "expression": expression}] if status == "candidate" else []),
            },
        )

    guidance = await load_knowledge_guidance(families=["momentum_reversal"], limit=5)
    empirical = guidance["cards"][0]["empirical"]
    assert empirical["trials"] == 3
    assert empirical["candidates"] == 2
    assert empirical["candidate_rate"] == pytest.approx(2 / 3, abs=0.0001)


class _FakeWQClient:
    def list_operator_names(self) -> set[str]:
        return {"rank", "ts_delta"}


@pytest.mark.asyncio
async def test_knowledge_seed_plan_attaches_traceable_card_id(knowledge_db):
    await upsert_knowledge_source(source_type="textbook", source_key="textbook:a", title="Textbook A")
    await upsert_knowledge_source(source_type="arxiv", source_key="arxiv:b", title="Paper B")
    card = await upsert_knowledge_card(
        {
            "concept": "reversal",
            "family": "momentum_reversal",
            "hypothesis": "Short-term reversal.",
            "evidence": [
                {"source_key": "textbook:a", "support": "positive"},
                {"source_key": "arxiv:b", "support": "positive"},
            ],
            "operators": ["rank", "ts_delta"],
            "expression_templates": ["-rank(ts_delta(close, 5))"],
            "confidence": 0.9,
        }
    )
    guidance = await load_knowledge_guidance(limit=5)
    plan = build_knowledge_seed_plan(
        _FakeWQClient(),
        {"knowledge_guidance": guidance},
        seen=set(),
        limit=2,
        hypothesis="find robust alpha",
    )
    assert len(plan) == 1
    assert plan[0]["planner_strategy"] == "multi_source_knowledge"
    assert plan[0]["knowledge_card_ids"] == [card["id"]]
