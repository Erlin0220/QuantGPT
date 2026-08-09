from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quantgpt.models import Base
from quantgpt.wq_correlation_proxy import (
    correlation_evidence_is_fresh,
    correlation_priority_multiplier,
    daily_changes_from_pnl,
    portfolio_correlation_proxy,
)
from quantgpt.wq_submission_policy import _priority_score, get_submission_policy_status, record_research_candidates


def _cumulative(values: list[float]) -> list[list[object]]:
    return [[f"2026-01-{index + 1:02d}", value] for index, value in enumerate(values)]


def test_daily_changes_uses_pnl_differences_not_cumulative_curve():
    assert list(daily_changes_from_pnl(_cumulative([10.0, 12.0, 11.0, 15.0])).values()) == [2.0, -1.0, 4.0]


def test_high_portfolio_correlation_is_detected_from_daily_changes():
    candidate = {str(index): float(index % 7 - 3) for index in range(30)}
    evidence = portfolio_correlation_proxy(candidate, {"active": {k: v * 2 for k, v in candidate.items()}})
    assert evidence["max_correlation"] == 1.0
    assert evidence["matching_alpha_id"] == "active"
    assert evidence["sample_length"] == 30
    assert evidence["high_correlation"] is True
    assert evidence["official_sc"] is False


def test_missing_short_or_stale_evidence_is_non_blocking():
    short = portfolio_correlation_proxy({"1": 0.1, "2": 0.2}, {"active": {"1": 0.1, "2": 0.2}})
    assert short["status"] == "unavailable"
    assert correlation_priority_multiplier(short) == 1.0
    now = datetime(2026, 8, 10, tzinfo=timezone.utc)
    stale = {"status": "available", "max_correlation": 0.95, "calculated_at": (now - timedelta(hours=72)).isoformat()}
    assert correlation_evidence_is_fresh(stale, now=now) is False
    assert correlation_priority_multiplier(stale) == 1.0


def test_high_local_correlation_only_lowers_priority_and_sc_pending_is_not_a_local_fail():
    base = {
        "expression": "rank(close)",
        "is_metrics": {"sharpe": 1.6, "fitness": 1.3, "returns": 0.12, "turnover": 0.2, "checks": [{"name": "SELF_CORRELATION", "result": "PENDING"}]},
        "validation": {"robustness_score": 0.9},
        "novelty_score": 0.8,
    }
    high = dict(base, local_correlation_proxy={"status": "available", "max_correlation": 0.9, "calculated_at": datetime.now(timezone.utc).isoformat(), "official_sc": False})
    assert _priority_score(base) > 0
    assert 0 < _priority_score(high) < _priority_score(base)


def test_research_agent_enriches_candidate_from_active_portfolio(monkeypatch):
    import quantgpt.wq_research_agent as agent
    monkeypatch.setattr(agent, "run_list_alphas", lambda *_args, **_kwargs: {"ok": True, "alphas": [{"alpha_id": "active-1", "status": "ACTIVE"}]})

    class FakeClient:
        def fetch_alpha_pnl(self, alpha_id, refresh=False):
            scale = 2 if alpha_id == "candidate-1" else 1
            total = 0.0
            records = []
            for index in range(30):
                total += float((index % 5) - 2) * scale
                records.append([str(index), total])
            return {"records": records}

    candidates = [{"alpha_id": "candidate-1", "validation": {"status": "ready"}}]
    agent._attach_local_correlation_proxy(FakeClient(), candidates)
    evidence = candidates[0]["local_correlation_proxy"]
    assert evidence["status"] == "available"
    assert evidence["matching_alpha_id"] == "active-1"
    assert candidates[0]["validation"]["local_correlation_proxy"] == evidence


@pytest_asyncio.fixture
async def correlation_db(monkeypatch):
    import quantgpt.db as db
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(db, "_session_factory", factory)
    yield
    await engine.dispose()


@pytest.mark.asyncio
async def test_correlation_evidence_persists_and_is_exposed_in_status(correlation_db):
    calculated_at = datetime.now(timezone.utc).isoformat()
    await record_research_candidates("primary", [{
        "alpha_id": "corr-1",
        "expression": "rank(close)",
        "is_metrics": {"sharpe": 1.7, "fitness": 1.3, "returns": 0.1, "turnover": 0.2},
        "validation": {"status": "ready", "robustness_score": 0.9},
        "local_correlation_proxy": {"status": "available", "max_correlation": 0.81, "matching_alpha_id": "active-9", "sample_length": 120, "calculated_at": calculated_at, "official_sc": False},
    }])
    status = await get_submission_policy_status("primary")
    evidence = status["candidate_queue_top"][0]["local_correlation_proxy"]
    assert evidence["max_correlation"] == pytest.approx(0.81)
    assert evidence["matching_alpha_id"] == "active-9"
    assert evidence["sample_length"] == 120
    assert evidence["official_sc"] is False
