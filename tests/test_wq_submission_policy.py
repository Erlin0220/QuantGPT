"""Tests for the local WQ submission budget and delayed-points reconciliation."""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quantgpt.models import Base
from quantgpt.wq_research_memory import load_research_memory, record_research_trials
from quantgpt.wq_submission_policy import (
    finalize_submission_attempt,
    get_submission_policy_status,
    observe_account_status,
    reconcile_candidate_platform_statuses,
    record_platform_candidates,
    record_research_candidates,
    reserve_submission,
)


@pytest_asyncio.fixture
async def policy_db(monkeypatch):
    import quantgpt.db as db

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(db, "_engine", engine)
    monkeypatch.setattr(db, "_session_factory", factory)
    monkeypatch.setenv("WQ_DAILY_SUBMISSION_BUDGET", "2")
    monkeypatch.setenv("WQ_SUBMISSION_TIMEZONE", "UTC")
    yield
    await engine.dispose()


@pytest.mark.asyncio
async def test_daily_budget_blocks_third_submission(policy_db):
    first = await reserve_submission("primary", "alpha-1")
    assert first["allowed"] is True
    await finalize_submission_attempt("primary", "alpha-1", {"ok": True, "final_status": "ACTIVE"})

    second = await reserve_submission("primary", "alpha-2")
    assert second["allowed"] is True
    await finalize_submission_attempt("primary", "alpha-2", {"ok": True, "final_status": "ACTIVE"})

    third = await reserve_submission("primary", "alpha-3")
    assert third["allowed"] is False
    assert third["reason"] == "daily_submission_budget_exhausted"

    status = await get_submission_policy_status("primary")
    assert status["daily_submission_budget"] == 2
    assert status["used_submission_slots"] == 2
    assert status["remaining_submission_slots"] == 0
    assert status["pending_score_submissions"] == 2


@pytest.mark.asyncio
async def test_lagging_points_do_not_settle_until_leaderboard_is_current(policy_db):
    await reserve_submission("primary", "alpha-1")
    await finalize_submission_attempt("primary", "alpha-1", {"ok": True, "final_status": "ACTIVE"})

    lagging = await observe_account_status(
        "primary",
        {"points": 2000, "points_status": "LEADERBOARD_LAGGING"},
    )
    assert lagging["pending_score_submissions"] == 1
    assert lagging["last_settled_delta"] is None
    assert lagging["daily_submission_budget"] == 2

    current = await observe_account_status(
        "primary",
        {"points": 4000, "points_status": "CURRENT"},
    )
    assert current["pending_score_submissions"] == 0
    assert current["last_settled_delta"] == 2000
    assert current["last_settled_submission_count"] == 1
    assert current["daily_submission_budget"] == 1


@pytest.mark.asyncio
async def test_untracked_leaderboard_gap_freezes_new_submissions(policy_db):
    lagging = await observe_account_status(
        "primary",
        {
            "points": 2000,
            "points_status": "LEADERBOARD_LAGGING",
            "leaderboard": {"active_alpha_gap": 1},
        },
    )
    assert lagging["submission_frozen"] is True
    assert lagging["untracked_active_gap"] == 1

    blocked = await reserve_submission("primary", "alpha-new")
    assert blocked["allowed"] is False
    assert blocked["reason"] == "leaderboard_lagging_with_untracked_submissions"

    current = await observe_account_status(
        "primary",
        {
            "points": 2000,
            "points_status": "CURRENT",
            "leaderboard": {"active_alpha_gap": 0},
        },
    )
    assert current["submission_frozen"] is False

    allowed = await reserve_submission("primary", "alpha-new")
    assert allowed["allowed"] is True


@pytest.mark.asyncio
async def test_platform_history_backfills_only_strong_unsubmitted_candidates(policy_db):
    saved = await record_platform_candidates(
        "primary",
        [
            {
                "alpha_id": "strong-1",
                "expression": "rank(close)",
                "status": "UNSUBMITTED",
                "sharpe": 1.7,
                "fitness": 1.2,
                "returns": 0.08,
                "turnover": 0.12,
            },
            {
                "alpha_id": "weak-1",
                "expression": "rank(open)",
                "status": "UNSUBMITTED",
                "sharpe": 1.1,
                "fitness": 1.1,
                "returns": 0.03,
                "turnover": 0.12,
            },
            {
                "alpha_id": "active-1",
                "expression": "rank(volume)",
                "status": "ACTIVE",
                "sharpe": 2.0,
                "fitness": 1.5,
                "returns": 0.10,
                "turnover": 0.20,
            },
        ],
    )
    assert saved == 1

    status = await get_submission_policy_status("primary")
    assert status["candidate_queue_count"] == 1
    assert status["candidate_queue_top"][0]["alpha_id"] == "strong-1"


@pytest.mark.asyncio
async def test_research_candidate_is_queued_then_removed_when_reserved(policy_db):
    saved = await record_research_candidates(
        "primary",
        [
            {
                "alpha_id": "candidate-1",
                "expression": "rank(close)",
                "is_metrics": {
                    "sharpe": 1.6,
                    "fitness": 1.3,
                    "returns": 0.12,
                    "turnover": 0.25,
                },
            }
        ],
        settings={"region": "USA", "universe": "TOP3000", "delay": 1},
        tag="agent-test",
    )
    assert saved == 1

    before = await get_submission_policy_status("primary")
    assert before["candidate_queue_count"] == 1
    assert before["candidate_queue_top"][0]["alpha_id"] == "candidate-1"

    decision = await reserve_submission("primary", "candidate-1")
    assert decision["allowed"] is True

    after = await get_submission_policy_status("primary")
    assert after["candidate_queue_count"] == 0


@pytest.mark.asyncio
async def test_platform_active_reconciles_stale_candidate_queue_entry(policy_db):
    await record_research_candidates(
        "primary",
        [
            {
                "alpha_id": "already-active",
                "expression": "rank(close)",
                "is_metrics": {"sharpe": 1.8, "fitness": 1.2, "returns": 0.1, "turnover": 0.2},
            }
        ],
    )
    assert (await get_submission_policy_status("primary"))["candidate_queue_count"] == 1

    updated = await reconcile_candidate_platform_statuses(
        "primary",
        {"already-active": {"status": "ACTIVE", "sc_result": None}},
    )

    assert updated == 1
    assert (await get_submission_policy_status("primary"))["candidate_queue_count"] == 0


@pytest.mark.asyncio
async def test_nearby_parameter_variants_keep_only_stronger_queue_candidate(policy_db):
    base = {
        "expression": "rank(ts_mean(returns, 10))",
        "is_metrics": {"sharpe": 1.4, "fitness": 1.1, "returns": 0.08, "turnover": 0.2},
        "research_meta": {"family": "momentum_reversal", "generation": 1},
    }
    await record_research_candidates("primary", [{"alpha_id": "weak-10", **base}])

    stronger = {
        "alpha_id": "strong-20",
        "expression": "rank(ts_mean(returns, 20))",
        "is_metrics": {"sharpe": 1.7, "fitness": 1.4, "returns": 0.11, "turnover": 0.2},
        "research_meta": {
            "family": "momentum_reversal",
            "generation": 2,
            "parent_expression": base["expression"],
            "mutation_type": "widen_windows",
        },
    }
    await record_research_candidates("primary", [stronger])

    status = await get_submission_policy_status("primary")
    assert status["candidate_queue_count"] == 1
    assert status["candidate_queue_top"][0]["alpha_id"] == "strong-20"
    assert status["candidate_queue_top"][0]["family"] == "momentum_reversal"
    assert status["candidate_queue_top"][0]["generation"] == 2


@pytest.mark.asyncio
async def test_research_memory_persists_family_lineage_and_failure_feedback(policy_db):
    result = {
        "tag": "memory-test",
        "settings": {"region": "USA", "universe": "TOP3000"},
        "results": [
            {
                "alpha_id": "trial-1",
                "expression": "rank(ts_mean(returns, 20))",
                "is_metrics": {
                    "sharpe": 0.9,
                    "fitness": 0.6,
                    "returns": 0.03,
                    "turnover": 0.2,
                    "checks": [{"name": "SELF_CORRELATION", "result": "FAIL"}],
                },
                "mutation_targets": ["improve_signal_sharpe", "reduce_self_correlation"],
                "research_meta": {
                    "family": "momentum_reversal",
                    "generation": 2,
                    "parent_expression": "rank(ts_mean(returns, 10))",
                    "mutation_type": "widen_windows",
                },
            }
        ],
        "candidates": [],
        "failed": [],
        "invalid": [],
    }

    saved = await record_research_trials("primary", result, hypothesis="memory hypothesis")
    memory = await load_research_memory("primary")

    assert saved == 1
    assert memory["trials"] == 1
    assert memory["family_counts"]["momentum_reversal"] == 1
    assert memory["self_correlation_family_counts"]["momentum_reversal"] == 1
    assert memory["status_counts"]["rejected"] == 1
