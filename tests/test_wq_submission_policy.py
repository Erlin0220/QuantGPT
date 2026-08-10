"""Tests for the local WQ submission budget and delayed-points reconciliation."""

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quantgpt.models import Base, WQSubmissionAttempt
from quantgpt.wq_research_memory import load_research_memory, record_research_trials
from quantgpt.wq_submission_policy import (
    finalize_submission_attempt,
    get_candidate_robustness_revalidation_payloads,
    get_submission_policy_status,
    observe_account_status,
    reconcile_candidate_platform_statuses,
    reconcile_submission_reservations,
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


def _current_points(points: float, *, active_alpha_gap: int = 0) -> dict:
    return {
        "points": points,
        "points_status": "CURRENT",
        "leaderboard": {"active_alpha_gap": active_alpha_gap},
    }


async def _seed_ready_candidate(
    alpha_id: str,
    *,
    family: str = "test_family",
    dataset_id: str = "test_dataset",
    expression: str | None = None,
    validation: dict | None = None,
    local_correlation_proxy: dict | None = None,
) -> None:
    field = "field_" + "".join(ch if ch.isalnum() else "_" for ch in alpha_id).lower()
    candidate = {
        "alpha_id": alpha_id,
        "expression": expression or f"rank(ts_mean({field}, 20))",
        "is_metrics": {
            "sharpe": 1.6,
            "fitness": 1.2,
            "returns": 0.07,
            "turnover": 0.2,
            "checks": [{"name": "SELF_CORRELATION", "result": "PASS", "value": 0.3}],
        },
        "research_meta": {"family": family, "dataset_id": dataset_id, "data_fields": [field]},
        "validation": validation or {"status": "ready", "robustness_score": 1.0},
    }
    if local_correlation_proxy is not None:
        candidate["local_correlation_proxy"] = local_correlation_proxy
    await record_research_candidates("primary", [candidate])


@pytest.mark.asyncio
async def test_daily_budget_blocks_third_submission(policy_db):
    for alpha_id in ("alpha-1", "alpha-2", "alpha-3"):
        await _seed_ready_candidate(alpha_id)
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
async def test_empty_inventory_enters_replenishment_mode(policy_db):
    status = await get_submission_policy_status("primary")
    assert status["inventory"]["floor"] == 30
    assert status["inventory"]["high_confidence_count"] == 0
    assert status["inventory"]["deficit"] == 30
    assert status["inventory"]["mode"] == "REPLENISHMENT"
    assert status["research_mode"] == "REPLENISHMENT"


@pytest.mark.asyncio
async def test_cold_start_inventory_uses_research_readiness_not_probability_tier(policy_db):
    await record_research_candidates(
        "primary",
        [
            {
                "alpha_id": "cold-ready",
                "expression": "rank(ts_mean(volume, 20))",
                "is_metrics": {"sharpe": 1.3, "fitness": 1.02, "returns": 0.03, "turnover": 0.2, "checks": []},
                "research_meta": {"family": "price_volume", "dataset_id": "pv1", "data_fields": ["volume"]},
                "validation": {"status": "ready", "robustness_score": 0.4},
            }
        ],
    )
    status = await get_submission_policy_status("primary")

    assert status["active_outcome_learning_gate"]["ready"] is False
    assert status["candidate_eligibility_mode"] == "research_readiness"
    assert status["research_readiness_eligible_count"] == 1
    assert status["inventory"]["high_confidence_count"] == 1
    assert status["inventory"]["deficit"] == 29
    assert status["candidate_queue_top"][0]["research_readiness_eligible"] is True


@pytest.mark.asyncio
async def test_lagging_points_do_not_settle_until_leaderboard_is_current(policy_db):
    await _seed_ready_candidate("alpha-1")
    await reserve_submission("primary", "alpha-1")
    await finalize_submission_attempt("primary", "alpha-1", {"ok": True, "final_status": "ACTIVE"})

    lagging = await observe_account_status(
        "primary",
        {"points": 2000, "points_status": "LEADERBOARD_LAGGING"},
    )
    assert lagging["pending_score_submissions"] == 1
    assert lagging["last_settled_delta"] is None
    assert lagging["daily_submission_budget"] == 2

    current = await observe_account_status("primary", _current_points(4000))
    assert current["pending_score_submissions"] == 0
    assert current["last_settled_delta"] == 2000
    assert current["last_settled_submission_count"] == 1
    assert current["daily_submission_budget"] == 2


@pytest.mark.asyncio
async def test_points_settlement_records_confidence_weighted_research_feedback(policy_db):
    import quantgpt.db as db

    await observe_account_status("primary", _current_points(1000))
    await record_research_candidates(
        "primary",
        [
            {
                "alpha_id": "points-a",
                "expression": "rank(ts_mean(field_a, 20))",
                "is_metrics": {"sharpe": 1.6, "fitness": 1.2, "returns": 0.07, "turnover": 0.2},
                "research_meta": {"family": "family_a", "dataset_id": "dataset_a", "data_fields": ["field_a"]},
                "validation": {"status": "ready", "robustness_score": 1.0},
            },
            {
                "alpha_id": "points-b",
                "expression": "rank(ts_mean(field_b, 20))",
                "is_metrics": {"sharpe": 1.7, "fitness": 1.3, "returns": 0.08, "turnover": 0.2},
                "research_meta": {"family": "family_b", "dataset_id": "dataset_b", "data_fields": ["field_b"]},
                "validation": {"status": "ready", "robustness_score": 1.0},
            },
        ],
    )
    for alpha_id in ("points-a", "points-b"):
        assert (await reserve_submission("primary", alpha_id))["allowed"] is True
        await finalize_submission_attempt("primary", alpha_id, {"ok": True, "final_status": "ACTIVE"})

    settled = await observe_account_status("primary", _current_points(3000))
    assert settled["last_settled_delta"] == 2000
    assert settled["last_settled_submission_count"] == 2

    factory = db._get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(WQSubmissionAttempt).where(WQSubmissionAttempt.alpha_id.in_(["points-a", "points-b"]))
        )
        attempts = list(result.scalars().all())
    assert len(attempts) == 2
    assert {attempt.attributed_points_share for attempt in attempts} == {1000.0}
    assert {attempt.attribution_confidence for attempt in attempts} == {0.5}

    memory = await load_research_memory("primary")
    assert memory["family_points_attribution"] == {"family_a": 1000.0, "family_b": 1000.0}
    assert memory["family_points_feedback"] == {"family_a": 500.0, "family_b": 500.0}
    assert memory["dataset_points_feedback"] == {"dataset_a": 500.0, "dataset_b": 500.0}
    assert memory["points_attribution_rule"] == "equal_share_confidence_weighted"


@pytest.mark.asyncio
async def test_untracked_leaderboard_gap_warns_but_does_not_freeze_new_submissions(policy_db):
    lagging = await observe_account_status(
        "primary",
        {
            "points": 2000,
            "points_status": "LEADERBOARD_LAGGING",
            "leaderboard": {"active_alpha_gap": 1},
        },
    )
    assert lagging["submission_frozen"] is False
    assert lagging["untracked_active_gap"] == 1
    assert lagging["submission_warning"] == "leaderboard_lagging_with_untracked_active_alphas"

    await _seed_ready_candidate("alpha-new")
    allowed_while_lagging = await reserve_submission("primary", "alpha-new")
    assert allowed_while_lagging["allowed"] is True

    current = await observe_account_status(
        "primary",
        {
            "points": 2000,
            "points_status": "CURRENT",
            "leaderboard": {"active_alpha_gap": 0},
        },
    )
    assert current["submission_frozen"] is False

    await _seed_ready_candidate("alpha-second")
    second = await reserve_submission("primary", "alpha-second")
    assert second["allowed"] is True


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
async def test_autonomous_candidate_waits_for_robustness_before_queue_and_submission(policy_db):
    saved = await record_research_candidates(
        "primary",
        [
            {
                "alpha_id": "pending-robustness",
                "expression": "rank(ts_mean(close, 20))",
                "is_metrics": {"sharpe": 1.7, "fitness": 1.2, "returns": 0.08, "turnover": 0.2},
                "validation": {"status": "validation_pending"},
                "research_meta": {"family": "momentum_reversal", "generation": 2},
            }
        ],
    )
    assert saved == 1
    status = await get_submission_policy_status("primary")
    assert status["candidate_queue_count"] == 0

    decision = await reserve_submission("primary", "pending-robustness")
    assert decision["allowed"] is False
    assert decision["reason"] == "candidate_not_ready"
    assert decision["candidate_status"] == "validation_pending"


@pytest.mark.asyncio
async def test_ready_candidate_persists_robustness_novelty_and_live_fields(policy_db):
    await record_research_candidates(
        "primary",
        [
            {
                "alpha_id": "peer",
                "expression": "rank(ts_mean(volume, 20))",
                "is_metrics": {"sharpe": 1.4, "fitness": 1.1, "returns": 0.05, "turnover": 0.2},
            }
        ],
    )
    saved = await record_research_candidates(
        "primary",
        [
            {
                "alpha_id": "ready-live",
                "expression": "rank(ts_mean(fresh_quality, 20))",
                "is_metrics": {"sharpe": 1.8, "fitness": 1.3, "returns": 0.09, "turnover": 0.18},
                "validation": {"status": "ready", "robustness_score": 1.0, "passed": 2, "completed": 2},
                "research_meta": {
                    "family": "fundamental_quality",
                    "generation": 1,
                    "mutation_type": "live_field_seed",
                    "data_fields": ["fresh_quality"],
                    "dataset_id": "fundamentalX",
                },
            }
        ],
    )
    assert saved == 1

    status = await get_submission_policy_status("primary")
    item = next(value for value in status["candidate_queue_top"] if value["alpha_id"] == "ready-live")
    assert item["validation_status"] == "ready"
    assert item["robustness_score"] == 1.0
    assert 0.0 <= item["novelty_score"] <= 1.0
    assert item["data_fields"] == ["fresh_quality"]
    assert item["dataset_id"] == "fundamentalX"


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
                "validation": {"status": "ready", "robustness_score": 1.0},
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
async def test_platform_preflight_does_not_bypass_missing_robustness(policy_db):
    await record_research_candidates(
        "primary",
        [
            {
                "alpha_id": "platform-no-robustness",
                "expression": "rank(close)",
                "is_metrics": {"sharpe": 1.8, "fitness": 1.3, "returns": 0.1, "turnover": 0.2},
                "validation": {"status": "platform_recheck"},
            }
        ],
    )
    updated = await reconcile_candidate_platform_statuses(
        "primary",
        {"platform-no-robustness": {"status": "UNSUBMITTED", "sc_result": "PASS", "sc_value": 0.40}},
    )
    assert updated == 1
    status = await get_submission_policy_status("primary")
    assert status["candidate_queue_count"] == 0
    decision = await reserve_submission("primary", "platform-no-robustness")
    assert decision["allowed"] is False
    assert decision["validation_status"] == "robustness_pending"

    payloads = await get_candidate_robustness_revalidation_payloads(
        "primary",
        ["platform-no-robustness"],
    )
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["expression"] == "rank(close)"
    payload["validation"] = {"status": "ready", "robustness_score": 1.0, "completed": 2, "passed": 2}
    await record_research_candidates(
        "primary",
        [payload],
        settings=payload["settings"],
        tag=payload["tag"],
    )
    promoted = await reserve_submission("primary", "platform-no-robustness")
    assert promoted["allowed"] is True


@pytest.mark.asyncio
async def test_platform_sc_value_updates_queued_candidate_and_priority(policy_db):
    await record_research_candidates(
        "primary",
        [
            {
                "alpha_id": "sc-valued",
                "expression": "rank(close)",
                "is_metrics": {"sharpe": 1.8, "fitness": 1.3, "returns": 0.1, "turnover": 0.2},
                "validation": {"status": "ready", "robustness_score": 1.0},
            }
        ],
    )
    before = await get_submission_policy_status("primary")
    before_item = before["candidate_queue_top"][0]
    before_priority = before_item["priority_score"]

    updated = await reconcile_candidate_platform_statuses(
        "primary",
        {"sc-valued": {"status": "UNSUBMITTED", "sc_result": "PASS", "sc_value": 0.58}},
    )

    assert updated == 1
    after = await get_submission_policy_status("primary")
    item = after["candidate_queue_top"][0]
    assert item["sc_status"] == "PASS"
    assert item["self_correlation"] == 0.58
    assert item["priority_score"] < before_priority


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


@pytest.mark.asyncio
async def test_points_settlement_persists_exact_cohort_and_timing_assumptions(policy_db):
    import quantgpt.db as db

    await observe_account_status("primary", _current_points(1000))
    for alpha_id in ("cohort-a", "cohort-b"):
        await _seed_ready_candidate(alpha_id)
        assert (await reserve_submission("primary", alpha_id))["allowed"] is True
        await finalize_submission_attempt("primary", alpha_id, {"ok": True, "final_status": "ACTIVE"})
    await observe_account_status("primary", _current_points(3000))

    factory = db._get_session_factory()
    async with factory() as session:
        attempts = list((await session.execute(
            select(WQSubmissionAttempt).where(WQSubmissionAttempt.alpha_id.in_(["cohort-a", "cohort-b"]))
        )).scalars().all())
    assert {attempt.attributed_points_share for attempt in attempts} == {1000.0}
    assert {attempt.attribution_confidence for attempt in attempts} == {0.5}
    assert all(attempt.settled_at is not None for attempt in attempts)
    for attempt in attempts:
        details = attempt.attribution_details
        assert details["cohort_alpha_ids"] == ["cohort-a", "cohort-b"]
        assert details["delta"] == 2000.0
        assert details["sync_evidence"] == {"active_alpha_gap_known": True, "active_alpha_gap": 0}
        assert details["attribution_rule"] == "equal_share_confidence_weighted"


@pytest.mark.asyncio
async def test_untracked_active_gap_reduces_attribution_confidence(policy_db):
    import quantgpt.db as db

    await observe_account_status("primary", _current_points(1000))
    await _seed_ready_candidate("ambiguous-a")
    assert (await reserve_submission("primary", "ambiguous-a"))["allowed"] is True
    await finalize_submission_attempt("primary", "ambiguous-a", {"ok": True, "final_status": "ACTIVE"})
    await observe_account_status(
        "primary",
        {"points": 1500, "points_status": "CURRENT", "leaderboard": {"active_alpha_gap": 2}},
    )
    factory = db._get_session_factory()
    async with factory() as session:
        attempt = (await session.execute(
            select(WQSubmissionAttempt).where(WQSubmissionAttempt.alpha_id == "ambiguous-a")
        )).scalar_one()
    assert attempt.attribution_confidence == pytest.approx(0.5)
    assert attempt.attribution_details["untracked_active_gap"] == 1


@pytest.mark.asyncio
async def test_points_feedback_gate_exposes_operator_feedback(policy_db, monkeypatch):
    monkeypatch.setenv("WQ_POINTS_FEEDBACK_MIN_CONFIDENCE", "0.5")
    monkeypatch.setenv("WQ_POINTS_FEEDBACK_MIN_SAMPLES", "1")
    await observe_account_status("primary", _current_points(1000))
    await record_research_candidates(
        "primary",
        [{
            "alpha_id": "feedback-a",
            "expression": "rank(ts_mean(field_feedback, 20))",
            "is_metrics": {"sharpe": 1.8, "fitness": 1.3, "returns": 0.09, "turnover": 0.2},
            "research_meta": {"family": "feedback_family", "dataset_id": "feedback_dataset"},
            "validation": {"status": "ready", "robustness_score": 1.0},
        }],
    )
    assert (await reserve_submission("primary", "feedback-a"))["allowed"] is True
    await finalize_submission_attempt("primary", "feedback-a", {"ok": True, "final_status": "ACTIVE"})
    await observe_account_status("primary", _current_points(1800))
    memory = await load_research_memory("primary")
    assert memory["family_points_feedback_usable"]["feedback_family"] == 800.0
    assert memory["dataset_points_feedback_usable"]["feedback_dataset"] == 800.0
    assert memory["operator_points_feedback_usable"]
    assert memory["points_feedback_coverage"]["usable_attempts"] == 1
    assert memory["points_feedback_gate"]["high_capacity_models_deferred"] == ["RUDDER", "ARES", "QR_DQN"]


@pytest.mark.asyncio
async def test_zero_delta_current_state_keeps_active_points_pending(policy_db):
    await observe_account_status("primary", _current_points(1000))
    await _seed_ready_candidate("historical-zero-delta")
    assert (await reserve_submission("primary", "historical-zero-delta"))["allowed"] is True
    await finalize_submission_attempt("primary", "historical-zero-delta", {"ok": True, "final_status": "ACTIVE"})
    status = await observe_account_status("primary", _current_points(1000))
    memory = await load_research_memory("primary")
    assert status["pending_score_submissions"] == 1
    assert status["points_score_unchanged_pending"] is True
    assert status["submission_warning"] == "leaderboard_alpha_count_current_but_score_unchanged_pending_points"
    assert memory["points_feedback_coverage"]["settled_attempts"] == 0
    assert memory["points_feedback_coverage"]["usable_attempts"] == 0
    assert memory["family_points_feedback_usable"] == {}
    assert memory["dataset_points_feedback_usable"] == {}
    assert memory["operator_points_feedback_usable"] == {}


@pytest.mark.asyncio
async def test_current_without_active_gap_never_settles_points(policy_db):
    await observe_account_status("primary", _current_points(1000))
    await _seed_ready_candidate("sync-unknown-a")
    assert (await reserve_submission("primary", "sync-unknown-a"))["allowed"] is True
    await finalize_submission_attempt("primary", "sync-unknown-a", {"ok": True, "final_status": "ACTIVE"})

    status = await observe_account_status(
        "primary",
        {"points": 1500, "points_status": "CURRENT", "leaderboard": {}},
    )

    assert status["last_points_status"] == "SYNC_UNKNOWN"
    assert status["points_sync_unknown"] is True
    assert status["pending_score_submissions"] == 1
    assert status["last_settled_delta"] is None


@pytest.mark.asyncio
async def test_untracked_alpha_cannot_consume_submission_slot(policy_db):
    decision = await reserve_submission("primary", "not-in-candidate-queue")

    assert decision["allowed"] is False
    assert decision["reason"] == "candidate_not_tracked"
    status = await get_submission_policy_status("primary")
    assert status["used_submission_slots"] == 0


@pytest.mark.asyncio
async def test_high_local_correlation_is_blocked_before_submission(policy_db):
    now = datetime.now(timezone.utc).isoformat()
    await _seed_ready_candidate(
        "corr-risk",
        local_correlation_proxy={
            "status": "available",
            "max_correlation": 0.82,
            "matching_alpha_id": "active-peer",
            "sample_length": 120,
            "calculated_at": now,
        },
    )

    status = await get_submission_policy_status("primary")
    assert status["candidate_queue_count"] == 0
    decision = await reserve_submission("primary", "corr-risk")
    assert decision["allowed"] is False
    assert "local_correlation_high" in decision["blockers"]
    assert status["used_submission_slots"] == 0


@pytest.mark.asyncio
async def test_weak_overfitting_evidence_is_blocked_before_submission(policy_db):
    await _seed_ready_candidate(
        "overfit-risk",
        validation={
            "status": "ready",
            "robustness_score": 1.0,
            "overfitting_evidence": {"status": "available", "score": 0.2, "sample_count": 252},
        },
    )

    status = await get_submission_policy_status("primary")
    assert status["candidate_queue_count"] == 0
    decision = await reserve_submission("primary", "overfit-risk")
    assert decision["allowed"] is False
    assert "overfitting_evidence_weak" in decision["blockers"]
    assert status["used_submission_slots"] == 0


@pytest.mark.asyncio
async def test_stale_reservation_keeps_slot_until_platform_recovery(policy_db, monkeypatch):
    import quantgpt.db as db

    monkeypatch.setenv("WQ_SUBMISSION_RESERVATION_TTL_SECONDS", "120")
    await _seed_ready_candidate("stale-reservation")
    assert (await reserve_submission("primary", "stale-reservation"))["allowed"] is True

    factory = db._get_session_factory()
    async with factory() as session:
        attempt = (await session.execute(
            select(WQSubmissionAttempt).where(WQSubmissionAttempt.alpha_id == "stale-reservation")
        )).scalar_one()
        attempt.created_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        await session.commit()

    status = await get_submission_policy_status("primary")
    assert status["used_submission_slots"] == 1
    assert "stale-reservation" in status["expired_reservation_ids"]
    assert status["reservation_ttl_seconds"] == 120

    retry = await reserve_submission("primary", "stale-reservation")
    assert retry["allowed"] is False
    assert retry["reason"] == "submission_reconciliation_required"
    assert retry["recovery_alpha_ids"] == ["stale-reservation"]


@pytest.mark.asyncio
async def test_expired_reservation_can_be_revalidated_after_platform_confirms_unsubmitted(policy_db, monkeypatch):
    import quantgpt.db as db

    monkeypatch.setenv("WQ_SUBMISSION_RESERVATION_TTL_SECONDS", "120")
    await _seed_ready_candidate("recovered-unsubmitted")
    assert (await reserve_submission("primary", "recovered-unsubmitted"))["allowed"] is True

    factory = db._get_session_factory()
    async with factory() as session:
        attempt = (await session.execute(
            select(WQSubmissionAttempt).where(WQSubmissionAttempt.alpha_id == "recovered-unsubmitted")
        )).scalar_one()
        attempt.created_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        await session.commit()
    await get_submission_policy_status("primary")

    platform = {
        "recovered-unsubmitted": {
            "ok": True,
            "status": "UNSUBMITTED",
            "sc_result": "PASS",
            "sc_value": 0.3,
            "sharpe": 1.6,
            "fitness": 1.2,
            "returns": 0.08,
            "turnover": 0.2,
        }
    }
    assert await reconcile_submission_reservations("primary", platform) == 1
    assert await reconcile_candidate_platform_statuses("primary", platform) == 1

    factory = db._get_session_factory()
    async with factory() as session:
        attempt = (await session.execute(
            select(WQSubmissionAttempt).where(WQSubmissionAttempt.alpha_id == "recovered-unsubmitted")
        )).scalar_one()
        assert attempt.status == "UNSUBMITTED_CONFIRMED"

    decision = await reserve_submission("primary", "recovered-unsubmitted")
    assert decision["allowed"] is True


@pytest.mark.asyncio
async def test_unknown_submit_outcome_is_fail_closed_until_platform_reconciliation(policy_db):
    import quantgpt.db as db

    await _seed_ready_candidate("unknown-submit")
    await _seed_ready_candidate("next-alpha")
    assert (await reserve_submission("primary", "unknown-submit"))["allowed"] is True

    await finalize_submission_attempt(
        "primary",
        "unknown-submit",
        {
            "ok": False,
            "platform_status": "UNKNOWN",
            "submission_uncertain": True,
            "detail": "POST timed out after request body was sent",
        },
    )

    factory = db._get_session_factory()
    async with factory() as session:
        attempt = (await session.execute(
            select(WQSubmissionAttempt).where(WQSubmissionAttempt.alpha_id == "unknown-submit")
        )).scalar_one()
        assert attempt.status == "SUBMIT_UNKNOWN"
        assert attempt.score_state == "PENDING"

    status = await get_submission_policy_status("primary")
    assert status["used_submission_slots"] == 1
    assert status["pending_score_submissions"] == 1
    assert status["submission_reconciliation_required"] is True
    assert status["submission_reconciliation_alpha_ids"] == ["unknown-submit"]

    blocked = await reserve_submission("primary", "next-alpha")
    assert blocked["allowed"] is False
    assert blocked["reason"] == "submission_reconciliation_required"
    assert blocked["recovery_alpha_ids"] == ["unknown-submit"]

    platform = {
        "unknown-submit": {
            "ok": True,
            "status": "UNSUBMITTED",
            "sc_result": "PASS",
            "sc_value": 0.2,
            "sharpe": 1.6,
            "fitness": 1.2,
            "returns": 0.08,
            "turnover": 0.2,
        }
    }
    assert await reconcile_submission_reservations("primary", platform) == 1
    assert await reconcile_candidate_platform_statuses("primary", platform) == 1

    status = await get_submission_policy_status("primary")
    assert status["used_submission_slots"] == 0
    assert (await reserve_submission("primary", "next-alpha"))["allowed"] is True


@pytest.mark.asyncio
async def test_active_submission_is_monotonic_against_later_uncertain_or_unsubmitted_observations(policy_db):
    import quantgpt.db as db

    await _seed_ready_candidate("monotonic-active")
    assert (await reserve_submission("primary", "monotonic-active"))["allowed"] is True
    await finalize_submission_attempt(
        "primary",
        "monotonic-active",
        {"ok": True, "final_status": "ACTIVE", "detail": "platform confirmed ACTIVE"},
    )

    await finalize_submission_attempt(
        "primary",
        "monotonic-active",
        {"ok": False, "final_status": "ERROR", "detail": "temporary platform read failure"},
    )
    await finalize_submission_attempt(
        "primary",
        "monotonic-active",
        {"ok": False, "final_status": "UNSUBMITTED", "detail": "stale platform observation"},
    )

    factory = db._get_session_factory()
    async with factory() as session:
        attempt = (await session.execute(
            select(WQSubmissionAttempt).where(WQSubmissionAttempt.alpha_id == "monotonic-active")
        )).scalar_one()
        assert attempt.status == "ACTIVE"
        assert attempt.score_state == "PENDING"
        assert attempt.detail == "platform confirmed ACTIVE"

    status = await get_submission_policy_status("primary")
    assert status["used_submission_slots"] == 1
    assert status["pending_score_submissions"] == 1
