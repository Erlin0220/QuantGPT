from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quantgpt.models import Base, WQResearchCandidate, WQResearchTrial, WQSubmissionAttempt
from quantgpt.wq_candidate_calibration import (
    calibrate_active_probability,
    calibration_report,
    confidence_tier,
)
from quantgpt.wq_submission_policy import (
    _load_conversion_feedback,
    get_submission_policy_status,
    record_research_candidates,
)


def _candidate(**overrides):
    candidate = {
        "alpha_id": "alpha-1",
        "expression": "rank(ts_mean(returns, 20))",
        "family": "momentum_reversal",
        "dataset_id": "pv1",
        "operator_pattern": "rank(ts_mean(*,#))",
        "research_meta": {
            "family": "momentum_reversal",
            "dataset_id": "pv1",
            "operator_pattern": "rank(ts_mean(*,#))",
        },
        "is_metrics": {
            "sharpe": 1.6,
            "fitness": 1.3,
            "returns": 0.12,
            "turnover": 0.2,
            "checks": [{"name": "SELF_CORRELATION", "result": "PENDING"}],
        },
        "validation": {"status": "ready", "robustness_score": 0.8},
        "novelty_score": 0.8,
    }
    candidate.update(overrides)
    return candidate


def test_sparse_evidence_keeps_probability_unavailable_and_is_deterministic():
    feedback = {"global": {"rate": 0.5, "samples": 0}, "family": {}, "dataset": {}, "cell": {}}
    first = calibrate_active_probability(_candidate(), feedback)
    second = calibrate_active_probability(_candidate(), feedback)
    assert first == second
    assert first["probability"] is None
    assert first["tier"] is None
    assert first["support"] == 0
    assert first["provenance"] == "cold_start_unavailable"
    assert first["outcome_weight"] == 0.0


def test_cell_feedback_wins_when_specific_support_is_sufficient():
    feedback = {
        "global": {"rate": 0.4, "samples": 20},
        "family": {"momentum_reversal": {"rate": 0.6, "samples": 8}},
        "dataset": {"pv1": {"rate": 0.55, "samples": 8}},
        "cell": {"momentum_reversal|pv1|rank(ts_mean(*,#))": {"rate": 0.8, "samples": 5}},
    }
    result = calibrate_active_probability(_candidate(), feedback)
    assert result["provenance"].startswith("cell:")
    assert result["support"] == 5
    assert result["empirical_rate"] == 0.8


def test_active_reference_does_not_create_probability_during_cold_start():
    feedback = {"global": {"rate": 0.5, "samples": 0}, "family": {}, "dataset": {}, "cell": {}}
    low = calibrate_active_probability(_candidate(active_prior_score=0.1), feedback)
    high = calibrate_active_probability(_candidate(active_prior_score=0.9), feedback)

    assert low["probability"] is None
    assert high["probability"] is None
    assert "active_reference" not in low["baseline"]["components"]
    assert "active_reference" not in high["baseline"]["components"]


def test_raw_candidate_evidence_is_preserved_without_magic_score():
    result = calibrate_active_probability(_candidate(), {"global": {"rate": 0.5, "samples": 0}})
    assert result["baseline"]["score"] is None
    assert result["baseline"]["components"]["fitness"] == 1.3
    assert result["baseline"]["method"] == "evidence_hierarchy_no_probability"


def test_sparse_cell_falls_back_to_family_dataset_groups():
    feedback = {
        "global": {"rate": 0.4, "samples": 20},
        "family": {"momentum_reversal": {"rate": 0.6, "samples": 3}},
        "dataset": {"pv1": {"rate": 0.5, "samples": 2}},
        "cell": {"momentum_reversal|pv1|rank(ts_mean(*,#))": {"rate": 0.9, "samples": 1}},
    }
    result = calibrate_active_probability(_candidate(), feedback)
    assert result["provenance"] == "family_dataset"
    assert result["support"] == 5
    assert result["empirical_rate"] == pytest.approx(0.55)


def test_official_sc_fail_forces_zero_without_training_on_pending():
    failed = _candidate()
    failed["is_metrics"]["checks"] = [{"name": "SELF_CORRELATION", "result": "FAIL", "value": 0.8}]
    result = calibrate_active_probability(failed, {"global": {"rate": 0.9, "samples": 50}})
    assert result["probability"] == 0.0
    assert result["tier"] == "B"


def test_confidence_thresholds_are_configurable_independently(monkeypatch):
    monkeypatch.setenv("WQ_CONFIDENCE_TIER_S", "0.8")
    monkeypatch.setenv("WQ_CONFIDENCE_TIER_A", "0.6")
    assert confidence_tier(0.81) == "S"
    assert confidence_tier(0.65) == "A"
    assert confidence_tier(0.59) == "B"


def test_calibration_report_waits_for_minimum_samples_then_reports_brier_and_buckets():
    assert calibration_report([(0.8, True)] * 4, minimum_samples=5)["status"] == "insufficient_samples"
    report = calibration_report([(0.8, True), (0.7, True), (0.2, False), (0.3, False), (0.6, True)], minimum_samples=5)
    assert report["status"] == "ready"
    assert report["brier_score"] is not None
    assert report["reliability_buckets"]


@pytest_asyncio.fixture
async def calibration_db(monkeypatch):
    import quantgpt.db as db
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(db, "_session_factory", factory)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_only_terminal_outcomes_train_feedback(calibration_db):
    factory = calibration_db
    async with factory() as session:
        for index, status in enumerate(("ACTIVE", "SC_FAIL", "SC_PENDING", "RESERVED"), start=1):
            alpha_id = f"a{index}"
            session.add(WQResearchCandidate(
                account="primary", alpha_id=alpha_id, expression=f"rank(close)+{index}",
                family="price_volume", dataset_id="pv", operator_pattern="rank(*)",
                validation_status="ready", robustness_score=0.8, sharpe=1.5, fitness=1.2,
                priority_score=1.0, active_probability=0.6, confidence_tier="A",
            ))
            session.add(WQSubmissionAttempt(account="primary", alpha_id=alpha_id, submission_day="2026-08-10", status=status))
        await session.commit()
        feedback = await _load_conversion_feedback(session, "primary")
    assert feedback["global"]["samples"] == 2
    assert feedback["global"]["active"] == 1
    assert feedback["global"]["failed"] == 1


@pytest.mark.asyncio
async def test_group_active_feedback_credits_originating_trial_not_current_candidate_metadata(calibration_db):
    factory = calibration_db
    async with factory() as session:
        session.add(WQResearchTrial(
            account="primary",
            alpha_id="origin-a",
            expression="rank(close)",
            expression_normalized="rank(close)",
            family="price_volume",
            status="candidate",
            dataset_id="pv1",
            provenance_state="resolved",
            operator_pattern="rank(*)",
        ))
        session.add(WQResearchCandidate(
            account="primary",
            alpha_id="origin-a",
            expression="rank(close)",
            family="other",
            dataset_id="analyst4",
            provenance_state="resolved",
            operator_pattern="group_rank(rank(*))",
            validation_status="ready",
            sharpe=1.5,
            fitness=1.2,
            priority_score=1.0,
        ))
        session.add(WQSubmissionAttempt(
            account="primary",
            alpha_id="origin-a",
            submission_day="2026-08-11",
            status="ACTIVE",
        ))
        await session.commit()
        feedback = await _load_conversion_feedback(session, "primary")

    assert "price_volume" in feedback["family"]
    assert "other" not in feedback["family"]
    assert "pv1" in feedback["dataset"]
    assert "analyst4" not in feedback["dataset"]
    assert feedback["credit_assignment"]["coverage"] == 1.0


@pytest.mark.asyncio
async def test_group_active_feedback_prefers_candidate_lineage_when_alpha_id_changed(calibration_db):
    factory = calibration_db
    async with factory() as session:
        session.add(WQResearchTrial(
            account="primary",
            alpha_id="origin-alpha",
            expression="group_rank(ts_zscore(cashflow_op/cap, 60), subindustry)",
            expression_normalized="group_rank(ts_zscore(cashflow_op/cap,60),subindustry)",
            family="fundamental_quality",
            status="candidate",
            dataset_id="fundamental6",
            provenance_state="resolved",
            operator_pattern="group_rank>ts_zscore>divide",
            lineage_id="lineage-fundamental-1",
        ))
        session.add(WQResearchCandidate(
            account="primary",
            alpha_id="platform-alpha",
            expression="group_rank(ts_zscore(cashflow_op/cap, 60), subindustry)",
            family="other",
            dataset_id="analyst4",
            provenance_state="resolved",
            operator_pattern="rank(*)",
            lineage_id="lineage-fundamental-1",
            validation_status="ready",
            sharpe=1.5,
            fitness=1.2,
            priority_score=1.0,
        ))
        session.add(WQSubmissionAttempt(
            account="primary",
            alpha_id="platform-alpha",
            submission_day="2026-08-11",
            status="ACTIVE",
        ))
        await session.commit()
        feedback = await _load_conversion_feedback(session, "primary")

    assert "fundamental_quality" in feedback["family"]
    assert "other" not in feedback["family"]
    assert "fundamental6" in feedback["dataset"]
    assert "analyst4" not in feedback["dataset"]
    assert feedback["credit_assignment"]["coverage"] == 1.0


@pytest.mark.asyncio
async def test_cold_start_probability_is_null_and_status_exposes_support(calibration_db):
    await record_research_candidates("primary", [_candidate()], settings={"region": "USA", "universe": "TOP3000"})
    status = await get_submission_policy_status("primary")
    top = status["candidate_queue_top"][0]
    assert top["active_probability"] is None
    assert top["confidence_tier"] is None
    assert top["probability_support"] == 0
    assert top["probability_provenance"] == "cold_start_unavailable"
    assert top["calibration_details"]["baseline"]["score"] is None
