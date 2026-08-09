"""Structured failure evidence for autonomous WorldQuant research."""

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quantgpt.models import Base, WQResearchTrial
from quantgpt.wq_failure_taxonomy import classify_research_failure
from quantgpt.wq_research_memory import load_research_memory, record_research_trials
from quantgpt.wq_submission_policy import finalize_submission_attempt, reserve_submission


@pytest_asyncio.fixture
async def research_db(monkeypatch):
    import quantgpt.db as db

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(db, "_engine", engine)
    monkeypatch.setattr(db, "_session_factory", factory)
    yield factory
    await engine.dispose()


def test_compile_failure_keeps_secondary_metric_evidence():
    diagnostic = classify_research_failure(
        {
            "error": "unknown operator: ts_magic",
            "is_metrics": {
                "sharpe": 0.4,
                "fitness": 0.2,
                "turnover": 0.8,
                "checks": [{"name": "LOW_SHARPE", "result": "FAIL"}],
            },
        },
        status="invalid",
    )

    assert diagnostic["failure_stage"] == "compile"
    assert diagnostic["failure_reason"] == "unknown_operator"
    reasons = {(item["stage"], item["reason"]) for item in diagnostic["failure_reasons"]}
    assert ("metrics", "low_sharpe") in reasons
    assert ("metrics", "low_fitness") in reasons
    assert ("metrics", "turnover_high") in reasons
    assert diagnostic["failure_evidence"]["error"] == "unknown operator: ts_magic"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("rank expects 1 args, got 2", "invalid_arity"),
        ("ts_mean: window must be 1..500, got 0", "invalid_parameter"),
        ('Attempted to use unknown variable "bad_field".', "incompatible_field_operator"),
        ("missing closing parenthesis", "syntax"),
    ],
)
def test_compile_reason_variants(error, expected):
    diagnostic = classify_research_failure({"error": error}, status="invalid")
    assert diagnostic["failure_stage"] == "compile"
    assert diagnostic["failure_reason"] == expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("request timed out after 180s", "timeout"),
        ("dataset unavailable for region", "unavailable_data"),
        ("HTTP 503 connection failed", "transport_or_platform_error"),
        ("simulation rejected by platform", "simulation_rejected"),
    ],
)
def test_simulation_failure_variants(error, expected):
    diagnostic = classify_research_failure({"error": error}, status="simulation_failed")
    assert diagnostic["failure_stage"] == "simulation"
    assert diagnostic["failure_reason"] == expected


def test_platform_checks_map_to_robustness_and_diversity():
    diagnostic = classify_research_failure(
        {
            "is_metrics": {
                "sharpe": 1.5,
                "fitness": 1.2,
                "turnover": 0.2,
                "checks": [
                    {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL", "value": 0.9},
                    {"name": "SELF_CORRELATION", "result": "FAIL", "value": 0.72},
                    {"name": "CONCENTRATED_WEIGHT", "result": "FAIL"},
                ],
            }
        },
        status="rejected",
    )

    assert diagnostic["failure_stage"] == "metrics"
    assert diagnostic["failure_reason"] == "weight_concentration"
    reasons = {(item["stage"], item["reason"]) for item in diagnostic["failure_reasons"]}
    assert ("robustness", "sub_universe_instability") in reasons
    assert ("diversity", "official_self_correlation") in reasons
    assert diagnostic["failure_evidence"]["checks"][1]["value"] == 0.72


def test_submission_sc_fail_is_not_confused_with_pending():
    failed = classify_research_failure({"final_status": "SC_FAIL"}, status="rejected")
    pending = classify_research_failure({"final_status": "SC_PENDING"}, status="rejected")

    assert any(item["reason"] == "sc_fail" for item in failed["failure_reasons"])
    assert all(item["reason"] != "sc_fail" for item in pending["failure_reasons"])


@pytest.mark.asyncio
async def test_record_and_memory_expose_structured_failure_counts(research_db):
    result = {
        "settings": {"universe": "TOP3000"},
        "results": [
            {
                "alpha_id": "metric-fail",
                "expression": "rank(volume)",
                "is_metrics": {
                    "sharpe": 0.8,
                    "fitness": 0.7,
                    "turnover": 0.2,
                    "checks": [{"name": "LOW_SHARPE", "result": "FAIL"}],
                },
                "research_meta": {"family": "price_volume", "dataset_id": "pv1"},
            }
        ],
        "invalid": [
            {
                "expression": "ts_magic(close, 5)",
                "error": "unknown operator: ts_magic",
                "research_meta": {"family": "momentum_reversal", "dataset_id": "price1"},
            }
        ],
    }

    assert await record_research_trials("primary", result) == 2
    memory = await load_research_memory("primary")

    assert memory["status_counts"] == {"invalid": 1, "rejected": 1}
    assert memory["failure_stage_counts"] == {"compile": 1, "metrics": 1}
    assert memory["failure_reason_counts"]["unknown_operator"] == 1
    assert memory["failure_reason_counts"]["low_fitness"] == 1
    assert memory["failure_reason_family_counts"]["unknown_operator"] == {"momentum_reversal": 1}
    assert memory["failure_reason_dataset_counts"]["low_fitness"] == {"pv1": 1}
    recent = {item["expression"]: item for item in memory["recent_trials"]}
    assert recent["ts_magic(close, 5)"]["failure_evidence"]["error"] == "unknown operator: ts_magic"
    assert recent["rank(volume)"]["failure_reasons"]


@pytest.mark.asyncio
async def test_candidate_has_no_failure_taxonomy(research_db):
    result = {
        "results": [
            {
                "alpha_id": "candidate-1",
                "expression": "rank(close)",
                "is_metrics": {"sharpe": 1.5, "fitness": 1.2, "turnover": 0.2, "checks": []},
            }
        ],
        "candidates": [{"alpha_id": "candidate-1"}],
    }
    await record_research_trials("primary", result)
    factory = research_db
    async with factory() as session:
        row = (await session.execute(sa.select(WQResearchTrial))).scalar_one()
        assert row.status == "candidate"
        assert row.failure_stage is None
        assert row.failure_reason is None
        assert row.failure_reasons == []


@pytest.mark.asyncio
async def test_terminal_submission_failure_updates_latest_research_trial(research_db):
    await record_research_trials(
        "primary",
        {
            "results": [
                {
                    "alpha_id": "sc-fail-alpha",
                    "expression": "rank(close)",
                    "is_metrics": {"sharpe": 1.6, "fitness": 1.3, "turnover": 0.2, "checks": []},
                }
            ],
            "candidates": [{"alpha_id": "sc-fail-alpha"}],
        },
    )
    reservation = await reserve_submission("primary", "sc-fail-alpha")
    assert reservation["allowed"] is True
    await finalize_submission_attempt(
        "primary",
        "sc-fail-alpha",
        {"ok": False, "final_status": "SC_FAIL", "detail": "SC FAIL: value=0.72 > limit=0.7"},
    )

    factory = research_db
    async with factory() as session:
        row = (await session.execute(sa.select(WQResearchTrial))).scalar_one()
        assert row.status == "candidate"
        assert row.failure_stage == "submission"
        assert row.failure_reason == "sc_fail"
        assert row.failure_evidence["final_status"] == "SC_FAIL"


def test_dev_migration_adds_failure_columns_to_existing_table():
    from quantgpt.db import _migrate_add_columns

    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE wq_research_trials ("
                "id VARCHAR PRIMARY KEY, account VARCHAR(20), expression TEXT, expression_normalized TEXT"
                ")"
            )
        )
        _migrate_add_columns(connection)
        columns = {column["name"] for column in sa.inspect(connection).get_columns("wq_research_trials")}

    assert {"failure_stage", "failure_reason", "failure_reasons", "failure_evidence"} <= columns
