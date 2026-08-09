"""Observable stage funnel for autonomous WorldQuant research."""

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quantgpt.models import Base, WQResearchStageEvent
from quantgpt.wq_candidate_funnel import (
    STAGE_CANDIDATE,
    STAGE_COMPILE,
    STAGE_DIAGNOSIS,
    STAGE_IDEA,
    STAGE_MUTATION,
    STAGE_ROBUSTNESS,
    STAGE_SIMULATION,
)
from quantgpt.wq_research_memory import load_research_memory, record_research_trials


@pytest_asyncio.fixture
async def funnel_db(monkeypatch):
    import quantgpt.db as db

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(db, "_engine", engine)
    monkeypatch.setattr(db, "_session_factory", factory)
    yield factory
    await engine.dispose()


async def _events(factory, alpha_lineage: str | None = None):
    async with factory() as session:
        query = select(WQResearchStageEvent).order_by(WQResearchStageEvent.created_at.asc())
        if alpha_lineage:
            query = query.where(WQResearchStageEvent.lineage_id == alpha_lineage)
        return list((await session.execute(query)).scalars().all())


@pytest.mark.asyncio
async def test_success_path_reaches_candidate(funnel_db):
    result = {
        "settings": {"delay": 1},
        "results": [
            {
                "alpha_id": "success-alpha",
                "expression": "rank(ts_mean(volume, 20))",
                "is_metrics": {"sharpe": 1.6, "fitness": 1.3, "turnover": 0.2, "checks": []},
                "validation": {"status": "ready", "robustness_score": 1.0},
                "novelty_score": 0.8,
            }
        ],
        "candidates": [{"alpha_id": "success-alpha"}],
    }
    await record_research_trials("primary", result)
    events = await _events(funnel_db)

    stages = [(event.stage, event.outcome) for event in events]
    assert stages[0:4] == [
        (STAGE_IDEA, "passed"),
        (STAGE_COMPILE, "passed"),
        (STAGE_SIMULATION, "passed"),
        (STAGE_DIAGNOSIS, "passed"),
    ]
    assert (STAGE_ROBUSTNESS, "passed") in stages
    assert stages[-1] == (STAGE_CANDIDATE, "passed")


@pytest.mark.asyncio
async def test_compile_failure_terminates_at_compile(funnel_db):
    await record_research_trials(
        "primary",
        {"invalid": [{"expression": "ts_magic(close, 5)", "error": "unknown operator: ts_magic"}]},
    )
    events = await _events(funnel_db)

    assert [(event.stage, event.outcome) for event in events] == [
        (STAGE_IDEA, "passed"),
        (STAGE_COMPILE, "failed"),
    ]
    assert events[-1].failure_reason == "unknown_operator"


@pytest.mark.asyncio
async def test_metric_rejection_terminates_at_failure_diagnosis(funnel_db):
    await record_research_trials(
        "primary",
        {
            "results": [
                {
                    "alpha_id": "weak",
                    "expression": "rank(close)",
                    "is_metrics": {"sharpe": 0.5, "fitness": 0.4, "turnover": 0.2, "checks": []},
                }
            ]
        },
    )
    events = await _events(funnel_db)
    stages = [(event.stage, event.outcome) for event in events]

    assert stages[-1] == (STAGE_DIAGNOSIS, "failed")
    assert events[-1].failure_reason in {"low_fitness", "low_sharpe"}
    assert STAGE_CANDIDATE not in [event.stage for event in events]


@pytest.mark.asyncio
async def test_directed_mutation_preserves_parent_context(funnel_db):
    await record_research_trials(
        "primary",
        {
            "settings": {"delay": 1},
            "results": [
                {
                    "alpha_id": "mutant",
                    "expression": "rank(ts_mean(close, 60))",
                    "is_metrics": {"sharpe": 1.4, "fitness": 1.1, "turnover": 0.2, "checks": []},
                    "research_meta": {
                        "parent_expression": "rank(ts_mean(close, 20))",
                        "generation": 1,
                        "mutation_type": "window_change",
                        "mutation_reason": "low_sharpe",
                    },
                    "validation": {"status": "ready"},
                }
            ],
            "candidates": [{"alpha_id": "mutant"}],
        },
    )
    events = await _events(funnel_db)
    mutation = next(event for event in events if event.stage == STAGE_MUTATION)

    assert mutation.outcome == "passed"
    assert mutation.parent_lineage_id
    assert mutation.details["parent_lineage_id"] == mutation.parent_lineage_id
    assert mutation.details["mutation_reason"] == "low_sharpe"


@pytest.mark.asyncio
async def test_memory_exposes_funnel_and_bottleneck(funnel_db):
    await record_research_trials(
        "primary",
        {
            "invalid": [
                {"expression": "bad1(close)", "error": "unknown operator: bad1"},
                {"expression": "bad2(close)", "error": "unknown operator: bad2"},
            ],
            "results": [
                {
                    "alpha_id": "ok",
                    "expression": "rank(volume)",
                    "is_metrics": {"sharpe": 1.5, "fitness": 1.2, "turnover": 0.2, "checks": []},
                    "validation": {"status": "ready"},
                }
            ],
            "candidates": [{"alpha_id": "ok"}],
        },
    )

    memory = await load_research_memory("primary")
    funnel = memory["candidate_funnel"]
    recent = funnel["recent_500_events"]

    assert recent["stages"][STAGE_IDEA]["entered"] == 3
    assert recent["stages"][STAGE_COMPILE]["failed"] == 2
    assert recent["stages"][STAGE_CANDIDATE]["passed"] == 1
    assert funnel["dominant_bottleneck_stage"] == STAGE_COMPILE
