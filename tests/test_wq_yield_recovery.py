from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quantgpt import wq_autonomous_research as autonomous
from quantgpt.models import Base
from quantgpt.wq_research_memory import load_research_memory, record_research_trials
from quantgpt.wq_submission_policy import (
    finalize_submission_attempt,
    get_submission_policy_status,
    record_research_candidates,
    reserve_submission,
)


@pytest_asyncio.fixture
async def recovery_db(monkeypatch):
    import quantgpt.db as db

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(db, "_engine", engine)
    monkeypatch.setattr(db, "_session_factory", factory)
    monkeypatch.setenv("WQ_DAILY_SUBMISSION_BUDGET", "2")
    monkeypatch.setenv("WQ_ACTIVE_DECISION_MIN_SAMPLES", "20")
    yield
    await engine.dispose()


@pytest.mark.asyncio
async def test_full_yield_recovery_grows_inventory_with_submission_slots_exhausted(recovery_db, monkeypatch):
    for alpha_id in ("already-live-1", "already-live-2"):
        assert (await reserve_submission("primary", alpha_id))["allowed"] is True
        await finalize_submission_attempt("primary", alpha_id, {"ok": True, "final_status": "ACTIVE"})
    before = await get_submission_policy_status("primary")
    assert before["remaining_submission_slots"] == 0
    assert before["inventory"]["deficit"] == 30

    monkeypatch.setattr(autonomous, "run_list_alphas", lambda *_args, **_kwargs: {"ok": True, "alphas": []})
    monkeypatch.setattr(
        autonomous,
        "build_live_field_plan",
        lambda *_args, **_kwargs: ([], {"available": False, "count": 0}, []),
    )
    monkeypatch.setattr(autonomous, "build_llm_live_plan", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        autonomous,
        "validate_candidate_robustness",
        lambda *_args, **_kwargs: {"status": "ready", "robustness_score": 0.8, "validation_simulations": 2},
    )

    calls = []

    def fake_research_batch(_client, expressions, **kwargs):
        generation = 2 if kwargs["tag"].endswith("g2") else 1
        calls.append((generation, list(expressions)))
        results = []
        candidates = []
        for index, expression in enumerate(expressions):
            good = generation == 2 and index == 0
            item = {
                "ok": True,
                "alpha_id": f"yield-g{generation}-{index}",
                "expression": expression,
                "is_metrics": {
                    "sharpe": 1.5 if good else 1.3,
                    "fitness": 1.2 if good else 0.8,
                    "returns": 0.08 if good else 0.02,
                    "turnover": 0.2 if good else 0.55,
                    "checks": [],
                },
                "passes_primary_thresholds": good,
                "mutation_targets": ["candidate_passes_primary_thresholds"] if good else ["improve_fitness"],
            }
            results.append(item)
            if good:
                candidates.append(item)
        return {
            "ok": True,
            "tag": kwargs["tag"],
            "settings": {
                "region": kwargs["region"],
                "universe": kwargs["universe"],
                "delay": kwargs["delay"],
                "decay": kwargs["decay"],
                "neutralization": kwargs["neutralization"],
                "truncation": kwargs["truncation"],
            },
            "summary": {"simulated": len(expressions)},
            "results": results,
            "candidates": candidates,
            "failed": [],
            "invalid": [],
        }

    monkeypatch.setattr(autonomous, "run_research_batch", fake_research_batch)
    memory = await load_research_memory("primary")
    memory["inventory"] = before["inventory"]
    memory["submission"] = before

    result = autonomous.run_autonomous_research(
        object(),
        memory=memory,
        max_simulations=8,
        generations=2,
        family_count=2,
        tag="yield-recovery",
    )

    assert [generation for generation, _ in calls] == [1, 2]
    assert result["summary"]["directed_mutation_routes"] > 0
    assert result["inventory_mode"] == "REPLENISHMENT"
    assert result["summary"]["formally_submitted"] == 0
    assert result["ready_candidates"]

    assert await record_research_trials("primary", result, hypothesis="yield recovery") > 0
    assert await record_research_candidates(
        "primary",
        result["candidates"],
        settings=result["settings"],
        tag=result["tag"],
    ) > 0

    after = await get_submission_policy_status("primary")
    assert after["remaining_submission_slots"] == 0
    assert after["used_submission_slots"] == 2
    assert after["candidate_queue_count"] >= 1
    assert after["research_readiness_eligible_count"] >= 1
    assert after["inventory"]["deficit"] < before["inventory"]["deficit"]

    learned = await load_research_memory("primary")
    diagnosis = learned["candidate_funnel"]["all_time"]["stages"]["failure_diagnosis"]
    mutation = learned["candidate_funnel"]["all_time"]["stages"]["directed_mutation"]
    assert diagnosis["routed"] > 0
    assert mutation["entered"] > 0
