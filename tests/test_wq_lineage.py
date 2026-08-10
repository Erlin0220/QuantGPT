"""Restart-safe WQ research lineage and conservative metadata recovery."""

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quantgpt.models import Base, WQResearchCandidate, WQResearchTrial
from quantgpt.wq_lineage import (
    build_field_metadata_registry,
    extract_expression_metadata,
    lineage_id_for,
    recover_research_metadata,
)
from quantgpt.wq_research_memory import (
    load_research_memory,
    reconcile_research_metadata,
    record_research_trials,
)
from quantgpt.wq_submission_policy import record_research_candidates


@pytest_asyncio.fixture
async def lineage_db(monkeypatch):
    import quantgpt.db as db

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(db, "_engine", engine)
    monkeypatch.setattr(db, "_session_factory", factory)
    yield factory
    await engine.dispose()


def test_lineage_is_format_stable_but_parameter_sensitive():
    settings = {"region": "USA", "universe": "TOP3000", "delay": 1}
    first = lineage_id_for(" rank( ts_mean(close, 20) ) ", settings=settings, metadata={})
    equivalent = lineage_id_for("rank(ts_mean(close,20))", settings=settings, metadata={})
    changed_window = lineage_id_for("rank(ts_mean(close,60))", settings=settings, metadata={})

    assert first == equivalent
    assert first != changed_window
    assert first.startswith("wql_")


def test_expression_metadata_extracts_operator_pattern_and_fields():
    metadata = extract_expression_metadata("rank(ts_delta(close, 5) / ts_mean(volume, 20))")

    assert metadata["operators"] == ["rank", "ts_delta", "ts_mean"]
    assert metadata["operator_pattern"] == "rank>ts_delta>ts_mean"
    assert metadata["data_fields"] == ["close", "volume"]
    assert "#" in metadata["structure_signature"]


def test_metadata_recovery_never_invents_dataset():
    recovered = recover_research_metadata("rank(ts_mean(volume, 20))", {})

    assert recovered["family"] == "price_volume"
    assert recovered["data_fields"] == ["volume"]
    assert recovered["dataset_id"] is None
    assert recovered["provenance_state"] == "partial"
    assert recovered["provenance_reason"] == "core_or_derived_fields_have_no_truthful_platform_dataset"


def test_local_field_registry_resolves_truthful_dataset_without_fabrication():
    registry = build_field_metadata_registry(
        [
            {"data_fields": ["analyst_eps_revision"], "dataset_id": "analyst42", "dataset_category": "analyst"},
            {"data_fields": ["close"], "dataset_id": None},
        ]
    )
    recovered = recover_research_metadata(
        "rank(ts_mean(analyst_eps_revision, 20))",
        {},
        field_registry=registry,
    )

    assert recovered["dataset_id"] == "analyst42"
    assert recovered["dataset_category"] == "analyst"
    assert recovered["provenance_state"] == "resolved"
    assert recovered["provenance_reason"] == "resolved_from_local_field_registry"


def test_multi_dataset_expression_is_truthfully_partial_not_unresolved():
    registry = build_field_metadata_registry(
        [
            {"data_fields": ["field_a"], "dataset_id": "dataset-a", "dataset_category": "analyst"},
            {"data_fields": ["field_b"], "dataset_id": "dataset-b", "dataset_category": "sentiment"},
        ]
    )
    recovered = recover_research_metadata(
        "rank(ts_corr(field_a, field_b, 20))",
        {},
        field_registry=registry,
    )

    assert recovered["dataset_id"] is None
    assert recovered["dataset_category"] == "multi_dataset"
    assert recovered["provenance_state"] == "partial"
    assert recovered["provenance_reason"] == "multiple_dataset_ids_in_expression"


def test_conflicting_registry_evidence_is_discarded_conservatively():
    registry = build_field_metadata_registry(
        [
            {"data_fields": ["shared_field"], "dataset_id": "dataset-a"},
            {"data_fields": ["shared_field"], "dataset_id": "dataset-b"},
        ]
    )

    assert "shared_field" not in registry


def test_explicit_dataset_and_fields_win_over_derived_evidence():
    recovered = recover_research_metadata(
        "rank(close)",
        {"dataset_id": "analyst42", "data_fields": ["analyst_eps_revision"], "family": "analyst_revision"},
    )

    assert recovered["dataset_id"] == "analyst42"
    assert recovered["data_fields"] == ["analyst_eps_revision"]
    assert recovered["family"] == "analyst_revision"


@pytest.mark.asyncio
async def test_trial_and_candidate_share_lineage_identity(lineage_db):
    settings = {
        "region": "USA",
        "universe": "TOP3000",
        "delay": 1,
        "decay": 5,
        "neutralization": "SUBINDUSTRY",
        "truncation": 0.08,
    }
    candidate = {
        "alpha_id": "lineage-alpha",
        "expression": "rank(ts_mean(volume, 20))",
        "is_metrics": {"sharpe": 1.6, "fitness": 1.2, "returns": 0.08, "turnover": 0.2, "checks": []},
        "research_meta": {
            "hypothesis": "volume persistence",
            "dataset_id": "pv-dataset",
            "generation": 2,
            "mutation_type": "decay_adjustment",
            "mutation_reason": "turnover_high",
            "planner_strategy": "thompson_exploit",
            "allocation_cell": "price_volume|pv-dataset|rank>ts_mean",
            "source_run_id": "run-123",
        },
        "validation": {"status": "ready", "robustness_score": 1.0},
    }
    batch = {"settings": settings, "results": [candidate], "candidates": [{"alpha_id": "lineage-alpha"}]}

    assert await record_research_trials("primary", batch) == 1
    assert await record_research_candidates("primary", [candidate], settings=settings) == 1

    factory = lineage_db
    async with factory() as session:
        trial = (await session.execute(select(WQResearchTrial))).scalar_one()
        queued = (await session.execute(select(WQResearchCandidate))).scalar_one()

    assert trial.lineage_id == queued.lineage_id
    assert trial.dataset_id == queued.dataset_id == "pv-dataset"
    assert trial.data_fields == queued.data_fields == ["volume"]
    assert trial.operator_pattern == queued.operator_pattern == "rank>ts_mean"
    assert trial.mutation_reason == queued.mutation_reason == "turnover_high"
    assert trial.planner_strategy == queued.planner_strategy == "thompson_exploit"
    assert trial.source_run_id == queued.source_run_id == "run-123"


@pytest.mark.asyncio
async def test_parent_child_lineage_is_queryable(lineage_db):
    parent_expression = "rank(ts_mean(close, 20))"
    parent = {
        "alpha_id": "parent-alpha",
        "expression": parent_expression,
        "is_metrics": {"sharpe": 1.3, "fitness": 1.0, "turnover": 0.2, "checks": []},
        "research_meta": {"generation": 0, "hypothesis": "base signal"},
    }
    child = {
        "alpha_id": "child-alpha",
        "expression": "rank(ts_mean(close, 60))",
        "is_metrics": {"sharpe": 1.4, "fitness": 1.1, "turnover": 0.2, "checks": []},
        "research_meta": {
            "parent_expression": parent_expression,
            "generation": 1,
            "mutation_type": "window_change",
            "mutation_reason": "low_sharpe",
        },
    }
    await record_research_trials(
        "primary",
        {"settings": {"delay": 1}, "results": [parent, child], "candidates": []},
    )

    factory = lineage_db
    async with factory() as session:
        rows = list((await session.execute(select(WQResearchTrial))).scalars().all())
    by_alpha = {row.alpha_id: row for row in rows}
    parent_row = by_alpha["parent-alpha"]
    child_row = by_alpha["child-alpha"]

    assert child_row.parent_lineage_id == parent_row.lineage_id
    assert child_row.parent_lineage_id != child_row.lineage_id
    ancestry = await sessionless_find_children(lineage_db, parent_row.lineage_id)
    assert ancestry == [child_row.lineage_id]


async def sessionless_find_children(factory, parent_lineage_id: str) -> list[str]:
    async with factory() as session:
        result = await session.execute(
            select(WQResearchTrial.lineage_id).where(WQResearchTrial.parent_lineage_id == parent_lineage_id)
        )
        return [str(value) for value in result.scalars().all()]


@pytest.mark.asyncio
async def test_conservative_backfill_enriches_old_rows_without_dataset_fabrication(lineage_db):
    factory = lineage_db
    async with factory() as session:
        session.add(
            WQResearchTrial(
                account="primary",
                alpha_id="legacy-trial",
                expression="rank(ts_mean(volume, 20))",
                expression_normalized="rank(ts_mean(volume,20))",
                family="unknown",
                status="rejected",
                data_fields=[],
                dataset_id=None,
            )
        )
        session.add(
            WQResearchCandidate(
                account="primary",
                alpha_id="legacy-candidate",
                expression="rank(ts_mean(close, 20))",
                family=None,
                data_fields=[],
                dataset_id=None,
            )
        )
        await session.commit()

    assert await reconcile_research_metadata("primary") == 2

    async with factory() as session:
        trial = (
            await session.execute(select(WQResearchTrial).where(WQResearchTrial.alpha_id == "legacy-trial"))
        ).scalar_one()
        candidate = (
            await session.execute(select(WQResearchCandidate).where(WQResearchCandidate.alpha_id == "legacy-candidate"))
        ).scalar_one()

    assert trial.family == "price_volume"
    assert trial.data_fields == ["volume"]
    assert trial.dataset_id is None
    assert trial.lineage_id and trial.operator_pattern == "rank>ts_mean"
    assert candidate.family == "momentum_reversal"
    assert candidate.data_fields == ["close"]
    assert candidate.dataset_id is None
    assert candidate.lineage_id


@pytest.mark.asyncio
async def test_reconcile_reclassifies_legacy_multi_dataset_rows_as_partial(lineage_db):
    factory = lineage_db
    async with factory() as session:
        session.add_all(
            [
                WQResearchTrial(
                    account="primary",
                    alpha_id="legacy-multi",
                    expression="rank(ts_corr(field_a, field_b, 20))",
                    expression_normalized="rank(ts_corr(field_a,field_b,20))",
                    family="other",
                    status="rejected",
                    data_fields=["field_a", "field_b"],
                    dataset_id=None,
                    provenance_state="unresolved",
                    provenance_reason="multiple_dataset_ids_in_expression",
                ),
                WQResearchCandidate(
                    account="primary",
                    alpha_id="registry-a",
                    expression="rank(field_a)",
                    family="other",
                    data_fields=["field_a"],
                    dataset_id="dataset-a",
                    dataset_category="analyst",
                    provenance_state="resolved",
                ),
                WQResearchCandidate(
                    account="primary",
                    alpha_id="registry-b",
                    expression="rank(field_b)",
                    family="other",
                    data_fields=["field_b"],
                    dataset_id="dataset-b",
                    dataset_category="sentiment",
                    provenance_state="resolved",
                ),
            ]
        )
        await session.commit()

    assert await reconcile_research_metadata("primary") >= 1

    async with factory() as session:
        row = (
            await session.execute(select(WQResearchTrial).where(WQResearchTrial.alpha_id == "legacy-multi"))
        ).scalar_one()

    assert row.dataset_id is None
    assert row.dataset_category == "multi_dataset"
    assert row.provenance_state == "partial"
    assert row.provenance_reason == "multiple_dataset_ids_in_expression"


@pytest.mark.asyncio
async def test_memory_reports_metadata_completeness(lineage_db):
    await record_research_trials(
        "primary",
        {
            "settings": {"delay": 1},
            "results": [
                {
                    "alpha_id": "complete",
                    "expression": "rank(volume)",
                    "is_metrics": {"sharpe": 0.8, "fitness": 0.7, "turnover": 0.2, "checks": []},
                    "research_meta": {"dataset_id": "pv1"},
                },
                {
                    "alpha_id": "no-dataset",
                    "expression": "rank(close)",
                    "is_metrics": {"sharpe": 0.7, "fitness": 0.6, "turnover": 0.2, "checks": []},
                },
            ],
        },
    )

    memory = await load_research_memory("primary")
    completeness = memory["metadata_completeness"]
    assert completeness["lineage_id"]["rate"] == 1.0
    assert completeness["family"]["rate"] == 1.0
    assert completeness["data_fields"]["rate"] == 1.0
    assert completeness["operator_pattern"]["rate"] == 1.0
    assert completeness["dataset_id"]["present"] == 1
    assert completeness["dataset_id"]["missing"] == 1
    assert memory["provenance"]["resolved"] == 1
    assert memory["provenance"]["partial"] == 1
