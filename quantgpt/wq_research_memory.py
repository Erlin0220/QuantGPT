"""Persistent research memory for WorldQuant autonomous Alpha exploration."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from sqlalchemy import select

from .db import _get_session_factory
from .models import WQResearchCandidate, WQResearchStageEvent, WQResearchTrial, WQSubmissionAttempt
from .wq_candidate_funnel import funnel_events_for_trial, summarize_funnel
from .wq_failure_taxonomy import classify_research_failure
from .wq_lineage import (
    classify_family_from_metadata,
    lineage_id_for,
    parent_lineage_id_for,
    recover_research_metadata,
)
from .wq_mutation_policy import compile_prevention_rules, summarize_mutation_outcomes
from .wq_operator_registry import canonicalize_wq_expression
from .wq_research_scheduler import summarize_research_cells


def normalize_wq_expression(expression: str) -> str:
    """Canonical normalized expression used for local research-memory dedupe."""
    value = str(expression or "").strip()
    if not value:
        return ""
    try:
        value = canonicalize_wq_expression(value)
    except Exception:
        pass
    return re.sub(r"\s+", "", value).lower()


def classify_wq_family(expression: str) -> str:
    """Classify an expression using deterministic recovered field/operator evidence."""
    return classify_family_from_metadata(expression)


def _self_correlation_failed(item: dict[str, Any]) -> bool:
    metrics = item.get("is_metrics") or {}
    for check in metrics.get("checks") or []:
        if str(check.get("name") or "").upper() != "SELF_CORRELATION":
            continue
        if str(check.get("result") or "").upper() == "FAIL":
            return True
    return False


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trial_from_item(
    account: str,
    item: dict[str, Any],
    *,
    status: str,
    settings: dict[str, Any],
    tag: str | None,
    default_family: str,
    default_hypothesis: str,
) -> WQResearchTrial | None:
    expression = str(item.get("expression") or "").strip()
    normalized = normalize_wq_expression(expression)
    if not expression or not normalized:
        return None

    metrics = item.get("is_metrics") or {}
    raw_meta = item.get("research_meta") or {}
    meta = recover_research_metadata(expression, raw_meta, platform_meta=item)
    mutation_targets = item.get("mutation_targets")
    if mutation_targets is None and status in {"invalid", "simulation_failed"}:
        error = str(item.get("error") or "").strip()
        mutation_targets = [f"{status}:{error}"] if error else [status]
    diagnostics = classify_research_failure(item, status=status)
    lineage_id = str(meta.get("lineage_id") or "") or lineage_id_for(expression, settings=settings, metadata=meta)
    parent_lineage_id = str(meta.get("parent_lineage_id") or "") or parent_lineage_id_for(
        meta.get("parent_expression"), settings=settings
    )

    return WQResearchTrial(
        account=account,
        alpha_id=str(item.get("alpha_id") or "").strip() or None,
        expression=expression,
        expression_normalized=normalized,
        family=str(meta.get("family") or default_family or classify_wq_family(expression)),
        hypothesis=str(meta.get("hypothesis") or default_hypothesis or "") or None,
        parent_expression=str(meta.get("parent_expression") or "") or None,
        generation=int(meta.get("generation") or 0),
        mutation_type=str(meta.get("mutation_type") or "") or None,
        status=status,
        sharpe=_safe_float(metrics.get("sharpe")),
        fitness=_safe_float(metrics.get("fitness")),
        returns=_safe_float(metrics.get("returns")),
        turnover=_safe_float(metrics.get("turnover")),
        self_correlation_failed=_self_correlation_failed(item),
        failure_stage=diagnostics["failure_stage"],
        failure_reason=diagnostics["failure_reason"],
        failure_reasons=list(diagnostics["failure_reasons"] or []),
        failure_evidence=diagnostics["failure_evidence"],
        mutation_targets=list(mutation_targets or []),
        settings=dict(item.get("settings") or settings or {}),
        data_fields=list(meta.get("data_fields") or []),
        dataset_id=str(meta.get("dataset_id") or "") or None,
        lineage_id=lineage_id,
        parent_lineage_id=parent_lineage_id,
        operator_pattern=str(meta.get("operator_pattern") or "") or None,
        operators=list(meta.get("operators") or []),
        mutation_reason=str(meta.get("mutation_reason") or "") or None,
        planner_strategy=str(meta.get("planner_strategy") or "") or None,
        allocation_cell=str(meta.get("allocation_cell") or "") or None,
        source_run_id=str(meta.get("source_run_id") or item.get("run_id") or item.get("task_id") or "") or None,
        tag=tag,
    )


async def record_research_trials(
    account: str,
    result: dict[str, Any],
    *,
    default_family: str = "external",
    hypothesis: str = "",
    tag: str | None = None,
) -> int:
    """Persist simulated, invalid, and failed expressions from a research batch."""
    if not result:
        return 0

    settings = dict(result.get("settings") or {})
    effective_tag = tag or result.get("tag")
    factory = _get_session_factory()
    rows: list[WQResearchTrial] = []
    stage_events: list[WQResearchStageEvent] = []

    candidate_ids = {
        str(item.get("alpha_id"))
        for item in (result.get("candidates") or [])
        if item.get("alpha_id")
        and (
            "validation" not in item
            or str((item.get("validation") or {}).get("status") or "").lower() == "ready"
        )
    }
    for item in result.get("results") or []:
        status = "candidate" if str(item.get("alpha_id")) in candidate_ids else "rejected"
        row = _trial_from_item(
            account,
            item,
            status=status,
            settings=settings,
            tag=effective_tag,
            default_family=default_family,
            default_hypothesis=hypothesis,
        )
        if row is not None:
            rows.append(row)
            for event in funnel_events_for_trial(row, item):
                stage_events.append(
                    WQResearchStageEvent(
                        account=account,
                        lineage_id=str(row.lineage_id),
                        parent_lineage_id=row.parent_lineage_id,
                        source_run_id=row.source_run_id,
                        stage=event["stage"],
                        outcome=event["outcome"],
                        failure_stage=row.failure_stage,
                        failure_reason=event.get("failure_reason"),
                        details=event.get("details"),
                    )
                )

    for status_key, status in (("failed", "simulation_failed"), ("invalid", "invalid")):
        for item in result.get(status_key) or []:
            row = _trial_from_item(
                account,
                item,
                status=status,
                settings=settings,
                tag=effective_tag,
                default_family=default_family,
                default_hypothesis=hypothesis,
            )
            if row is not None:
                rows.append(row)
                for event in funnel_events_for_trial(row, item):
                    stage_events.append(
                        WQResearchStageEvent(
                            account=account,
                            lineage_id=str(row.lineage_id),
                            parent_lineage_id=row.parent_lineage_id,
                            source_run_id=row.source_run_id,
                            stage=event["stage"],
                            outcome=event["outcome"],
                            failure_stage=row.failure_stage,
                            failure_reason=event.get("failure_reason"),
                            details=event.get("details"),
                        )
                    )

    if not rows:
        return 0

    async with factory() as session:
        session.add_all(rows)
        session.add_all(stage_events)
        await session.commit()
    return len(rows)


async def reconcile_research_metadata(account: str = "primary") -> int:
    """Conservatively backfill lineage/metadata only when stored evidence is sufficient."""
    factory = _get_session_factory()
    updated = 0
    async with factory() as session:
        trial_result = await session.execute(select(WQResearchTrial).where(WQResearchTrial.account == account))
        for row in trial_result.scalars().all():
            recovered = recover_research_metadata(
                row.expression,
                {
                    "family": row.family,
                    "dataset_id": row.dataset_id,
                    "data_fields": list(row.data_fields or []),
                    "hypothesis": row.hypothesis,
                    "parent_expression": row.parent_expression,
                    "generation": row.generation,
                    "mutation_type": row.mutation_type,
                    "mutation_reason": row.mutation_reason,
                    "planner_strategy": row.planner_strategy,
                    "allocation_cell": row.allocation_cell,
                    "parent_lineage_id": row.parent_lineage_id,
                },
            )
            values = {
                "family": recovered.get("family"),
                "data_fields": list(recovered.get("data_fields") or []),
                "lineage_id": row.lineage_id or lineage_id_for(row.expression, settings=dict(row.settings or {}), metadata=recovered),
                "parent_lineage_id": row.parent_lineage_id
                or parent_lineage_id_for(row.parent_expression, settings=dict(row.settings or {})),
                "operator_pattern": row.operator_pattern or recovered.get("operator_pattern"),
                "operators": list(row.operators or recovered.get("operators") or []),
            }
            changed = False
            for key, value in values.items():
                current = getattr(row, key)
                if key == "family":
                    if str(current or "unknown") not in {"", "unknown"}:
                        continue
                    if value not in (None, "", "unknown"):
                        setattr(row, key, value)
                        changed = True
                    continue
                if key == "data_fields" and current:
                    continue
                if current in (None, "", []) and value not in (None, "", []):
                    setattr(row, key, value)
                    changed = True
            updated += int(changed)

        candidate_result = await session.execute(
            select(WQResearchCandidate).where(WQResearchCandidate.account == account)
        )
        for row in candidate_result.scalars().all():
            settings = {
                "region": row.region,
                "universe": row.universe,
                "delay": row.delay,
                "decay": row.decay,
                "neutralization": row.neutralization,
                "truncation": row.truncation,
            }
            recovered = recover_research_metadata(
                row.expression,
                {
                    "family": row.family,
                    "dataset_id": row.dataset_id,
                    "data_fields": list(row.data_fields or []),
                    "hypothesis": row.hypothesis,
                    "parent_expression": row.parent_expression,
                    "generation": row.generation,
                    "mutation_type": row.mutation_type,
                    "mutation_reason": row.mutation_reason,
                    "planner_strategy": row.planner_strategy,
                    "allocation_cell": row.allocation_cell,
                    "parent_lineage_id": row.parent_lineage_id,
                },
            )
            values = {
                "family": recovered.get("family"),
                "data_fields": list(recovered.get("data_fields") or []),
                "lineage_id": row.lineage_id or lineage_id_for(row.expression, settings=settings, metadata=recovered),
                "parent_lineage_id": row.parent_lineage_id or parent_lineage_id_for(row.parent_expression, settings=settings),
                "operator_pattern": row.operator_pattern or recovered.get("operator_pattern"),
                "operators": list(row.operators or recovered.get("operators") or []),
                "structure_signature": row.structure_signature or recovered.get("structure_signature"),
            }
            changed = False
            for key, value in values.items():
                current = getattr(row, key)
                if key == "family":
                    if str(current or "unknown") not in {"", "unknown"}:
                        continue
                    if value not in (None, "", "unknown"):
                        setattr(row, key, value)
                        changed = True
                    continue
                if key == "data_fields" and current:
                    continue
                if current in (None, "", []) and value not in (None, "", []):
                    setattr(row, key, value)
                    changed = True
            updated += int(changed)
        if updated:
            await session.commit()
    return updated


async def load_research_memory(account: str = "primary", limit: int = 2000) -> dict[str, Any]:
    """Load a bounded summary used by the autonomous planner."""
    await reconcile_research_metadata(account)
    factory = _get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(WQResearchTrial)
            .where(WQResearchTrial.account == account)
            .order_by(WQResearchTrial.created_at.desc())
            .limit(max(1, min(5000, int(limit))))
        )
        rows = list(result.scalars().all())
        funnel_result = await session.execute(
            select(WQResearchStageEvent)
            .where(WQResearchStageEvent.account == account)
            .order_by(WQResearchStageEvent.created_at.desc())
        )
        funnel_events = list(funnel_result.scalars().all())
        attribution_result = await session.execute(
            select(WQSubmissionAttempt, WQResearchCandidate)
            .join(
                WQResearchCandidate,
                (WQResearchCandidate.account == WQSubmissionAttempt.account)
                & (WQResearchCandidate.alpha_id == WQSubmissionAttempt.alpha_id),
            )
            .where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.score_state == "SETTLED",
                WQSubmissionAttempt.attributed_points_share.is_not(None),
            )
        )
        attribution_rows = list(attribution_result.all())
        candidate_result = await session.execute(
            select(WQResearchCandidate).where(WQResearchCandidate.account == account)
        )
        candidate_rows = list(candidate_result.scalars().all())
        formal_result = await session.execute(
            select(WQSubmissionAttempt, WQResearchCandidate)
            .join(
                WQResearchCandidate,
                (WQResearchCandidate.account == WQSubmissionAttempt.account)
                & (WQResearchCandidate.alpha_id == WQSubmissionAttempt.alpha_id),
            )
            .where(WQSubmissionAttempt.account == account)
        )
        formal_rows = list(formal_result.all())
        from .wq_submission_policy import _load_conversion_feedback

        conversion_feedback = await _load_conversion_feedback(session, account)

    research_cells = summarize_research_cells(rows, candidate_rows, formal_rows)
    funnel_summary_all = summarize_funnel(funnel_events)
    funnel_summary_recent = summarize_funnel(funnel_events[:500])
    family_counts = Counter(str(row.family or "unknown") for row in rows)
    status_counts = Counter(str(row.status or "unknown") for row in rows)
    candidate_family_counts = Counter(
        str(row.family or "unknown") for row in rows if str(row.status or "") == "candidate"
    )
    self_corr_family_counts = Counter(
        str(row.family or "unknown") for row in rows if bool(row.self_correlation_failed)
    )
    failure_stage_counts = Counter(str(row.failure_stage) for row in rows if row.failure_stage)
    failure_reason_counts = Counter(str(row.failure_reason) for row in rows if row.failure_reason)
    failure_reason_family_counts: dict[str, Counter[str]] = {}
    failure_reason_dataset_counts: dict[str, Counter[str]] = {}
    for row in rows:
        if not row.failure_reason:
            continue
        reason = str(row.failure_reason)
        failure_reason_family_counts.setdefault(reason, Counter())[str(row.family or "unknown")] += 1
        failure_reason_dataset_counts.setdefault(reason, Counter())[str(row.dataset_id or "unknown")] += 1
    normalized = {str(row.expression_normalized or "") for row in rows if row.expression_normalized}
    metadata_fields = (
        "lineage_id",
        "family",
        "data_fields",
        "operator_pattern",
    )
    completeness_counts = {
        field: sum(1 for row in rows if getattr(row, field, None) not in (None, "", [], "unknown"))
        for field in metadata_fields
    }
    metadata_completeness = {
        field: {
            "present": count,
            "missing": len(rows) - count,
            "rate": round(count / max(1, len(rows)), 4),
        }
        for field, count in completeness_counts.items()
    }
    dataset_present = sum(1 for row in rows if row.dataset_id)
    metadata_completeness["dataset_id"] = {
        "present": dataset_present,
        "missing": len(rows) - dataset_present,
        "rate": round(dataset_present / max(1, len(rows)), 4),
        "note": "unknown_is_valid_when_dataset_provenance_is_not_derivable",
    }
    family_points_attribution: Counter[str] = Counter()
    family_points_feedback: Counter[str] = Counter()
    dataset_points_attribution: Counter[str] = Counter()
    dataset_points_feedback: Counter[str] = Counter()
    for attempt, candidate in attribution_rows:
        share = float(attempt.attributed_points_share or 0.0)
        confidence = max(0.0, min(1.0, float(attempt.attribution_confidence or 0.0)))
        weighted = share * confidence
        family = str(candidate.family or "unknown")
        dataset_id = str(candidate.dataset_id or "unknown")
        family_points_attribution[family] += share
        family_points_feedback[family] += weighted
        dataset_points_attribution[dataset_id] += share
        dataset_points_feedback[dataset_id] += weighted

    mutation_rows = [
        {
            "expression": row.expression,
            "status": row.status,
            "mutation_type": row.mutation_type,
            "failure_reason": row.failure_reason,
            "failure_reasons": list(row.failure_reasons or []),
            "operator_pattern": row.operator_pattern,
            "data_fields": list(row.data_fields or []),
        }
        for row in rows
    ]
    prevention_rules = compile_prevention_rules(mutation_rows)
    mutation_outcomes = summarize_mutation_outcomes(mutation_rows)

    recent_trials = [
        {
            "expression": row.expression,
            "family": row.family,
            "hypothesis": row.hypothesis,
            "parent_expression": row.parent_expression,
            "generation": int(row.generation or 0),
            "mutation_type": row.mutation_type,
            "status": row.status,
            "sharpe": row.sharpe,
            "fitness": row.fitness,
            "returns": row.returns,
            "turnover": row.turnover,
            "self_correlation_failed": bool(row.self_correlation_failed),
            "failure_stage": row.failure_stage,
            "failure_reason": row.failure_reason,
            "failure_reasons": list(row.failure_reasons or []),
            "failure_evidence": row.failure_evidence,
            "mutation_targets": list(row.mutation_targets or []),
            "data_fields": list(row.data_fields or []),
            "dataset_id": row.dataset_id,
            "lineage_id": row.lineage_id,
            "parent_lineage_id": row.parent_lineage_id,
            "operator_pattern": row.operator_pattern,
            "operators": list(row.operators or []),
            "mutation_reason": row.mutation_reason,
            "planner_strategy": row.planner_strategy,
            "allocation_cell": row.allocation_cell,
            "source_run_id": row.source_run_id,
        }
        for row in rows[:200]
        if row.expression
    ]

    return {
        "trials": len(rows),
        "family_counts": dict(family_counts),
        "candidate_family_counts": dict(candidate_family_counts),
        "self_correlation_family_counts": dict(self_corr_family_counts),
        "status_counts": dict(status_counts),
        "failure_stage_counts": dict(failure_stage_counts),
        "failure_reason_counts": dict(failure_reason_counts),
        "failure_reason_family_counts": {
            reason: dict(counts) for reason, counts in failure_reason_family_counts.items()
        },
        "failure_reason_dataset_counts": {
            reason: dict(counts) for reason, counts in failure_reason_dataset_counts.items()
        },
        "metadata_completeness": metadata_completeness,
        "research_cells": research_cells,
        "candidate_funnel": {
            "all_time": funnel_summary_all,
            "recent_500_events": funnel_summary_recent,
            "dominant_bottleneck_stage": funnel_summary_recent.get("bottleneck_stage")
            or funnel_summary_all.get("bottleneck_stage"),
        },
        "family_points_attribution": dict(family_points_attribution),
        "family_points_feedback": dict(family_points_feedback),
        "dataset_points_attribution": dict(dataset_points_attribution),
        "dataset_points_feedback": dict(dataset_points_feedback),
        "points_attribution_rule": "equal_share_confidence_weighted",
        "active_conversion": conversion_feedback,
        "active_conversion_rate": (conversion_feedback.get("global") or {}).get("rate"),
        "terminal_submission_samples": (conversion_feedback.get("global") or {}).get("samples", 0),
        "family_active_conversion": conversion_feedback.get("family", {}),
        "dataset_active_conversion": conversion_feedback.get("dataset", {}),
        "normalized_expressions": sorted(normalized),
        "prevention_rules": prevention_rules,
        "mutation_outcomes": mutation_outcomes,
        "recent_trials": recent_trials,
    }


def record_research_trials_sync(*args, **kwargs) -> int:
    from .wq_submission_policy import _run_coro_sync

    return int(_run_coro_sync(record_research_trials(*args, **kwargs)))


def load_research_memory_sync(account: str = "primary", limit: int = 2000) -> dict[str, Any]:
    from .wq_submission_policy import _run_coro_sync

    return dict(_run_coro_sync(load_research_memory(account, limit=limit)))
