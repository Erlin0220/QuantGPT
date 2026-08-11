"""Persistent research memory for WorldQuant autonomous Alpha exploration."""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from sqlalchemy import select

from .db import _get_session_factory
from .models import WQResearchCandidate, WQResearchStageEvent, WQResearchTrial, WQSubmissionAttempt
from .wq_candidate_funnel import funnel_events_for_trial, summarize_funnel
from .wq_failure_taxonomy import classify_research_failure
from .wq_learning_maturity import build_learning_maturity
from .wq_lineage import (
    build_field_metadata_registry,
    classify_family_from_metadata,
    extract_expression_metadata,
    lineage_id_for,
    parent_lineage_id_for,
    recover_research_metadata,
)
from .wq_mutation_policy import compile_prevention_rules, summarize_mutation_outcomes
from .wq_operator_registry import canonicalize_wq_expression
from .wq_research_scheduler import summarize_research_cells

_DEFAULT_POINTS_FEEDBACK_MIN_CONFIDENCE = 0.5
_DEFAULT_POINTS_FEEDBACK_MIN_SAMPLES = 2


def _points_feedback_thresholds() -> tuple[float, int]:
    try:
        confidence = float(os.environ.get("WQ_POINTS_FEEDBACK_MIN_CONFIDENCE", str(_DEFAULT_POINTS_FEEDBACK_MIN_CONFIDENCE)))
    except (TypeError, ValueError):
        confidence = _DEFAULT_POINTS_FEEDBACK_MIN_CONFIDENCE
    try:
        samples = int(os.environ.get("WQ_POINTS_FEEDBACK_MIN_SAMPLES", str(_DEFAULT_POINTS_FEEDBACK_MIN_SAMPLES)))
    except (TypeError, ValueError):
        samples = _DEFAULT_POINTS_FEEDBACK_MIN_SAMPLES
    return max(0.0, min(1.0, confidence)), max(1, samples)


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
        dataset_category=str(meta.get("dataset_category") or "") or None,
        provenance_state=str(meta.get("provenance_state") or "") or None,
        provenance_reason=str(meta.get("provenance_reason") or "") or None,
        lineage_id=lineage_id,
        parent_lineage_id=parent_lineage_id,
        operator_pattern=str(meta.get("operator_pattern") or "") or None,
        operators=list(meta.get("operators") or []),
        mutation_reason=str(meta.get("mutation_reason") or "") or None,
        planner_strategy=str(meta.get("planner_strategy") or "") or None,
        allocation_cell=str(meta.get("allocation_cell") or "") or None,
        source_run_id=str(meta.get("source_run_id") or item.get("run_id") or item.get("task_id") or "") or None,
        knowledge_card_ids=list(meta.get("knowledge_card_ids") or []),
        failure_signature=dict(meta.get("failure_signature") or item.get("failure_signature") or {}),
        diversity_case=dict(meta.get("diversity_case") or item.get("diversity_case") or {}),
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

    result_items_by_alpha = {
        str(item.get("alpha_id")): item
        for item in (result.get("results") or [])
        if item.get("alpha_id")
    }
    candidate_ids: set[str] = set()
    for candidate in result.get("candidates") or []:
        alpha_id = str(candidate.get("alpha_id") or "")
        if not alpha_id:
            continue
        evidence_item = candidate if "validation" in candidate else result_items_by_alpha.get(alpha_id, candidate)
        validation = evidence_item.get("validation") if isinstance(evidence_item.get("validation"), dict) else None
        if validation is None:
            candidate_ids.add(alpha_id)
            continue
        if (
            str(validation.get("status") or "").lower() in {"ready", "evidence_collected"}
            and (validation.get("submission_gate") or {}).get("ready", True)
        ):
            candidate_ids.add(alpha_id)
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
    """Conservatively backfill lineage/provenance from persisted truthful evidence."""
    factory = _get_session_factory()
    updated = 0
    async with factory() as session:
        trial_result = await session.execute(select(WQResearchTrial).where(WQResearchTrial.account == account))
        candidate_result = await session.execute(select(WQResearchCandidate).where(WQResearchCandidate.account == account))
        trials = list(trial_result.scalars().all())
        candidates = list(candidate_result.scalars().all())
        field_registry = build_field_metadata_registry([*trials, *candidates])

        def apply(row: Any, *, settings: dict[str, Any]) -> bool:
            recovered = recover_research_metadata(
                row.expression,
                {
                    "family": row.family,
                    "dataset_id": row.dataset_id,
                    "dataset_category": getattr(row, "dataset_category", None),
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
                field_registry=field_registry,
            )
            values = {
                "family": recovered.get("family"),
                "data_fields": list(recovered.get("data_fields") or []),
                "dataset_id": recovered.get("dataset_id"),
                "dataset_category": recovered.get("dataset_category"),
                "provenance_state": recovered.get("provenance_state"),
                "provenance_reason": recovered.get("provenance_reason"),
                "lineage_id": row.lineage_id or lineage_id_for(row.expression, settings=settings, metadata=recovered),
                "parent_lineage_id": row.parent_lineage_id or parent_lineage_id_for(row.parent_expression, settings=settings),
                "operator_pattern": row.operator_pattern or recovered.get("operator_pattern"),
                "operators": list(row.operators or recovered.get("operators") or []),
            }
            if hasattr(row, "structure_signature"):
                values["structure_signature"] = row.structure_signature or recovered.get("structure_signature")
            changed = False
            for key, value in values.items():
                current = getattr(row, key)
                if key == "family" and str(current or "unknown") not in {"", "unknown"}:
                    continue
                if key == "data_fields" and current:
                    continue
                if key in {"dataset_id", "dataset_category"} and current not in (None, "", "unknown"):
                    continue
                if key in {"provenance_state", "provenance_reason"}:
                    if current != value and value not in (None, ""):
                        setattr(row, key, value)
                        changed = True
                    continue
                if current in (None, "", [], "unknown") and value not in (None, "", [], "unknown"):
                    setattr(row, key, value)
                    changed = True
            return changed

        for row in trials:
            updated += int(apply(row, settings=dict(row.settings or {})))
        for row in candidates:
            updated += int(apply(row, settings={
                "region": row.region,
                "universe": row.universe,
                "delay": row.delay,
                "decay": row.decay,
                "neutralization": row.neutralization,
                "truncation": row.truncation,
            }))
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
            .outerjoin(
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
        formal_rows = [(attempt, candidate) for attempt, candidate in formal_result.all()]
        from .wq_submission_policy import _load_conversion_feedback

        conversion_feedback = await _load_conversion_feedback(session, account)

    research_cells = summarize_research_cells(rows, candidate_rows, formal_rows)
    overfitting_rows = []
    local_correlation_available = 0
    local_correlation_high_risk = 0
    for candidate in candidate_rows:
        if candidate.local_correlation is not None:
            local_correlation_available += 1
            if float(candidate.local_correlation) >= 0.70:
                local_correlation_high_risk += 1
        details = candidate.validation_details if isinstance(candidate.validation_details, dict) else {}
        evidence = details.get("overfitting_evidence") if isinstance(details, dict) else None
        if isinstance(evidence, dict):
            overfitting_rows.append({"alpha_id": candidate.alpha_id, **evidence})
    overfitting_available = [item for item in overfitting_rows if item.get("status") == "available"]
    formal_attempts = len(formal_rows)
    active_attempts = sum(1 for attempt, _candidate in formal_rows if str(attempt.status or "").upper() == "ACTIVE")
    settled_active = sum(
        1
        for attempt, _candidate in formal_rows
        if str(attempt.status or "").upper() == "ACTIVE" and str(attempt.score_state or "").upper() == "SETTLED"
    )
    trial_candidates = sum(1 for row in rows if str(row.status or "").lower() == "candidate")
    high_confidence_candidates = sum(
        1
        for candidate in candidate_rows
        if str(candidate.confidence_tier or "B").upper() in {"S", "A"}
        and str(candidate.status or "").lower() == "queued"
        and str(candidate.validation_status or "").lower() in {"ready", "evidence_collected"}
    )

    def conversion(numerator: int, denominator: int) -> dict[str, Any]:
        return {
            "count": numerator,
            "denominator": denominator,
            "rate": round(numerator / denominator, 4) if denominator > 0 else None,
        }

    learning_funnel = {
        "trial_to_candidate": conversion(trial_candidates, len(rows)),
        "candidate_to_high_confidence": conversion(high_confidence_candidates, len(candidate_rows)),
        "candidate_to_submit": conversion(formal_attempts, len(candidate_rows)),
        "submit_to_active": conversion(active_attempts, formal_attempts),
        "active_to_points_settled": conversion(settled_active, active_attempts),
    }
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
        if str(row.provenance_state or "").lower() == "resolved" and row.dataset_id:
            failure_reason_dataset_counts.setdefault(reason, Counter())[str(row.dataset_id)] += 1
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
    provenance_counts = Counter(str(row.provenance_state or "unresolved").lower() for row in rows)
    unresolved_reason_counts = Counter(
        str(row.provenance_reason or "unspecified")
        for row in rows
        if str(row.provenance_state or "").lower() == "unresolved"
    )
    partial_reason_counts = Counter(
        str(row.provenance_reason or "unspecified")
        for row in rows
        if str(row.provenance_state or "").lower() == "partial"
    )
    provenance_resolved = int(provenance_counts.get("resolved", 0))
    provenance_partial = int(provenance_counts.get("partial", 0))
    provenance_classified = provenance_resolved + provenance_partial
    metadata_completeness["dataset_id"] = {
        "present": dataset_present,
        "missing": len(rows) - dataset_present,
        "rate": round(dataset_present / max(1, len(rows)), 4),
        "note": "unresolved_dataset_identity_is_excluded_from_dataset_specific_learning",
    }
    provenance_summary = {
        "resolved": provenance_resolved,
        "partial": provenance_partial,
        "unresolved": int(provenance_counts.get("unresolved", 0)),
        "resolved_rate": round(provenance_resolved / max(1, len(rows)), 4),
        "classified": provenance_classified,
        "classified_rate": round(provenance_classified / max(1, len(rows)), 4),
        "classified_note": "resolved_and_truthfully_partial_rows_are_valid_provenance;_only_resolved_rows_feed_dataset_specific_learning",
        "unresolved_reasons": dict(unresolved_reason_counts),
        "partial_reasons": dict(partial_reason_counts),
    }
    min_points_confidence, min_points_samples = _points_feedback_thresholds()
    family_points_attribution: defaultdict[str, float] = defaultdict(float)
    family_points_feedback: defaultdict[str, float] = defaultdict(float)
    dataset_points_attribution: defaultdict[str, float] = defaultdict(float)
    dataset_points_feedback: defaultdict[str, float] = defaultdict(float)
    operator_points_attribution: defaultdict[str, float] = defaultdict(float)
    operator_points_feedback: defaultdict[str, float] = defaultdict(float)
    usable_family_feedback: defaultdict[str, float] = defaultdict(float)
    usable_dataset_feedback: defaultdict[str, float] = defaultdict(float)
    usable_operator_feedback: defaultdict[str, float] = defaultdict(float)
    usable_family_samples: Counter[str] = Counter()
    usable_dataset_samples: Counter[str] = Counter()
    usable_operator_samples: Counter[str] = Counter()
    confidence_total = 0.0
    settled_points_total = 0.0
    confidence_weighted_points_total = 0.0
    usable_attempts = 0
    settlement_cohorts: set[str] = set()
    for attempt, candidate in attribution_rows:
        share = float(attempt.attributed_points_share or 0.0)
        confidence = max(0.0, min(1.0, float(attempt.attribution_confidence or 0.0)))
        weighted = share * confidence
        confidence_total += confidence
        settled_points_total += share
        confidence_weighted_points_total += weighted
        details = attempt.attribution_details if isinstance(attempt.attribution_details, dict) else {}
        cohort_ids = details.get("cohort_alpha_ids") or [attempt.alpha_id]
        settlement_cohorts.add("|".join(sorted(str(value) for value in cohort_ids if value)))
        if candidate is None:
            continue
        family = str(candidate.family or "unknown")
        dataset_id = str(candidate.dataset_id or "")
        dataset_resolved = str(candidate.provenance_state or "").lower() == "resolved" and bool(dataset_id)
        operator_pattern = str(candidate.operator_pattern or "unknown")
        family_points_attribution[family] += share
        family_points_feedback[family] += weighted
        if dataset_resolved:
            dataset_points_attribution[dataset_id] += share
            dataset_points_feedback[dataset_id] += weighted
        operator_points_attribution[operator_pattern] += share
        operator_points_feedback[operator_pattern] += weighted
        if share > 0.0 and confidence >= min_points_confidence:
            usable_attempts += 1
            usable_family_feedback[family] += weighted
            if dataset_resolved:
                usable_dataset_feedback[dataset_id] += weighted
                usable_dataset_samples[dataset_id] += 1
            usable_operator_feedback[operator_pattern] += weighted
            usable_family_samples[family] += 1
            usable_operator_samples[operator_pattern] += 1

    def usable_groups(feedback: Mapping[str, float], samples: Counter[str]) -> dict[str, float]:
        return {
            key: round(float(value), 4)
            for key, value in feedback.items()
            if int(samples.get(key, 0)) >= min_points_samples
        }

    family_points_feedback_usable = usable_groups(usable_family_feedback, usable_family_samples)
    dataset_points_feedback_usable = usable_groups(usable_dataset_feedback, usable_dataset_samples)
    operator_points_feedback_usable = usable_groups(usable_operator_feedback, usable_operator_samples)

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

    points_coverage = {
        "settled_attempts": len(attribution_rows),
        "settlement_cohorts": len(settlement_cohorts),
        "settled_points_total": round(settled_points_total, 4),
        "confidence_weighted_points_total": round(confidence_weighted_points_total, 4),
        "mean_confidence": round(confidence_total / max(1, len(attribution_rows)), 4),
        "usable_attempts": usable_attempts,
        "usable_family_groups": len(family_points_feedback_usable),
        "usable_dataset_groups": len(dataset_points_feedback_usable),
        "usable_operator_groups": len(operator_points_feedback_usable),
    }
    learning_maturity = build_learning_maturity(
        trials=len(rows),
        provenance_resolved=provenance_resolved,
        provenance_classified=provenance_classified,
        active_feedback=conversion_feedback,
        points_coverage=points_coverage,
    )
    field_registry = build_field_metadata_registry([*rows, *candidate_rows])
    from .wq_knowledge import load_knowledge_guidance

    knowledge_guidance = await load_knowledge_guidance(limit=24)

    positive_rows = sorted(
        [row for row in rows if str(row.status or "").lower() == "candidate" or (float(row.fitness or 0.0) >= 0.85 and float(row.sharpe or 0.0) >= 1.0)],
        key=lambda row: (float(row.fitness or 0.0), float(row.sharpe or 0.0)),
        reverse=True,
    )[:16]
    positive_memory = [
        {
            "family": row.family,
            "dataset_id": row.dataset_id if str(row.provenance_state or "").lower() == "resolved" else None,
            "operator_pattern": row.operator_pattern,
            "structure_signature": extract_expression_metadata(row.expression).get("structure_signature"),
            "mutation_type": row.mutation_type,
            "status": row.status,
            "fitness": row.fitness,
            "sharpe": row.sharpe,
        }
        for row in positive_rows
    ]
    negative_counter: Counter[tuple[str, str | None, str, str]] = Counter()
    for row in rows:
        if not row.failure_reason:
            continue
        negative_counter[(
            str(row.family or "unknown"),
            str(row.dataset_id) if str(row.provenance_state or "").lower() == "resolved" and row.dataset_id else None,
            str(row.operator_pattern or extract_expression_metadata(row.expression).get("operator_pattern") or "unknown"),
            str(row.failure_reason),
        )] += 1
    negative_memory = [
        {
            "family": family,
            "dataset_id": dataset_id,
            "operator_pattern": operator_pattern,
            "failure_reason": failure_reason,
            "count": count,
        }
        for (family, dataset_id, operator_pattern, failure_reason), count in negative_counter.most_common(16)
        if count >= 2
    ]
    structure_counts = Counter(str(row.operator_pattern or extract_expression_metadata(row.expression).get("operator_pattern") or "unknown") for row in rows)
    research_memory_guidance = {
        "positive": positive_memory,
        "negative": negative_memory,
        "overused_structures": [
            {"operator_pattern": pattern, "trials": count}
            for pattern, count in structure_counts.most_common(12)
            if pattern != "unknown"
        ],
        "policy": "bounded_positive_negative_structural_memory",
    }

    recent_trials = [
        {
            "alpha_id": row.alpha_id,
            "expression": row.expression,
            "settings": dict(row.settings or {}),
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
            "dataset_category": row.dataset_category,
            "provenance_state": row.provenance_state,
            "provenance_reason": row.provenance_reason,
            "lineage_id": row.lineage_id,
            "parent_lineage_id": row.parent_lineage_id,
            "operator_pattern": row.operator_pattern,
            "operators": list(row.operators or []),
            "mutation_reason": row.mutation_reason,
            "planner_strategy": row.planner_strategy,
            "allocation_cell": row.allocation_cell,
            "source_run_id": row.source_run_id,
            "knowledge_card_ids": list(row.knowledge_card_ids or []),
            "failure_signature": dict(row.failure_signature or {}),
            "diversity_case": dict(row.diversity_case or {}),
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
        "provenance": provenance_summary,
        "field_registry": field_registry,
        "research_cells": research_cells,
        "learning_funnel": learning_funnel,
        "learning_maturity": learning_maturity,
        "research_memory_guidance": research_memory_guidance,
        "knowledge_guidance": knowledge_guidance,
        "local_correlation_risk": {
            "available": local_correlation_available,
            "high_risk": local_correlation_high_risk,
            "high_risk_threshold": 0.70,
            "official_platform_check": False,
        },
        "overfitting_evidence": {
            "available": len(overfitting_available),
            "unavailable": len(overfitting_rows) - len(overfitting_available),
            "mean_dsr_score": round(
                sum(float(item.get("score") or 0.0) for item in overfitting_available) / max(1, len(overfitting_available)),
                6,
            ),
            "recent": overfitting_rows[:20],
            "official_platform_check": False,
            "pbo_policy": "bounded_optional_for_coherent_variant_families",
        },
        "candidate_funnel": {
            "all_time": funnel_summary_all,
            "recent_500_events": funnel_summary_recent,
            "dominant_bottleneck_stage": funnel_summary_recent.get("bottleneck_stage")
            or funnel_summary_all.get("bottleneck_stage"),
        },
        "family_points_attribution": dict(family_points_attribution),
        "family_points_feedback": dict(family_points_feedback),
        "family_points_feedback_usable": family_points_feedback_usable,
        "dataset_points_attribution": dict(dataset_points_attribution),
        "dataset_points_feedback": dict(dataset_points_feedback),
        "dataset_points_feedback_usable": dataset_points_feedback_usable,
        "operator_points_attribution": dict(operator_points_attribution),
        "operator_points_feedback": dict(operator_points_feedback),
        "operator_points_feedback_usable": operator_points_feedback_usable,
        "points_attribution_rule": "equal_share_confidence_weighted",
        "points_feedback_gate": {
            "min_confidence": min_points_confidence,
            "min_samples_per_group": min_points_samples,
            "planner_uses_only_gated_feedback": True,
            "high_capacity_models_deferred": ["RUDDER", "ARES", "QR_DQN"],
        },
        "points_feedback_coverage": points_coverage,
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
