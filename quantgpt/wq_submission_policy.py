"""Local submission budget, candidate queue, and delayed-points reconciliation for WQ BRAIN.

BRAIN leaderboard points can lag formal Alpha submissions.  This module makes
submission limiting deterministic and local: formal submissions consume a
per-day slot before the remote request is sent, while research candidates stay
queued for later days.  Points are only reconciled when account status reports
that the leaderboard is current.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, select

from .db import _get_session_factory
from .models import WQResearchCandidate, WQResearchTrial, WQSubmissionAttempt, WQSubmissionState
from .wq_candidate_calibration import (
    calibrate_active_probability,
    calibration_report,
    configured_confidence_thresholds,
)
from .wq_failure_taxonomy import classify_research_failure
from .wq_learning_maturity import active_feedback_gate
from .wq_lineage import lineage_id_for, parent_lineage_id_for, recover_research_metadata
from .wq_research_scheduler import research_cell_key
from .wq_submission_evidence import configured_submission_evidence_policy, submission_evidence_blockers

_DEFAULT_DAILY_BUDGET = 2
_MAX_CONFIGURED_BUDGET = 5
_DEFAULT_INVENTORY_FLOOR = 30
_DEFAULT_INVENTORY_TARGET_LOW = 40
_DEFAULT_INVENTORY_TARGET_HIGH = 50
_DEFAULT_FRESHNESS_HOURS = 24.0
_DEFAULT_RESERVATION_TTL_SECONDS = 15 * 60
_TERMINAL_POSITIVE_STATUSES = {"ACTIVE"}
_TERMINAL_NEGATIVE_STATUSES = {"SC_FAIL", "OTHER_FAIL"}
_SLOT_CONSUMING_STATUSES = {"RESERVED", "RESERVATION_EXPIRED", "SUBMIT_UNKNOWN", "SC_PENDING", "ACTIVE"}
_RECONCILIATION_REQUIRED_STATUSES = {"RESERVATION_EXPIRED", "SUBMIT_UNKNOWN"}
_CONVERSION_PRIOR_ALPHA = 2.0
_CONVERSION_PRIOR_BETA = 2.0
_CONVERSION_GROUP_PRIOR_STRENGTH = 4.0
_reservation_lock = threading.Lock()


def _configured_daily_budget() -> int:
    """Daily ACTIVE target; legacy env name is retained for compatibility."""
    raw = os.environ.get("WQ_DAILY_ACTIVE_TARGET", os.environ.get("WQ_DAILY_SUBMISSION_BUDGET", str(_DEFAULT_DAILY_BUDGET)))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = _DEFAULT_DAILY_BUDGET
    return max(1, min(_MAX_CONFIGURED_BUDGET, value))


def _configured_inventory_policy() -> dict[str, float | int]:
    def integer(name: str, default: int) -> int:
        try:
            return max(1, int(os.environ.get(name, str(default))))
        except (TypeError, ValueError):
            return default

    floor = integer("WQ_CANDIDATE_INVENTORY_FLOOR", _DEFAULT_INVENTORY_FLOOR)
    target_low = max(floor, integer("WQ_CANDIDATE_INVENTORY_TARGET_LOW", _DEFAULT_INVENTORY_TARGET_LOW))
    target_high = max(target_low, integer("WQ_CANDIDATE_INVENTORY_TARGET_HIGH", _DEFAULT_INVENTORY_TARGET_HIGH))
    try:
        freshness_hours = float(os.environ.get("WQ_CANDIDATE_FRESHNESS_HOURS", str(_DEFAULT_FRESHNESS_HOURS)))
    except (TypeError, ValueError):
        freshness_hours = _DEFAULT_FRESHNESS_HOURS
    return {
        "floor": floor,
        "target_low": target_low,
        "target_high": target_high,
        "freshness_hours": max(0.25, freshness_hours),
    }


def _submission_timezone():
    name = os.environ.get("WQ_SUBMISSION_TIMEZONE", "Asia/Shanghai").strip() or "Asia/Shanghai"
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return timezone.utc


def submission_day(now: datetime | None = None) -> str:
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(_submission_timezone()).date().isoformat()


def _normalized_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_is_fresh(candidate: WQResearchCandidate, *, now: datetime | None = None) -> bool:
    validated_at = _normalized_datetime(candidate.last_validated_at)
    if validated_at is None:
        return False
    current = _normalized_datetime(now or datetime.now(timezone.utc)) or datetime.now(timezone.utc)
    age_hours = max(0.0, (current - validated_at).total_seconds() / 3600.0)
    return age_hours <= float(_configured_inventory_policy()["freshness_hours"])


def _configured_reservation_ttl_seconds() -> int:
    try:
        value = int(os.environ.get("WQ_SUBMISSION_RESERVATION_TTL_SECONDS", str(_DEFAULT_RESERVATION_TTL_SECONDS)))
    except (TypeError, ValueError):
        value = _DEFAULT_RESERVATION_TTL_SECONDS
    return max(120, value)


def _candidate_submission_blockers(
    candidate: WQResearchCandidate,
    *,
    now: datetime | None = None,
) -> list[str]:
    """Return deterministic reasons a tracked candidate must not consume a formal slot."""
    blockers: list[str] = []
    if str(candidate.status or "").lower() != "queued":
        blockers.append("candidate_not_queued")
    if str(candidate.validation_status or "").lower() not in {"ready", "evidence_collected"}:
        blockers.append("validation_not_ready")
    if float(candidate.sharpe or 0.0) < 1.25:
        blockers.append("sharpe_below_threshold")
    if float(candidate.fitness or 0.0) < 1.0:
        blockers.append("fitness_below_threshold")
    if not 0.01 <= float(candidate.turnover or 0.0) <= 0.7:
        blockers.append("turnover_out_of_range")
    if str(candidate.sc_status or "").upper() == "FAIL":
        blockers.append("official_self_correlation_fail")
    if not _candidate_is_fresh(candidate, now=now):
        blockers.append("validation_stale")

    local_evidence = {
        "status": "available" if candidate.local_correlation is not None else "unavailable",
        "max_correlation": candidate.local_correlation,
        "calculated_at": candidate.local_correlation_at.isoformat() if candidate.local_correlation_at else None,
    }
    validation_details = candidate.validation_details if isinstance(candidate.validation_details, dict) else {}
    blockers.extend(
        submission_evidence_blockers(
            local_evidence,
            validation_details.get("overfitting_evidence"),
            now=now,
        )
    )
    return blockers


def _candidate_fallback_submission_blockers(
    candidate: WQResearchCandidate,
    *,
    now: datetime | None = None,
) -> list[str]:
    """Hard blockers for filling the daily ACTIVE target.

    The fallback path intentionally lets BRAIN be the final judge while the
    local daily ACTIVE target is still unmet. Local robustness, correlation,
    and overfitting evidence remain ranking/risk signals rather than vetoes.
    Platform eligibility thresholds and an explicit official SC failure remain
    hard blockers so we do not spend requests on Alphas BRAIN cannot accept.
    """
    blockers: list[str] = []
    candidate_status = str(candidate.status or "").lower()

    if candidate_status not in {"queued", "validation_pending", "robustness_fail"}:
        blockers.append("candidate_not_available")
    if float(candidate.sharpe or 0.0) < 1.25:
        blockers.append("sharpe_below_threshold")
    if float(candidate.fitness or 0.0) < 1.0:
        blockers.append("fitness_below_threshold")
    if not 0.01 <= float(candidate.turnover or 0.0) <= 0.7:
        blockers.append("turnover_out_of_range")
    if str(candidate.sc_status or "").upper() == "FAIL":
        blockers.append("official_self_correlation_fail")
    return blockers


def _candidate_research_ready(candidate: WQResearchCandidate) -> bool:
    """Cold-start inventory eligibility from deterministic submission-readiness evidence."""
    return not _candidate_submission_blockers(candidate)


def _candidate_fallback_submission_ready(candidate: WQResearchCandidate) -> bool:
    return not _candidate_fallback_submission_blockers(candidate)


def _structure_signature(expression: str) -> str:
    """Collapse lookback constants so nearby parameter variants share one structure."""
    value = re.sub(r"\s+", "", str(expression or "").lower())
    value = re.sub(r"(?<![a-z_])\d+(?:\.\d+)?", "#", value)
    return value


def _expression_tokens(expression: str) -> set[str]:
    return set(re.findall(r"[a-z_][a-z0-9_]*", str(expression or "").lower()))


def _novelty_score(expression: str, peers: list[str]) -> float:
    """Cheap structural novelty proxy used before a true return-correlation API is available."""
    tokens = _expression_tokens(expression)
    if not tokens or not peers:
        return 1.0
    max_similarity = 0.0
    for peer in peers:
        other = _expression_tokens(peer)
        if not other:
            continue
        union = tokens | other
        similarity = len(tokens & other) / max(1, len(union))
        max_similarity = max(max_similarity, similarity)
    return round(max(0.0, 1.0 - max_similarity), 4)


def _priority_score(candidate: dict[str, Any]) -> float:
    """Compatibility/display score only; candidate ordering uses evidence hierarchy.

    Keep the stored field for API/backward compatibility without pretending an
    opaque weighted metric blend is a calibrated preference or probability.
    """
    metrics = candidate.get("is_metrics") or {}
    raw_checks = list(metrics.get("checks") or []) if isinstance(metrics, dict) else []
    if any(
        str(check.get("name") or "").upper() == "SELF_CORRELATION"
        and str(check.get("result") or "").upper() == "FAIL"
        for check in raw_checks
    ):
        return -10000.0
    try:
        return float(metrics.get("fitness") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _smoothed_rate(successes: int, samples: int, *, prior_rate: float, prior_strength: float) -> float:
    return (float(successes) + float(prior_rate) * float(prior_strength)) / max(1.0, float(samples) + float(prior_strength))


async def _load_conversion_feedback(session, account: str) -> dict[str, Any]:
    """Load small-sample-safe ACTIVE conversion feedback used by research memory."""
    terminal_statuses = sorted(_TERMINAL_POSITIVE_STATUSES | _TERMINAL_NEGATIVE_STATUSES)
    attempts_result = await session.execute(
        select(WQSubmissionAttempt).where(
            WQSubmissionAttempt.account == account,
            WQSubmissionAttempt.status.in_(terminal_statuses),
        )
    )
    attempts = list(attempts_result.scalars().all())
    positive = sum(1 for item in attempts if str(item.status or "").upper() in _TERMINAL_POSITIVE_STATUSES)
    samples = len(attempts)
    global_rate = _smoothed_rate(
        positive,
        samples,
        prior_rate=_CONVERSION_PRIOR_ALPHA / (_CONVERSION_PRIOR_ALPHA + _CONVERSION_PRIOR_BETA),
        prior_strength=_CONVERSION_PRIOR_ALPHA + _CONVERSION_PRIOR_BETA,
    )
    grouped_result = await session.execute(
        select(WQSubmissionAttempt, WQResearchCandidate)
        .join(
            WQResearchCandidate,
            (WQResearchCandidate.account == WQSubmissionAttempt.account)
            & (WQResearchCandidate.alpha_id == WQSubmissionAttempt.alpha_id),
        )
        .where(
            WQSubmissionAttempt.account == account,
            WQSubmissionAttempt.status.in_(terminal_statuses),
        )
    )
    family_counts: dict[str, list[int]] = {}
    dataset_counts: dict[str, list[int]] = {}
    cell_counts: dict[str, list[int]] = {}
    resolved_predictions: list[tuple[float, bool]] = []
    for attempt, candidate in grouped_result.all():
        is_positive = int(str(attempt.status or "").upper() in _TERMINAL_POSITIVE_STATUSES)
        cell_key = research_cell_key(candidate)
        family_key = str(candidate.family or "unknown")
        values = family_counts.setdefault(family_key, [0, 0])
        values[0] += is_positive
        values[1] += 1
        if str(candidate.provenance_state or "").lower() == "resolved" and candidate.dataset_id:
            for mapping, key in ((dataset_counts, str(candidate.dataset_id)), (cell_counts, cell_key)):
                grouped = mapping.setdefault(key, [0, 0])
                grouped[0] += is_positive
                grouped[1] += 1
        if candidate.active_probability is not None:
            resolved_predictions.append((float(candidate.active_probability), bool(is_positive)))

    def summarize(mapping: dict[str, list[int]]) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "rate": round(
                    _smoothed_rate(values[0], values[1], prior_rate=global_rate, prior_strength=_CONVERSION_GROUP_PRIOR_STRENGTH),
                    4,
                ),
                "samples": values[1],
                "active": values[0],
                "failed": values[1] - values[0],
            }
            for key, values in mapping.items()
        }

    return {
        "global": {"rate": round(global_rate, 4), "samples": samples, "active": positive, "failed": samples - positive},
        "family": summarize(family_counts),
        "dataset": summarize(dataset_counts),
        "cell": summarize(cell_counts),
        "calibration": calibration_report(resolved_predictions),
        "smoothing": {"global_prior": "beta(2,2)", "group_prior_strength": _CONVERSION_GROUP_PRIOR_STRENGTH},
    }


async def _get_or_create_state(session, account: str) -> WQSubmissionState:
    result = await session.execute(select(WQSubmissionState).where(WQSubmissionState.account == account))
    state = result.scalar_one_or_none()
    if state is None:
        state = WQSubmissionState(account=account, daily_budget=_configured_daily_budget())
        session.add(state)
        await session.flush()
    return state


async def _expire_stale_reservations(
    session,
    account: str,
    *,
    now: datetime | None = None,
) -> list[str]:
    """Release stale local slot leases without making the same Alpha retryable blindly."""
    current = _normalized_datetime(now or datetime.now(timezone.utc)) or datetime.now(timezone.utc)
    cutoff = current - timedelta(seconds=_configured_reservation_ttl_seconds())
    result = await session.execute(
        select(WQSubmissionAttempt).where(
            WQSubmissionAttempt.account == account,
            WQSubmissionAttempt.status == "RESERVED",
            WQSubmissionAttempt.created_at <= cutoff,
        )
    )
    stale = list(result.scalars().all())
    for attempt in stale:
        attempt.status = "RESERVATION_EXPIRED"
        attempt.score_state = "NOT_ELIGIBLE"
        attempt.detail = "local reservation lease expired; platform recheck required before retry"
        candidate_result = await session.execute(
            select(WQResearchCandidate).where(
                WQResearchCandidate.account == account,
                WQResearchCandidate.alpha_id == attempt.alpha_id,
            )
        )
        candidate = candidate_result.scalar_one_or_none()
        if candidate is not None and str(candidate.status or "").lower() == "reserved":
            candidate.status = "validation_pending"
            candidate.validation_status = "reservation_recovery_required"
    return [str(attempt.alpha_id) for attempt in stale]


async def get_submission_reservation_recovery_ids(account: str = "primary", *, limit: int = 20) -> list[str]:
    """Return stale/expired reservations that need a platform-side status recheck."""
    factory = _get_session_factory()
    async with factory() as session:
        expired_now = await _expire_stale_reservations(session, account)
        result = await session.execute(
            select(WQSubmissionAttempt.alpha_id)
            .where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.status.in_(sorted(_RECONCILIATION_REQUIRED_STATUSES)),
            )
            .order_by(WQSubmissionAttempt.created_at.desc())
            .limit(max(1, min(100, int(limit))))
        )
        ids = [str(value) for value in result.scalars().all() if value]
        await session.commit()
    return list(dict.fromkeys([*expired_now, *ids]))[: max(1, min(100, int(limit)))]


async def reconcile_submission_reservations(
    account: str,
    platform_alphas: dict[str, dict[str, Any]],
) -> int:
    """Recover expired submission leases from authoritative platform status."""
    if not platform_alphas:
        return 0
    factory = _get_session_factory()
    updated = 0
    async with factory() as session:
        for alpha_id, info in platform_alphas.items():
            if (info or {}).get("ok") is False:
                continue
            attempt_result = await session.execute(
                select(WQSubmissionAttempt)
                .where(
                    WQSubmissionAttempt.account == account,
                    WQSubmissionAttempt.alpha_id == str(alpha_id),
                    WQSubmissionAttempt.status.in_(["RESERVED", "RESERVATION_EXPIRED", "SUBMIT_UNKNOWN", "SC_PENDING"]),
                )
                .order_by(WQSubmissionAttempt.created_at.desc())
                .limit(1)
            )
            attempt = attempt_result.scalar_one_or_none()
            if attempt is None:
                continue
            platform_status = str((info or {}).get("status") or "").upper()
            sc_result = str((info or {}).get("sc_result") or "").upper()
            if platform_status == "ACTIVE":
                attempt.status = "ACTIVE"
                attempt.score_state = "PENDING"
                attempt.detail = "recovered stale reservation: platform reports ACTIVE"
            elif sc_result == "FAIL":
                attempt.status = "SC_FAIL"
                attempt.score_state = "NOT_ELIGIBLE"
                attempt.detail = "recovered stale reservation: platform SELF_CORRELATION failed"
            elif platform_status == "UNSUBMITTED":
                attempt.status = "UNSUBMITTED_CONFIRMED"
                attempt.score_state = "NOT_ELIGIBLE"
                attempt.detail = "platform confirms UNSUBMITTED; formal slot released and candidate may be revalidated"
            elif platform_status:
                attempt.status = "SC_PENDING"
                attempt.score_state = "PENDING"
                attempt.detail = f"recovered stale reservation: platform status={platform_status}"
            else:
                continue
            updated += 1
        await session.commit()
    return updated


async def record_research_candidates(
    account: str,
    candidates: list[dict[str, Any]],
    *,
    settings: dict[str, Any] | None = None,
    tag: str | None = None,
) -> int:
    """Upsert passing research candidates into the persistent queue."""
    if not candidates:
        return 0

    settings = settings or {}
    factory = _get_session_factory()
    saved = 0
    async with factory() as session:
        conversion_feedback = await _load_conversion_feedback(session, account)
        for candidate in candidates:
            alpha_id = str(candidate.get("alpha_id") or "").strip()
            expression = str(candidate.get("expression") or "").strip()
            if not alpha_id or not expression:
                continue

            existing_result = await session.execute(
                select(WQResearchCandidate).where(
                    WQResearchCandidate.account == account,
                    WQResearchCandidate.alpha_id == alpha_id,
                )
            )
            existing = existing_result.scalar_one_or_none()
            metrics = candidate.get("is_metrics") or {}
            raw_meta = candidate.get("research_meta") or {}
            meta = recover_research_metadata(expression, raw_meta, platform_meta=candidate)
            raw_validation = candidate.get("validation") or {}
            validation = raw_validation if isinstance(raw_validation, dict) else {}
            validation_status = str(validation.get("status") or "legacy_unvalidated")
            sc_check = next(
                (
                    check
                    for check in (metrics.get("checks") or [])
                    if str(check.get("name") or "").upper() == "SELF_CORRELATION"
                ),
                None,
            )
            self_correlation = _optional_float((sc_check or {}).get("value"))
            sc_status = str((sc_check or {}).get("result") or "").upper() or None
            raw_local_correlation = candidate.get("local_correlation_proxy") or validation.get("local_correlation_proxy") or {}
            local_proxy = raw_local_correlation if isinstance(raw_local_correlation, dict) else {}
            local_correlation = _optional_float(local_proxy.get("max_correlation"))
            local_correlation_alpha_id = str(local_proxy.get("matching_alpha_id") or "") or None
            try:
                local_correlation_samples = int(local_proxy.get("sample_length") or 0)
            except (TypeError, ValueError):
                local_correlation_samples = 0
            local_correlation_at = None
            if local_proxy.get("calculated_at"):
                try:
                    local_correlation_at = datetime.fromisoformat(str(local_proxy["calculated_at"]).replace("Z", "+00:00"))
                except (TypeError, ValueError):
                    local_correlation_at = None
            evidence_blockers = submission_evidence_blockers(
                local_proxy,
                validation.get("overfitting_evidence"),
            )
            validation["submission_gate"] = {
                "ready": not evidence_blockers,
                "blockers": list(evidence_blockers),
                "policy": configured_submission_evidence_policy(),
            }
            peer_result = await session.execute(
                select(WQResearchCandidate.expression).where(
                    WQResearchCandidate.account == account,
                    WQResearchCandidate.alpha_id != alpha_id,
                    WQResearchCandidate.status.in_(["queued", "reserved", "submitted"]),
                )
            )
            novelty_score = _novelty_score(expression, [str(value) for value in peer_result.scalars().all() if value])
            candidate["novelty_score"] = novelty_score
            candidate["research_meta"] = meta
            calibration = calibrate_active_probability(candidate, conversion_feedback)
            raw_probability = calibration.get("probability")
            active_probability = float(raw_probability) if raw_probability is not None else None
            priority_score = _priority_score(candidate)
            structure_signature = str(meta.get("structure_signature") or _structure_signature(expression))
            lineage_id = str(meta.get("lineage_id") or "") or lineage_id_for(expression, settings=settings, metadata=meta)
            parent_lineage_id = str(meta.get("parent_lineage_id") or "") or parent_lineage_id_for(
                meta.get("parent_expression"), settings=settings
            )
            values = {
                "alpha_id": alpha_id,
                "expression": expression,
                "region": settings.get("region", "USA"),
                "universe": settings.get("universe", "TOP3000"),
                "delay": int(settings.get("delay", 1)),
                "decay": int(settings.get("decay", 0)),
                "neutralization": settings.get("neutralization", "SUBINDUSTRY"),
                "truncation": float(settings.get("truncation", 0.08)),
                "sharpe": metrics.get("sharpe"),
                "fitness": metrics.get("fitness"),
                "returns": metrics.get("returns"),
                "turnover": metrics.get("turnover"),
                "priority_score": priority_score,
                "active_probability": active_probability,
                "confidence_tier": calibration.get("tier"),
                "probability_support": int(calibration["support"]),
                "probability_provenance": str(calibration["provenance"]),
                "calibration_details": calibration,
                "last_validated_at": (
                    datetime.now(timezone.utc)
                    if validation_status in {"ready", "evidence_collected"}
                    else (existing.last_validated_at if existing is not None else None)
                ),
                "family": meta.get("family"),
                "hypothesis": meta.get("hypothesis"),
                "parent_expression": meta.get("parent_expression"),
                "generation": int(meta.get("generation") or 0),
                "mutation_type": meta.get("mutation_type"),
                "structure_signature": structure_signature,
                "data_fields": list(meta.get("data_fields") or []),
                "dataset_id": str(meta.get("dataset_id") or "") or None,
                "dataset_category": str(meta.get("dataset_category") or "") or None,
                "provenance_state": str(meta.get("provenance_state") or "") or None,
                "provenance_reason": str(meta.get("provenance_reason") or "") or None,
                "lineage_id": lineage_id,
                "parent_lineage_id": parent_lineage_id,
                "operator_pattern": str(meta.get("operator_pattern") or "") or None,
                "operators": list(meta.get("operators") or []),
                "mutation_reason": str(meta.get("mutation_reason") or "") or None,
                "planner_strategy": str(meta.get("planner_strategy") or "") or None,
                "allocation_cell": str(meta.get("allocation_cell") or "") or None,
                "source_run_id": str(meta.get("source_run_id") or candidate.get("run_id") or candidate.get("task_id") or "") or None,
                "knowledge_card_ids": list(meta.get("knowledge_card_ids") or []),
                "validation_status": validation_status,
                "robustness_score": validation.get("robustness_score"),
                "novelty_score": novelty_score,
                "self_correlation": self_correlation,
                "sc_status": sc_status,
                "local_correlation": local_correlation,
                "local_correlation_alpha_id": local_correlation_alpha_id,
                "local_correlation_samples": local_correlation_samples or None,
                "local_correlation_at": local_correlation_at,
                "validation_details": dict(validation),
                "tag": tag,
            }

            # Primary-pass Alphas are not automatically submission-ready. Autonomous
            # candidates must clear the robustness funnel; manual/platform legacy
            # candidates retain their old queue semantics and are still rechecked before submit.
            if validation_status == "robustness_fail":
                target_status = "robustness_fail"
            elif validation_status in {"validation_pending", "robustness_unavailable"} or evidence_blockers:
                target_status = "validation_pending"
            else:
                target_status = "queued"

            # Keep only the strongest nearby parameterization in the live queue.
            similar_result = await session.execute(
                select(WQResearchCandidate)
                .where(
                    WQResearchCandidate.account == account,
                    WQResearchCandidate.alpha_id != alpha_id,
                    WQResearchCandidate.structure_signature == structure_signature,
                    WQResearchCandidate.status.in_(["queued", "reserved", "submitted"]),
                )
                .order_by(WQResearchCandidate.priority_score.desc())
                .limit(1)
            )
            similar = similar_result.scalar_one_or_none()
            if target_status == "queued" and similar is not None:
                if similar.status == "queued" and priority_score > float(similar.priority_score or 0.0):
                    similar.status = "redundant"
                else:
                    target_status = "redundant"

            if existing is None:
                session.add(WQResearchCandidate(account=account, status=target_status, **values))
            else:
                for key, value in values.items():
                    setattr(existing, key, value)
                # Never put an already-attempted Alpha back into the submission queue.
                if existing.status not in {"reserved", "submitted", "sc_fail", "submit_failed"}:
                    if (
                        existing.status in {"robustness_fail", "validation_pending"}
                        and validation_status in {"legacy_unvalidated", "platform_recheck"}
                    ):
                        pass
                    else:
                        existing.status = target_status
            saved += 1
        await session.commit()
    return saved


async def record_platform_candidates(
    account: str,
    alphas: list[dict[str, Any]],
    *,
    min_sharpe: float = 1.25,
    min_fitness: float = 1.0,
) -> int:
    """Backfill strong UNSUBMITTED platform alphas into the local candidate queue.

    SC is deliberately not required here: queueing is inventory management only.
    Formal submission still rechecks SC PASS through ``wq_brain_check_alphas``.
    """
    candidates: list[dict[str, Any]] = []
    for alpha in alphas or []:
        if str(alpha.get("status") or "").upper() != "UNSUBMITTED":
            continue
        try:
            sharpe = float(alpha.get("sharpe") or 0.0)
            fitness = float(alpha.get("fitness") or 0.0)
            turnover = float(alpha.get("turnover") or 0.0)
        except (TypeError, ValueError):
            continue
        if sharpe < min_sharpe or fitness < min_fitness or not (0.01 <= turnover <= 0.7):
            continue
        candidates.append(
            {
                "alpha_id": alpha.get("alpha_id"),
                "expression": alpha.get("expression"),
                "is_metrics": {
                    "sharpe": sharpe,
                    "fitness": fitness,
                    "returns": alpha.get("returns"),
                    "turnover": turnover,
                    "checks": [],
                },
                "research_meta": {
                    "family": alpha.get("family"),
                    "dataset_id": alpha.get("dataset_id") or alpha.get("datasetId"),
                    "data_fields": alpha.get("data_fields") or alpha.get("fields") or [],
                    "hypothesis": "recovered_from_platform_history",
                    "generation": 0,
                    "mutation_type": "platform_backfill",
                    "mutation_reason": "platform_inventory_recovery",
                    "planner_strategy": "platform_backfill",
                    "source_run_id": alpha.get("task_id") or alpha.get("run_id"),
                },
                "validation": {"status": "platform_recheck"},
            }
        )
    return await record_research_candidates(
        account,
        candidates,
        settings={},
        tag="platform-backfill",
    )


async def reconcile_candidate_platform_statuses(
    account: str,
    platform_alphas: dict[str, dict[str, Any]],
) -> int:
    """Remove stale queue entries when BRAIN reports a terminal/non-queue status."""
    if not platform_alphas:
        return 0

    factory = _get_session_factory()
    updated = 0
    async with factory() as session:
        conversion_feedback = await _load_conversion_feedback(session, account)
        for alpha_id, info in platform_alphas.items():
            if (info or {}).get("ok") is False:
                continue
            platform_status = str((info or {}).get("status") or "").upper()
            sc_result = str((info or {}).get("sc_result") or "").upper()
            sc_value = _optional_float((info or {}).get("sc_value"))
            if platform_status == "ACTIVE":
                target_status = "submitted"
            elif sc_result == "FAIL":
                target_status = "sc_fail"
            elif platform_status == "UNSUBMITTED":
                target_status = None
            else:
                target_status = None

            result = await session.execute(
                select(WQResearchCandidate).where(
                    WQResearchCandidate.account == account,
                    WQResearchCandidate.alpha_id == str(alpha_id),
                    WQResearchCandidate.status.in_([
                        "queued",
                        "reserved",
                        "validation_pending",
                        "robustness_fail",
                        "submission_unknown",
                    ]),
                )
            )
            candidate = result.scalar_one_or_none()
            if candidate is not None:
                for field in ("sharpe", "fitness", "returns", "turnover"):
                    value = (info or {}).get(field)
                    if value is not None:
                        try:
                            setattr(candidate, field, float(value))
                        except (TypeError, ValueError):
                            pass
                candidate.sc_status = sc_result or candidate.sc_status
                if sc_value is not None:
                    candidate.self_correlation = sc_value
                candidate.last_validated_at = datetime.now(timezone.utc)
                if target_status is not None:
                    candidate.status = target_status
                elif platform_status == "UNSUBMITTED":
                    metrics_ready = (
                        float(candidate.sharpe or 0.0) >= 1.25
                        and float(candidate.fitness or 0.0) >= 1.0
                        and 0.01 <= float(candidate.turnover or 0.0) <= 0.7
                        and sc_result != "FAIL"
                    )
                    validation_details = (
                        dict(candidate.validation_details)
                        if isinstance(candidate.validation_details, dict)
                        else {}
                    )
                    robustness_ready = (
                        str(validation_details.get("status") or candidate.validation_status or "").lower()
                        in {"ready", "evidence_collected"}
                    )
                    ready = metrics_ready and robustness_ready
                    candidate.status = "queued" if ready else "validation_pending"
                    if ready and str(candidate.validation_status or "").lower() != "evidence_collected":
                        candidate.validation_status = "ready"
                    elif metrics_ready and not robustness_ready:
                        candidate.validation_status = "robustness_pending"
                        validation_details.update(
                            {
                                "status": "robustness_pending",
                                "reason": "platform_preflight_requires_skill_defined_cross_setting_evidence_before_submission",
                            }
                        )
                        candidate.validation_details = validation_details
                    else:
                        candidate.validation_status = "platform_readiness_failed"
                    if ready:
                        evidence_blockers = [
                            blocker
                            for blocker in _candidate_submission_blockers(candidate)
                            if blocker in {"local_correlation_high", "overfitting_evidence_weak"}
                        ]
                        if evidence_blockers:
                            candidate.status = "validation_pending"
                evidence = {
                    "expression": candidate.expression,
                    "family": candidate.family,
                    "dataset_id": candidate.dataset_id,
                    "research_meta": {
                        "family": candidate.family,
                        "dataset_id": candidate.dataset_id,
                        "operator_pattern": candidate.operator_pattern,
                    },
                    "is_metrics": {
                        "fitness": candidate.fitness,
                        "sharpe": candidate.sharpe,
                        "returns": candidate.returns,
                        "turnover": candidate.turnover,
                        "checks": [{
                            "name": "SELF_CORRELATION",
                            "result": candidate.sc_status,
                            "value": candidate.self_correlation,
                        }],
                    },
                    "validation": {
                        **(
                            dict(candidate.validation_details)
                            if isinstance(candidate.validation_details, dict)
                            else {}
                        ),
                        "status": candidate.validation_status,
                        "robustness_score": candidate.robustness_score,
                    },
                    "local_correlation_proxy": {
                        "status": "available" if candidate.local_correlation is not None else "unavailable",
                        "max_correlation": candidate.local_correlation,
                        "matching_alpha_id": candidate.local_correlation_alpha_id,
                        "sample_length": candidate.local_correlation_samples or 0,
                        "calculated_at": candidate.local_correlation_at.isoformat() if candidate.local_correlation_at else None,
                        "official_sc": False,
                    },
                    "novelty_score": candidate.novelty_score,
                }
                calibration = calibrate_active_probability(evidence, conversion_feedback)
                raw_probability = calibration.get("probability")
                candidate.active_probability = float(raw_probability) if raw_probability is not None else None
                candidate.confidence_tier = str(calibration["tier"]) if calibration.get("tier") else None
                candidate.probability_support = int(calibration["support"])
                candidate.probability_provenance = str(calibration["provenance"])
                candidate.calibration_details = calibration
                candidate.priority_score = _priority_score(evidence)
                updated += 1
        await session.commit()
    return updated


async def get_candidate_robustness_revalidation_payloads(
    account: str,
    alpha_ids: list[str],
) -> list[dict[str, Any]]:
    """Load tracked candidates that need cross-setting robustness before formal submission."""
    requested = {str(alpha_id or "").strip() for alpha_id in alpha_ids if str(alpha_id or "").strip()}
    if not requested:
        return []
    factory = _get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(WQResearchCandidate).where(
                WQResearchCandidate.account == account,
                WQResearchCandidate.alpha_id.in_(sorted(requested)),
                WQResearchCandidate.status.in_(["queued", "validation_pending"]),
            )
        )
        candidates = list(result.scalars().all())

    payloads: list[dict[str, Any]] = []
    for candidate in candidates:
        validation_status = str(candidate.validation_status or "").lower()
        if validation_status in {"ready", "evidence_collected"}:
            continue
        if validation_status not in {
            "platform_recheck",
            "robustness_pending",
            "validation_pending",
            "robustness_unavailable",
        }:
            continue
        validation = dict(candidate.validation_details) if isinstance(candidate.validation_details, dict) else {}
        local_proxy = {
            "status": "available" if candidate.local_correlation is not None else "unavailable",
            "max_correlation": candidate.local_correlation,
            "matching_alpha_id": candidate.local_correlation_alpha_id,
            "sample_length": candidate.local_correlation_samples or 0,
            "calculated_at": candidate.local_correlation_at.isoformat() if candidate.local_correlation_at else None,
            "official_sc": False,
        }
        payloads.append(
            {
                "alpha_id": str(candidate.alpha_id),
                "expression": str(candidate.expression),
                "is_metrics": {
                    "sharpe": candidate.sharpe,
                    "fitness": candidate.fitness,
                    "returns": candidate.returns,
                    "turnover": candidate.turnover,
                    "checks": [
                        {
                            "name": "SELF_CORRELATION",
                            "result": candidate.sc_status,
                            "value": candidate.self_correlation,
                        }
                    ],
                },
                "research_meta": {
                    "family": candidate.family,
                    "hypothesis": candidate.hypothesis,
                    "parent_expression": candidate.parent_expression,
                    "generation": candidate.generation,
                    "mutation_type": candidate.mutation_type,
                    "data_fields": list(candidate.data_fields or []),
                    "dataset_id": candidate.dataset_id,
                    "dataset_category": candidate.dataset_category,
                    "provenance_state": candidate.provenance_state,
                    "provenance_reason": candidate.provenance_reason,
                    "lineage_id": candidate.lineage_id,
                    "parent_lineage_id": candidate.parent_lineage_id,
                    "operator_pattern": candidate.operator_pattern,
                    "operators": list(candidate.operators or []),
                    "mutation_reason": candidate.mutation_reason,
                    "planner_strategy": candidate.planner_strategy,
                    "allocation_cell": candidate.allocation_cell,
                    "source_run_id": candidate.source_run_id,
                },
                "validation": validation,
                "local_correlation_proxy": local_proxy,
                "settings": {
                    "region": candidate.region,
                    "universe": candidate.universe,
                    "delay": candidate.delay,
                    "decay": candidate.decay,
                    "neutralization": candidate.neutralization,
                    "truncation": candidate.truncation,
                },
                "tag": candidate.tag,
            }
        )
    return payloads


async def reserve_submission(
    account: str,
    alpha_id: str,
    *,
    points: float | None = None,
    points_status: str | None = None,
) -> dict[str, Any]:
    """Consume one local daily submission slot before a remote formal submit."""
    alpha_id = str(alpha_id or "").strip()
    if not alpha_id:
        return {"allowed": False, "reason": "missing_alpha_id"}

    day = submission_day()
    factory = _get_session_factory()
    async with factory() as session:
        state = await _get_or_create_state(session, account)
        budget = _configured_daily_budget()
        state.daily_budget = budget
        await _expire_stale_reservations(session, account)

        unresolved_result = await session.execute(
            select(WQSubmissionAttempt.alpha_id).where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.status.in_(sorted(_RECONCILIATION_REQUIRED_STATUSES)),
            )
        )
        unresolved_ids = [str(value) for value in unresolved_result.scalars().all() if value]
        if unresolved_ids:
            await session.commit()
            return {
                "allowed": False,
                "reason": "submission_reconciliation_required",
                "alpha_id": alpha_id,
                "recovery_alpha_ids": sorted(set(unresolved_ids)),
                "detail": "unresolved prior formal submissions must be reconciled with BRAIN before reserving another slot",
            }

        readiness_result = await session.execute(
            select(WQResearchCandidate).where(
                WQResearchCandidate.account == account,
                WQResearchCandidate.alpha_id == alpha_id,
            )
        )
        readiness_candidate = readiness_result.scalar_one_or_none()
        if readiness_candidate is None:
            await session.commit()
            return {
                "allowed": False,
                "reason": "candidate_not_tracked",
                "alpha_id": alpha_id,
                "detail": "formal submission requires a tracked Candidate Queue entry",
            }
        strict_blockers = _candidate_submission_blockers(readiness_candidate)
        fallback_blockers = _candidate_fallback_submission_blockers(readiness_candidate)
        if fallback_blockers:
            await session.commit()
            return {
                "allowed": False,
                "reason": "candidate_not_ready",
                "alpha_id": alpha_id,
                "candidate_status": readiness_candidate.status,
                "validation_status": readiness_candidate.validation_status,
                "blockers": fallback_blockers,
                "strict_blockers": strict_blockers,
            }
        submission_mode = "strict_ready" if not strict_blockers else "active_target_fallback"

        prior_result = await session.execute(
            select(WQSubmissionAttempt)
            .where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.alpha_id == alpha_id,
            )
            .order_by(WQSubmissionAttempt.created_at.desc())
            .limit(1)
        )
        prior = prior_result.scalar_one_or_none()
        prior_status = str(prior.status or "").upper() if prior is not None else ""
        if prior is not None and prior_status != "UNSUBMITTED_CONFIRMED":
            return {
                "allowed": False,
                "reason": "alpha_already_submitted_or_pending",
                "alpha_id": alpha_id,
                "previous_status": prior.status,
                "submission_day": day,
                "daily_budget": budget,
            }

        count_result = await session.execute(
            select(func.count()).where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.submission_day == day,
                WQSubmissionAttempt.status.in_(sorted(_SLOT_CONSUMING_STATUSES)),
            )
        )
        used = int(count_result.scalar() or 0)
        if used >= budget:
            return {
                "allowed": False,
                "reason": "daily_active_target_filled_or_inflight",
                "alpha_id": alpha_id,
                "submission_day": day,
                "daily_active_target": budget,
                "daily_budget": budget,
                "used_slots": used,
                "remaining_slots": 0,
            }

        if prior is not None and prior_status == "UNSUBMITTED_CONFIRMED" and prior.submission_day == day:
            # Keep one audit row per Alpha/day. A platform-confirmed UNSUBMITTED
            # attempt did not consume a remote slot, so the same-day row can be
            # safely renewed. Cross-day retries get a new audit row.
            attempt = prior
            now = datetime.now(timezone.utc)
            attempt.status = "RESERVED"
            attempt.score_state = "PENDING"
            attempt.detail = "reservation renewed after platform-confirmed UNSUBMITTED"
            attempt.points_at_reservation = points
            attempt.points_status_at_reservation = points_status
            attempt.attributed_points_share = None
            attempt.attribution_confidence = None
            attempt.attribution_details = None
            attempt.settled_at = None
            attempt.created_at = now
            attempt.updated_at = now
        else:
            attempt = WQSubmissionAttempt(
                account=account,
                alpha_id=alpha_id,
                submission_day=day,
                status="RESERVED",
                score_state="PENDING",
                points_at_reservation=points,
                points_status_at_reservation=points_status,
            )
            session.add(attempt)

        readiness_candidate.status = "reserved"

        await session.commit()
        return {
            "allowed": True,
            "reason": "slot_reserved",
            "submission_mode": submission_mode,
            "soft_blockers_waived": strict_blockers if submission_mode == "active_target_fallback" else [],
            "alpha_id": alpha_id,
            "submission_day": day,
            "daily_active_target": budget,
            "daily_budget": budget,
            "used_slots": used + 1,
            "remaining_slots": max(0, budget - used - 1),
        }


def _authoritative_platform_check_failure(result: dict[str, Any]) -> str | None:
    """Return a named BRAIN check that explicitly rejected an unsubmitted Alpha."""
    try:
        status_code = int(result.get("status_code") or 0)
    except (TypeError, ValueError):
        status_code = 0
    if status_code != 403:
        return None
    detail = str(result.get("detail") or "")
    for match in re.finditer(
        r'"name"\s*:\s*"([^"]+)"\s*,\s*"result"\s*:\s*"FAIL"',
        detail,
        flags=re.IGNORECASE,
    ):
        check_name = str(match.group(1) or "").upper()
        if check_name and check_name != "SELF_CORRELATION":
            return check_name
    return None

def _normalized_submission_result_status(result: dict[str, Any]) -> str:
    """Map remote submission observations to fail-closed local ledger states."""
    explicit = str(result.get("final_status") or "").upper()
    platform_status = str(result.get("platform_status") or "").upper()
    detail = str(result.get("detail") or "")
    platform_check_failure = _authoritative_platform_check_failure(result)

    if explicit in {"UNSUBMITTED", "UNSUBMITTED_CONFIRMED"}:
        return "UNSUBMITTED_CONFIRMED"
    if explicit in {"ERROR", "TIMEOUT", "UNKNOWN", "SUBMIT_UNKNOWN"}:
        if platform_check_failure:
            return "OTHER_FAIL"
        return "SUBMIT_UNKNOWN"
    if explicit == "OTHER_FAIL" and not result.get("confirmed_not_submitted"):
        return "SUBMIT_UNKNOWN"
    if explicit:
        return explicit
    if result.get("ok"):
        return "ACTIVE"
    if "SC FAIL" in detail.upper():
        return "SC_FAIL"
    if result.get("confirmed_not_submitted") or platform_status == "UNSUBMITTED":
        return "UNSUBMITTED_CONFIRMED"
    return "SUBMIT_UNKNOWN"


def _monotonic_submission_status(current: str, incoming: str) -> str:
    """Prevent uncertain observations from downgrading authoritative terminal states."""
    current = str(current or "").upper()
    incoming = str(incoming or "").upper()
    if current == "ACTIVE":
        return "ACTIVE"
    if incoming == "ACTIVE":
        return "ACTIVE"
    if current == "SC_FAIL" and incoming != "ACTIVE":
        return "SC_FAIL"
    if incoming == "SC_FAIL":
        return "SC_FAIL"
    if current == "SC_PENDING" and incoming == "SUBMIT_UNKNOWN":
        return "SC_PENDING"
    return incoming or current or "SUBMIT_UNKNOWN"


async def finalize_submission_attempt(account: str, alpha_id: str, result: dict[str, Any]) -> None:
    """Persist a remote outcome without letting uncertainty release or corrupt a formal slot."""
    factory = _get_session_factory()
    async with factory() as session:
        attempt_result = await session.execute(
            select(WQSubmissionAttempt)
            .where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.alpha_id == alpha_id,
            )
            .order_by(WQSubmissionAttempt.created_at.desc())
            .limit(1)
        )
        attempt = attempt_result.scalar_one_or_none()
        if attempt is None:
            return

        observed_status = _normalized_submission_result_status(result)
        previous_status = str(attempt.status or "").upper()
        final_status = _monotonic_submission_status(previous_status, observed_status)
        transition_applied = final_status != previous_status or final_status == observed_status

        attempt.status = final_status
        if transition_applied:
            attempt.detail = str(result.get("detail") or "")[:4000]
        attempt.score_state = (
            "PENDING"
            if final_status in {"ACTIVE", "SC_PENDING", "SUBMIT_UNKNOWN"}
            else "NOT_ELIGIBLE"
        )

        candidate_result = await session.execute(
            select(WQResearchCandidate).where(
                WQResearchCandidate.account == account,
                WQResearchCandidate.alpha_id == alpha_id,
            )
        )
        candidate = candidate_result.scalar_one_or_none()
        if candidate is not None:
            if final_status in {"ACTIVE", "SC_PENDING"}:
                candidate.status = "submitted"
            elif final_status == "SUBMIT_UNKNOWN":
                candidate.status = "submission_unknown"
            elif final_status == "UNSUBMITTED_CONFIRMED":
                candidate.status = "validation_pending"
                candidate.validation_status = "platform_recheck"
            elif final_status == "SC_FAIL":
                candidate.status = "sc_fail"
            else:
                candidate.status = "submit_failed"

        if final_status in {"SC_FAIL", "OTHER_FAIL"}:
            trial_result = await session.execute(
                select(WQResearchTrial)
                .where(
                    WQResearchTrial.account == account,
                    WQResearchTrial.alpha_id == alpha_id,
                )
                .order_by(WQResearchTrial.created_at.desc())
                .limit(1)
            )
            trial = trial_result.scalar_one_or_none()
            if trial is not None:
                diagnostic = classify_research_failure(
                    {"final_status": final_status, "error": attempt.detail},
                    status="rejected",
                )
                trial.failure_stage = diagnostic["failure_stage"]
                trial.failure_reason = diagnostic["failure_reason"]
                trial.failure_reasons = list(diagnostic["failure_reasons"] or [])
                trial.failure_evidence = diagnostic["failure_evidence"]

        await session.commit()


async def observe_account_status(account: str, status: dict[str, Any]) -> dict[str, Any]:
    """Reconcile delayed leaderboard points only after BRAIN reports CURRENT."""
    points = status.get("points")
    points_status = str(status.get("points_status") or "UNAVAILABLE").upper()
    leaderboard = status.get("leaderboard") or {}
    raw_active_alpha_gap = leaderboard.get("active_alpha_gap")
    if points_status == "CURRENT" and raw_active_alpha_gap is None:
        # Reconciliation must independently require synchronization evidence;
        # never trust a CURRENT label without a known ACTIVE-vs-leaderboard gap.
        points_status = "SYNC_UNKNOWN"
    try:
        active_alpha_gap = max(0, int(raw_active_alpha_gap)) if raw_active_alpha_gap is not None else 0
    except (TypeError, ValueError):
        active_alpha_gap = 0
        if points_status == "CURRENT":
            points_status = "SYNC_UNKNOWN"
    try:
        points_value = float(points) if points is not None else None
    except (TypeError, ValueError):
        points_value = None

    factory = _get_session_factory()
    async with factory() as session:
        state = await _get_or_create_state(session, account)
        pending_result = await session.execute(
            select(WQSubmissionAttempt).where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.status == "ACTIVE",
                WQSubmissionAttempt.score_state == "PENDING",
            )
        )
        pending = list(pending_result.scalars().all())
        state.untracked_active_gap = max(0, active_alpha_gap - len(pending))

        if points_value is not None:
            if state.last_settled_points is None:
                state.last_settled_points = points_value
            state.last_observed_points = points_value

            if points_status == "CURRENT":
                baseline = float(state.last_settled_points or 0.0)
                if pending and points_value > baseline:
                    delta = points_value - baseline
                    cohort_ids = sorted(str(attempt.alpha_id) for attempt in pending)
                    attributed_share = delta / len(pending)
                    uncertainty_slots = max(0, int(state.untracked_active_gap or 0))
                    attribution_confidence = 1.0 / max(1, len(pending) + uncertainty_slots)
                    settled_at = datetime.now(timezone.utc)
                    active_times = [
                        value
                        for value in (_normalized_datetime(attempt.updated_at) for attempt in pending)
                        if value is not None
                    ]
                    assumptions = [
                        "leaderboard_current_closes_pending_local_active_cohort",
                        "equal_share_confidence_weighted_fallback",
                    ]
                    if uncertainty_slots:
                        assumptions.append("untracked_active_gap_reduces_confidence")
                    attribution_details = {
                        "cohort_alpha_ids": cohort_ids,
                        "cohort_size": len(cohort_ids),
                        "baseline_points": baseline,
                        "observed_points": points_value,
                        "delta": delta,
                        "points_status": points_status,
                        "active_alpha_gap": active_alpha_gap,
                        "sync_evidence": {
                            "active_alpha_gap_known": raw_active_alpha_gap is not None,
                            "active_alpha_gap": active_alpha_gap,
                        },
                        "untracked_active_gap": uncertainty_slots,
                        "first_active_at": min(active_times).isoformat() if active_times else None,
                        "last_active_at": max(active_times).isoformat() if active_times else None,
                        "settled_at": settled_at.isoformat(),
                        "attribution_rule": "equal_share_confidence_weighted",
                        "assumptions": assumptions,
                    }
                    for attempt in pending:
                        attempt.score_state = "SETTLED"
                        attempt.attributed_points_share = attributed_share
                        attempt.attribution_confidence = attribution_confidence
                        attempt.attribution_details = dict(attribution_details)
                        attempt.settled_at = settled_at
                    state.last_settled_delta = delta
                    state.last_settled_submission_count = len(pending)
                    # The daily target is user-configured and must not shrink just
                    # because one ACTIVE Alpha happened to settle a large point gain.
                    state.daily_budget = _configured_daily_budget()
                    state.last_settled_points = points_value
                elif not pending and points_value != state.last_settled_points:
                    # Catch up the baseline for score changes not created by this
                    # local ledger (e.g. submissions made before this feature).
                    state.last_settled_points = points_value

        state.last_points_status = points_status
        await session.commit()

    return await get_submission_policy_status(account)


async def get_submission_policy_status(account: str = "primary") -> dict[str, Any]:
    day = submission_day()
    factory = _get_session_factory()
    async with factory() as session:
        state = await _get_or_create_state(session, account)
        expired_reservation_ids = await _expire_stale_reservations(session, account)
        recovery_result = await session.execute(
            select(WQSubmissionAttempt.alpha_id).where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.status.in_(sorted(_RECONCILIATION_REQUIRED_STATUSES)),
            )
        )
        reconciliation_required_ids = sorted({str(value) for value in recovery_result.scalars().all() if value})

        attempts_result = await session.execute(
            select(func.count()).where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.submission_day == day,
                WQSubmissionAttempt.status.in_(sorted(_SLOT_CONSUMING_STATUSES)),
            )
        )
        used_slots = int(attempts_result.scalar() or 0)
        active_today_result = await session.execute(
            select(func.count()).where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.submission_day == day,
                WQSubmissionAttempt.status == "ACTIVE",
            )
        )
        active_today = int(active_today_result.scalar() or 0)
        inflight_today = max(0, used_slots - active_today)
        failed_result = await session.execute(
            select(func.count()).where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.submission_day == day,
                WQSubmissionAttempt.status.in_(["SC_FAIL", "OTHER_FAIL"]),
            )
        )
        failed_attempts = int(failed_result.scalar() or 0)
        daily_budget = _configured_daily_budget()
        state.daily_budget = daily_budget

        pending_result = await session.execute(
            select(func.count()).where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.score_state == "PENDING",
                WQSubmissionAttempt.status.in_(sorted(_SLOT_CONSUMING_STATUSES)),
            )
        )
        pending_score = int(pending_result.scalar() or 0)
        conversion_feedback = await _load_conversion_feedback(session, account)
        active_gate = active_feedback_gate(conversion_feedback)
        # Candidate Evidence Skill: official eligibility is enforced separately;
        # preference uses calibrated ACTIVE probability only when mature, then a
        # transparent lexicographic BRAIN-metric hierarchy. No opaque weighted sum.
        candidate_order = [
            WQResearchCandidate.fitness.desc(),
            WQResearchCandidate.sharpe.desc(),
            WQResearchCandidate.returns.desc(),
            WQResearchCandidate.created_at.asc(),
        ]
        if active_gate["ready"]:
            candidate_order.insert(0, WQResearchCandidate.active_probability.desc().nullslast())

        queue_result = await session.execute(
            select(WQResearchCandidate)
            .where(
                WQResearchCandidate.account == account,
                WQResearchCandidate.status == "queued",
            )
            .order_by(*candidate_order)
        )
        queued_candidates = list(queue_result.scalars().all())
        queued_candidates.sort(
            key=lambda candidate: (not _candidate_research_ready(candidate),),
        )
        queue_count = len(queued_candidates)

        submission_pool_result = await session.execute(
            select(WQResearchCandidate)
            .where(
                WQResearchCandidate.account == account,
                WQResearchCandidate.status.in_(["queued", "validation_pending", "robustness_fail"]),
            )
            .order_by(*candidate_order)
        )
        submission_candidates = list(submission_pool_result.scalars().all())
        submission_candidates.sort(
            key=lambda candidate: (
                not _candidate_research_ready(candidate),
                not _candidate_fallback_submission_ready(candidate),
                -(float(candidate.active_probability) if active_gate["ready"] and candidate.active_probability is not None else -1.0),
                -(float(candidate.fitness or 0.0)),
                -(float(candidate.sharpe or 0.0)),
                -(float(candidate.returns or 0.0)),
            ),
        )

        def candidate_payload(item: WQResearchCandidate) -> dict[str, Any]:
            strict_ready = _candidate_research_ready(item)
            fallback_ready = _candidate_fallback_submission_ready(item)
            return {
                "alpha_id": item.alpha_id,
                "fitness": item.fitness,
                "sharpe": item.sharpe,
                "turnover": item.turnover,
                "priority_score": item.priority_score,
                "active_probability": item.active_probability,
                "confidence_tier": item.confidence_tier,
                "probability_support": int(item.probability_support or 0),
                "probability_provenance": item.probability_provenance,
                "calibration_details": item.calibration_details,
                "family": item.family,
                "generation": item.generation,
                "mutation_type": item.mutation_type,
                "validation_status": item.validation_status,
                "robustness_score": item.robustness_score,
                "novelty_score": item.novelty_score,
                "self_correlation": item.self_correlation,
                "sc_status": item.sc_status,
                "local_correlation_proxy": {
                    "status": "available" if item.local_correlation is not None else "unavailable",
                    "max_correlation": item.local_correlation,
                    "matching_alpha_id": item.local_correlation_alpha_id,
                    "sample_length": item.local_correlation_samples or 0,
                    "calculated_at": item.local_correlation_at.isoformat() if item.local_correlation_at else None,
                    "official_sc": False,
                },
                "data_fields": list(item.data_fields or []),
                "dataset_id": item.dataset_id,
                "dataset_category": item.dataset_category,
                "provenance_state": item.provenance_state,
                "provenance_reason": item.provenance_reason,
                "research_readiness_eligible": strict_ready,
                "submission_blockers": _candidate_submission_blockers(item),
                "fallback_submission_eligible": fallback_ready,
                "fallback_submission_blockers": _candidate_fallback_submission_blockers(item),
                "submission_mode": "strict_ready" if strict_ready else ("active_target_fallback" if fallback_ready else "blocked"),
                "tag": item.tag,
            }

        top_candidates = [candidate_payload(item) for item in queued_candidates[:5]]
        fallback_submission_candidates = [
            item for item in submission_candidates if _candidate_fallback_submission_ready(item)
        ]
        submission_candidate_top = [candidate_payload(item) for item in fallback_submission_candidates[:10]]

        confidence_counts = {"S": 0, "A": 0, "B": 0}
        ready_confidence_counts = {"S": 0, "A": 0, "B": 0}
        calibrated_high_confidence_count = 0
        stale_high_confidence_count = 0
        research_readiness_eligible_count = 0
        for candidate in queued_candidates:
            tier = str(candidate.confidence_tier or "B").upper()
            if tier not in confidence_counts:
                tier = "B"
            confidence_counts[tier] += 1
            if str(candidate.validation_status or "").lower() in {"ready", "evidence_collected"}:
                ready_confidence_counts[tier] += 1
                if _candidate_research_ready(candidate):
                    research_readiness_eligible_count += 1
                if tier in {"S", "A"}:
                    if _candidate_is_fresh(candidate):
                        calibrated_high_confidence_count += 1
                    else:
                        stale_high_confidence_count += 1

        eligibility_mode = "calibrated_active_probability" if active_gate["ready"] else "research_readiness"
        high_confidence_count = calibrated_high_confidence_count if active_gate["ready"] else research_readiness_eligible_count
        inventory_policy = _configured_inventory_policy()
        inventory_floor = int(inventory_policy["floor"])
        target_low = int(inventory_policy["target_low"])
        target_high = int(inventory_policy["target_high"])
        inventory_deficit = max(0, inventory_floor - high_confidence_count)
        if high_confidence_count < inventory_floor:
            research_mode = "REPLENISHMENT"
        elif high_confidence_count > target_high:
            research_mode = "EXPLORATION"
        else:
            research_mode = "NORMAL"

        last_points_status = str(state.last_points_status or "").upper()
        leaderboard_lagging = last_points_status == "LEADERBOARD_LAGGING"
        points_sync_unknown = last_points_status == "SYNC_UNKNOWN"
        untracked_active_gap = int(state.untracked_active_gap or 0)
        points_score_unchanged_pending = (
            pending_score > 0
            and last_points_status == "CURRENT"
            and state.last_observed_points is not None
            and state.last_settled_points is not None
            and float(state.last_observed_points) <= float(state.last_settled_points)
        )
        if reconciliation_required_ids:
            submission_warning = "submission_reconciliation_required_before_new_formal_submit"
        elif points_sync_unknown:
            submission_warning = "points_sync_unknown_alpha_counts_unavailable"
        elif points_score_unchanged_pending:
            submission_warning = "leaderboard_alpha_count_current_but_score_unchanged_pending_points"
        elif leaderboard_lagging and untracked_active_gap > 0:
            submission_warning = "leaderboard_lagging_with_untracked_active_alphas"
        else:
            submission_warning = None
        submission_frozen = False
        budget_remaining = max(0, daily_budget - used_slots)
        remaining_active_target = max(0, daily_budget - active_today)
        daily_active_target_met = remaining_active_target == 0
        inventory_status = {
            "floor": inventory_floor,
            "target_low": target_low,
            "target_high": target_high,
            "freshness_hours": float(inventory_policy["freshness_hours"]),
            "high_confidence_count": high_confidence_count,
            "calibrated_high_confidence_count": calibrated_high_confidence_count,
            "research_readiness_eligible_count": research_readiness_eligible_count,
            "eligibility_mode": eligibility_mode,
            "stale_high_confidence_count": stale_high_confidence_count,
            "deficit": inventory_deficit,
            "mode": research_mode,
            "tier_counts": confidence_counts,
            "ready_tier_counts": ready_confidence_counts,
            "queue_count": queue_count,
            "top_candidates": top_candidates,
            "active_outcome_learning_gate": active_gate,
        }

        await session.commit()
        return {
            "submission_day": day,
            "submission_timezone": os.environ.get("WQ_SUBMISSION_TIMEZONE", "Asia/Shanghai") or "Asia/Shanghai",
            "daily_active_target": daily_budget,
            "daily_active_count": active_today,
            "daily_active_target_met": daily_active_target_met,
            "remaining_active_target": remaining_active_target,
            "inflight_submission_count": inflight_today,
            "daily_submission_budget": daily_budget,
            "used_submission_slots": used_slots,
            "failed_submission_attempts": failed_attempts,
            "budget_remaining_slots": budget_remaining,
            "remaining_submission_slots": budget_remaining,
            "pending_score_submissions": pending_score,
            "expired_reservation_ids": expired_reservation_ids,
            "submission_reconciliation_required": bool(reconciliation_required_ids),
            "submission_reconciliation_alpha_ids": reconciliation_required_ids,
            "reservation_ttl_seconds": _configured_reservation_ttl_seconds(),
            "candidate_queue_count": queue_count,
            "candidate_confidence_counts": confidence_counts,
            "candidate_confidence_thresholds": {
                "S": configured_confidence_thresholds()[0],
                "A": configured_confidence_thresholds()[1],
            },
            "candidate_queue_top": top_candidates,
            "submission_candidate_top": submission_candidate_top,
            "fallback_submission_candidate_count": len(fallback_submission_candidates),
            "high_confidence_candidate_count": high_confidence_count,
            "research_readiness_eligible_count": research_readiness_eligible_count,
            "candidate_eligibility_mode": eligibility_mode,
            "active_outcome_learning_gate": active_gate,
            "inventory_deficit": inventory_deficit,
            "research_mode": research_mode,
            "inventory": inventory_status,
            "last_observed_points": state.last_observed_points,
            "last_points_status": state.last_points_status,
            "last_settled_points": state.last_settled_points,
            "last_settled_delta": state.last_settled_delta,
            "last_settled_submission_count": state.last_settled_submission_count,
            "untracked_active_gap": untracked_active_gap,
            "leaderboard_lagging": leaderboard_lagging,
            "points_sync_unknown": points_sync_unknown,
            "points_score_unchanged_pending": points_score_unchanged_pending,
            "submission_warning": submission_warning,
            "submission_frozen": submission_frozen,
            "rule": "target_two_active_per_day; official_brain_checks_hard; local_robustness_correlation_overfit_soft_for_target_fill; uncertain_submissions_fail_closed",
        }


def _run_coro_sync(coro):
    from . import task_store

    loop = task_store.main_loop
    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=20)
    return asyncio.run(coro)


def get_candidate_robustness_revalidation_payloads_sync(
    account: str,
    alpha_ids: list[str],
) -> list[dict[str, Any]]:
    return _run_coro_sync(get_candidate_robustness_revalidation_payloads(account, alpha_ids))


def get_submission_reservation_recovery_ids_sync(account: str = "primary", *, limit: int = 20) -> list[str]:
    return _run_coro_sync(get_submission_reservation_recovery_ids(account, limit=limit))


def reconcile_submission_reservations_sync(
    account: str,
    platform_alphas: dict[str, dict[str, Any]],
) -> int:
    return _run_coro_sync(reconcile_submission_reservations(account, platform_alphas))


def reconcile_candidate_platform_statuses_sync(
    account: str,
    platform_alphas: dict[str, dict[str, Any]],
) -> int:
    return _run_coro_sync(reconcile_candidate_platform_statuses(account, platform_alphas))


def reserve_submission_sync(account: str, alpha_id: str) -> dict[str, Any]:
    # The lock keeps count+insert atomic for concurrent local worker threads.
    with _reservation_lock:
        return _run_coro_sync(reserve_submission(account, alpha_id))


def finalize_submission_attempt_sync(account: str, alpha_id: str, result: dict[str, Any]) -> None:
    _run_coro_sync(finalize_submission_attempt(account, alpha_id, result))


def record_research_candidates_sync(
    account: str,
    candidates: list[dict[str, Any]],
    *,
    settings: dict[str, Any] | None = None,
    tag: str | None = None,
) -> int:
    return _run_coro_sync(record_research_candidates(account, candidates, settings=settings, tag=tag))
