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
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, select

from .db import _get_session_factory
from .models import WQResearchCandidate, WQResearchTrial, WQSubmissionAttempt, WQSubmissionState
from .wq_candidate_calibration import calibrate_active_probability, calibration_report
from .wq_correlation_proxy import correlation_priority_multiplier
from .wq_failure_taxonomy import classify_research_failure
from .wq_lineage import lineage_id_for, parent_lineage_id_for, recover_research_metadata
from .wq_overfitting import overfitting_priority_multiplier
from .wq_research_scheduler import research_cell_key

_DEFAULT_DAILY_BUDGET = 2
_MAX_CONFIGURED_BUDGET = 5
_TERMINAL_POSITIVE_STATUSES = {"ACTIVE"}
_TERMINAL_NEGATIVE_STATUSES = {"SC_FAIL", "OTHER_FAIL"}
_CONVERSION_PRIOR_ALPHA = 2.0
_CONVERSION_PRIOR_BETA = 2.0
_CONVERSION_GROUP_PRIOR_STRENGTH = 4.0
_reservation_lock = threading.Lock()


def _configured_daily_budget() -> int:
    try:
        value = int(os.environ.get("WQ_DAILY_SUBMISSION_BUDGET", str(_DEFAULT_DAILY_BUDGET)))
    except (TypeError, ValueError):
        value = _DEFAULT_DAILY_BUDGET
    return max(1, min(_MAX_CONFIGURED_BUDGET, value))


def _submission_timezone():
    name = os.environ.get("WQ_SUBMISSION_TIMEZONE", "UTC").strip() or "UTC"
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
    metrics = candidate.get("is_metrics") or {}

    def number(key: str) -> float:
        try:
            return float(metrics.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    raw_checks = list(metrics.get("checks") or [])
    checks = {
        str(check.get("name") or "").upper(): str(check.get("result") or "").upper()
        for check in raw_checks
    }
    sc_check = next((check for check in raw_checks if str(check.get("name") or "").upper() == "SELF_CORRELATION"), None)
    if checks.get("SELF_CORRELATION") == "FAIL":
        return -10000.0
    try:
        self_correlation = float((sc_check or {}).get("value"))
    except (TypeError, ValueError):
        self_correlation = 0.0

    turnover = number("turnover")
    turnover_penalty = 0.0
    if turnover > 0.5:
        turnover_penalty += (turnover - 0.5) * 60.0
    elif 0 < turnover < 0.03:
        turnover_penalty += (0.03 - turnover) * 60.0

    expression = str(candidate.get("expression") or "")
    complexity_penalty = max(0, expression.count("(") - 8) * 1.5
    raw_validation = candidate.get("validation") or {}
    validation = raw_validation if isinstance(raw_validation, dict) else {}
    try:
        robustness = float(validation.get("robustness_score") or 0.0)
    except (TypeError, ValueError):
        robustness = 0.0
    try:
        novelty = float(candidate.get("novelty_score") or 0.0)
    except (TypeError, ValueError):
        novelty = 0.0
    score = (
        number("fitness") * 100.0
        + number("sharpe") * 20.0
        + number("returns") * 50.0
        + robustness * 20.0
        + novelty * 10.0
        - max(0.0, self_correlation) * 20.0
        - turnover_penalty
        - complexity_penalty
    )
    overfitting_multiplier = overfitting_priority_multiplier(validation.get("overfitting_evidence"))
    return score * correlation_priority_multiplier(candidate.get("local_correlation_proxy")) * overfitting_multiplier


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
        for mapping, key in (
            (family_counts, str(candidate.family or "unknown")),
            (dataset_counts, str(candidate.dataset_id or "unknown")),
            (cell_counts, cell_key),
        ):
            values = mapping.setdefault(key, [0, 0])
            values[0] += is_positive
            values[1] += 1
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
            try:
                self_correlation = float((sc_check or {}).get("value"))
            except (TypeError, ValueError):
                self_correlation = None
            sc_status = str((sc_check or {}).get("result") or "").upper() or None
            raw_local_correlation = candidate.get("local_correlation_proxy") or validation.get("local_correlation_proxy") or {}
            local_proxy = raw_local_correlation if isinstance(raw_local_correlation, dict) else {}
            try:
                local_correlation = float(local_proxy.get("max_correlation"))
            except (TypeError, ValueError):
                local_correlation = None
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
            active_probability = float(calibration["probability"])
            priority_score = _priority_score(candidate) * (0.75 + active_probability * 0.5)
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
                "confidence_tier": calibration["tier"],
                "probability_support": int(calibration["support"]),
                "probability_provenance": str(calibration["provenance"]),
                "calibration_details": calibration,
                "family": meta.get("family"),
                "hypothesis": meta.get("hypothesis"),
                "parent_expression": meta.get("parent_expression"),
                "generation": int(meta.get("generation") or 0),
                "mutation_type": meta.get("mutation_type"),
                "structure_signature": structure_signature,
                "data_fields": list(meta.get("data_fields") or []),
                "dataset_id": str(meta.get("dataset_id") or "") or None,
                "lineage_id": lineage_id,
                "parent_lineage_id": parent_lineage_id,
                "operator_pattern": str(meta.get("operator_pattern") or "") or None,
                "operators": list(meta.get("operators") or []),
                "mutation_reason": str(meta.get("mutation_reason") or "") or None,
                "planner_strategy": str(meta.get("planner_strategy") or "") or None,
                "allocation_cell": str(meta.get("allocation_cell") or "") or None,
                "source_run_id": str(meta.get("source_run_id") or candidate.get("run_id") or candidate.get("task_id") or "") or None,
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
            elif validation_status in {"validation_pending", "robustness_unavailable"}:
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
        for alpha_id, info in platform_alphas.items():
            platform_status = str((info or {}).get("status") or "").upper()
            sc_result = str((info or {}).get("sc_result") or "").upper()
            try:
                sc_value = float((info or {}).get("sc_value"))
            except (TypeError, ValueError):
                sc_value = None
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
                    WQResearchCandidate.status.in_(["queued", "reserved", "validation_pending", "robustness_fail"]),
                )
            )
            candidate = result.scalar_one_or_none()
            if candidate is not None:
                if target_status is not None:
                    candidate.status = target_status
                candidate.sc_status = sc_result or candidate.sc_status
                if sc_value is not None:
                    candidate.self_correlation = sc_value
                candidate.priority_score = _priority_score(
                    {
                        "expression": candidate.expression,
                        "is_metrics": {
                            "fitness": candidate.fitness,
                            "sharpe": candidate.sharpe,
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
                        "validation": {"robustness_score": candidate.robustness_score},
                        "novelty_score": candidate.novelty_score,
                    }
                )
                updated += 1
        await session.commit()
    return updated


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
        budget = max(1, int(state.daily_budget or _configured_daily_budget()))
        if str(state.last_points_status or "").upper() == "LEADERBOARD_LAGGING" and int(state.untracked_active_gap or 0) > 0:
            return {
                "allowed": False,
                "reason": "leaderboard_lagging_with_untracked_submissions",
                "alpha_id": alpha_id,
                "submission_day": day,
                "daily_budget": budget,
                "untracked_active_gap": int(state.untracked_active_gap or 0),
            }

        readiness_result = await session.execute(
            select(WQResearchCandidate).where(
                WQResearchCandidate.account == account,
                WQResearchCandidate.alpha_id == alpha_id,
            )
        )
        readiness_candidate = readiness_result.scalar_one_or_none()
        if readiness_candidate is not None and readiness_candidate.status in {"robustness_fail", "validation_pending"}:
            return {
                "allowed": False,
                "reason": "candidate_not_ready",
                "alpha_id": alpha_id,
                "candidate_status": readiness_candidate.status,
                "validation_status": readiness_candidate.validation_status,
            }

        prior_result = await session.execute(
            select(WQSubmissionAttempt)
            .where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.alpha_id == alpha_id,
                WQSubmissionAttempt.status.in_(["RESERVED", "ACTIVE", "SC_PENDING", "SC_FAIL"]),
            )
            .order_by(WQSubmissionAttempt.created_at.desc())
            .limit(1)
        )
        prior = prior_result.scalar_one_or_none()
        if prior is not None:
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
            )
        )
        used = int(count_result.scalar() or 0)
        if used >= budget:
            return {
                "allowed": False,
                "reason": "daily_submission_budget_exhausted",
                "alpha_id": alpha_id,
                "submission_day": day,
                "daily_budget": budget,
                "used_slots": used,
                "remaining_slots": 0,
            }

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

        candidate_result = await session.execute(
            select(WQResearchCandidate).where(
                WQResearchCandidate.account == account,
                WQResearchCandidate.alpha_id == alpha_id,
            )
        )
        candidate = candidate_result.scalar_one_or_none()
        if candidate is not None:
            candidate.status = "reserved"

        await session.commit()
        return {
            "allowed": True,
            "reason": "slot_reserved",
            "alpha_id": alpha_id,
            "submission_day": day,
            "daily_budget": budget,
            "used_slots": used + 1,
            "remaining_slots": max(0, budget - used - 1),
        }


async def finalize_submission_attempt(account: str, alpha_id: str, result: dict[str, Any]) -> None:
    """Persist the remote outcome for a previously reserved submission slot."""
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

        final_status = str(result.get("final_status") or "").upper()
        if not final_status:
            detail = str(result.get("detail") or "")
            if result.get("ok"):
                final_status = "ACTIVE"
            elif "SC FAIL" in detail.upper():
                final_status = "SC_FAIL"
            elif str(result.get("platform_status") or "").upper() == "TIMEOUT":
                final_status = "SC_PENDING"
            else:
                final_status = "OTHER_FAIL"

        attempt.status = final_status
        attempt.detail = str(result.get("detail") or "")[:4000]
        attempt.score_state = "PENDING" if final_status in {"ACTIVE", "SC_PENDING"} else "NOT_ELIGIBLE"

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
    try:
        active_alpha_gap = max(0, int(leaderboard.get("active_alpha_gap") or 0))
    except (TypeError, ValueError):
        active_alpha_gap = 0
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
                if pending:
                    baseline = float(state.last_settled_points or 0.0)
                    delta = max(0.0, points_value - baseline)
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
                    # Only auto-calibrate within the conservative 1..2 range and
                    # only after the leaderboard has caught up with active Alphas.
                    if len(pending) == 1 and delta >= 1800:
                        state.daily_budget = 1
                    else:
                        state.daily_budget = min(2, _configured_daily_budget())
                    state.last_settled_points = points_value
                elif points_value != state.last_settled_points:
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

        attempts_result = await session.execute(
            select(func.count()).where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.submission_day == day,
            )
        )
        used_slots = int(attempts_result.scalar() or 0)
        daily_budget = max(1, int(state.daily_budget or _configured_daily_budget()))

        pending_result = await session.execute(
            select(func.count()).where(
                WQSubmissionAttempt.account == account,
                WQSubmissionAttempt.score_state == "PENDING",
                WQSubmissionAttempt.status.in_(["RESERVED", "ACTIVE", "SC_PENDING"]),
            )
        )
        pending_score = int(pending_result.scalar() or 0)

        queue_count_result = await session.execute(
            select(func.count()).where(
                WQResearchCandidate.account == account,
                WQResearchCandidate.status == "queued",
            )
        )
        queue_count = int(queue_count_result.scalar() or 0)

        queue_result = await session.execute(
            select(WQResearchCandidate)
            .where(
                WQResearchCandidate.account == account,
                WQResearchCandidate.status == "queued",
            )
            .order_by(WQResearchCandidate.priority_score.desc(), WQResearchCandidate.created_at.asc())
            .limit(5)
        )
        top_candidates = [
            {
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
                "tag": item.tag,
            }
            for item in queue_result.scalars().all()
        ]

        submission_frozen = bool(
            str(state.last_points_status or "").upper() == "LEADERBOARD_LAGGING"
            and int(state.untracked_active_gap or 0) > 0
        )
        budget_remaining = max(0, daily_budget - used_slots)

        await session.commit()
        return {
            "submission_day": day,
            "submission_timezone": os.environ.get("WQ_SUBMISSION_TIMEZONE", "UTC") or "UTC",
            "daily_submission_budget": daily_budget,
            "used_submission_slots": used_slots,
            "budget_remaining_slots": budget_remaining,
            "remaining_submission_slots": 0 if submission_frozen else budget_remaining,
            "pending_score_submissions": pending_score,
            "candidate_queue_count": queue_count,
            "candidate_queue_top": top_candidates,
            "last_observed_points": state.last_observed_points,
            "last_points_status": state.last_points_status,
            "last_settled_points": state.last_settled_points,
            "last_settled_delta": state.last_settled_delta,
            "last_settled_submission_count": state.last_settled_submission_count,
            "untracked_active_gap": int(state.untracked_active_gap or 0),
            "submission_frozen": submission_frozen,
            "rule": "points_are_feedback_not_realtime_rate_limit",
        }


def _run_coro_sync(coro):
    from . import task_store

    loop = task_store.main_loop
    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=20)
    return asyncio.run(coro)


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
