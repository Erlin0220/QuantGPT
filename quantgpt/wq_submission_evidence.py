"""Shared pre-submission evidence gates for WorldQuant candidates."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from .wq_correlation_proxy import correlation_evidence_is_fresh

_DEFAULT_SUBMISSION_MAX_LOCAL_CORRELATION = 0.70
_DEFAULT_SUBMISSION_MIN_OVERFIT_SCORE = 0.50


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def configured_submission_evidence_policy() -> dict[str, float]:
    try:
        max_local_correlation = float(
            os.environ.get("WQ_SUBMISSION_MAX_LOCAL_CORRELATION", str(_DEFAULT_SUBMISSION_MAX_LOCAL_CORRELATION))
        )
    except (TypeError, ValueError):
        max_local_correlation = _DEFAULT_SUBMISSION_MAX_LOCAL_CORRELATION
    try:
        min_overfit_score = float(
            os.environ.get("WQ_SUBMISSION_MIN_OVERFIT_SCORE", str(_DEFAULT_SUBMISSION_MIN_OVERFIT_SCORE))
        )
    except (TypeError, ValueError):
        min_overfit_score = _DEFAULT_SUBMISSION_MIN_OVERFIT_SCORE
    return {
        "max_local_correlation": max(0.0, min(1.0, max_local_correlation)),
        "min_overfit_score": max(0.0, min(1.0, min_overfit_score)),
    }


def submission_evidence_blockers(
    local_evidence: dict[str, Any] | None,
    overfitting: dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> list[str]:
    blockers: list[str] = []
    evidence_policy = configured_submission_evidence_policy()
    local_evidence = local_evidence if isinstance(local_evidence, dict) else {}
    local_correlation = _optional_float(local_evidence.get("max_correlation"))
    if (
        local_correlation is not None
        and correlation_evidence_is_fresh(local_evidence, now=now)
        and local_correlation >= evidence_policy["max_local_correlation"]
    ):
        blockers.append("local_correlation_high")

    overfitting = overfitting if isinstance(overfitting, dict) else {}
    if str(overfitting.get("status") or "").lower() == "available":
        overfit_score = _optional_float(overfitting.get("score"))
        if overfit_score is not None and overfit_score < evidence_policy["min_overfit_score"]:
            blockers.append("overfitting_evidence_weak")
    return blockers
