"""Evidence gates for sparse WorldQuant research-learning feedback."""
from __future__ import annotations

import os
from typing import Any


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(0.0, min(1.0, value))


def configured_learning_thresholds() -> dict[str, int | float]:
    return {
        "active_terminal_samples": _env_int("WQ_ACTIVE_DECISION_MIN_SAMPLES", 20, 5),
        "points_usable_attempts": _env_int("WQ_POINTS_DECISION_MIN_ATTEMPTS", 2, 1),
        "scheduler_trials": _env_int("WQ_SCHEDULER_ADAPTIVE_MIN_TRIALS", 50, 10),
        "scheduler_provenance_rate": _env_float("WQ_SCHEDULER_MIN_PROVENANCE_RATE", 0.80),
    }


def active_feedback_gate(feedback: dict[str, Any] | None) -> dict[str, Any]:
    feedback = feedback or {}
    samples = int(((feedback.get("global") or {}).get("samples")) or 0)
    minimum = int(configured_learning_thresholds()["active_terminal_samples"])
    ready = samples >= minimum
    return {
        "status": "ready" if ready else "cold_start",
        "ready": ready,
        "samples": samples,
        "minimum_samples": minimum,
        "decision_weight_enabled": ready,
        "reason": None if ready else "insufficient_terminal_active_outcomes",
    }


def points_feedback_gate(coverage: dict[str, Any] | None) -> dict[str, Any]:
    coverage = coverage or {}
    usable = int(coverage.get("usable_attempts") or 0)
    settled = int(coverage.get("settled_attempts") or 0)
    minimum = int(configured_learning_thresholds()["points_usable_attempts"])
    ready = settled > 0 and usable >= minimum
    return {
        "status": "ready" if ready else "cold_start",
        "ready": ready,
        "settled_attempts": settled,
        "usable_attempts": usable,
        "minimum_usable_attempts": minimum,
        "decision_weight_enabled": ready,
        "reason": None if ready else ("no_settled_points" if settled == 0 else "insufficient_usable_points_feedback"),
    }


def scheduler_learning_gate(
    *,
    trials: int,
    provenance_resolved: int,
    active_gate: dict[str, Any],
) -> dict[str, Any]:
    thresholds = configured_learning_thresholds()
    trials = max(0, int(trials))
    provenance_resolved = max(0, int(provenance_resolved))
    provenance_rate = provenance_resolved / max(1, trials)
    reasons: list[str] = []
    if trials < int(thresholds["scheduler_trials"]):
        reasons.append("insufficient_research_trials")
    if provenance_rate < float(thresholds["scheduler_provenance_rate"]):
        reasons.append("insufficient_resolved_provenance")
    if not bool(active_gate.get("ready")):
        reasons.append("active_outcome_learning_not_mature")
    ready = not reasons
    return {
        "status": "ready" if ready else "cold_start",
        "ready": ready,
        "trials": trials,
        "minimum_trials": int(thresholds["scheduler_trials"]),
        "provenance_resolved": provenance_resolved,
        "provenance_rate": round(provenance_rate, 4),
        "minimum_provenance_rate": float(thresholds["scheduler_provenance_rate"]),
        "reasons": reasons,
        "policy": "adaptive" if ready else "coverage_first",
    }


def build_learning_maturity(
    *,
    trials: int,
    provenance_resolved: int,
    active_feedback: dict[str, Any] | None,
    points_coverage: dict[str, Any] | None,
) -> dict[str, Any]:
    active = active_feedback_gate(active_feedback)
    points = points_feedback_gate(points_coverage)
    scheduler = scheduler_learning_gate(
        trials=trials,
        provenance_resolved=provenance_resolved,
        active_gate=active,
    )
    return {
        "active_outcome_weighting": active,
        "points_planner_weighting": points,
        "scheduler_adaptation": scheduler,
        "high_capacity_models_deferred": ["RUDDER", "ARES", "QR_DQN"],
    }
