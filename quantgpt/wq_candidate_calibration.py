"""Outcome-grounded ACTIVE probability calibration for WQ candidates.

Candidate metrics are evidence, not a probability model. A numeric P(ACTIVE) is
only emitted after enough resolved formal submission outcomes exist to calibrate
against. During cold start the probability is intentionally unavailable.
"""
from __future__ import annotations

import os
from typing import Any

from .wq_learning_maturity import active_feedback_gate
from .wq_research_scheduler import research_cell_key


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def configured_confidence_thresholds() -> tuple[float, float]:
    """Presentation-only tiers; they do not define candidate ordering."""
    try:
        s_threshold = float(os.environ.get("WQ_CONFIDENCE_TIER_S", "0.70"))
    except ValueError:
        s_threshold = 0.70
    try:
        a_threshold = float(os.environ.get("WQ_CONFIDENCE_TIER_A", "0.50"))
    except ValueError:
        a_threshold = 0.50
    s_threshold = _clamp01(s_threshold)
    a_threshold = min(s_threshold, _clamp01(a_threshold))
    return s_threshold, a_threshold


def confidence_tier(probability: float) -> str:
    s_threshold, a_threshold = configured_confidence_thresholds()
    if probability >= s_threshold:
        return "S"
    if probability >= a_threshold:
        return "A"
    return "B"


def quality_baseline(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return transparent candidate evidence without converting it to probability."""
    metrics = candidate.get("is_metrics") or {}
    validation = candidate.get("validation") or {}
    if not isinstance(validation, dict):
        validation = {}
    checks = list(metrics.get("checks") or []) if isinstance(metrics, dict) else []
    sc_check = next((check for check in checks if str(check.get("name") or "").upper() == "SELF_CORRELATION"), {})
    if str(sc_check.get("result") or "").upper() == "FAIL":
        return {
            "score": None,
            "components": {"official_self_correlation": "FAIL"},
            "hard_fail": "official_self_correlation",
            "method": "evidence_hierarchy_no_probability",
        }

    evidence_policy = validation.get("candidate_evidence_policy")
    if not isinstance(evidence_policy, dict):
        evidence_policy = (candidate.get("research_meta") or {}).get("candidate_evidence_policy") or {}
    return {
        "score": None,
        "components": {
            "fitness": _optional_float(metrics.get("fitness")),
            "sharpe": _optional_float(metrics.get("sharpe")),
            "returns": _optional_float(metrics.get("returns")),
            "turnover": _optional_float(metrics.get("turnover")),
            "official_self_correlation": sc_check.get("value"),
            "robustness_status": validation.get("status"),
            "robustness_score": validation.get("robustness_score"),
            "overfitting_evidence": validation.get("overfitting_evidence"),
            "local_correlation_proxy": candidate.get("local_correlation_proxy") or validation.get("local_correlation_proxy"),
            "novelty_score": candidate.get("novelty_score"),
        },
        "hard_fail": None,
        "method": "evidence_hierarchy_no_probability",
        "candidate_evidence_policy": dict(evidence_policy) if isinstance(evidence_policy, dict) else {},
    }


def _feedback_choice(candidate: dict[str, Any], feedback: dict[str, Any]) -> tuple[float, int, str]:
    """Choose the most specific empirically supported smoothed ACTIVE rate."""
    global_feedback = feedback.get("global") or {}
    global_rate = _safe_float(global_feedback.get("rate"), 0.5)
    global_samples = int(global_feedback.get("samples") or 0)
    meta = candidate.get("research_meta") or {}
    family = str(meta.get("family") or candidate.get("family") or "unknown")
    dataset = str(meta.get("dataset_id") or candidate.get("dataset_id") or "unknown")
    cell_key = research_cell_key(
        {
            "family": family,
            "dataset_id": dataset,
            "operator_pattern": meta.get("operator_pattern") or candidate.get("operator_pattern"),
        }
    )
    cell = (feedback.get("cell") or {}).get(cell_key)
    if cell and int(cell.get("samples") or 0) >= 3:
        return _safe_float(cell.get("rate"), global_rate), int(cell.get("samples") or 0), f"cell:{cell_key}"

    group_rates: list[float] = []
    group_support = 0
    family_feedback = (feedback.get("family") or {}).get(family)
    dataset_feedback = (feedback.get("dataset") or {}).get(dataset)
    for item in (family_feedback, dataset_feedback):
        if item and int(item.get("samples") or 0) >= 2:
            group_rates.append(_safe_float(item.get("rate"), global_rate))
            group_support += int(item.get("samples") or 0)
    if group_rates:
        return sum(group_rates) / len(group_rates), group_support, "family_dataset"
    return global_rate, global_samples, "global"


def calibrate_active_probability(candidate: dict[str, Any], feedback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return empirical P(ACTIVE) only when resolved-outcome evidence is mature."""
    feedback = feedback or {}
    baseline = quality_baseline(candidate)
    gate = active_feedback_gate(feedback)
    if baseline.get("hard_fail"):
        return {
            "probability": 0.0,
            "tier": "B",
            "baseline": baseline,
            "support": 0,
            "provenance": baseline["hard_fail"],
            "empirical_rate": None,
            "outcome_weight": 0.0,
            "raw_outcome_weight": 0.0,
            "decision_weight_enabled": False,
            "learning_gate": gate,
        }
    if not gate["ready"]:
        return {
            "probability": None,
            "tier": None,
            "baseline": baseline,
            "support": int(((feedback.get("global") or {}).get("samples")) or 0),
            "provenance": "cold_start_unavailable",
            "empirical_rate": None,
            "outcome_weight": 0.0,
            "raw_outcome_weight": 0.0,
            "decision_weight_enabled": False,
            "learning_gate": gate,
        }

    empirical_rate, support, provenance = _feedback_choice(candidate, feedback)
    probability = _clamp01(empirical_rate)
    return {
        "probability": round(probability, 4),
        "tier": confidence_tier(probability),
        "baseline": baseline,
        "support": support,
        "provenance": provenance,
        "empirical_rate": round(probability, 4),
        "outcome_weight": 1.0,
        "raw_outcome_weight": 1.0,
        "decision_weight_enabled": True,
        "learning_gate": gate,
    }


def calibration_report(resolved_predictions: list[tuple[float, bool]], *, minimum_samples: int | None = None) -> dict[str, Any]:
    if minimum_samples is None:
        try:
            minimum_samples = max(5, int(os.environ.get("WQ_CALIBRATION_MIN_SAMPLES", "20")))
        except ValueError:
            minimum_samples = 20
    samples = len(resolved_predictions)
    if samples < minimum_samples:
        return {
            "status": "insufficient_samples",
            "samples": samples,
            "minimum_samples": minimum_samples,
            "brier_score": None,
            "reliability_buckets": [],
        }
    brier = sum((float(probability) - (1.0 if active else 0.0)) ** 2 for probability, active in resolved_predictions) / samples
    buckets = []
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        upper = lower + 0.2
        members = [(p, y) for p, y in resolved_predictions if lower <= p < upper or (upper >= 1.0 and p == 1.0)]
        if not members:
            continue
        buckets.append(
            {
                "lower": lower,
                "upper": round(upper, 1),
                "samples": len(members),
                "mean_probability": round(sum(p for p, _ in members) / len(members), 4),
                "active_rate": round(sum(1 for _, y in members if y) / len(members), 4),
            }
        )
    return {
        "status": "ready",
        "samples": samples,
        "minimum_samples": minimum_samples,
        "brier_score": round(brier, 6),
        "reliability_buckets": buckets,
    }
