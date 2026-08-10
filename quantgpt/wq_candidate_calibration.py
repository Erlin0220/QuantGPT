"""Deterministic, dependency-free ACTIVE probability calibration for WQ candidates."""
from __future__ import annotations

import os
from typing import Any

from .wq_correlation_proxy import correlation_evidence_is_fresh, correlation_priority_multiplier
from .wq_learning_maturity import active_feedback_gate
from .wq_overfitting import overfitting_priority_multiplier
from .wq_research_scheduler import research_cell_key


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def configured_confidence_thresholds() -> tuple[float, float]:
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
    metrics = candidate.get("is_metrics") or {}
    validation = candidate.get("validation") or {}
    if not isinstance(validation, dict):
        validation = {}
    checks = list(metrics.get("checks") or []) if isinstance(metrics, dict) else []
    sc_check = next((check for check in checks if str(check.get("name") or "").upper() == "SELF_CORRELATION"), {})
    if str(sc_check.get("result") or "").upper() == "FAIL":
        return {"score": 0.0, "components": {"official_sc": 0.0}, "hard_fail": "official_self_correlation"}

    fitness = _safe_float(metrics.get("fitness"))
    sharpe = _safe_float(metrics.get("sharpe"))
    returns = _safe_float(metrics.get("returns"))
    turnover = _safe_float(metrics.get("turnover"))
    robustness = _clamp01(_safe_float(validation.get("robustness_score")))
    novelty = _clamp01(_safe_float(candidate.get("novelty_score")))
    raw_self_correlation = sc_check.get("value")
    if raw_self_correlation is None:
        official_correlation = 0.65
    else:
        try:
            self_correlation = float(raw_self_correlation)
            official_correlation = _clamp01(1.0 - max(0.0, self_correlation))
        except (TypeError, ValueError):
            official_correlation = 0.65

    local_proxy = candidate.get("local_correlation_proxy") or {}
    local_multiplier = correlation_priority_multiplier(local_proxy)
    if 0.03 <= turnover <= 0.5:
        turnover_component = 1.0
    elif 0 < turnover < 0.03:
        turnover_component = _clamp01(turnover / 0.03)
    elif turnover > 0.5:
        turnover_component = _clamp01(1.0 - (turnover - 0.5) / 0.2)
    else:
        turnover_component = 0.0
    expression = str(candidate.get("expression") or "")
    complexity = _clamp01(1.0 - max(0, expression.count("(") - 4) / 12.0)
    raw_active_prior = candidate.get("active_prior_score")
    active_reference = 0.5 if raw_active_prior is None else _clamp01(_safe_float(raw_active_prior, 0.5))
    validation_ready = str(validation.get("status") or "").lower() == "ready"
    freshness = 1.0 if validation_ready else 0.7
    if local_proxy.get("status") == "available":
        freshness *= 1.0 if correlation_evidence_is_fresh(local_proxy) else 0.8

    components = {
        "fitness": _clamp01((fitness - 0.8) / 1.0),
        "sharpe": _clamp01((sharpe - 1.0) / 1.5),
        "returns": _clamp01(returns / 0.20),
        "robustness": robustness,
        "novelty": novelty,
        "official_correlation": official_correlation,
        "local_correlation_multiplier": local_multiplier,
        "turnover": turnover_component,
        "complexity": complexity,
        "freshness": freshness,
        "active_reference": active_reference,
    }
    score = (
        components["fitness"] * 0.21
        + components["sharpe"] * 0.17
        + components["returns"] * 0.07
        + components["robustness"] * 0.13
        + components["novelty"] * 0.08
        + components["official_correlation"] * 0.10
        + components["turnover"] * 0.07
        + components["complexity"] * 0.05
        + components["freshness"] * 0.04
        + components["active_reference"] * 0.08
    )
    overfitting = validation.get("overfitting_evidence")
    overfitting_multiplier = overfitting_priority_multiplier(overfitting)
    components["overfitting_multiplier"] = overfitting_multiplier
    return {
        "score": round(_clamp01(score * local_multiplier * overfitting_multiplier), 6),
        "components": components,
        "hard_fail": None,
    }


def _feedback_choice(candidate: dict[str, Any], feedback: dict[str, Any]) -> tuple[float, int, str]:
    global_feedback = feedback.get("global") or {}
    global_rate = _safe_float(global_feedback.get("rate"), 0.5)
    global_samples = int(global_feedback.get("samples") or 0)
    meta = candidate.get("research_meta") or {}
    family = str(meta.get("family") or candidate.get("family") or "unknown")
    dataset = str(meta.get("dataset_id") or candidate.get("dataset_id") or "unknown")
    cell_key = research_cell_key({
        "family": family,
        "dataset_id": dataset,
        "operator_pattern": meta.get("operator_pattern") or candidate.get("operator_pattern"),
    })
    cell = (feedback.get("cell") or {}).get(cell_key)
    if cell and int(cell.get("samples") or 0) >= 3:
        return _safe_float(cell.get("rate"), global_rate), int(cell.get("samples") or 0), f"cell:{cell_key}"

    group_rates: list[float] = []
    group_support = 0
    family_feedback = (feedback.get("family") or {}).get(family)
    dataset_feedback = (feedback.get("dataset") or {}).get(dataset)
    for _label, item in (("family", family_feedback), ("dataset", dataset_feedback)):
        if item and int(item.get("samples") or 0) >= 2:
            group_rates.append(_safe_float(item.get("rate"), global_rate))
            group_support += int(item.get("samples") or 0)
    if group_rates:
        return sum(group_rates) / len(group_rates), group_support, "family_dataset"
    return global_rate, global_samples, "global_prior" if global_samples == 0 else "global"


def calibrate_active_probability(candidate: dict[str, Any], feedback: dict[str, Any] | None = None) -> dict[str, Any]:
    feedback = feedback or {}
    baseline = quality_baseline(candidate)
    gate = active_feedback_gate(feedback)
    if baseline.get("hard_fail"):
        probability = 0.0
        return {
            "probability": probability,
            "tier": confidence_tier(probability),
            "baseline": baseline,
            "support": 0,
            "provenance": baseline["hard_fail"],
            "empirical_rate": None,
            "outcome_weight": 0.0,
            "decision_weight_enabled": False,
            "learning_gate": gate,
        }
    empirical_rate, support, provenance = _feedback_choice(candidate, feedback)
    raw_outcome_weight = min(0.45, 0.45 * support / (support + 8.0)) if support > 0 else 0.0
    outcome_weight = raw_outcome_weight if gate["ready"] else 0.0
    probability = _clamp01(float(baseline["score"]) * (1.0 - outcome_weight) + empirical_rate * outcome_weight)
    return {
        "probability": round(probability, 4),
        "tier": confidence_tier(probability),
        "baseline": baseline,
        "support": support,
        "provenance": provenance,
        "empirical_rate": round(empirical_rate, 4),
        "outcome_weight": round(outcome_weight, 4),
        "raw_outcome_weight": round(raw_outcome_weight, 4),
        "decision_weight_enabled": bool(gate["ready"]),
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
        return {"status": "insufficient_samples", "samples": samples, "minimum_samples": minimum_samples, "brier_score": None, "reliability_buckets": []}
    brier = sum((float(probability) - (1.0 if active else 0.0)) ** 2 for probability, active in resolved_predictions) / samples
    buckets = []
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        upper = lower + 0.2
        members = [(p, y) for p, y in resolved_predictions if lower <= p < upper or (upper >= 1.0 and p == 1.0)]
        if not members:
            continue
        buckets.append({
            "lower": lower,
            "upper": round(upper, 1),
            "samples": len(members),
            "mean_probability": round(sum(p for p, _ in members) / len(members), 4),
            "active_rate": round(sum(1 for _, y in members if y) / len(members), 4),
        })
    return {"status": "ready", "samples": samples, "minimum_samples": minimum_samples, "brier_score": round(brier, 6), "reliability_buckets": buckets}
