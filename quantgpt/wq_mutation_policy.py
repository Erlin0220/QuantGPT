"""Deterministic failure-directed mutation policy for WQ research."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .wq_operator_registry import canonicalize_wq_expression


def normalize_wq_expression(expression: str) -> str:
    value = str(expression or "").strip()
    if not value:
        return ""
    try:
        value = canonicalize_wq_expression(value)
    except Exception:
        pass
    return "".join(value.lower().split())


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _failure_reasons(item: dict[str, Any]) -> set[str]:
    reasons: set[str] = set()
    if item.get("failure_reason"):
        reasons.add(str(item["failure_reason"]))
    for entry in item.get("failure_reasons") or []:
        if isinstance(entry, dict) and entry.get("reason"):
            reasons.add(str(entry["reason"]))
        elif entry:
            reasons.add(str(entry))
    targets = " ".join(str(value) for value in (item.get("mutation_targets") or [])).lower()
    if "turnover" in targets:
        reasons.add("turnover_high" if "reduce" in targets else "turnover_low")
    if "fitness" in targets:
        reasons.add("low_fitness")
    if "self_correlation" in targets:
        reasons.add("official_self_correlation")
    if "sub_universe" in targets or "robustness" in targets:
        reasons.add("robustness_instability")
    validation = item.get("validation") or {}
    if isinstance(validation, dict):
        status = str(validation.get("status") or "").lower()
        if status in {"correlation_fail", "local_correlation_fail"}:
            reasons.add("local_correlation_risk")
        if status == "robustness_fail":
            reasons.add("robustness_instability")
    proxy = item.get("local_correlation_proxy") or {}
    if isinstance(proxy, dict) and proxy.get("high_correlation"):
        reasons.add("local_correlation_risk")
    return reasons


def preferred_mutation_classes(item: dict[str, Any]) -> list[dict[str, str]]:
    """Return ordered structural mutation classes from normalized failure evidence."""
    reasons = _failure_reasons(item)
    metrics = item.get("is_metrics") or {}
    sharpe = _safe_float(metrics.get("sharpe") if isinstance(metrics, dict) else item.get("sharpe"))
    fitness = _safe_float(metrics.get("fitness") if isinstance(metrics, dict) else item.get("fitness"))
    turnover = _safe_float(metrics.get("turnover") if isinstance(metrics, dict) else item.get("turnover"))
    out: list[dict[str, str]] = []

    def add(mutation_class: str, rationale: str) -> None:
        if mutation_class not in {entry["mutation_class"] for entry in out}:
            out.append({"mutation_class": mutation_class, "rationale": rationale})

    correlation_failure = bool(reasons & {"official_self_correlation", "local_correlation_risk", "sc_fail"})
    robustness_failure = bool(reasons & {"sub_universe_instability", "robustness_instability"})
    compile_failure = bool(reasons & {"unknown_operator", "invalid_arity", "invalid_parameter", "incompatible_field_operator", "syntax"})

    if compile_failure:
        add("compile_prevention", "known compile/field incompatibility should be prevented before another simulation")
        return out
    if correlation_failure:
        add("economic_reseed", "correlation failure requires a different data source/economic hypothesis before cosmetic parameter changes")
        add("operator_family_switch", "change the transformation family after changing economic logic")
        add("neutralization_shift", "change portfolio exposure only after structural signal changes")
        return out
    if robustness_failure:
        add("stable_data_reseed", "sub-universe/robustness failure favors more liquid and slower data categories")
        add("neutralization_shift", "remove unstable broad exposures")
        add("weight_control", "reduce concentration sensitivity through ranking/truncation-compatible structure")
        return out
    high_turnover = "turnover_high" in reasons or (turnover is not None and turnover > 0.7)
    if high_turnover:
        add("widen_windows", "high turnover should first be reduced through slower lookbacks/decay-equivalent smoothing")
    if "low_fitness" in reasons or (fitness is not None and fitness < 1.0):
        add("smooth_signal", "low Fitness should first test lower trading intensity while preserving the hypothesis")
        if sharpe is not None and sharpe >= 1.25:
            add("cost_reduction", "acceptable Sharpe with low Fitness is usually a turnover/cost problem")
        else:
            add("signal_quality", "if smoothing does not rescue Fitness, improve signal quality next")
    if high_turnover:
        add("conditional_execution", "trade only in informative regimes before unrelated random mutation")
    if "turnover_low" in reasons or (turnover is not None and 0 < turnover < 0.01):
        add("increase_responsiveness", "very low turnover needs a more responsive signal")
    if not out:
        add("signal_quality", "no specialized failure route was available")
    return out


def compile_prevention_rules(trials: Iterable[dict[str, Any]]) -> dict[str, list[Any]]:
    exact: set[str] = set()
    operator_patterns: set[str] = set()
    field_pairs: set[tuple[str, tuple[str, ...]]] = set()
    compile_reasons = {"unknown_operator", "invalid_arity", "invalid_parameter", "incompatible_field_operator", "syntax"}
    for trial in trials:
        reasons = _failure_reasons(trial)
        if not reasons & compile_reasons:
            continue
        expression = normalize_wq_expression(str(trial.get("expression") or ""))
        if expression:
            exact.add(expression)
        pattern = str(trial.get("operator_pattern") or "").strip()
        fields = tuple(sorted(str(value) for value in (trial.get("data_fields") or []) if value))
        if pattern and reasons & {"unknown_operator", "invalid_arity", "invalid_parameter"}:
            operator_patterns.add(pattern)
        if pattern and fields and "incompatible_field_operator" in reasons:
            field_pairs.add((pattern, fields))
    return {
        "blocked_exact_expressions": sorted(exact),
        "blocked_operator_patterns": sorted(operator_patterns),
        "blocked_operator_field_pairs": [
            {"operator_pattern": pattern, "data_fields": list(fields)} for pattern, fields in sorted(field_pairs)
        ],
    }


def summarize_mutation_outcomes(trials: Iterable[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"trials": 0, "rescued": 0})
    for trial in trials:
        mutation_type = str(trial.get("mutation_type") or "").strip()
        if not mutation_type:
            continue
        counts[mutation_type]["trials"] += 1
        if str(trial.get("status") or "").lower() == "candidate":
            counts[mutation_type]["rescued"] += 1
    return {
        mutation_type: {
            "trials": values["trials"],
            "rescued": values["rescued"],
            "failed": values["trials"] - values["rescued"],
            "rescue_rate": round(values["rescued"] / max(1, values["trials"]), 4),
        }
        for mutation_type, values in sorted(counts.items())
    }
