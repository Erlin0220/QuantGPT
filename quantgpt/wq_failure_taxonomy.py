"""Deterministic failure taxonomy for WorldQuant research evidence.

Lifecycle status (candidate/rejected/invalid/simulation_failed) remains separate
from these diagnostic labels so historical workflow semantics stay stable.
"""

from __future__ import annotations

import re
from typing import Any

from .wq_brain_client import SUBMIT_THRESHOLDS

_CHECK_REASON_MAP: dict[str, tuple[str, str]] = {
    "LOW_SHARPE": ("metrics", "low_sharpe"),
    "LOW_FITNESS": ("metrics", "low_fitness"),
    "LOW_TURNOVER": ("metrics", "turnover_low"),
    "HIGH_TURNOVER": ("metrics", "turnover_high"),
    "CONCENTRATED_WEIGHT": ("metrics", "weight_concentration"),
    "LOW_SUB_UNIVERSE_SHARPE": ("robustness", "sub_universe_instability"),
    "SUB_UNIVERSE": ("robustness", "sub_universe_instability"),
    "ROBUSTNESS": ("robustness", "robustness_instability"),
    "SELF_CORRELATION": ("diversity", "official_self_correlation"),
}

_PRIMARY_PRIORITY = {
    "compile": 0,
    "simulation": 1,
    "metrics": 2,
    "robustness": 3,
    "diversity": 4,
    "submission": 5,
}


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _compile_reason(error: str) -> str | None:
    text = error.lower()
    if not text:
        return None
    if "unknown operator" in text or "unsupported operator" in text:
        return "unknown_operator"
    if re.search(r"expects?\s+\d+\s+args?|arity|argument count|wrong number of arguments", text):
        return "invalid_arity"
    if any(token in text for token in ("lookback", "window must", "invalid parameter", "parameter must", "decay must")):
        return "invalid_parameter"
    if any(
        token in text
        for token in (
            "unknown variable",
            "unknown field",
            "invalid field",
            "data field",
            "incompatible input",
            "incompatible field",
            "field type",
        )
    ):
        return "incompatible_field_operator"
    if any(
        token in text
        for token in (
            "syntax",
            "parse",
            "unexpected token",
            "unexpected character",
            "missing closing parenthesis",
            "expected ','",
            "invalid expression",
        )
    ):
        return "syntax"
    return None


def _simulation_reason(error: str) -> str:
    text = error.lower()
    if any(token in text for token in ("timeout", "timed out", "deadline exceeded")):
        return "timeout"
    if any(token in text for token in ("unavailable data", "data unavailable", "no data", "dataset unavailable")):
        return "unavailable_data"
    if any(token in text for token in ("connection", "transport", "network", "http ", "502", "503", "504")):
        return "transport_or_platform_error"
    return "simulation_rejected"


def _append_unique(items: list[dict[str, str]], stage: str, reason: str, source: str) -> None:
    entry = {"stage": stage, "reason": reason, "source": source}
    if entry not in items:
        items.append(entry)


def classify_research_failure(item: dict[str, Any], *, status: str) -> dict[str, Any]:
    """Return normalized diagnostic evidence without changing lifecycle status.

    ``failure_reasons`` retains all deterministic findings. ``failure_stage`` /
    ``failure_reason`` select one stable primary finding for aggregation.
    """
    if status == "candidate":
        return {
            "failure_stage": None,
            "failure_reason": None,
            "failure_reasons": [],
            "failure_evidence": None,
        }

    metrics = dict(item.get("is_metrics") or {})
    raw_checks = [dict(check) for check in (metrics.get("checks") or []) if isinstance(check, dict)]
    validation = dict(item.get("validation") or {})
    error = str(item.get("error") or "").strip()
    mutation_targets = [str(value) for value in (item.get("mutation_targets") or [])]
    findings: list[dict[str, str]] = []

    compile_reason = _compile_reason(error)
    if status == "invalid" or compile_reason:
        _append_unique(findings, "compile", compile_reason or "syntax", "error")
    elif status == "simulation_failed":
        _append_unique(findings, "simulation", _simulation_reason(error), "error")

    for check in raw_checks:
        if str(check.get("result") or "").upper() != "FAIL":
            continue
        name = str(check.get("name") or "").upper()
        mapped = _CHECK_REASON_MAP.get(name)
        if mapped:
            _append_unique(findings, mapped[0], mapped[1], f"check:{name}")
        elif "SUB_UNIVERSE" in name or "SUBUNIVERSE" in name:
            _append_unique(findings, "robustness", "sub_universe_instability", f"check:{name}")
        elif "WEIGHT" in name:
            _append_unique(findings, "metrics", "weight_concentration", f"check:{name}")
        elif "ROBUST" in name:
            _append_unique(findings, "robustness", "robustness_instability", f"check:{name}")
        else:
            _append_unique(findings, "metrics", "platform_metric_check_failed", f"check:{name or 'UNKNOWN'}")

    sharpe = _safe_float(metrics.get("sharpe"))
    fitness = _safe_float(metrics.get("fitness"))
    turnover = _safe_float(metrics.get("turnover"))
    if sharpe is not None and sharpe < float(SUBMIT_THRESHOLDS["sharpe"]):
        _append_unique(findings, "metrics", "low_sharpe", "metric:sharpe")
    if fitness is not None and fitness < float(SUBMIT_THRESHOLDS["fitness"]):
        _append_unique(findings, "metrics", "low_fitness", "metric:fitness")
    if turnover is not None and turnover < float(SUBMIT_THRESHOLDS["turnover_min"]):
        _append_unique(findings, "metrics", "turnover_low", "metric:turnover")
    if turnover is not None and turnover > float(SUBMIT_THRESHOLDS["turnover_max"]):
        _append_unique(findings, "metrics", "turnover_high", "metric:turnover")

    validation_status = str(validation.get("status") or "").lower()
    if validation_status == "robustness_fail":
        _append_unique(findings, "robustness", "robustness_instability", "validation")
    if validation_status in {"correlation_fail", "local_correlation_fail"}:
        _append_unique(findings, "diversity", "local_correlation_risk", "validation")
    raw_submission_gate = validation.get("submission_gate")
    submission_gate: dict[str, Any] = dict(raw_submission_gate) if isinstance(raw_submission_gate, dict) else {}
    for blocker in submission_gate.get("blockers") or []:
        if blocker == "overfitting_evidence_weak":
            _append_unique(findings, "robustness", "overfitting_evidence_weak", "submission_gate")
        elif blocker == "local_correlation_high":
            _append_unique(findings, "diversity", "local_correlation_risk", "submission_gate")

    joined_targets = " ".join(mutation_targets).lower()
    if "self_correlation" in joined_targets:
        _append_unique(findings, "diversity", "official_self_correlation", "mutation_target")
    if "sub_universe" in joined_targets or "robustness" in joined_targets:
        _append_unique(findings, "robustness", "sub_universe_instability", "mutation_target")
    if "weight_concentration" in joined_targets:
        _append_unique(findings, "metrics", "weight_concentration", "mutation_target")

    final_status = str(item.get("final_status") or item.get("submission_status") or "").upper()
    if final_status == "SC_FAIL":
        _append_unique(findings, "submission", "sc_fail", "submission")
    elif final_status and final_status not in {"ACTIVE", "UNSUBMITTED", "PENDING", "SC_PENDING"}:
        _append_unique(findings, "submission", "terminal_submission_rejection", "submission")

    if status == "rejected" and not findings:
        _append_unique(findings, "metrics", "primary_threshold_rejection", "lifecycle")
    elif status == "simulation_failed" and not findings:
        _append_unique(findings, "simulation", "simulation_rejected", "lifecycle")

    findings.sort(key=lambda value: (_PRIMARY_PRIORITY.get(value["stage"], 99), value["reason"], value["source"]))
    primary = findings[0] if findings else None
    evidence = {
        "error": error or None,
        "checks": raw_checks,
        "validation": validation or None,
        "mutation_targets": mutation_targets,
        "final_status": final_status or None,
    }
    return {
        "failure_stage": primary["stage"] if primary else None,
        "failure_reason": primary["reason"] if primary else None,
        "failure_reasons": findings,
        "failure_evidence": evidence if any(value for value in evidence.values()) else None,
    }
