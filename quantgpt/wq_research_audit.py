"""Per-round audit metrics for WorldQuant Skill-first research.

The audit is intentionally derived from persisted Tasks + ResearchTrial/Candidate/
SubmissionAttempt rows. It measures whether the research workflow followed its
Skill contracts and whether the resulting cohort actually converted. It does
not influence BRAIN eligibility or invent scoring thresholds.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable

_REQUIRED_BASE_SKILLS = {
    "wq-alpha-hypothesis",
    "wq-alpha-review",
    "wq-robustness-validation",
    "wq-candidate-evidence",
}


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 4)


def _skill_contract(candidate: dict[str, Any]) -> dict[str, Any]:
    chain = {str(value) for value in (candidate.get("skill_chain") or []) if str(value)}
    missing = sorted(_REQUIRED_BASE_SKILLS - chain)
    review_run = str(candidate.get("review_decision") or "").upper() == "RUN"
    robustness = candidate.get("robustness_plan")
    robustness_ok = isinstance(robustness, dict) and str(robustness.get("mode") or "") == "skill_defined"
    evidence = candidate.get("candidate_evidence_policy")
    evidence_ok = isinstance(evidence, dict) and str(evidence.get("mode") or "") == "calibrated_evidence_hierarchy"

    is_repair = "wq-alpha-repair" in chain
    failure_signature = candidate.get("failure_signature")
    failure_signature_ok = (not is_repair) or (
        "wq-failure-diagnosis" in chain and isinstance(failure_signature, dict) and bool(failure_signature)
    )

    is_diversify = "wq-alpha-diversify" in chain
    diversity_case = candidate.get("diversity_case")
    changed_dimensions = (
        diversity_case.get("changed_dimensions")
        if isinstance(diversity_case, dict)
        else None
    )
    diversity_case_ok = (not is_diversify) or (
        isinstance(diversity_case, dict)
        and bool(diversity_case)
        and isinstance(changed_dimensions, list)
        and bool(changed_dimensions)
    )

    return {
        "ok": not missing and review_run and robustness_ok and evidence_ok and failure_signature_ok and diversity_case_ok,
        "missing_base_skills": missing,
        "review_run": review_run,
        "robustness_plan": robustness_ok,
        "candidate_evidence_policy": evidence_ok,
        "repair_requires_failure_signature": is_repair,
        "failure_signature": failure_signature_ok,
        "diversify_requires_diversity_case": is_diversify,
        "diversity_case": diversity_case_ok,
        "route": "repair" if is_repair else ("diversify" if is_diversify else "new_hypothesis"),
    }


def build_research_round_audit(
    task: Any,
    *,
    trials: Iterable[Any] = (),
    candidates: Iterable[Any] = (),
    attempts: Iterable[Any] = (),
    attribution_mode: str = "source_run_id",
) -> dict[str, Any]:
    params = dict(_get(task, "params") or {})
    result = dict(_get(task, "result") or {})
    summary = dict(result.get("summary") or {})
    skill_candidates = [dict(item) for item in (params.get("skill_candidates") or []) if isinstance(item, dict)]
    contracts = [_skill_contract(item) for item in skill_candidates]
    route_counts = Counter(item["route"] for item in contracts)

    trial_rows = list(trials)
    candidate_rows = list(candidates)
    attempt_rows = list(attempts)
    statuses = Counter(str(_get(attempt, "status") or "UNKNOWN").upper() for attempt in attempt_rows)
    formal_statuses = {"RESERVED", "RESERVATION_EXPIRED", "SUBMIT_UNKNOWN", "SC_PENDING", "ACTIVE", "SC_FAIL", "OTHER_FAIL"}
    formal_attempts = sum(count for status, count in statuses.items() if status in formal_statuses)
    terminal_attempts = sum(count for status, count in statuses.items() if status in {"ACTIVE", "SC_FAIL", "OTHER_FAIL"})
    active = int(statuses.get("ACTIVE", 0))

    primary_simulations = int(summary.get("simulated") or 0)
    robustness_simulations = int(summary.get("validation_simulations") or 0)
    total_simulations = int(summary.get("total_simulations") or (primary_simulations + robustness_simulations))
    persisted_candidates = len(candidate_rows)

    diversity_states = Counter()
    for candidate in candidate_rows:
        case = _get(candidate, "diversity_case")
        if isinstance(case, dict) and case:
            state = str(case.get("empirical_state") or case.get("state") or "designed_diverse")
            diversity_states[state] += 1

    valid_skill_candidates = sum(1 for item in contracts if item["ok"])
    repair_required = sum(1 for item in contracts if item["repair_requires_failure_signature"])
    repair_compliant = sum(1 for item in contracts if item["repair_requires_failure_signature"] and item["failure_signature"])
    diversify_required = sum(1 for item in contracts if item["diversify_requires_diversity_case"])
    diversify_compliant = sum(1 for item in contracts if item["diversify_requires_diversity_case"] and item["diversity_case"])

    return {
        "source_run_id": str(_get(task, "id") or _get(task, "task_id") or ""),
        "tag": params.get("tag") or result.get("tag"),
        "status": str(_get(task, "status") or "unknown"),
        "mode": result.get("mode") or ("manual" if params.get("expressions") else "unknown"),
        "attribution_mode": attribution_mode,
        "created_at": _iso(_get(task, "created_at")),
        "updated_at": _iso(_get(task, "updated_at")),
        "skill_compliance": {
            "planned": len(skill_candidates),
            "valid": valid_skill_candidates,
            "rate": _ratio(valid_skill_candidates, len(skill_candidates)),
            "routes": dict(route_counts),
            "repair_required": repair_required,
            "repair_with_failure_signature": repair_compliant,
            "diversify_required": diversify_required,
            "diversify_with_diversity_case": diversify_compliant,
        },
        "execution": {
            "persisted_trials": len(trial_rows),
            "primary_simulations": primary_simulations,
            "robustness_simulations": robustness_simulations,
            "total_simulations": total_simulations,
            "persisted_candidates": persisted_candidates,
            "simulation_to_candidate_rate": _ratio(persisted_candidates, primary_simulations),
            "simulations_per_candidate": round(total_simulations / persisted_candidates, 4) if persisted_candidates else None,
        },
        "submission": {
            "formal_attempts": formal_attempts,
            "terminal_attempts": terminal_attempts,
            "active": active,
            "candidate_to_formal_rate": _ratio(formal_attempts, persisted_candidates),
            "candidate_to_active_rate": _ratio(active, persisted_candidates),
            "terminal_to_active_rate": _ratio(active, terminal_attempts),
            "status_counts": dict(statuses),
        },
        "diversity": {
            "state_counts": dict(diversity_states),
            "empirically_supported": int(diversity_states.get("empirically_supported", 0)),
            "not_diverse": int(diversity_states.get("not_diverse", 0)),
        },
    }
