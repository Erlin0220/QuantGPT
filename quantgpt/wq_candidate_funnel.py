"""Observable stage evidence for the autonomous WorldQuant candidate factory."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

STAGE_IDEA = "idea_hypothesis"
STAGE_COMPILE = "compile_semantic_validation"
STAGE_SIMULATION = "primary_simulation"
STAGE_DIAGNOSIS = "failure_diagnosis"
STAGE_MUTATION = "directed_mutation"
STAGE_ROBUSTNESS = "robustness_anti_overfit"
STAGE_CORRELATION = "correlation_novelty"
STAGE_CANDIDATE = "candidate"

STAGE_ORDER = (
    STAGE_IDEA,
    STAGE_COMPILE,
    STAGE_SIMULATION,
    STAGE_DIAGNOSIS,
    STAGE_MUTATION,
    STAGE_ROBUSTNESS,
    STAGE_CORRELATION,
    STAGE_CANDIDATE,
)

STAGE_LABELS = {
    STAGE_IDEA: "Idea/Hypothesis",
    STAGE_COMPILE: "Compile/Semantic Validation",
    STAGE_SIMULATION: "Primary Simulation",
    STAGE_DIAGNOSIS: "Failure Diagnosis",
    STAGE_MUTATION: "Directed Mutation",
    STAGE_ROBUSTNESS: "Robustness/Anti-Overfit",
    STAGE_CORRELATION: "Correlation/Novelty",
    STAGE_CANDIDATE: "Candidate",
}


def _event(stage: str, outcome: str, *, reason: str | None = None, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "stage": stage,
        "outcome": outcome,
        "failure_reason": reason,
        "details": dict(details or {}),
    }


def funnel_events_for_trial(trial: Any, item: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive one deterministic stage path from persisted trial evidence.

    This is deliberately post-hoc instrumentation of the existing orchestration,
    not a second execution framework.
    """
    events: list[dict[str, Any]] = [
        _event(
            STAGE_IDEA,
            "passed",
            details={"hypothesis": getattr(trial, "hypothesis", None), "family": getattr(trial, "family", None)},
        )
    ]
    status = str(getattr(trial, "status", "") or "")
    failure_stage = str(getattr(trial, "failure_stage", "") or "") or None
    failure_reason = str(getattr(trial, "failure_reason", "") or "") or None

    if status == "invalid" or failure_stage == "compile":
        events.append(_event(STAGE_COMPILE, "failed", reason=failure_reason))
        return events
    events.append(_event(STAGE_COMPILE, "passed", details={"operator_pattern": getattr(trial, "operator_pattern", None)}))

    if status == "simulation_failed" or failure_stage == "simulation":
        events.append(_event(STAGE_SIMULATION, "failed", reason=failure_reason))
        return events
    events.append(
        _event(
            STAGE_SIMULATION,
            "passed",
            details={
                "sharpe": getattr(trial, "sharpe", None),
                "fitness": getattr(trial, "fitness", None),
                "turnover": getattr(trial, "turnover", None),
            },
        )
    )

    generation = int(getattr(trial, "generation", 0) or 0)
    mutation_type = str(getattr(trial, "mutation_type", "") or "") or None
    if generation > 0 or mutation_type:
        events.append(
            _event(
                STAGE_MUTATION,
                "passed",
                details={
                    "generation": generation,
                    "mutation_type": mutation_type,
                    "mutation_reason": getattr(trial, "mutation_reason", None),
                    "parent_lineage_id": getattr(trial, "parent_lineage_id", None),
                },
            )
        )

    if failure_stage == "metrics":
        if item.get("directed_mutation_routed"):
            events.append(
                _event(
                    STAGE_DIAGNOSIS,
                    "routed",
                    reason=failure_reason,
                    details={
                        "mutation_children": int(item.get("mutation_children") or 0),
                        "route_reason": item.get("mutation_route_reason"),
                    },
                )
            )
            return events
        events.append(_event(STAGE_DIAGNOSIS, "failed", reason=failure_reason))
        return events
    events.append(_event(STAGE_DIAGNOSIS, "passed"))

    validation = item.get("validation") if isinstance(item.get("validation"), dict) else {}
    validation_status = str((validation or {}).get("status") or "").lower()
    has_robustness_evidence = bool(validation) or failure_stage == "robustness"
    if failure_stage == "robustness" or validation_status == "robustness_fail":
        events.append(_event(STAGE_ROBUSTNESS, "failed", reason=failure_reason, details=dict(validation or {})))
        return events
    if has_robustness_evidence:
        events.append(_event(STAGE_ROBUSTNESS, "passed", details=dict(validation or {})))

    checks = list((item.get("is_metrics") or {}).get("checks") or [])
    has_correlation_evidence = failure_stage == "diversity" or any(
        str(check.get("name") or "").upper() == "SELF_CORRELATION" for check in checks if isinstance(check, dict)
    ) or item.get("novelty_score") is not None
    if failure_stage in {"diversity", "submission"}:
        events.append(_event(STAGE_CORRELATION, "failed", reason=failure_reason))
        return events
    if has_correlation_evidence:
        events.append(
            _event(
                STAGE_CORRELATION,
                "passed",
                details={"novelty_score": item.get("novelty_score"), "checks": checks},
            )
        )

    if status == "candidate":
        events.append(_event(STAGE_CANDIDATE, "passed"))
    return events


def summarize_funnel(events: Iterable[Any]) -> dict[str, Any]:
    """Summarize entered/passed/failed counts and identify the dominant bottleneck."""
    counts: dict[str, Counter[str]] = {stage: Counter() for stage in STAGE_ORDER}
    for event in events:
        stage = str(getattr(event, "stage", None) or (event.get("stage") if isinstance(event, dict) else ""))
        outcome = str(getattr(event, "outcome", None) or (event.get("outcome") if isinstance(event, dict) else ""))
        if stage not in counts:
            continue
        counts[stage]["entered"] += 1
        if outcome in {"passed", "failed", "routed"}:
            counts[stage][outcome] += 1

    stages: dict[str, dict[str, Any]] = {}
    bottleneck_stage = None
    bottleneck_failure_rate = -1.0
    for stage in STAGE_ORDER:
        entered = int(counts[stage]["entered"])
        passed = int(counts[stage]["passed"])
        failed = int(counts[stage]["failed"])
        routed = int(counts[stage]["routed"])
        pass_rate = round((passed + routed) / entered, 4) if entered else None
        failure_rate = failed / entered if entered else 0.0
        stages[stage] = {
            "label": STAGE_LABELS[stage],
            "entered": entered,
            "passed": passed,
            "failed": failed,
            "routed": routed,
            "pass_rate": pass_rate,
        }
        if entered and failed and failure_rate > bottleneck_failure_rate:
            bottleneck_stage = stage
            bottleneck_failure_rate = failure_rate

    return {
        "stage_order": list(STAGE_ORDER),
        "stages": stages,
        "bottleneck_stage": bottleneck_stage,
        "bottleneck_label": STAGE_LABELS.get(bottleneck_stage) if bottleneck_stage else None,
        "bottleneck_failure_rate": round(bottleneck_failure_rate, 4) if bottleneck_stage else None,
    }
