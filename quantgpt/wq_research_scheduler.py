"""Dependency-free adaptive allocation for autonomous WorldQuant research."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import ceil
from typing import Any, Iterable

_GLOBAL_PRIOR_ALPHA = 2.0
_GLOBAL_PRIOR_BETA = 18.0
_DEFAULT_EXPLORATION_SHARE = 0.25


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def research_cell_key(item: Any) -> str:
    """Stable family × dataset/category × operator-pattern research cell key."""
    family = str(_get(item, "family") or "unknown")
    dataset = str(_get(item, "dataset_id") or _get(item, "dataset_category") or "unknown")
    pattern = str(_get(item, "operator_pattern") or "unknown")
    return f"{family}|{dataset}|{pattern}"


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def summarize_research_cells(
    trials: Iterable[Any],
    candidates: Iterable[Any] = (),
    attempts: Iterable[tuple[Any, Any]] = (),
) -> list[dict[str, Any]]:
    """Aggregate research and formal outcomes into scheduler-ready cells."""
    cells: dict[str, dict[str, Any]] = {}

    def ensure(item: Any) -> dict[str, Any]:
        key = research_cell_key(item)
        if key not in cells:
            family, dataset, pattern = key.split("|", 2)
            cells[key] = {
                "cell_key": key,
                "family": family,
                "dataset": dataset,
                "operator_pattern": pattern,
                "trials": 0,
                "candidates": 0,
                "high_confidence_candidates": 0,
                "formal_submissions": 0,
                "active": 0,
                "terminal_failures": 0,
                "recent_failure_reasons": {},
                "last_sampled_at": None,
            }
        return cells[key]

    for trial in trials:
        cell = ensure(trial)
        cell["trials"] += 1
        if str(_get(trial, "status") or "").lower() == "candidate":
            cell["candidates"] += 1
        reason = str(_get(trial, "failure_reason") or "")
        if reason:
            reasons = Counter(cell["recent_failure_reasons"])
            reasons[reason] += 1
            cell["recent_failure_reasons"] = dict(reasons)
        sampled = _iso(_get(trial, "created_at"))
        if sampled and (not cell["last_sampled_at"] or sampled > cell["last_sampled_at"]):
            cell["last_sampled_at"] = sampled

    candidate_by_alpha: dict[str, Any] = {}
    for candidate in candidates:
        alpha_id = str(_get(candidate, "alpha_id") or "")
        if alpha_id:
            candidate_by_alpha[alpha_id] = candidate
        cell = ensure(candidate)
        ready = str(_get(candidate, "validation_status") or "").lower() == "ready"
        tier = str(_get(candidate, "confidence_tier") or "").upper()
        robustness = float(_get(candidate, "robustness_score") or 0.0)
        sharpe = float(_get(candidate, "sharpe") or 0.0)
        fitness = float(_get(candidate, "fitness") or 0.0)
        calibrated_high_confidence = tier in {"S", "A"}
        legacy_high_confidence = not tier and robustness >= 0.5 and sharpe >= 1.25 and fitness >= 1.0
        if ready and (calibrated_high_confidence or legacy_high_confidence):
            cell["high_confidence_candidates"] += 1

    terminal_failures = {"SC_FAIL", "OTHER_FAIL"}
    for attempt, candidate in attempts:
        candidate = candidate or candidate_by_alpha.get(str(_get(attempt, "alpha_id") or ""))
        if candidate is None:
            continue
        cell = ensure(candidate)
        status = str(_get(attempt, "status") or "").upper()
        if status in {"RESERVED", "SC_PENDING", "ACTIVE", *terminal_failures}:
            cell["formal_submissions"] += 1
        if status == "ACTIVE":
            cell["active"] += 1
        elif status in terminal_failures:
            cell["terminal_failures"] += 1

    return sorted(cells.values(), key=lambda item: item["cell_key"])


def _posterior(cell: dict[str, Any]) -> tuple[float, float, float]:
    successes = int(cell.get("candidates") or 0)
    trials = int(cell.get("trials") or 0)
    alpha = _GLOBAL_PRIOR_ALPHA + successes
    beta = _GLOBAL_PRIOR_BETA + max(0, trials - successes)
    return alpha, beta, alpha / (alpha + beta)


def _cooldown(cell: dict[str, Any]) -> tuple[bool, float, str | None]:
    trials = int(cell.get("trials") or 0)
    if trials < 8:
        return False, 1.0, None
    candidates = int(cell.get("candidates") or 0)
    reasons = Counter(cell.get("recent_failure_reasons") or {})
    dominant_reason, dominant_count = reasons.most_common(1)[0] if reasons else (None, 0)
    poor_yield = candidates / max(1, trials) < 0.03
    repeated_failure = dominant_count / max(1, trials) >= 0.6
    if not (poor_yield or repeated_failure):
        return False, 1.0, None
    last_sampled = cell.get("last_sampled_at")
    recovery = False
    if last_sampled:
        try:
            sampled = datetime.fromisoformat(str(last_sampled).replace("Z", "+00:00"))
            if sampled.tzinfo is None:
                sampled = sampled.replace(tzinfo=timezone.utc)
            recovery = (datetime.now(timezone.utc) - sampled.astimezone(timezone.utc)).total_seconds() >= 24 * 3600
        except (TypeError, ValueError):
            pass
    penalty = 0.7 if recovery else 0.35
    reason = dominant_reason if repeated_failure else "persistently_poor_yield"
    return True, penalty, reason


def allocate_research_cells(
    cells: Iterable[dict[str, Any]],
    *,
    budget: int,
    exploration_share: float = _DEFAULT_EXPLORATION_SHARE,
    inventory_mode: str = "NORMAL",
) -> dict[str, Any]:
    """Allocate budget using smoothed posterior yield plus forced exploration.

    This intentionally uses posterior means instead of stochastic Thompson draws:
    it has the same Beta-Binomial shrinkage semantics while remaining deterministic
    for scheduled-agent tests and reproducible research lineage.
    """
    budget = max(0, int(budget))
    rows = [dict(cell) for cell in cells]
    mode = str(inventory_mode or "NORMAL").upper()
    share = max(0.0, min(0.8, float(exploration_share)))
    if mode in {"EXPLORATION", "OVER_TARGET", "HEALTHY"}:
        share = max(share, 0.4)
    elif mode in {"REPLENISHMENT", "DEFICIT"}:
        share = min(share, 0.2)
    enriched: list[dict[str, Any]] = []
    for cell in rows:
        alpha, beta, posterior = _posterior(cell)
        cooling, penalty, cooldown_reason = _cooldown(cell)
        scored = dict(cell)
        scored.update({
            "posterior_alpha": round(alpha, 4),
            "posterior_beta": round(beta, 4),
            "posterior_yield": round(posterior, 6),
            "cooldown": cooling,
            "cooldown_penalty": penalty,
            "cooldown_reason": cooldown_reason,
            "allocation_score": round(posterior * penalty, 6),
        })
        enriched.append(scored)
    if budget == 0 or not enriched:
        return {"budget": budget, "exploration_share": share, "inventory_mode": mode, "selected_cells": [], "cell_summaries": enriched}

    exploration_slots = min(budget, max(1, ceil(budget * share)))
    exploitation_slots = budget - exploration_slots
    exploit_order = sorted(enriched, key=lambda item: (-item["allocation_score"], int(item.get("trials") or 0), item["cell_key"]))
    explore_order = sorted(enriched, key=lambda item: (int(item.get("trials") or 0), item.get("last_sampled_at") or "", item["cell_key"]))
    allocations: Counter[str] = Counter()
    # Greedy diminishing-return allocation: stronger posterior cells receive more
    # exploitation slots, while the denominator prevents one cell from swallowing
    # the entire non-exploration budget.
    for _ in range(exploitation_slots):
        chosen = max(
            exploit_order,
            key=lambda item: (
                item["allocation_score"] / (1.0 + allocations[item["cell_key"]] * 0.45),
                -int(item.get("trials") or 0),
                item["cell_key"],
            ),
        )
        allocations[chosen["cell_key"]] += 1
    for index in range(exploration_slots):
        allocations[explore_order[index % len(explore_order)]["cell_key"]] += 1

    by_key = {item["cell_key"]: item for item in enriched}
    selected = []
    for key, slots in sorted(allocations.items(), key=lambda pair: (-pair[1], -by_key[pair[0]]["allocation_score"], pair[0])):
        item = dict(by_key[key])
        item["slots"] = slots
        item["rationale"] = "forced_exploration" if key in {row["cell_key"] for row in explore_order[:exploration_slots]} else "posterior_exploitation"
        selected.append(item)
    return {
        "budget": budget,
        "exploration_share": share,
        "exploration_slots": exploration_slots,
        "exploitation_slots": exploitation_slots,
        "inventory_mode": mode,
        "selected_cells": selected,
        "cell_summaries": enriched,
        "prior": {"alpha": _GLOBAL_PRIOR_ALPHA, "beta": _GLOBAL_PRIOR_BETA},
    }
