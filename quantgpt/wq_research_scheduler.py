"""Dependency-free adaptive and cold-start allocation for autonomous WorldQuant research."""
from __future__ import annotations

from collections import Counter
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


def _resolved_dataset(item: Any) -> str | None:
    dataset = str(_get(item, "dataset_id") or "").strip()
    state = str(_get(item, "provenance_state") or "").strip().lower()
    if not dataset or dataset.lower() == "unknown" or state in {"unresolved", "partial"}:
        return None
    return dataset


def research_cell_key(item: Any) -> str:
    """Stable family × resolved dataset/category × operator-pattern research cell key."""
    family = str(_get(item, "family") or "unknown")
    dataset = _resolved_dataset(item) or str(_get(item, "dataset_category") or "unknown")
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


def _latest_trial_by_alpha(trials: Iterable[Any]) -> dict[str, Any]:
    """Resolve submission credit to the actual research trial that produced an alpha_id."""
    latest: dict[str, Any] = {}
    for trial in trials:
        alpha_id = str(_get(trial, "alpha_id") or "").strip()
        if not alpha_id:
            continue
        current = latest.get(alpha_id)
        if current is None:
            latest[alpha_id] = trial
            continue
        current_created = _iso(_get(current, "created_at")) or ""
        trial_created = _iso(_get(trial, "created_at")) or ""
        if trial_created >= current_created:
            latest[alpha_id] = trial
    return latest


def summarize_research_credit_assignment(
    trials: Iterable[Any],
    candidates: Iterable[Any] = (),
    attempts: Iterable[tuple[Any, Any]] = (),
) -> dict[str, Any]:
    """Report whether candidate/submission outcomes can be traced to a real research trial."""
    trial_rows = list(trials)
    latest_trial = _latest_trial_by_alpha(trial_rows)
    candidate_rows = list(candidates)
    candidate_by_alpha = {
        str(_get(candidate, "alpha_id") or ""): candidate
        for candidate in candidate_rows
        if str(_get(candidate, "alpha_id") or "")
    }
    candidate_attributed = 0
    candidate_unattributed = 0
    candidate_metadata_mismatch = 0
    for alpha_id, candidate in candidate_by_alpha.items():
        trial = latest_trial.get(alpha_id)
        if trial is None:
            candidate_unattributed += 1
            continue
        candidate_attributed += 1
        if research_cell_key(candidate) != research_cell_key(trial):
            candidate_metadata_mismatch += 1

    formal_total = formal_attributed = formal_unattributed = 0
    active_total = active_attributed = active_unattributed = 0
    terminal_total = terminal_attributed = terminal_unattributed = 0
    for attempt, candidate in attempts:
        status = str(_get(attempt, "status") or "").upper()
        alpha_id = str(_get(attempt, "alpha_id") or "")
        is_formal = status in {"RESERVED", "RESERVATION_EXPIRED", "SUBMIT_UNKNOWN", "SC_PENDING", "ACTIVE", "SC_FAIL", "OTHER_FAIL"}
        is_active = status == "ACTIVE"
        is_terminal = status in {"ACTIVE", "SC_FAIL", "OTHER_FAIL"}
        if is_formal:
            formal_total += 1
        if is_active:
            active_total += 1
        if is_terminal:
            terminal_total += 1
        attributed = bool(alpha_id and latest_trial.get(alpha_id) is not None)
        if is_formal:
            if attributed:
                formal_attributed += 1
            else:
                formal_unattributed += 1
        if is_active:
            if attributed:
                active_attributed += 1
            else:
                active_unattributed += 1
        if is_terminal:
            if attributed:
                terminal_attributed += 1
            else:
                terminal_unattributed += 1

    return {
        "policy": "submission_outcomes_credit_only_to_originating_trial",
        "candidate": {
            "total": len(candidate_by_alpha),
            "attributed": candidate_attributed,
            "unattributed": candidate_unattributed,
            "metadata_cell_mismatch": candidate_metadata_mismatch,
            "coverage": round(candidate_attributed / max(1, len(candidate_by_alpha)), 4),
        },
        "formal_submission": {
            "total": formal_total,
            "attributed": formal_attributed,
            "unattributed": formal_unattributed,
            "coverage": round(formal_attributed / max(1, formal_total), 4),
        },
        "terminal_outcome": {
            "total": terminal_total,
            "attributed": terminal_attributed,
            "unattributed": terminal_unattributed,
            "coverage": round(terminal_attributed / max(1, terminal_total), 4),
        },
        "active": {
            "total": active_total,
            "attributed": active_attributed,
            "unattributed": active_unattributed,
            "coverage": round(active_attributed / max(1, active_total), 4),
        },
    }


def summarize_research_cells(
    trials: Iterable[Any],
    candidates: Iterable[Any] = (),
    attempts: Iterable[tuple[Any, Any]] = (),
) -> list[dict[str, Any]]:
    """Aggregate provenance-resolved evidence, crediting outcomes to their originating trial cell."""
    trial_rows = list(trials)
    cells: dict[str, dict[str, Any]] = {}

    def ensure(item: Any) -> dict[str, Any] | None:
        dataset = _resolved_dataset(item)
        if not dataset:
            return None
        key = research_cell_key(item)
        if key not in cells:
            family, dataset_value, pattern = key.split("|", 2)
            cells[key] = {
                "cell_key": key,
                "family": family,
                "dataset": dataset_value,
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

    latest_trial = _latest_trial_by_alpha(trial_rows)

    for trial in trial_rows:
        cell = ensure(trial)
        if cell is None:
            continue
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
        origin_trial = latest_trial.get(alpha_id)
        if origin_trial is None:
            continue
        cell = ensure(origin_trial)
        if cell is None:
            continue
        ready = str(_get(candidate, "validation_status") or "").lower() in {"ready", "evidence_collected"}
        tier = str(_get(candidate, "confidence_tier") or "").upper()
        sharpe = float(_get(candidate, "sharpe") or 0.0)
        fitness = float(_get(candidate, "fitness") or 0.0)
        turnover = float(_get(candidate, "turnover") or 0.0)
        calibrated_high_confidence = tier in {"S", "A"}
        cold_start_research_ready = not tier and sharpe >= 1.25 and fitness >= 1.0 and 0.01 <= turnover <= 0.7
        if ready and (calibrated_high_confidence or cold_start_research_ready):
            cell["high_confidence_candidates"] += 1

    terminal_failures = {"SC_FAIL", "OTHER_FAIL"}
    for attempt, candidate in attempts:
        alpha_id = str(_get(attempt, "alpha_id") or "")
        origin_trial = latest_trial.get(alpha_id)
        if origin_trial is None:
            continue
        cell = ensure(origin_trial)
        if cell is None:
            continue
        status = str(_get(attempt, "status") or "").upper()
        if status in {"RESERVED", "RESERVATION_EXPIRED", "SUBMIT_UNKNOWN", "SC_PENDING", "ACTIVE", *terminal_failures}:
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


def _coverage_first(enriched: list[dict[str, Any]], budget: int, mode: str) -> dict[str, Any]:
    order = sorted(
        enriched,
        key=lambda item: (
            int(item.get("trials") or 0),
            int(item.get("candidates") or 0),
            item.get("last_sampled_at") or "",
            item["cell_key"],
        ),
    )
    allocations: Counter[str] = Counter()
    for index in range(budget):
        allocations[order[index % len(order)]["cell_key"]] += 1
    by_key = {item["cell_key"]: item for item in enriched}
    selected: list[dict[str, Any]] = []
    for key, slots in sorted(allocations.items(), key=lambda pair: (-pair[1], by_key[pair[0]]["cell_key"])):
        item = dict(by_key[key])
        item["slots"] = slots
        item["rationale"] = "cold_start_coverage_first"
        selected.append(item)
    return {
        "budget": budget,
        "policy": "coverage_first",
        "learning_status": "cold_start",
        "exploration_share": 1.0,
        "exploration_slots": budget,
        "exploitation_slots": 0,
        "inventory_mode": mode,
        "selected_cells": selected,
        "cell_summaries": enriched,
        "prior": {"alpha": _GLOBAL_PRIOR_ALPHA, "beta": _GLOBAL_PRIOR_BETA},
        "cooldown_enabled": False,
    }


def allocate_research_cells(
    cells: Iterable[dict[str, Any]],
    *,
    budget: int,
    exploration_share: float = _DEFAULT_EXPLORATION_SHARE,
    inventory_mode: str = "NORMAL",
    learning_maturity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Allocate research budget; use coverage-first policy until evidence is mature."""
    budget = max(0, int(budget))
    rows = [
        dict(cell)
        for cell in cells
        if str(cell.get("dataset") or cell.get("dataset_id") or "").lower() not in {"", "unknown"}
    ]
    mode = str(inventory_mode or "NORMAL").upper()
    scheduler_gate = (learning_maturity or {}).get("scheduler_adaptation") or {}
    cold_start = bool(learning_maturity) and not bool(scheduler_gate.get("ready"))

    enriched: list[dict[str, Any]] = []
    for cell in rows:
        alpha, beta, posterior = _posterior(cell)
        cooling, penalty, cooldown_reason = (False, 1.0, None) if cold_start else _cooldown(cell)
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
        return {
            "budget": budget,
            "policy": "coverage_first" if cold_start else "adaptive",
            "learning_status": "cold_start" if cold_start else "ready",
            "exploration_share": 1.0 if cold_start else max(0.0, min(0.8, float(exploration_share))),
            "inventory_mode": mode,
            "selected_cells": [],
            "cell_summaries": enriched,
            "cooldown_enabled": not cold_start,
        }
    if cold_start:
        return _coverage_first(enriched, budget, mode)

    share = max(0.0, min(0.8, float(exploration_share)))
    if mode in {"EXPLORATION", "OVER_TARGET", "HEALTHY"}:
        share = max(share, 0.4)
    elif mode in {"REPLENISHMENT", "DEFICIT"}:
        share = min(share, 0.2)

    exploration_slots = min(budget, max(1, ceil(budget * share)))
    exploitation_slots = budget - exploration_slots
    exploit_order = sorted(enriched, key=lambda item: (-item["allocation_score"], int(item.get("trials") or 0), item["cell_key"]))
    explore_order = sorted(enriched, key=lambda item: (int(item.get("trials") or 0), item.get("last_sampled_at") or "", item["cell_key"]))
    allocations: Counter[str] = Counter()
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
    exploration_keys = {row["cell_key"] for row in explore_order[:exploration_slots]}
    selected = []
    for key, slots in sorted(allocations.items(), key=lambda pair: (-pair[1], -by_key[pair[0]]["allocation_score"], pair[0])):
        item = dict(by_key[key])
        item["slots"] = slots
        item["rationale"] = "forced_exploration" if key in exploration_keys else "posterior_exploitation"
        selected.append(item)
    return {
        "budget": budget,
        "policy": "adaptive",
        "learning_status": "ready",
        "exploration_share": share,
        "exploration_slots": exploration_slots,
        "exploitation_slots": exploitation_slots,
        "inventory_mode": mode,
        "selected_cells": selected,
        "cell_summaries": enriched,
        "prior": {"alpha": _GLOBAL_PRIOR_ALPHA, "beta": _GLOBAL_PRIOR_BETA},
        "cooldown_enabled": True,
    }
