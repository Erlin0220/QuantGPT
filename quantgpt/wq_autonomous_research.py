"""Bounded autonomous WorldQuant Alpha planning and evolutionary research."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any, Callable

from .wq_brain_service import run_list_alphas, run_single_simulation
from .wq_lineage import extract_expression_metadata
from .wq_mutation_policy import preferred_mutation_classes
from .wq_operator_registry import validate_wq_expression
from .wq_research_agent import diagnose_wq_result, run_research_batch
from .wq_research_memory import classify_wq_family, normalize_wq_expression
from .wq_research_scheduler import allocate_research_cells
from .wq_submission_evidence import configured_submission_evidence_policy, submission_evidence_blockers

FAMILY_SEEDS: dict[str, tuple[str, ...]] = {
    "price_volume": (
        "-1 * rank(ts_decay_linear(returns * volume / adv20, 5))",
        "-1 * rank(ts_decay_linear(returns * volume / adv20, 10))",
        "-1 * rank(ts_decay_linear(close / vwap, 5))",
        "-1 * rank(ts_decay_linear(close / vwap, 10))",
        "rank(ts_corr(ts_delta(close, 5), ts_delta(volume, 5), 10))",
        "trade_when(volume > 1.2 * ts_mean(volume, 20), -rank(ts_corr(close, volume, 10)), -1)",
    ),
    "momentum_reversal": (
        "-1 * rank(ts_delta(close, 5) / ts_delay(close, 5))",
        "rank(ts_delta(close, 10) / ts_delay(close, 10))",
        "rank(ts_mean(returns, 20) / (ts_std_dev(returns, 20) + 0.001))",
        "-1 * rank(ts_mean(returns, 20) / (ts_std_dev(returns, 20) + 0.001))",
        "rank(ts_mean(open / close, 20))",
    ),
    "volatility_structure": (
        "-rank(ts_std_dev(returns, 20))",
        "-rank(ts_std_dev(returns, 60))",
        "rank(ts_arg_max(close, 60) - ts_arg_min(close, 60))",
        "rank(ts_arg_min(low, 60) - ts_arg_max(high, 60))",
        "-rank(ts_std_dev(volume / (ts_mean(volume, 20) + 1), 60))",
    ),
    "analyst_revision": (
        "rank(ts_mean(ts_backfill(analyst_revision_rank_derivative, 60), 20))",
        "-rank(ts_mean(ts_backfill(analyst_revision_rank_derivative, 60), 20))",
        "rank(ts_corr(analyst_revision_rank_derivative, ts_delay(analyst_revision_rank_derivative, 20), 20))",
        "rank(-ts_std_dev(ts_backfill(cashflow_efficiency_rank_derivative, 60), 20))",
    ),
    "options_volatility": (
        "rank(ts_mean(ts_backfill(implied_volatility_call_30 - historical_volatility_30, 60), 20))",
        "-rank(ts_mean(ts_backfill(implied_volatility_call_30 - historical_volatility_30, 60), 20))",
        "rank(ts_std_dev(ts_backfill(implied_volatility_call_30 - historical_volatility_30, 20), 20))",
        "rank(ts_corr(ts_backfill(implied_volatility_call_30, 20), ts_backfill(historical_volatility_30, 20), 20))",
    ),
    "sentiment": (
        "rank(ts_mean(ts_backfill(snt_social_value_fast_d1, 30), 20))",
        "-rank(ts_mean(ts_backfill(snt_social_value_fast_d1, 30), 20))",
        "rank(ts_backfill(snt_social_value_fast_d1, 30) - ts_backfill(snt1_d1_nettargetpercent, 30))",
    ),
    "fundamental_quality": (
        "rank(ts_mean(ts_backfill(mdf_quality, 120), 40))",
        "rank(ts_mean(ts_backfill(mdf_bp, 120), 40))",
        "rank(-ts_mean(ts_backfill(mdf_leverage, 120), 40))",
        "rank(ts_mean(ts_backfill(cashflow_efficiency_rank_derivative, 60), 20))",
    ),
}


def _safe_metric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_platform_alpha_history(client, *, limit: int = 500) -> dict[str, Any]:
    """Load a bounded multi-page slice of BRAIN Alpha history.

    Candidate recovery must not assume the newest 100 rows contain the strongest
    UNSUBMITTED near-misses.  Scan a few cheap list pages and de-duplicate by
    alpha_id so older simulated inventory remains available to the rescue path.
    """
    limit = max(1, min(500, int(limit)))
    page_size = 100
    alphas: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    pages = 0
    errors: list[str] = []
    for offset in range(0, limit, page_size):
        requested = min(page_size, limit - offset)
        page = run_list_alphas(client, limit=requested, offset=offset)
        pages += 1
        if not page.get("ok"):
            errors.append(str(page.get("error") or f"offset {offset} failed"))
            break
        rows = list(page.get("alphas") or [])
        for alpha in rows:
            alpha_id = str(alpha.get("alpha_id") or "").strip()
            dedupe_key = alpha_id or normalize_wq_expression(str(alpha.get("expression") or ""))
            if dedupe_key and dedupe_key in seen_ids:
                continue
            if dedupe_key:
                seen_ids.add(dedupe_key)
            alphas.append(alpha)
            if len(alphas) >= limit:
                break
        if len(alphas) >= limit or len(rows) < requested:
            break
    return {
        "ok": bool(alphas) or not errors,
        "alphas": alphas,
        "pages_scanned": pages,
        "errors": errors,
    }


def _trial_promise_score(
    item: dict[str, Any],
    *,
    min_sharpe: float = 1.25,
    min_fitness: float = 1.0,
) -> float:
    """Score how close a failed trial is to being useful without rewarding raw parameter chasing."""
    sharpe = _safe_metric(item.get("sharpe"))
    fitness = _safe_metric(item.get("fitness"))
    turnover = _safe_metric(item.get("turnover"))
    sharpe_progress = max(0.0, min(1.5, sharpe / max(0.01, min_sharpe)))
    fitness_progress = max(0.0, min(1.5, fitness / max(0.01, min_fitness)))
    score = sharpe_progress * 0.55 + fitness_progress * 0.45
    if turnover and not 0.01 <= turnover <= 0.7:
        score -= 0.25
    if item.get("self_correlation_failed"):
        score -= 0.5
    failure_text = " ".join(
        [str(item.get("failure_reason") or "")]
        + [str(value) for value in (item.get("mutation_targets") or [])]
        + [str(value) for value in (item.get("failure_reasons") or [])]
    ).lower()
    validation = item.get("validation") or {}
    robustness_failed = (
        "robustness" in failure_text
        or "sub_universe" in failure_text
        or (isinstance(validation, dict) and str(validation.get("status") or "").lower() == "robustness_fail")
    )
    if robustness_failed:
        score -= 0.45
    return score


def _family_priority(memory: dict[str, Any], family: str) -> tuple[float, str]:
    trials = float((memory.get("family_counts") or {}).get(family, 0))
    candidates = float((memory.get("candidate_family_counts") or {}).get(family, 0))
    self_corr = float((memory.get("self_correlation_family_counts") or {}).get(family, 0))
    recent = [item for item in (memory.get("recent_trials") or []) if item.get("family") == family]
    top_promises = sorted((_trial_promise_score(item) for item in recent), reverse=True)[:3]
    promise = sum(top_promises) / len(top_promises) if top_promises else 0.0
    candidate_rate = candidates / max(1.0, trials)
    self_corr_rate = self_corr / max(1.0, trials)
    points_gate = ((memory.get("learning_maturity") or {}).get("points_planner_weighting") or {})
    points_feedback = 0.0
    if points_gate.get("ready"):
        points_feedback = max(
            0.0,
            float((memory.get("family_points_feedback_usable") or {}).get(family, 0.0) or 0.0),
        )
    points_reward = min(1.0, math.log1p(points_feedback / 500.0) * 0.4)
    # Lower is better: retain exploration pressure while exploiting families that
    # have produced candidates, several near-threshold trials, or conservative
    # confidence-weighted evidence of settled leaderboard Points.
    score = math.log1p(trials) - candidate_rate * 8.0 - promise * 1.5 - points_reward + self_corr_rate * 4.0
    return score, family


def select_research_families(memory: dict[str, Any] | None = None, count: int = 3) -> list[str]:
    memory = memory or {}
    adaptive = memory.get("adaptive_allocation") or {}
    selected: list[str] = []
    for cell in adaptive.get("selected_cells") or []:
        family = str(cell.get("family") or "")
        if family in FAMILY_SEEDS and family not in selected:
            selected.append(family)
    scheduler_gate = ((memory.get("learning_maturity") or {}).get("scheduler_adaptation") or {})
    if scheduler_gate and not scheduler_gate.get("ready"):
        family_counts = memory.get("family_counts") or {}
        ordered = sorted(FAMILY_SEEDS, key=lambda family: (int(family_counts.get(family, 0)), family))
    else:
        ordered = sorted(FAMILY_SEEDS, key=lambda family: _family_priority(memory, family))
    for family in ordered:
        if family not in selected:
            selected.append(family)
    return selected[: max(1, min(len(selected), int(count)))]


def _unknown_fields_from_memory(memory: dict[str, Any]) -> set[str]:
    """Learn account-specific unsupported BRAIN fields from prior remote errors."""
    unknown: set[str] = set()
    pattern = re.compile(r'unknown variable ["\']([^"\']+)["\']', re.IGNORECASE)
    for trial in memory.get("recent_trials") or []:
        for target in trial.get("mutation_targets") or []:
            match = pattern.search(str(target))
            if match:
                unknown.add(match.group(1).strip().lower())
    return unknown


def _uses_unknown_field(expression: str, unknown_fields: set[str]) -> bool:
    normalized = normalize_wq_expression(expression)
    return any(re.search(rf"(?<![a-z0-9_]){re.escape(field)}(?![a-z0-9_])", normalized) for field in unknown_fields)


_CORE_WQ_FIELDS = {
    "open", "high", "low", "close", "volume", "vwap", "returns", "cap",
    "market", "sector", "industry", "subindustry",
}


def _classify_live_field(field: dict[str, Any]) -> str:
    dataset = field.get("dataset") or {}
    text = " ".join(
        str(value or "").lower()
        for value in (
            field.get("id"), field.get("name"), field.get("description"),
            dataset.get("id") if isinstance(dataset, dict) else dataset,
            dataset.get("name") if isinstance(dataset, dict) else "",
        )
    )
    if any(token in text for token in ("analyst", "estimate", "revision", "recommendation")):
        return "analyst_revision"
    if any(token in text for token in ("option", "implied_vol", "historical_vol")):
        return "options_volatility"
    if any(token in text for token in ("sentiment", "social", "news", "buzz", "nws")):
        return "sentiment"
    if any(token in text for token in ("cashflow", "fundamental", "asset", "debt", "earning", "book", "quality", "leverage")):
        return "fundamental_quality"
    if any(token in text for token in ("volume", "vwap", "price", "turnover")):
        return "price_volume"
    if any(token in text for token in ("volatility", "variance", "std", "risk")):
        return "volatility_structure"
    return "live_data"


def _live_field_id(field: dict[str, Any]) -> str:
    field_id = str(field.get("id") or "").strip()
    return field_id if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field_id) else ""


def _dataset_category(dataset: dict[str, Any]) -> str:
    category = dataset.get("category") or {}
    if isinstance(category, dict):
        return str(category.get("id") or category.get("name") or "unknown").lower()
    return str(category or "unknown").lower()


def _select_live_datasets(datasets: list[dict[str, Any]], memory: dict[str, Any], *, limit: int = 6) -> list[dict[str, Any]]:
    """Prefer underused datasets while spreading a research round across categories."""
    usage = Counter(str(item.get("dataset_id") or "") for item in (memory.get("recent_trials") or []))
    points_gate = ((memory.get("learning_maturity") or {}).get("points_planner_weighting") or {})
    points_feedback = (
        {
            str(key): max(0.0, float(value or 0.0))
            for key, value in (memory.get("dataset_points_feedback") or {}).items()
        }
        if points_gate.get("ready")
        else {}
    )
    by_category: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        dataset_id = str(dataset.get("id") or "").strip()
        if not dataset_id:
            continue
        by_category.setdefault(_dataset_category(dataset), []).append(dataset)
    def dataset_rank(item: dict[str, Any]) -> tuple[float, str]:
        dataset_id = str(item.get("id") or "")
        feedback_reward = min(2.0, points_feedback.get(dataset_id, 0.0) / 1000.0)
        return usage[dataset_id] - feedback_reward, dataset_id

    for values in by_category.values():
        values.sort(key=dataset_rank)

    selected: list[dict[str, Any]] = []
    categories = sorted(by_category, key=lambda category: min(dataset_rank(item)[0] for item in by_category[category]))
    while len(selected) < max(1, int(limit)):
        added = False
        for category in categories:
            values = by_category[category]
            if not values:
                continue
            selected.append(values.pop(0))
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


def _registered_skill_fields(
    skill_candidates: list[dict[str, Any]],
    memory: dict[str, Any],
) -> list[dict[str, Any]]:
    """Seed strict Skill-first validation from restart-safe truthful field evidence.

    ``field_registry`` is built only from persisted trials/candidates with a
    non-ambiguous dataset attribution.  Reusing those observations avoids
    re-querying Data Explorer for fields the project has already executed while
    unknown fields still fall through to build_skill_plan's exact live lookup.
    """
    registry = memory.get("field_registry") or {}
    declared = {
        str(field).strip().lower()
        for candidate in skill_candidates
        if isinstance(candidate, dict)
        for field in (candidate.get("data_fields") or [])
        if str(field).strip()
    }
    fields: list[dict[str, Any]] = []
    for field_id in sorted(declared):
        metadata = registry.get(field_id)
        if not isinstance(metadata, dict):
            continue
        dataset_id = str(metadata.get("dataset_id") or "").strip()
        if not dataset_id:
            continue
        category = str(metadata.get("dataset_category") or "").strip()
        dataset: dict[str, Any] = {"id": dataset_id}
        if category:
            dataset["category"] = {"id": category}
        fields.append({
            "id": field_id,
            "type": "MATRIX",
            "dataset": dataset,
            "registry_source": metadata.get("source") or "persisted_local_registry",
        })
    return fields


def _live_field_candidates(client, memory: dict[str, Any], *, region: str, universe: str, delay: int, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch account-visible MATRIX fields and rank underused fields first."""
    if limit <= 0 or not hasattr(client, "list_data_fields"):
        return [], {"available": False, "count": 0, "error": "data-field catalog unavailable"}

    raw: list[dict[str, Any]] = []
    selected_datasets: list[dict[str, Any]] = []
    if hasattr(client, "list_datasets"):
        try:
            datasets = client.list_datasets(region=region, universe=universe, delay=delay, limit=100)
            selected_datasets = _select_live_datasets(datasets, memory, limit=min(6, max(3, limit)))
            per_dataset = max(6, math.ceil(max(36, limit * 8) / max(1, len(selected_datasets))))
            for dataset in selected_datasets:
                dataset_id = str(dataset.get("id") or "").strip()
                try:
                    fields = client.list_data_fields(
                        region=region,
                        universe=universe,
                        delay=delay,
                        dataset_id=dataset_id,
                        limit=per_dataset,
                    )
                except Exception:
                    continue
                for field in fields:
                    item = dict(field)
                    if not item.get("dataset"):
                        item["dataset"] = {"id": dataset_id, "name": dataset.get("name")}
                    raw.append(item)
        except Exception:
            selected_datasets = []

    if not raw:
        try:
            raw = client.list_data_fields(region=region, universe=universe, delay=delay, limit=max(80, limit * 12))
        except Exception as exc:
            return [], {"available": False, "count": 0, "error": str(exc)[:300]}

    unknown_fields = _unknown_fields_from_memory(memory)
    previous = [str(item.get("expression") or "").lower() for item in (memory.get("recent_trials") or [])]
    ranked_by_dataset: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
    for field in raw:
        field_id = _live_field_id(field)
        if not field_id or field_id.lower() in unknown_fields:
            continue
        field_type = str(field.get("type") or "MATRIX").upper()
        if field_type != "MATRIX":
            continue
        usage = sum(1 for expression in previous if re.search(rf"(?<![a-z0-9_]){re.escape(field_id.lower())}(?![a-z0-9_])", expression))
        dataset = field.get("dataset") or {}
        dataset_id = str(dataset.get("id") or "unknown") if isinstance(dataset, dict) else str(dataset or "unknown")
        ranked_by_dataset.setdefault(dataset_id, []).append((usage, field_id.lower(), field))
    for values in ranked_by_dataset.values():
        values.sort(key=lambda item: (item[0], item[1]))

    # Round-robin datasets so an alphabetically early fundamental dataset cannot
    # monopolize the live-field budget when several data sources are available.
    fields: list[dict[str, Any]] = []
    dataset_order = sorted(
        ranked_by_dataset,
        key=lambda dataset_id: (
            min((item[0] for item in ranked_by_dataset[dataset_id]), default=0),
            dataset_id,
        ),
    )
    while True:
        added = False
        for dataset_id in dataset_order:
            values = ranked_by_dataset[dataset_id]
            if not values:
                continue
            fields.append(values.pop(0)[2])
            added = True
        if not added:
            break
    return fields, {
        "available": True,
        "count": len(fields),
        "raw_count": len(raw),
        "selected_datasets": [str(item.get("id") or "") for item in selected_datasets],
        "selected_categories": sorted({_dataset_category(item) for item in selected_datasets}),
        "sample_ids": [_live_field_id(item) for item in fields[:10]],
    }


def _live_field_templates(field_id: str) -> list[str]:
    base = f"ts_backfill({field_id}, 60)"
    return [
        f"rank(ts_mean({base}, 20))",
        f"-rank(ts_mean({base}, 20))",
        f"rank(ts_delta({base}, 20))",
        f"-rank(ts_std_dev({base}, 20))",
    ]


def _build_live_field_plan_from_fields(
    client,
    fields: list[dict[str, Any]],
    *,
    seen: set[str],
    limit: int,
    hypothesis: str,
) -> list[dict[str, Any]]:
    if not fields or limit <= 0:
        return []
    try:
        supported = client.list_operator_names()
    except Exception:
        supported = set()

    plan: list[dict[str, Any]] = []
    for template_index in range(4):
        for field in fields:
            field_id = _live_field_id(field)
            templates = _live_field_templates(field_id)
            expression = templates[template_index]
            validation = validate_wq_expression(expression, supported) if supported else None
            if validation is not None and not validation.ok:
                continue
            expression = validation.expression if validation is not None else expression
            normalized = normalize_wq_expression(expression)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            dataset = field.get("dataset") or {}
            dataset_id = dataset.get("id") if isinstance(dataset, dict) else dataset
            dataset_category = _dataset_category(dataset) if isinstance(dataset, dict) else None
            plan.append(
                {
                    "expression": expression,
                    "family": _classify_live_field(field),
                    "generation": 1,
                    "hypothesis": hypothesis,
                    "parent_expression": None,
                    "mutation_type": "live_field_seed",
                    "planner_strategy": "live_catalog_coverage",
                    "data_fields": [field_id],
                    "dataset_id": dataset_id,
                    "dataset_category": dataset_category,
                    "provenance_state": "resolved" if dataset_id else "unresolved",
                    "provenance_reason": "live_field_catalog" if dataset_id else "live_field_missing_dataset_id",
                }
            )
            if len(plan) >= limit:
                return plan
    return plan


def build_live_field_plan(
    client,
    memory: dict[str, Any] | None,
    *,
    seen: set[str],
    limit: int,
    region: str,
    universe: str,
    delay: int,
    hypothesis: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    memory = memory or {}
    fields, catalog = _live_field_candidates(client, memory, region=region, universe=universe, delay=delay, limit=limit)
    plan = _build_live_field_plan_from_fields(client, fields, seen=seen, limit=limit, hypothesis=hypothesis)
    return plan, catalog, fields


def _active_motif_names(active_alphas: list[dict[str, Any]]) -> list[str]:
    """Extract only broad motifs that have already survived BRAIN formal submission."""
    motifs: list[str] = []
    expressions = [str(alpha.get("expression") or "").lower() for alpha in active_alphas]
    if any("ts_decay_linear" in expression and "rank(-returns)" in expression for expression in expressions):
        motifs.append("slow_field_x_reversal")
    if any("ts_std_dev(ts_backfill" in expression and "/" in expression for expression in expressions):
        motifs.append("field_stability_ratio")
    if any("rank(-ts_mean(ts_backfill" in expression or "-rank(ts_mean(ts_backfill" in expression for expression in expressions):
        motifs.append("slow_negative_field")
    if any("ts_mean(ts_backfill" in expression for expression in expressions):
        motifs.append("slow_ranked_field")
    return motifs


def _active_reference_profile(active_alphas: list[dict[str, Any]]) -> dict[str, float]:
    """Build a robust median metric profile from platform-authoritative ACTIVE alphas."""
    profile: dict[str, float] = {}
    for key in ("sharpe", "fitness", "returns", "turnover"):
        values = sorted(
            _safe_metric(alpha.get(key), math.nan)
            for alpha in active_alphas
            if math.isfinite(_safe_metric(alpha.get(key), math.nan))
        )
        if values:
            middle = len(values) // 2
            profile[key] = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
    return profile


def _active_reference_score(item: dict[str, Any], profile: dict[str, float]) -> float:
    """Score candidate quality against real ACTIVE outcomes without requiring many labels."""
    if not profile:
        return 0.0
    metrics = item.get("is_metrics") or item
    sharpe = _safe_metric(metrics.get("sharpe"))
    fitness = _safe_metric(metrics.get("fitness"))
    returns = _safe_metric(metrics.get("returns"))
    turnover = _safe_metric(metrics.get("turnover"))

    def progress(value: float, target: float) -> float:
        if target <= 0:
            return 0.0
        return max(0.0, min(1.0, value / target))

    turnover_component = 1.0 if 0.03 <= turnover <= 0.5 else max(0.0, 1.0 - abs(turnover - profile.get("turnover", 0.2)) / 0.7)
    return round(
        progress(fitness, profile.get("fitness", 1.0)) * 0.45
        + progress(sharpe, profile.get("sharpe", 1.5)) * 0.30
        + progress(returns, profile.get("returns", 0.10)) * 0.10
        + turnover_component * 0.15,
        4,
    )


def _active_dataset_sibling_fields(
    client,
    active_alphas: list[dict[str, Any]],
    *,
    region: str,
    universe: str,
    delay: int,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover sibling fields from datasets that already produced ACTIVE Alpha.

    Replenishment used to prefer globally underused datasets, which can starve the
    few datasets that have already demonstrated real ACTIVE conversion.  This
    bounded probe keeps the successful dataset while changing the field, so the
    existing ACTIVE-motif transfer can exploit proven information sources without
    cloning the submitted expression itself.
    """
    if limit <= 0 or not active_alphas or not hasattr(client, "list_data_fields"):
        return [], {"available": False, "count": 0, "source_fields": [], "dataset_ids": []}

    ranked_active = sorted(
        active_alphas,
        key=lambda item: (_safe_metric(item.get("fitness")), _safe_metric(item.get("sharpe"))),
        reverse=True,
    )
    source_fields: list[str] = []
    for alpha in ranked_active:
        expression = str(alpha.get("expression") or "")
        for field_id in extract_expression_metadata(expression).get("data_fields") or []:
            field_id = str(field_id or "").strip()
            if not field_id or field_id.lower() in _CORE_WQ_FIELDS or field_id in source_fields:
                continue
            source_fields.append(field_id)
            if len(source_fields) >= 8:
                break
        if len(source_fields) >= 8:
            break

    active_field_set = set(source_fields)
    dataset_sources: dict[str, set[str]] = {}
    dataset_meta: dict[str, dict[str, Any]] = {}
    for source_field in source_fields:
        try:
            matches = client.list_data_fields(
                region=region,
                universe=universe,
                delay=delay,
                search=source_field,
                limit=20,
            )
        except Exception:
            continue
        exact = next((item for item in matches if _live_field_id(item) == source_field), None)
        if exact is None:
            continue
        dataset = exact.get("dataset") or {}
        dataset_id = str(dataset.get("id") or "") if isinstance(dataset, dict) else str(dataset or "")
        if not dataset_id:
            continue
        dataset_sources.setdefault(dataset_id, set()).add(source_field)
        if isinstance(dataset, dict):
            dataset_meta[dataset_id] = dataset

    def lexical_similarity(source: str, candidate: str) -> float:
        source_tokens = {token for token in source.lower().split("_") if token and not token.isdigit()}
        candidate_tokens = {token for token in candidate.lower().split("_") if token and not token.isdigit()}
        if not source_tokens or not candidate_tokens:
            return 0.0
        return len(source_tokens & candidate_tokens) / max(1, len(source_tokens | candidate_tokens))

    ranked_siblings: list[tuple[float, str, dict[str, Any]]] = []
    for dataset_id, sources in dataset_sources.items():
        try:
            siblings = client.list_data_fields(
                region=region,
                universe=universe,
                delay=delay,
                dataset_id=dataset_id,
                limit=max(20, limit * 4),
            )
        except Exception:
            continue
        for item in siblings:
            field_id = _live_field_id(item)
            if not field_id or field_id in active_field_set or str(item.get("type") or "MATRIX").upper() != "MATRIX":
                continue
            score = max((lexical_similarity(source, field_id) for source in sources), default=0.0)
            ranked_siblings.append((score, field_id, item))

    ranked_siblings.sort(key=lambda item: (-item[0], item[1]))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for _score, field_id, item in ranked_siblings:
        if field_id in selected_ids:
            continue
        selected_ids.add(field_id)
        selected.append(item)
        if len(selected) >= limit:
            break

    dataset_ids = sorted(dataset_sources)
    return selected, {
        "available": bool(dataset_ids),
        "count": len(selected),
        "source_fields": source_fields,
        "dataset_ids": dataset_ids,
        "datasets": [dataset_meta.get(dataset_id, {"id": dataset_id}) for dataset_id in dataset_ids],
        "sample_ids": [_live_field_id(item) for item in selected[:10]],
    }


def build_active_motif_plan(
    client,
    active_alphas: list[dict[str, Any]],
    fields: list[dict[str, Any]],
    *,
    seen: set[str],
    limit: int,
    hypothesis: str,
) -> list[dict[str, Any]]:
    """Transfer proven ACTIVE operator motifs onto different live fields.

    This deliberately avoids cloning the ACTIVE expression itself: the successful
    transformation shape is reused while the information source changes, reducing
    the chance that exploitation simply creates a self-correlation failure.
    """
    if limit <= 0 or not active_alphas or not fields:
        return []
    motifs = _active_motif_names(active_alphas)
    if not motifs:
        return []
    try:
        supported = client.list_operator_names()
    except Exception:
        supported = set()

    plan: list[dict[str, Any]] = []
    active_expressions = [str(alpha.get("expression") or "") for alpha in active_alphas]
    active_fields = set().union(*(set(extract_expression_metadata(expression).get("data_fields") or []) for expression in active_expressions))
    for motif in motifs:
        for field in fields:
            field_id = _live_field_id(field)
            if not field_id or field_id in active_fields:
                continue
            base = f"ts_backfill({field_id}, 60)"
            if motif == "slow_field_x_reversal":
                expression = f"ts_decay_linear(rank(ts_mean({base}, 20)) * rank(-returns), 5)"
            elif motif == "field_stability_ratio":
                expression = f"rank(ts_mean({base}, 20) / (ts_std_dev({base}, 60) + 0.001))"
            elif motif == "slow_negative_field":
                expression = f"rank(-ts_mean({base}, 20))"
            else:
                expression = f"rank(ts_mean({base}, 20))"
            validation = validate_wq_expression(expression, supported) if supported else None
            if validation is not None and not validation.ok:
                continue
            expression = validation.expression if validation is not None else expression
            normalized = normalize_wq_expression(expression)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            dataset = field.get("dataset") or {}
            dataset_id = dataset.get("id") if isinstance(dataset, dict) else dataset
            dataset_category = _dataset_category(dataset) if isinstance(dataset, dict) else None
            plan.append(
                {
                    "expression": expression,
                    "family": _classify_live_field(field),
                    "generation": 1,
                    "hypothesis": hypothesis,
                    "parent_expression": None,
                    "mutation_type": "active_motif_transfer",
                    "mutation_reason": f"transfer proven ACTIVE motif {motif} onto a different live field",
                    "planner_strategy": "active_motif_transfer",
                    "data_fields": [field_id],
                    "dataset_id": dataset_id,
                    "dataset_category": dataset_category,
                    "provenance_state": "resolved" if dataset_id else "unresolved",
                    "provenance_reason": "active_motif_live_field_transfer",
                }
            )
            if len(plan) >= limit:
                return plan
    return plan


def build_structural_live_plan(
    client,
    fields: list[dict[str, Any]],
    *,
    seen: set[str],
    limit: int,
    hypothesis: str,
) -> list[dict[str, Any]]:
    """Generate bounded cross-field structural motifs from live BRAIN fields."""
    if limit <= 0 or len(fields) < 2:
        return []
    try:
        supported = client.list_operator_names()
    except Exception:
        return []
    if not supported:
        return []

    valid_fields = [item for item in fields if _live_field_id(item)]
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for item in valid_fields:
        dataset = item.get("dataset") or {}
        dataset_id = str(dataset.get("id") or "") if isinstance(dataset, dict) else str(dataset or "")
        if dataset_id:
            by_dataset.setdefault(dataset_id, []).append(item)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for dataset_id in sorted(by_dataset):
        values = by_dataset[dataset_id]
        for index in range(0, len(values) - 1, 2):
            pairs.append((values[index], values[index + 1]))
    if not pairs:
        pairs = list(zip(valid_fields[::2], valid_fields[1::2]))

    def templates(left: str, right: str) -> list[str]:
        left_base = f"ts_backfill({left}, 60)"
        right_base = f"ts_backfill({right}, 60)"
        return [
            f"rank(ts_corr({left_base}, {right_base}, 20))",
            f"group_rank(({left_base} - {right_base}), subindustry)",
            f"rank(ts_mean(({left_base} - {right_base}), 20))",
            f"rank(ts_delta({left_base}, 20) - ts_delta({right_base}, 20))",
        ]

    plan: list[dict[str, Any]] = []
    for template_index in range(4):
        for left_item, right_item in pairs:
            left = _live_field_id(left_item)
            right = _live_field_id(right_item)
            if not left or not right or left == right:
                continue
            expression = templates(left, right)[template_index]
            validation = validate_wq_expression(expression, supported)
            if not validation.ok:
                continue
            expression = validation.expression
            normalized = normalize_wq_expression(expression)
            if not normalized or normalized in seen:
                continue
            if not _expression_uses_only_catalog_fields(expression, supported, {left, right}):
                continue
            seen.add(normalized)
            left_dataset = left_item.get("dataset") or {}
            right_dataset = right_item.get("dataset") or {}
            left_dataset_id = str(left_dataset.get("id") or "") if isinstance(left_dataset, dict) else str(left_dataset or "")
            right_dataset_id = str(right_dataset.get("id") or "") if isinstance(right_dataset, dict) else str(right_dataset or "")
            dataset_id = left_dataset_id if left_dataset_id and left_dataset_id == right_dataset_id else None
            category = _dataset_category(left_dataset) if isinstance(left_dataset, dict) else None
            right_category = _dataset_category(right_dataset) if isinstance(right_dataset, dict) else None
            dataset_category = category if category and category == right_category else None
            plan.append(
                {
                    "expression": expression,
                    "family": _classify_live_field(left_item),
                    "generation": 1,
                    "hypothesis": hypothesis,
                    "parent_expression": None,
                    "mutation_type": "structural_live_seed",
                    "planner_strategy": "cross_field_structural_diversity",
                    "data_fields": [left, right],
                    "dataset_id": dataset_id,
                    "dataset_category": dataset_category,
                    "provenance_state": "resolved" if dataset_id else "partial",
                    "provenance_reason": "same_live_dataset_pair" if dataset_id else "cross_dataset_pair",
                }
            )
            if len(plan) >= limit:
                return plan
    return plan


def _expression_uses_only_catalog_fields(expression: str, operators: set[str], allowed_fields: set[str]) -> bool:
    tokens = {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)}
    allowed = {value.lower() for value in allowed_fields} | _CORE_WQ_FIELDS | {f"adv{days}" for days in (5, 10, 20, 60, 120)}
    return all(token in operators or token in allowed for token in tokens)


REQUIRED_WQ_SKILL_CHAIN = (
    "wq-alpha-hypothesis",
    "wq-alpha-review",
    "wq-robustness-validation",
    "wq-candidate-evidence",
)

_SIMULATION_SETTING_KEYS = ("region", "universe", "delay", "decay", "neutralization", "truncation")


def _simulation_variant_key(expression: str, settings: dict[str, Any] | None) -> tuple[str, ...] | None:
    normalized = normalize_wq_expression(expression)
    if not normalized or not isinstance(settings, dict):
        return None
    values: list[str] = [normalized]
    for key in _SIMULATION_SETTING_KEYS:
        value = settings.get(key)
        if key in {"region", "universe", "neutralization"}:
            values.append(str(value or "").upper())
        elif key in {"delay", "decay"}:
            try:
                values.append(str(int(value)))
            except (TypeError, ValueError):
                values.append(str(value or ""))
        elif key == "truncation":
            try:
                values.append(f"{float(value):.10g}")
            except (TypeError, ValueError):
                values.append(str(value or ""))
    return tuple(values)


def _settings_only_repair_allowed(
    candidate: dict[str, Any],
    *,
    current_settings: dict[str, Any] | None,
    seen_variants: set[tuple[str, ...]] | None,
) -> bool:
    """Allow an exact-expression repair only for one explicit, not-yet-tested setting delta."""
    chain = {str(value).strip() for value in (candidate.get("skill_chain") or []) if str(value).strip()}
    if "wq-alpha-repair" not in chain or not isinstance(current_settings, dict):
        return False
    delta = candidate.get("settings_delta")
    if not isinstance(delta, dict) or len(delta) != 1:
        return False
    setting, change = next(iter(delta.items()))
    if setting not in _SIMULATION_SETTING_KEYS or not isinstance(change, dict):
        return False
    if "from" not in change or "to" not in change or change.get("from") == change.get("to"):
        return False

    target_settings = dict(current_settings)
    target_settings[setting] = change.get("to")
    source_settings = dict(current_settings)
    source_settings[setting] = change.get("from")
    expression = str(candidate.get("expression") or "")
    target_key = _simulation_variant_key(expression, target_settings)
    source_key = _simulation_variant_key(expression, source_settings)
    current_key = _simulation_variant_key(expression, current_settings)
    # The top-level autonomous-research settings must actually represent the requested target.
    if target_key is None or source_key is None or current_key != target_key:
        return False
    prior_variants = seen_variants or set()
    if prior_variants and source_key not in prior_variants:
        return False
    return target_key not in prior_variants


def validate_skill_candidate_contract(candidate: dict[str, Any]) -> str | None:
    """Validate client-side DevSpace skill provenance before spending BRAIN budget."""
    expression = str(candidate.get("expression") or "").strip()
    hypothesis = str(candidate.get("hypothesis") or "").strip()
    skill_chain = [str(value).strip() for value in (candidate.get("skill_chain") or []) if str(value).strip()]
    review_decision = str(candidate.get("review_decision") or "").strip().upper()
    if not expression:
        return "missing expression"
    if not hypothesis:
        return "missing hypothesis"
    missing = [name for name in REQUIRED_WQ_SKILL_CHAIN if name not in skill_chain]
    if missing:
        return f"missing required skill(s): {', '.join(missing)}"
    if review_decision != "RUN":
        return "review_decision must be RUN"
    robustness_plan = candidate.get("robustness_plan")
    if not isinstance(robustness_plan, dict) or str(robustness_plan.get("mode") or "") != "skill_defined":
        return "missing skill-defined robustness_plan"
    checks = robustness_plan.get("checks") or []
    if not isinstance(checks, list) or not 1 <= len(checks) <= 4:
        return "robustness_plan.checks must contain 1-4 targeted checks"
    evidence_policy = candidate.get("candidate_evidence_policy")
    if not isinstance(evidence_policy, dict) or str(evidence_policy.get("mode") or "") != "calibrated_evidence_hierarchy":
        return "missing candidate_evidence_policy"

    if "wq-alpha-repair" in skill_chain:
        for required in ("wq-failure-diagnosis", "wq-experiment-allocation"):
            if required not in skill_chain:
                return f"repair candidate missing required skill: {required}"
        signature = candidate.get("failure_signature")
        if not isinstance(signature, dict):
            return "repair candidate missing failure_signature"
        if not isinstance(signature.get("observed_symptoms"), list) or not isinstance(signature.get("plausible_causes"), list):
            return "failure_signature must preserve observed_symptoms and plausible_causes"

    if "wq-alpha-diversify" in skill_chain:
        diversity_case = candidate.get("diversity_case")
        if not isinstance(diversity_case, dict):
            return "diversified candidate missing diversity_case"
        changed = [str(value) for value in (diversity_case.get("changed_dimensions") or []) if str(value)]
        allowed = {"information_source", "economic_mechanism", "horizon_delay", "structure", "factor_exposure"}
        if not changed or not set(changed).issubset(allowed):
            return "diversity_case.changed_dimensions must name supported diversity dimensions"
        if not str(diversity_case.get("why_independent") or "").strip():
            return "diversity_case missing why_independent"
    return None


def build_skill_plan(
    client,
    candidates: list[dict[str, Any]] | None,
    fields: list[dict[str, Any]],
    *,
    seen: set[str],
    limit: int,
    hypothesis: str,
    current_settings: dict[str, Any] | None = None,
    seen_variants: set[tuple[str, ...]] | None = None,
    rejections: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Validate DevSpace skill-authored candidates and preserve their research provenance."""
    if limit <= 0 or not candidates:
        return []
    try:
        supported = client.list_operator_names()
    except Exception:
        return []

    usable_fields = [_live_field_id(field) for field in fields if _live_field_id(field)][:48]
    field_items: dict[str, dict[str, Any]] = {
        _live_field_id(field).lower(): field
        for field in fields
        if _live_field_id(field)
    }
    out: list[dict[str, Any]] = []

    def reject(raw_candidate: Any, reason: str, detail: str | None = None) -> None:
        if rejections is None:
            return
        item = raw_candidate if isinstance(raw_candidate, dict) else {}
        row: dict[str, Any] = {
            "expression": str(item.get("expression") or "").strip(),
            "family": str(item.get("family") or "").strip(),
            "reason": reason,
        }
        if detail:
            row["detail"] = detail[:300]
        rejections.append(row)

    for raw_candidate in candidates:
        if len(out) >= limit:
            reject(raw_candidate, "batch_limit_exceeded")
            continue
        if not isinstance(raw_candidate, dict):
            reject(raw_candidate, "invalid_candidate_type")
            continue
        contract_error = validate_skill_candidate_contract(raw_candidate)
        if contract_error:
            reject(raw_candidate, "skill_contract_rejected", contract_error)
            continue
        expression = str(raw_candidate.get("expression") or "").strip()
        validation = validate_wq_expression(expression, supported)
        if not validation.ok:
            reject(raw_candidate, "expression_validation_rejected", str(validation.error or "invalid expression"))
            continue
        expression = validation.expression
        normalized = normalize_wq_expression(expression)
        if not normalized:
            reject(raw_candidate, "empty_normalized_expression")
            continue
        if normalized in seen and not _settings_only_repair_allowed(
            raw_candidate, current_settings=current_settings, seen_variants=seen_variants
        ):
            reject(raw_candidate, "duplicate_expression_history")
            continue

        # Skill-first candidates are authored after ChatGPT has read the live Data
        # Explorer.  The autonomous planner's small coverage sample is therefore
        # not an authoritative allow-list.  Re-resolve declared fields against
        # BRAIN so valid skill-authored ideas are not silently dropped merely
        # because their field was outside the planner's unrelated sample.
        declared_fields = [
            str(value).strip()
            for value in (raw_candidate.get("data_fields") or [])
            if str(value).strip()
        ]
        for field_id in declared_fields:
            key = field_id.lower()
            if key in field_items or key in _CORE_WQ_FIELDS or re.fullmatch(r"adv(?:5|10|20|60|120)", key):
                continue
            try:
                matches = client.list_data_fields(search=field_id, limit=20)
            except Exception:
                matches = []
            exact = next(
                (
                    item for item in matches
                    if _live_field_id(item).lower() == key
                    and str(item.get("type") or "MATRIX").upper() == "MATRIX"
                ),
                None,
            )
            if exact is not None:
                field_items[key] = exact
                usable_fields.append(_live_field_id(exact))

        allowed_fields = set(usable_fields) | set(declared_fields)
        if not _expression_uses_only_catalog_fields(expression, supported, allowed_fields):
            reject(raw_candidate, "field_catalog_rejected")
            continue
        # Do not trust caller-declared field names by themselves: every non-core
        # field used by the expression must have been observed in either the
        # planner catalog or the exact BRAIN lookup above.
        expression_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)
        }
        unresolved = {
            token
            for token in expression_tokens
            if token not in supported
            and token not in _CORE_WQ_FIELDS
            and not re.fullmatch(r"adv(?:5|10|20|60|120)", token)
            and token not in field_items
        }
        if unresolved:
            reject(raw_candidate, "unresolved_data_fields", ",".join(sorted(unresolved)))
            continue
        seen.add(normalized)

        used = [
            field_id
            for key, item in field_items.items()
            if key in expression_tokens
            for field_id in [_live_field_id(item)]
            if field_id
        ]
        used_items = [field_items[field.lower()] for field in used if field.lower() in field_items]
        dataset_ids = {
            str((item.get("dataset") or {}).get("id") or "")
            for item in used_items
            if isinstance(item.get("dataset") or {}, dict) and (item.get("dataset") or {}).get("id")
        }
        categories = {
            _dataset_category(item.get("dataset") or {})
            for item in used_items
            if isinstance(item.get("dataset") or {}, dict)
        }
        dataset_id = next(iter(dataset_ids)) if len(dataset_ids) == 1 else raw_candidate.get("dataset_id")
        dataset_category = next(iter(categories)) if len(categories) == 1 else raw_candidate.get("dataset_category")
        out.append(
            {
                "expression": expression,
                "family": str(raw_candidate.get("family") or classify_wq_family(expression)),
                "generation": 1,
                "hypothesis": str(raw_candidate.get("hypothesis") or hypothesis),
                "parent_expression": raw_candidate.get("parent_expression"),
                "mutation_type": str(raw_candidate.get("mutation_type") or "skill_seed"),
                "mutation_reason": raw_candidate.get("mutation_reason"),
                "planner_strategy": "devspace_skill",
                "data_fields": used or list(raw_candidate.get("data_fields") or []),
                "dataset_id": dataset_id,
                "dataset_category": dataset_category,
                "provenance_state": "resolved" if dataset_id else "unresolved",
                "provenance_reason": "devspace_skill_live_catalog" if dataset_id else "devspace_skill_core_or_multi_dataset_expression",
                "generation_source": "devspace_skill",
                "skill_chain": list(raw_candidate.get("skill_chain") or []),
                "skill_review_decision": "RUN",
                "skill_review_notes": raw_candidate.get("review_notes"),
                "skill_provenance_verified": True,
                "knowledge_card_ids": list(raw_candidate.get("knowledge_card_ids") or []),
                "robustness_plan": dict(raw_candidate.get("robustness_plan") or {}),
                "candidate_evidence_policy": dict(raw_candidate.get("candidate_evidence_policy") or {}),
                "failure_signature": dict(raw_candidate.get("failure_signature") or {}),
                "diversity_case": dict(raw_candidate.get("diversity_case") or {}),
                "settings_delta": dict(raw_candidate.get("settings_delta") or {}),
            }
        )
    return out


def enforce_active_fill_batch_diversity(
    plan: list[dict[str, Any]],
    *,
    rejections: list[dict[str, Any]] | None = None,
    max_structure_share: float = 0.4,
    max_repairs_per_parent: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bound same-structure concentration before ACTIVE_FILL spends simulations.

    This is an execution-budget guard, not an Alpha-quality veto: rejected near-
    clones should be replaced by another Skill-authored causal bucket.
    """
    if not plan:
        return [], {
            "policy": "active_fill_structure_neighborhood_cap",
            "max_structure_share": max_structure_share,
            "unique_structure_neighborhoods": 0,
            "max_structure_count": 0,
            "rejected_for_concentration": 0,
            "rejected_for_parent_repair_concentration": 0,
            "max_repairs_per_parent": max(1, int(max_repairs_per_parent)),
        }
    signatures = [
        str(extract_expression_metadata(item["expression"]).get("structure_signature") or item["expression"])
        for item in plan
    ]
    cap = max(2, math.ceil(len(plan) * max(0.2, min(0.6, float(max_structure_share))))) if len(plan) >= 4 else None
    accepted: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    repair_parent_counts: Counter[str] = Counter()
    rejected = 0
    parent_rejected = 0
    repair_parent_cap = max(1, int(max_repairs_per_parent))
    for item, signature in zip(plan, signatures):
        chain = set(item.get("skill_chain") or [])
        parent_expression = str(item.get("parent_expression") or "").strip()
        parent_key = None
        if "wq-alpha-repair" in chain and parent_expression:
            parent_key = normalize_wq_expression(parent_expression)
            if repair_parent_counts[parent_key] >= repair_parent_cap:
                parent_rejected += 1
                if rejections is not None:
                    rejections.append({
                        "expression": str(item.get("expression") or ""),
                        "family": str(item.get("family") or ""),
                        "reason": "active_fill_parent_repair_concentration",
                        "detail": f"repair parent capped at {repair_parent_cap} candidates per batch",
                    })
                continue
        if cap is not None and counts[signature] >= cap:
            rejected += 1
            if rejections is not None:
                rejections.append({
                    "expression": str(item.get("expression") or ""),
                    "family": str(item.get("family") or ""),
                    "reason": "active_fill_structure_concentration",
                    "detail": f"structure neighborhood capped at {cap}/{len(plan)} supplied candidates",
                })
            continue
        if parent_key is not None:
            repair_parent_counts[parent_key] += 1
        counts[signature] += 1
        accepted.append(item)

    accepted_signatures = [
        str(extract_expression_metadata(item["expression"]).get("structure_signature") or item["expression"])
        for item in accepted
    ]
    accepted_counts = Counter(accepted_signatures)
    return accepted, {
        "policy": "active_fill_structure_neighborhood_cap",
        "max_structure_share": max_structure_share,
        "structure_cap": cap,
        "unique_structure_neighborhoods": len(accepted_counts),
        "max_structure_count": max(accepted_counts.values()) if accepted_counts else 0,
        "rejected_for_concentration": rejected,
        "rejected_for_parent_repair_concentration": parent_rejected,
        "max_repairs_per_parent": repair_parent_cap,
        "repair_parent_counts": dict(repair_parent_counts),
        "family_counts": dict(Counter(str(item.get("family") or "unknown") for item in accepted)),
        "route_counts": dict(Counter(
            "repair" if "wq-alpha-repair" in (item.get("skill_chain") or [])
            else "diversify" if "wq-alpha-diversify" in (item.get("skill_chain") or [])
            else "new_hypothesis"
            for item in accepted
        )),
    }


def build_chatgpt_plan(
    client,
    expressions: list[str] | None,
    fields: list[dict[str, Any]],
    *,
    seen: set[str],
    limit: int,
    hypothesis: str,
) -> list[dict[str, Any]]:
    """Validate legacy ChatGPT-authored FASTEXPR ideas without server-side LLM calls."""
    if limit <= 0 or not expressions:
        return []
    try:
        supported = client.list_operator_names()
    except Exception:
        return []

    usable_fields = [_live_field_id(field) for field in fields if _live_field_id(field)][:48]
    allowed_fields = set(usable_fields)
    out: list[dict[str, Any]] = []
    for raw_expression in expressions:
        if len(out) >= limit:
            break
        expression = str(raw_expression or "").strip()
        if not expression:
            continue
        validation = validate_wq_expression(expression, supported)
        if not validation.ok:
            continue
        expression = validation.expression
        normalized = normalize_wq_expression(expression)
        if not normalized or normalized in seen:
            continue
        if not _expression_uses_only_catalog_fields(expression, supported, allowed_fields):
            continue
        seen.add(normalized)

        used = [
            field
            for field in usable_fields
            if re.search(rf"(?<![a-z0-9_]){re.escape(field.lower())}(?![a-z0-9_])", normalized)
        ]
        used_items = [item for item in fields if _live_field_id(item) in used]
        dataset_ids = {
            str((item.get("dataset") or {}).get("id") or "")
            for item in used_items
            if isinstance(item.get("dataset") or {}, dict) and (item.get("dataset") or {}).get("id")
        }
        categories = {
            _dataset_category(item.get("dataset") or {})
            for item in used_items
            if isinstance(item.get("dataset") or {}, dict)
        }
        dataset_id = next(iter(dataset_ids)) if len(dataset_ids) == 1 else None
        dataset_category = next(iter(categories)) if len(categories) == 1 else None
        out.append(
            {
                "expression": expression,
                "family": classify_wq_family(expression),
                "generation": 1,
                "hypothesis": hypothesis,
                "parent_expression": None,
                "mutation_type": "chatgpt_seed",
                "planner_strategy": "chatgpt_client",
                "data_fields": used,
                "dataset_id": dataset_id,
                "dataset_category": dataset_category,
                "provenance_state": "resolved" if dataset_id else "unresolved",
                "provenance_reason": "live_field_catalog" if dataset_id else "chatgpt_core_or_multi_dataset_expression",
            }
        )
    return out


def build_knowledge_seed_plan(
    client,
    memory: dict[str, Any] | None,
    *,
    seen: set[str],
    limit: int,
    hypothesis: str,
    generation: int = 1,
) -> list[dict[str, Any]]:
    """Turn corroborated multi-source knowledge cards into bounded live-valid WQ seeds."""
    if limit <= 0:
        return []
    guidance = (memory or {}).get("knowledge_guidance") or {}
    templates = list(guidance.get("preferred_templates") or [])
    if not templates:
        return []
    try:
        supported = client.list_operator_names()
    except Exception:
        return []
    if not supported:
        return []
    templates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    out: list[dict[str, Any]] = []
    for item in templates:
        expression = str(item.get("expression") or "").strip()
        if not expression:
            continue
        validation = validate_wq_expression(expression, supported)
        if not validation.ok:
            continue
        expression = validation.expression
        normalized = normalize_wq_expression(expression)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(
            {
                "expression": expression,
                "family": str(item.get("family") or classify_wq_family(expression)),
                "generation": generation,
                "hypothesis": str(item.get("hypothesis") or hypothesis),
                "parent_expression": None,
                "mutation_type": "knowledge_seed",
                "mutation_reason": "multi_source_research_prior",
                "planner_strategy": "multi_source_knowledge",
                "knowledge_card_ids": [str(item.get("card_id"))] if item.get("card_id") else [],
                "knowledge_source_keys": list(item.get("source_keys") or []),
                "knowledge_failure_modes": list(item.get("failure_modes") or []),
                "knowledge_mutation_strategies": list(item.get("mutation_strategies") or []),
                "knowledge_empirical": dict(item.get("empirical") or {}),
                "knowledge_confidence": item.get("confidence"),
            }
        )
        if len(out) >= limit:
            break
    return out


def _seed_meta(
    expression: str,
    family: str,
    generation: int,
    hypothesis: str,
    *,
    parent_expression: str | None = None,
    mutation_type: str = "seed",
) -> dict[str, Any]:
    return {
        "expression": expression,
        "family": family,
        "generation": generation,
        "hypothesis": hypothesis,
        "parent_expression": parent_expression,
        "mutation_type": mutation_type,
    }


def _seed_expansion_variants(expression: str) -> list[tuple[str, str]]:
    """Create a small structural fallback pool after the curated seed catalog is exhausted."""
    variants: list[tuple[str, str]] = []
    widened = _widen_windows(expression)
    narrowed = _narrow_windows(expression)
    if normalize_wq_expression(widened) != normalize_wq_expression(expression):
        variants.append((widened, "seed_widen_windows"))
    if normalize_wq_expression(narrowed) != normalize_wq_expression(expression):
        variants.append((narrowed, "seed_narrow_windows"))
    variants.append((f"rank(ts_mean(({expression}), 10))", "seed_smooth_signal"))
    return variants


def _family_seed_pool(family: str, generation: int, hypothesis: str) -> list[dict[str, Any]]:
    seeds = FAMILY_SEEDS[family]
    pool = [_seed_meta(expression, family, generation, hypothesis) for expression in seeds]
    for expression in seeds:
        for variant, mutation_type in _seed_expansion_variants(expression):
            pool.append(
                _seed_meta(
                    variant,
                    family,
                    generation,
                    hypothesis,
                    parent_expression=expression,
                    mutation_type=mutation_type,
                )
            )
    return pool


def build_seed_plan(
    memory: dict[str, Any] | None,
    *,
    limit: int,
    family_count: int = 3,
    generation: int = 1,
    hypothesis: str = "",
) -> list[dict[str, Any]]:
    """Round-robin low-coverage families while excluding already researched expressions."""
    memory = memory or {}
    seen = set(memory.get("normalized_expressions") or [])
    unknown_fields = _unknown_fields_from_memory(memory)
    families = select_research_families(memory, family_count)
    adaptive = memory.get("adaptive_allocation") or {}
    family_slots = Counter()
    for cell in adaptive.get("selected_cells") or []:
        family = str(cell.get("family") or "")
        if family in families:
            family_slots[family] += int(cell.get("slots") or 0)
    family_cycle = [family for family in families for _ in range(max(1, family_slots[family]))]
    plan: list[dict[str, Any]] = []
    offsets = Counter()
    pools = {family: _family_seed_pool(family, generation, hypothesis) for family in families}
    guidance = memory.get("research_memory_guidance") or {}
    negative_counts = Counter(
        str(item.get("operator_pattern") or "")
        for item in (guidance.get("negative") or [])
        for _ in range(max(1, int(item.get("count") or 1)))
        if item.get("operator_pattern")
    )
    positive_patterns = Counter(
        str(item.get("operator_pattern") or "")
        for item in (guidance.get("positive") or [])
        if item.get("operator_pattern")
    )
    overused_counts = {
        str(item.get("operator_pattern") or ""): int(item.get("trials") or 0)
        for item in (guidance.get("overused_structures") or [])
        if item.get("operator_pattern")
    }

    def memory_penalty(item: dict[str, Any]) -> tuple[float, str]:
        pattern = str(extract_expression_metadata(item["expression"]).get("operator_pattern") or "")
        penalty = negative_counts.get(pattern, 0) * 2.0 + min(5.0, overused_counts.get(pattern, 0) / 20.0)
        penalty -= min(2.0, positive_patterns.get(pattern, 0) * 0.5)
        return penalty, normalize_wq_expression(item["expression"])

    for family in pools:
        pools[family].sort(key=memory_penalty)

    while len(plan) < max(1, int(limit)):
        added = False
        for family in family_cycle:
            pool = pools[family]
            while offsets[family] < len(pool):
                item = pool[offsets[family]]
                offsets[family] += 1
                normalized = normalize_wq_expression(item["expression"])
                if normalized in seen or _uses_unknown_field(item["expression"], unknown_fields):
                    continue
                seen.add(normalized)
                plan.append(item)
                added = True
                break
            if len(plan) >= limit:
                break
        if not added:
            break
    return plan


def _widen_windows(expression: str) -> str:
    replacements = ((", 120)", ", 180)"), (", 60)", ", 120)"), (", 40)", ", 60)"), (", 30)", ", 60)"), (", 20)", ", 40)"), (", 10)", ", 20)"), (", 5)", ", 10)"))
    value = expression
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def _narrow_windows(expression: str) -> str:
    # Small-to-large order prevents a replacement from being narrowed twice.
    replacements = ((", 10)", ", 5)"), (", 20)", ", 10)"), (", 40)", ", 20)"), (", 60)", ", 30)"), (", 120)", ", 60)"), (", 180)", ", 120)"))
    value = expression
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def _step_knowledge_decay_window(expression: str) -> str | None:
    """Increase only the decay window of our knowledge rescue wrapper.

    Avoid ``_widen_windows`` here because that would also change the economic
    signal's own lookbacks (for example volatility 20 -> 40) at the same time,
    making it impossible to learn whether lower turnover came from execution
    smoothing or from changing the underlying hypothesis.
    """
    match = re.fullmatch(r"rank\(ts_decay_linear\(\((.*)\),\s*(\d+)\)\)", expression.strip())
    if not match:
        return None
    current = int(match.group(2))
    ladder = (5, 10, 20, 40)
    next_window = next((value for value in ladder if value > current), None)
    if next_window is None:
        return None
    return f"rank(ts_decay_linear(({match.group(1)}), {next_window}))"


def _alternate_family_seed(family: str, seen: set[str], *, stable_only: bool = False) -> tuple[str, str] | None:
    preferred = ("fundamental_quality", "analyst_revision", "volatility_structure") if stable_only else tuple(FAMILY_SEEDS)
    for alternate_family in preferred:
        if alternate_family == family:
            continue
        for expression in FAMILY_SEEDS.get(alternate_family, ()):
            normalized = normalize_wq_expression(expression)
            if normalized and normalized not in seen:
                return expression, alternate_family
    return None


def _violates_prevention_rules(item: dict[str, Any], rules: dict[str, Any] | None) -> bool:
    rules = rules or {}
    expression = normalize_wq_expression(str(item.get("expression") or ""))
    if expression in set(rules.get("blocked_exact_expressions") or []):
        return True
    pattern = str(item.get("operator_pattern") or (item.get("research_meta") or {}).get("operator_pattern") or "")
    if pattern and pattern in set(rules.get("blocked_operator_patterns") or []):
        return True
    fields = set(str(value) for value in (item.get("data_fields") or (item.get("research_meta") or {}).get("data_fields") or []))
    for blocked in rules.get("blocked_operator_field_pairs") or []:
        if pattern and pattern == str(blocked.get("operator_pattern") or "") and fields.intersection(blocked.get("data_fields") or []):
            return True
    return False


def build_targeted_mutations(
    result: dict[str, Any],
    *,
    seen: set[str],
    limit: int,
    hypothesis: str = "",
) -> list[dict[str, Any]]:
    """Turn diagnostics into bounded structural mutations instead of blind parameter sweeps."""
    expression = str(result.get("expression") or "").strip()
    if not expression or limit <= 0:
        return []
    try:
        max_children = max(1, min(4, int(os.environ.get("WQ_MUTATION_MAX_CHILDREN", "2"))))
    except (TypeError, ValueError):
        max_children = 2
    try:
        max_generation = max(1, min(3, int(os.environ.get("WQ_MUTATION_MAX_GENERATION", "3"))))
    except (TypeError, ValueError):
        max_generation = 3
    limit = min(int(limit), max_children)

    meta = result.get("research_meta") or {}
    family = str(meta.get("family") or classify_wq_family(expression))
    targets = list(result.get("mutation_targets") or [])
    metrics = result.get("is_metrics") or {}
    sharpe = _safe_metric(metrics.get("sharpe"))
    fitness = _safe_metric(metrics.get("fitness"))
    turnover = _safe_metric(metrics.get("turnover"))
    knowledge_card_ids = [str(value) for value in (meta.get("knowledge_card_ids") or []) if value]
    parent_generation = int(meta.get("generation") or 1)
    generation = parent_generation + 1
    if generation > max_generation:
        result["mutation_route_terminal_reason"] = "mutation_generation_budget_exhausted"
        return []
    directives = preferred_mutation_classes(result)
    variants: list[tuple[str, str, str, str | None]] = []

    # Knowledge-backed parents get a deterministic implementation-cost rescue
    # before generic mutation routes.  This matters for short-horizon effects:
    # a strong raw Sharpe can be economically real while still being unusable
    # because the naive expression trades too aggressively.  Keep the economic
    # hypothesis/card lineage, but slow position changes first.
    knowledge_strategies = [str(value) for value in (meta.get("knowledge_mutation_strategies") or []) if value]
    stepped_decay = _step_knowledge_decay_window(expression)
    knowledge_cost_refinement = bool(
        knowledge_card_ids
        and turnover is not None
        and (
            turnover > 0.7
            or (
                stepped_decay
                and sharpe is not None
                and sharpe >= 1.25
                and fitness is not None
                and 0.7 <= fitness < 1.0
                and 0.01 <= turnover <= 0.7
            )
        )
    )
    if knowledge_cost_refinement:
        strategy_hint = "; ".join(knowledge_strategies[:2])
        suffix = f"; knowledge guidance: {strategy_hint}" if strategy_hint else ""
        if stepped_decay:
            variants.append(
                (
                    stepped_decay,
                    "knowledge_decay_step",
                    "knowledge-backed near-threshold turnover: increase only the execution decay window while preserving signal lookbacks" + suffix,
                    None,
                )
            )
            variants.append(
                (
                    f"rank(ts_mean(({expression}), 3))",
                    "knowledge_post_decay_smoothing",
                    "knowledge-backed near-threshold turnover: add light post-decay smoothing without changing the economic hypothesis" + suffix,
                    None,
                )
            )
        else:
            variants.append(
                (
                    f"rank(ts_decay_linear(({expression}), 5))",
                    "knowledge_decay_smoothing",
                    "knowledge-backed high-turnover signal: decay/smooth the same hypothesis to improve implementation efficiency" + suffix,
                    None,
                )
            )
            variants.append(
                (
                    f"rank(ts_mean(({expression}), 3))",
                    "knowledge_short_smoothing",
                    "knowledge-backed high-turnover signal: apply light smoothing before more aggressive turnover controls" + suffix,
                    None,
                )
            )
            variants.append(
                (
                    f"hump(rank(({expression})), hump=0.01)",
                    "knowledge_turnover_hump",
                    "knowledge-backed high-turnover fallback: strongly constrain position changes if lighter smoothing is insufficient" + suffix,
                    None,
                )
            )
    for directive in directives:
        mutation_class = directive["mutation_class"]
        rationale = directive["rationale"]
        if mutation_class == "compile_prevention":
            continue
        if mutation_class in {"economic_reseed", "stable_data_reseed"}:
            reseed = _alternate_family_seed(family, seen, stable_only=mutation_class == "stable_data_reseed")
            if reseed:
                variants.append((reseed[0], mutation_class, rationale, reseed[1]))
        elif mutation_class == "widen_windows":
            variants.append((_widen_windows(expression), mutation_class, rationale, None))
        elif mutation_class == "smooth_signal":
            variants.append((f"rank(ts_mean(({expression}), 10))", mutation_class, rationale, None))
        elif mutation_class == "cost_reduction":
            variants.append((f"rank(ts_decay_linear(({expression}), 5))", mutation_class, rationale, None))
        elif mutation_class == "conditional_execution":
            variants.append((f"trade_when(volume > ts_mean(volume, 20), ({expression}), -1)", mutation_class, rationale, None))
        elif mutation_class == "operator_family_switch":
            variants.append((f"zscore({expression})", mutation_class, rationale, None))
        elif mutation_class == "neutralization_shift":
            variants.append((f"group_rank(({expression}), subindustry)", mutation_class, rationale, None))
        elif mutation_class == "weight_control" and not expression.lstrip().startswith("rank("):
            variants.append((f"rank({expression})", mutation_class, rationale, None))
        elif mutation_class == "increase_responsiveness":
            variants.append((_narrow_windows(expression), mutation_class, rationale, None))
        elif mutation_class == "signal_quality":
            if sharpe < 0:
                variants.append((f"-1 * ({expression})", "invert_signal", rationale, None))
            else:
                variants.append((f"rank(ts_mean(({expression}), 10))", "smooth_signal", rationale, None))

    # Keep legacy diagnostic compatibility only when the normalized router has no
    # structural expression available; correlation/robustness routes intentionally
    # never fall back to cosmetic window-only changes.
    if not variants and "increase_turnover" in targets:
        variants.append((_narrow_windows(expression), "increase_responsiveness", "legacy turnover diagnostic", None))

    out: list[dict[str, Any]] = []
    for variant, mutation_type, rationale, family_override in variants:
        normalized = normalize_wq_expression(variant)
        if not normalized or normalized in seen or normalized == normalize_wq_expression(expression):
            continue
        seen.add(normalized)
        out.append(
            {
                "expression": variant,
                "family": family_override or family,
                "generation": generation,
                "hypothesis": hypothesis or str(meta.get("hypothesis") or ""),
                "parent_expression": expression,
                "parent_lineage_id": meta.get("lineage_id") or result.get("lineage_id"),
                "mutation_type": mutation_type,
                "mutation_reason": rationale,
                "planner_strategy": "failure_directed",
                "data_fields": [] if family_override else list(meta.get("data_fields") or []),
                "dataset_id": None if family_override else meta.get("dataset_id"),
                "dataset_category": None if family_override else meta.get("dataset_category"),
                "provenance_state": "unresolved" if family_override else meta.get("provenance_state"),
                "provenance_reason": "economic_reseed_requires_new_provenance" if family_override else meta.get("provenance_reason"),
                "knowledge_card_ids": [] if family_override else list(meta.get("knowledge_card_ids") or []),
                "failure_trigger": sorted({entry["reason"] if isinstance(entry, dict) else str(entry) for entry in (result.get("failure_reasons") or [])} | ({str(result.get("failure_reason"))} if result.get("failure_reason") else set())),
            }
        )
        if len(out) >= limit:
            break
    return out


def build_memory_mutation_plan(
    memory: dict[str, Any] | None,
    *,
    seen: set[str],
    limit: int,
    hypothesis: str = "",
    min_sharpe: float = 1.25,
    min_fitness: float = 1.0,
    min_parent_promise: float = 0.55,
) -> list[dict[str, Any]]:
    """Reuse recent failed trials as parents across autonomous runs.

    Only shallow lineages are evolved automatically. This keeps the search
    continuous without turning an old weak Alpha into an indefinitely nested
    expression tree.
    """
    if limit <= 0:
        return []
    memory = memory or {}
    recent = list(memory.get("recent_trials") or [])
    knowledge_cards_by_id = {
        str(card.get("id")): card
        for card in ((memory.get("knowledge_guidance") or {}).get("cards") or [])
        if card.get("id")
    }
    def memory_parent_rank(item: dict[str, Any]) -> tuple[int, float]:
        promise = _trial_promise_score(item, min_sharpe=min_sharpe, min_fitness=min_fitness)
        linked_ids = [str(value) for value in (item.get("knowledge_card_ids") or []) if value]
        # Reserve the first memory-mutation slot for a viable knowledge-backed
        # parent when one exists. Otherwise a large legacy trial inventory can
        # permanently crowd out the very follow-up experiments needed to turn a
        # literature prior into an implementable Alpha.
        knowledge_followup = int(
            promise >= min_parent_promise
            and any(card_id in knowledge_cards_by_id for card_id in linked_ids)
        )
        return knowledge_followup, promise

    recent.sort(key=memory_parent_rank, reverse=True)

    plan: list[dict[str, Any]] = []
    for trial in recent:
        generation = int(trial.get("generation") or 0)
        trial_card_ids = [str(value) for value in (trial.get("knowledge_card_ids") or []) if value]
        linked_cards = [knowledge_cards_by_id[value] for value in trial_card_ids if value in knowledge_cards_by_id]
        if generation >= 3:
            continue
        targets = list(trial.get("mutation_targets") or [])
        if not targets or targets == ["candidate_passes_primary_thresholds"]:
            continue
        promise = _trial_promise_score(trial, min_sharpe=min_sharpe, min_fitness=min_fitness)
        if promise < min_parent_promise:
            continue
        knowledge_failure_modes = [
            str(value)
            for card in linked_cards
            for value in (card.get("failure_modes") or [])
            if value
        ]
        knowledge_mutation_strategies = [
            str(value)
            for card in linked_cards
            for value in (card.get("mutation_strategies") or [])
            if value
        ]
        synthetic = {
            "expression": trial.get("expression"),
            "is_metrics": {
                "sharpe": trial.get("sharpe"),
                "fitness": trial.get("fitness"),
                "returns": trial.get("returns"),
                "turnover": trial.get("turnover"),
                "checks": list(((trial.get("failure_evidence") or {}).get("checks")) or []),
            },
            "mutation_targets": targets,
            "failure_reason": trial.get("failure_reason"),
            "failure_reasons": list(trial.get("failure_reasons") or []),
            "local_correlation_proxy": trial.get("local_correlation_proxy"),
            "research_meta": {
                "family": trial.get("family") or classify_wq_family(str(trial.get("expression") or "")),
                "generation": generation,
                "hypothesis": trial.get("hypothesis") or hypothesis,
                "lineage_id": trial.get("lineage_id"),
                "data_fields": list(trial.get("data_fields") or []),
                "dataset_id": trial.get("dataset_id"),
                "dataset_category": trial.get("dataset_category"),
                "provenance_state": trial.get("provenance_state"),
                "provenance_reason": trial.get("provenance_reason"),
                "operator_pattern": trial.get("operator_pattern"),
                "knowledge_card_ids": trial_card_ids,
                "knowledge_failure_modes": knowledge_failure_modes[:12],
                "knowledge_mutation_strategies": knowledge_mutation_strategies[:12],
            },
        }
        variants = build_targeted_mutations(
            synthetic,
            seen=seen,
            limit=min(2, limit - len(plan)),
            hypothesis=hypothesis,
        )
        plan.extend(variants)
        if len(plan) >= limit:
            break
    return plan


def build_platform_near_miss_plan(
    alphas: list[dict[str, Any]],
    *,
    seen: set[str],
    limit: int,
    hypothesis: str,
    min_sharpe: float = 1.25,
    min_fitness: float = 1.0,
) -> list[dict[str, Any]]:
    """Exploit strong UNSUBMITTED near-misses before spending budget on fresh random seeds."""
    if limit <= 0:
        return []
    parents: list[tuple[float, dict[str, Any]]] = []
    for alpha in alphas or []:
        if str(alpha.get("status") or "").upper() != "UNSUBMITTED":
            continue
        expression = str(alpha.get("expression") or "").strip()
        sharpe = _safe_metric(alpha.get("sharpe"), -999.0)
        fitness = _safe_metric(alpha.get("fitness"), -999.0)
        turnover = _safe_metric(alpha.get("turnover"), 0.0)
        returns = _safe_metric(alpha.get("returns"), -999.0)
        if not expression or sharpe < min_sharpe * 0.88 or not (min_fitness * 0.65 <= fitness < min_fitness):
            continue
        if returns <= 0 or not 0.01 <= turnover <= 1.2:
            continue
        turnover_score = 1.0 if turnover <= 0.5 else max(0.0, 1.0 - (turnover - 0.5) / 0.7)
        promise = min(1.25, sharpe / max(0.01, min_sharpe)) * 0.35 + min(1.0, fitness / max(0.01, min_fitness)) * 0.5 + turnover_score * 0.15
        parents.append((promise, alpha))
    parents.sort(key=lambda item: (item[0], str(item[1].get("alpha_id") or "")), reverse=True)

    plan: list[dict[str, Any]] = []
    # One child per parent first; only then allow a second child from the best
    # parent. This keeps near-miss exploitation diversified instead of parameter
    # sweeping a single Alpha lineage.
    for per_parent_limit in (1, 2):
        for _promise, alpha in parents:
            expression = str(alpha.get("expression") or "")
            turnover = _safe_metric(alpha.get("turnover"), 0.0)
            targets = ["improve_fitness"]
            if turnover > 0.5:
                targets.insert(0, "reduce_turnover")
            synthetic = {
                "expression": expression,
                "alpha_id": alpha.get("alpha_id"),
                "is_metrics": {
                    "sharpe": alpha.get("sharpe"),
                    "fitness": alpha.get("fitness"),
                    "returns": alpha.get("returns"),
                    "turnover": alpha.get("turnover"),
                    "checks": [],
                },
                "mutation_targets": targets,
                "failure_reason": "low_fitness",
                "research_meta": {
                    "family": classify_wq_family(expression),
                    "generation": 1,
                    "hypothesis": hypothesis,
                    "mutation_type": "platform_near_miss_parent",
                    "planner_strategy": "platform_near_miss",
                    "source_run_id": str(alpha.get("alpha_id") or ""),
                },
            }
            variants = build_targeted_mutations(
                synthetic,
                seen=seen,
                limit=min(per_parent_limit, limit - len(plan)),
                hypothesis=hypothesis,
            )
            for variant in variants:
                variant["planner_strategy"] = "platform_near_miss"
                variant["mutation_reason"] = (
                    f"repair platform near-miss {alpha.get('alpha_id')}: "
                    + str(variant.get("mutation_reason") or "directed mutation")
                )
                plan.append(variant)
                if len(plan) >= limit:
                    return plan
    return plan


def _build_next_generation_plan(
    ranked: list[dict[str, Any]],
    *,
    seen: set[str],
    limit: int,
    hypothesis: str,
) -> list[dict[str, Any]]:
    """Allocate one child per promising parent before giving any parent a second child."""
    if limit <= 0:
        return []
    plan: list[dict[str, Any]] = []
    for per_parent_limit in (1, 2):
        for item in ranked:
            mutations = build_targeted_mutations(
                item,
                seen=seen,
                limit=per_parent_limit,
                hypothesis=hypothesis,
            )
            if mutations:
                item["directed_mutation_routed"] = True
                item["mutation_children"] = int(item.get("mutation_children") or 0) + len(mutations)
                item["mutation_route_reason"] = ",".join(
                    sorted({str(value.get("mutation_reason") or "") for value in mutations if value.get("mutation_reason")})
                )[:500]
            elif item.get("mutation_targets") and not item.get("mutation_route_terminal_reason"):
                item["mutation_route_terminal_reason"] = "no_novel_actionable_mutation"
            for mutation in mutations:
                if mutation not in plan:
                    plan.append(mutation)
                    if len(plan) >= limit:
                        return plan
    return plan


def _annotate_batch(batch: dict[str, Any], plan: list[dict[str, Any]]) -> None:
    meta_by_expr = {normalize_wq_expression(item["expression"]): item for item in plan}
    for key in ("results", "candidates", "failed", "invalid"):
        for item in batch.get(key) or []:
            meta = meta_by_expr.get(normalize_wq_expression(item.get("expression", "")))
            if meta:
                item["research_meta"] = {k: v for k, v in meta.items() if k != "expression"}


def validate_candidate_robustness(
    client,
    candidate: dict[str, Any],
    *,
    region: str,
    universe: str,
    delay: int,
    decay: int,
    neutralization: str,
    truncation: float,
    check_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Execute a Skill-authored targeted robustness plan and preserve raw evidence.

    The Skill owns which perturbations are informative. This function deliberately
    avoids inventing a universal local pass ratio; BRAIN metrics/checks and later
    evidence review decide how much confidence to place in the candidate.
    """
    expression = str(candidate.get("expression") or "").strip()
    if not expression:
        return {"status": "robustness_unavailable", "robustness_score": None, "reason": "missing_expression"}

    meta = candidate.get("research_meta") or {}
    raw_plan = candidate.get("robustness_plan") or (meta.get("robustness_plan") if isinstance(meta, dict) else None)
    plan = dict(raw_plan) if isinstance(raw_plan, dict) else {}
    checks = plan.get("checks") or []
    if str(plan.get("mode") or "") != "skill_defined" or not isinstance(checks, list) or not checks:
        return {
            "status": "validation_pending",
            "robustness_score": None,
            "reason": "skill_robustness_plan_missing",
            "validation_simulations": 0,
            "policy": "skill_defined_required",
        }

    details: list[dict[str, Any]] = []
    for raw_check in checks[:4]:
        if check_cancelled and check_cancelled():
            break
        check = dict(raw_check) if isinstance(raw_check, dict) else {}
        settings = {
            "region": str(check.get("region") or region),
            "universe": str(check.get("universe") or universe),
            "delay": int(check.get("delay") if check.get("delay") is not None else delay),
            "decay": int(check.get("decay") if check.get("decay") is not None else decay),
            "neutralization": str(check.get("neutralization") or neutralization),
            "truncation": float(check.get("truncation") if check.get("truncation") is not None else truncation),
        }
        try:
            result = run_single_simulation(
                client,
                expression,
                region=settings["region"],
                universe=settings["universe"],
                delay=settings["delay"],
                decay=settings["decay"],
                neutralization=settings["neutralization"],
                truncation=settings["truncation"],
                auto_submit=False,
                check_cancelled=check_cancelled,
            )
        except Exception as exc:
            result = {"ok": False, "error": str(exc)[:300]}
        is_metrics = result.get("is_metrics") or {}
        details.append(
            {
                "status": "completed" if result.get("ok") else ("cancelled" if result.get("cancelled") else "failed"),
                "purpose": str(check.get("purpose") or "targeted robustness check"),
                "settings": settings,
                "alpha_id": result.get("alpha_id"),
                "sharpe": _safe_metric(is_metrics.get("sharpe"), None),
                "fitness": _safe_metric(is_metrics.get("fitness"), None),
                "returns": _safe_metric(is_metrics.get("returns"), None),
                "turnover": _safe_metric(is_metrics.get("turnover"), None),
                "checks": list(is_metrics.get("checks") or []),
                "error": result.get("error"),
            }
        )

    completed = [item for item in details if item.get("status") == "completed"]
    return {
        "status": "evidence_collected" if completed else "robustness_unavailable",
        "robustness_score": None,
        "validation_simulations": len(details),
        "completed": len(completed),
        "details": details,
        "robustness_plan": plan,
        "knowledge_card_ids": list(plan.get("knowledge_card_ids") or []),
        "interpretation_policy": "skill_review_required_no_magic_pass_ratio",
    }


def _merge_candidate_validation(
    candidate: dict[str, Any],
    robustness_validation: dict[str, Any],
    *,
    mirrored: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge robustness and submission evidence, preserving earlier evidence."""
    raw_existing_validation = candidate.get("validation")
    validation: dict[str, Any] = (
        dict(raw_existing_validation) if isinstance(raw_existing_validation, dict) else {}
    )
    validation.update(robustness_validation)
    meta = candidate.get("research_meta") or {}
    if isinstance(meta, dict) and isinstance(meta.get("candidate_evidence_policy"), dict):
        validation["candidate_evidence_policy"] = dict(meta["candidate_evidence_policy"])
    raw_local_evidence = candidate.get("local_correlation_proxy") or validation.get("local_correlation_proxy")
    local_evidence: dict[str, Any] = dict(raw_local_evidence) if isinstance(raw_local_evidence, dict) else {}
    raw_overfitting = validation.get("overfitting_evidence")
    overfitting_evidence: dict[str, Any] = dict(raw_overfitting) if isinstance(raw_overfitting, dict) else {}
    evidence_blockers = submission_evidence_blockers(local_evidence, overfitting_evidence)
    validation["submission_gate"] = {
        "ready": not evidence_blockers,
        "blockers": list(evidence_blockers),
        "policy": configured_submission_evidence_policy(),
    }
    candidate["validation"] = validation
    target = mirrored or candidate
    target["validation"] = dict(validation)
    if validation.get("status") == "robustness_fail":
        targets = list(target.get("mutation_targets") or [])
        if "improve_cross_setting_robustness" not in targets:
            targets.append("improve_cross_setting_robustness")
        target["mutation_targets"] = targets
    return validation


def _rank_result(item: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = item.get("is_metrics") or {}
    def num(key: str) -> float:
        try:
            return float(metrics.get(key) or -999.0)
        except (TypeError, ValueError):
            return -999.0
    active_prior = _safe_metric(item.get("active_prior_score"), 0.0)
    # ACTIVE similarity is a weak tiebreaker, not a replacement for raw Fitness.
    # This ensures real platform outcomes influence scarce robustness-validation
    # slots without turning the search into cloning historical Alphas.
    return num("fitness") + active_prior * 0.12, num("sharpe"), num("returns"), active_prior


def _select_native_decay_rescue_parent(
    memory: dict[str, Any],
    *,
    min_sharpe: float,
    min_fitness: float,
    current_decay: int,
) -> tuple[dict[str, Any] | None, list[int]]:
    """Select one near-threshold knowledge trial for a tiny native-decay sweep."""
    if int(current_decay) != 0:
        return None, []
    recent = list(memory.get("recent_trials") or [])
    tested_by_expression: dict[str, set[int]] = {}
    for trial in recent:
        norm = normalize_wq_expression(str(trial.get("expression") or ""))
        settings = trial.get("settings") or {}
        if not norm or not isinstance(settings, dict):
            continue
        try:
            tested_by_expression.setdefault(norm, set()).add(int(settings.get("decay", 0) or 0))
        except (TypeError, ValueError):
            continue

    eligible: list[dict[str, Any]] = []
    for trial in recent:
        if not list(trial.get("knowledge_card_ids") or []):
            continue
        sharpe = _safe_metric(trial.get("sharpe"))
        fitness = _safe_metric(trial.get("fitness"))
        turnover = _safe_metric(trial.get("turnover"))
        returns = _safe_metric(trial.get("returns"))
        if (
            sharpe >= min_sharpe
            and min_fitness * 0.7 <= fitness < min_fitness
            and 0.01 <= turnover <= 0.8
            and returns > 0
        ):
            eligible.append(trial)
    if not eligible:
        return None, []
    eligible.sort(
        key=lambda item: (
            _safe_metric(item.get("fitness")),
            _safe_metric(item.get("sharpe")),
            -abs(_safe_metric(item.get("turnover")) - 0.5),
        ),
        reverse=True,
    )
    parent = eligible[0]
    norm = normalize_wq_expression(str(parent.get("expression") or ""))
    tested = tested_by_expression.get(norm, {0})
    return parent, [value for value in (2, 4) if value not in tested][:2]


def _select_active_target_native_decay_rescue_parent(
    alphas: list[dict[str, Any]],
    *,
    min_sharpe: float,
    min_fitness: float,
    current_decay: int,
) -> tuple[dict[str, Any] | None, list[int]]:
    """Pick the strongest platform near-miss for a tiny native decay rescue.

    This path is intentionally only used while the daily ACTIVE target is still
    open.  A near-miss is not formally submittable yet; instead of discarding it
    or repeatedly wrapping the FASTEXPR, reuse BRAIN's native execution decay to
    try to lift Fitness across the real platform threshold with two bounded sims.
    """
    if int(current_decay) != 0:
        return None, []

    eligible: list[dict[str, Any]] = []
    for alpha in alphas or []:
        if str(alpha.get("status") or "").upper() != "UNSUBMITTED":
            continue
        expression = str(alpha.get("expression") or "").strip()
        sharpe = _safe_metric(alpha.get("sharpe"), -999.0)
        fitness = _safe_metric(alpha.get("fitness"), -999.0)
        returns = _safe_metric(alpha.get("returns"), -999.0)
        turnover = _safe_metric(alpha.get("turnover"), 0.0)
        if not expression:
            continue
        if sharpe < min_sharpe or not (min_fitness * 0.85 <= fitness < min_fitness):
            continue
        if returns <= 0 or not 0.01 <= turnover <= 0.7:
            continue
        item = dict(alpha)
        item["family"] = classify_wq_family(expression)
        item["generation"] = 1
        item["parameter_rescue_source"] = "platform_near_miss"
        eligible.append(item)

    if not eligible:
        return None, []
    eligible.sort(
        key=lambda item: (
            _safe_metric(item.get("fitness")),
            _safe_metric(item.get("sharpe")),
            _safe_metric(item.get("returns")),
            -abs(_safe_metric(item.get("turnover")) - 0.25),
        ),
        reverse=True,
    )
    return eligible[0], [2, 4]


def _run_native_decay_rescue(
    client,
    parent: dict[str, Any],
    decay_values: list[int],
    *,
    goal: str,
    tag: str,
    region: str,
    universe: str,
    delay: int,
    neutralization: str,
    truncation: float,
    min_sharpe: float,
    min_fitness: float,
    check_cancelled: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    expression = str(parent.get("expression") or "").strip()
    if not expression or not decay_values:
        return []
    base_meta = {
        "family": parent.get("family") or classify_wq_family(expression),
        "generation": int(parent.get("generation") or 0),
        "hypothesis": parent.get("hypothesis") or goal,
        "parent_expression": expression,
        "parent_alpha_id": parent.get("alpha_id"),
        "mutation_type": "native_decay_rescue",
        "mutation_reason": (
            "platform near-miss: bounded BRAIN-native decay before deeper FASTEXPR mutation"
            if parent.get("parameter_rescue_source") == "platform_near_miss"
            else "near-threshold knowledge Alpha: bounded BRAIN-native decay before deeper FASTEXPR mutation"
        ),
        "planner_strategy": (
            "active_target_native_parameter_rescue"
            if parent.get("parameter_rescue_source") == "platform_near_miss"
            else "knowledge_native_parameter_rescue"
        ),
        "knowledge_card_ids": list(parent.get("knowledge_card_ids") or []),
        "data_fields": list(parent.get("data_fields") or []),
        "dataset_id": parent.get("dataset_id"),
        "dataset_category": parent.get("dataset_category"),
        "provenance_state": parent.get("provenance_state"),
        "provenance_reason": parent.get("provenance_reason"),
    }
    out: list[dict[str, Any]] = []
    for native_decay in decay_values[:2]:
        if check_cancelled and check_cancelled():
            break
        result = run_single_simulation(
            client,
            expression,
            region=region,
            universe=universe,
            delay=delay,
            decay=int(native_decay),
            neutralization=neutralization,
            truncation=truncation,
            auto_submit=False,
            tag=f"{tag}-native-decay-{native_decay}",
            check_cancelled=check_cancelled,
        )
        if not result.get("ok"):
            continue
        metrics = result.get("is_metrics") or {}
        sharpe = _safe_metric(metrics.get("sharpe"))
        fitness = _safe_metric(metrics.get("fitness"))
        turnover = _safe_metric(metrics.get("turnover"))
        failed_checks = {
            str(item.get("name"))
            for item in (metrics.get("checks") or [])
            if str(item.get("result") or "").upper() == "FAIL"
        }
        result["passes_primary_thresholds"] = bool(
            sharpe >= min_sharpe
            and fitness >= min_fitness
            and 0.01 <= turnover <= 0.7
            and not failed_checks
        )
        result["mutation_targets"] = diagnose_wq_result(result, min_sharpe, min_fitness)
        result["research_meta"] = dict(base_meta)
        result["research_meta"]["native_decay"] = int(native_decay)
        result["parameter_rescue"] = {
            "type": "brain_native_decay",
            "parent_alpha_id": parent.get("alpha_id"),
            "parent_expression": expression,
            "decay": int(native_decay),
        }
        out.append(result)
    return out


def run_autonomous_research(
    client,
    *,
    memory: dict[str, Any] | None = None,
    skill_candidates: list[dict[str, Any]] | None = None,
    chatgpt_expressions: list[str] | None = None,
    allow_deterministic_fallback: bool = True,
    goal: str = "maximize robust low-correlation WorldQuant candidates",
    tag: str = "wq-autonomous",
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: int = 0,
    neutralization: str = "SUBINDUSTRY",
    truncation: float = 0.08,
    max_simulations: int = 20,
    generations: int = 2,
    family_count: int = 3,
    min_sharpe: float = 1.25,
    min_fitness: float = 1.0,
    on_progress: Callable[[int, int, str], None] | None = None,
    check_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Plan, simulate, diagnose, and mutate a bounded autonomous research round."""
    max_simulations = max(4, min(40, int(max_simulations)))
    generations = max(1, min(3, int(generations)))
    skill_candidates = list(skill_candidates or [])
    strict_skill_mode = not allow_deterministic_fallback
    if strict_skill_mode and not skill_candidates:
        return {
            "ok": False,
            "mode": "skill_first",
            "error": "skill_candidates_required",
            "required_skill_chain": list(REQUIRED_WQ_SKILL_CHAIN),
        }
    if strict_skill_mode:
        # Repair/diversification after real BRAIN feedback must come back through
        # DevSpace skills on the next client turn rather than hidden server templates.
        generations = 1
    memory = dict(memory or {})
    family_counts = Counter(memory.get("family_counts") or {})
    normalized = set(memory.get("normalized_expressions") or [])

    # Seed the persistent memory with recent platform research so the first autonomous
    # run does not simply rediscover the last externally-created expressions.
    try:
        platform_history_limit = max(100, min(500, int(os.environ.get("WQ_PLATFORM_HISTORY_LIMIT", "500"))))
    except (TypeError, ValueError):
        platform_history_limit = 500
    recent = _load_platform_alpha_history(client, limit=platform_history_limit)
    recent_alphas = list(recent.get("alphas") or []) if recent.get("ok") else []
    active_alphas = [alpha for alpha in recent_alphas if str(alpha.get("status") or "").upper() == "ACTIVE"]
    active_reference_profile = _active_reference_profile(active_alphas)
    if recent.get("ok"):
        for alpha in recent_alphas:
            expression = str(alpha.get("expression") or "")
            norm = normalize_wq_expression(expression)
            if norm:
                normalized.add(norm)
                family_counts[classify_wq_family(expression)] += 1
    memory["family_counts"] = dict(family_counts)
    memory["normalized_expressions"] = sorted(normalized)
    effective_settings = {
        "region": region,
        "universe": universe,
        "delay": delay,
        "decay": decay,
        "neutralization": neutralization,
        "truncation": truncation,
    }
    seen_variants: set[tuple[str, ...]] = set()
    for trial in memory.get("recent_trials") or []:
        key = _simulation_variant_key(str(trial.get("expression") or ""), trial.get("settings"))
        if key is not None:
            seen_variants.add(key)
    for alpha in recent_alphas:
        key = _simulation_variant_key(str(alpha.get("expression") or ""), alpha.get("settings"))
        if key is not None:
            seen_variants.add(key)

    first_budget = max_simulations if generations == 1 else max(4, math.ceil(max_simulations * 0.65))
    if strict_skill_mode:
        first_budget = min(first_budget, len(skill_candidates))
    inventory = memory.get("inventory") or {}
    explicit_inventory_mode = str(inventory.get("mode") or memory.get("inventory_mode") or "").upper()
    high_confidence_inventory = sum(
        int(cell.get("high_confidence_candidates") or 0) for cell in (memory.get("research_cells") or [])
    )
    inventory_mode = explicit_inventory_mode or (
        "REPLENISHMENT" if high_confidence_inventory < 30 else "EXPLORATION" if high_confidence_inventory > 50 else "NORMAL"
    )

    # Two different objectives share this engine but must not share the same
    # allocation policy. Before two daily ACTIVE outcomes are reached, optimize
    # short-horizon ACTIVE conversion; afterwards rebuild strict inventory.
    submission_state = memory.get("submission") or {}
    try:
        remaining_active_target = max(0, int(submission_state.get("remaining_active_target") or 0))
    except (TypeError, ValueError):
        remaining_active_target = 0
    active_target_mode = remaining_active_target > 0
    research_strategy = "ACTIVE_FILL" if active_target_mode else "INVENTORY_BUILD"
    allocation_mode = "ACTIVE_FILL" if active_target_mode else inventory_mode
    adaptive_allocation = allocate_research_cells(
        memory.get("research_cells") or [],
        budget=first_budget,
        inventory_mode=allocation_mode,
        remaining_active_target=remaining_active_target if active_target_mode else None,
        learning_maturity=memory.get("learning_maturity") or {},
    )
    memory["adaptive_allocation"] = adaptive_allocation
    seen = set(normalized)

    # In inventory replenishment, exploit two sources that are much closer to
    # ACTIVE than random seeds: motifs from actual platform ACTIVE alphas and
    # strong UNSUBMITTED near-misses. Any unused quota automatically flows back
    # to fresh exploration, so sparse platform history never starves discovery.
    if strict_skill_mode:
        active_share, near_miss_share, memory_share = 0.0, 0.0, 0.0
    elif active_target_mode:
        active_share, near_miss_share, memory_share = 0.65, 0.25, 0.05
    elif inventory_mode == "REPLENISHMENT":
        active_share, near_miss_share, memory_share = 0.45, 0.30, 0.10
    elif inventory_mode == "NORMAL":
        active_share, near_miss_share, memory_share = 0.25, 0.20, 0.20
    else:
        active_share, near_miss_share, memory_share = 0.15, 0.15, 0.20

    active_motif_budget = max(0, math.ceil(first_budget * active_share))
    near_miss_budget = max(0, math.ceil(first_budget * near_miss_share))
    memory_parent_budget = max(0, math.ceil(first_budget * memory_share))

    near_miss_plan = build_platform_near_miss_plan(
        recent_alphas,
        seen=seen,
        limit=min(first_budget, near_miss_budget),
        hypothesis=goal,
        min_sharpe=min_sharpe,
        min_fitness=min_fitness,
    )
    memory_plan = build_memory_mutation_plan(
        memory,
        seen=seen,
        limit=min(max(0, first_budget - len(near_miss_plan)), memory_parent_budget),
        hypothesis=goal,
        min_sharpe=min_sharpe,
        min_fitness=min_fitness,
    )

    field_probe_budget = max(3, min(first_budget, max(active_motif_budget, math.ceil(first_budget * 0.20))))
    if strict_skill_mode:
        # Skill-first candidates already declare the fields ChatGPT selected from
        # the Data Explorer. Reuse restart-safe field provenance from prior real
        # trials first; build_skill_plan performs an exact BRAIN lookup only for a
        # declared field that is still unknown locally. This keeps the strict gate
        # while removing repeated catalog traffic from the per-batch hot path.
        live_fields = _registered_skill_fields(skill_candidates, memory)
        live_catalog = {
            "available": True,
            "count": len(live_fields),
            "broad_probe_skipped": True,
            "validation_mode": "persisted_registry_then_exact_live_lookup",
            "sample_ids": [_live_field_id(item) for item in live_fields[:10]],
        }
        active_sibling_catalog = {
            "available": True,
            "count": 0,
            "broad_probe_skipped": True,
        }
    else:
        live_fields, live_catalog = _live_field_candidates(
            client,
            memory,
            region=region,
            universe=universe,
            delay=delay,
            limit=field_probe_budget,
        )
        active_sibling_fields, active_sibling_catalog = _active_dataset_sibling_fields(
            client,
            active_alphas,
            region=region,
            universe=universe,
            delay=delay,
            limit=max(field_probe_budget, active_motif_budget * 2),
        )
        if active_sibling_fields:
            sibling_ids = {_live_field_id(item) for item in active_sibling_fields}
            live_fields = active_sibling_fields + [
                item for item in live_fields if _live_field_id(item) not in sibling_ids
            ]
    live_catalog["active_dataset_siblings"] = active_sibling_catalog
    active_motif_plan = build_active_motif_plan(
        client,
        active_alphas,
        live_fields,
        seen=seen,
        limit=min(max(0, first_budget - len(near_miss_plan) - len(memory_plan)), active_motif_budget),
        hypothesis=goal,
    )

    exploration_budget = max(0, first_budget - len(active_motif_plan) - len(near_miss_plan) - len(memory_plan))

    # Skill-first candidates get first claim on the entire fresh research budget.
    # In strict mode they are the only source of new expressions; deterministic
    # server templates remain available only behind an explicit fallback opt-in.
    # ACTIVE_FILL therefore depends on ChatGPT supplying a *batch* of reviewed
    # skill candidates rather than one boutique expression per research round.
    skill_plan_rejections: list[dict[str, Any]] = []
    skill_plan = build_skill_plan(
        client,
        skill_candidates,
        live_fields,
        seen=seen,
        limit=exploration_budget,
        hypothesis=goal,
        current_settings=effective_settings,
        seen_variants=seen_variants,
        rejections=skill_plan_rejections,
    )
    skill_batch_diversity = None
    if active_target_mode and strict_skill_mode:
        skill_plan, skill_batch_diversity = enforce_active_fill_batch_diversity(
            skill_plan,
            rejections=skill_plan_rejections,
        )
    legacy_budget = 0 if strict_skill_mode else max(0, exploration_budget - len(skill_plan))
    chatgpt_plan = build_chatgpt_plan(
        client,
        chatgpt_expressions,
        live_fields,
        seen=seen,
        limit=legacy_budget,
        hypothesis=goal,
    )
    deterministic_budget = 0 if strict_skill_mode else max(0, legacy_budget - len(chatgpt_plan))
    knowledge_budget = min(6, deterministic_budget, max(0, math.ceil(deterministic_budget * 0.25)))
    knowledge_plan = build_knowledge_seed_plan(
        client,
        memory,
        seen=seen,
        limit=knowledge_budget,
        hypothesis=goal,
        generation=1,
    )
    live_remaining = max(0, deterministic_budget - len(knowledge_plan))
    live_budget = min(live_remaining, max(0, math.ceil(deterministic_budget * 0.35)))
    live_plan = _build_live_field_plan_from_fields(
        client,
        live_fields,
        seen=seen,
        limit=live_budget,
        hypothesis=goal,
    )
    structural_budget = min(
        max(0, deterministic_budget - len(knowledge_plan) - len(live_plan)),
        max(0, math.ceil(deterministic_budget * 0.35)),
    )
    structural_plan = build_structural_live_plan(
        client,
        live_fields,
        seen=seen,
        limit=structural_budget,
        hypothesis=goal,
    )
    seed_memory = dict(memory)
    seed_memory["normalized_expressions"] = sorted(seen)
    seed_plan = build_seed_plan(
        seed_memory,
        limit=max(0, deterministic_budget - len(knowledge_plan) - len(live_plan) - len(structural_plan)),
        family_count=family_count,
        generation=1,
        hypothesis=goal,
    )
    for item in seed_plan:
        seen.add(normalize_wq_expression(item["expression"]))

    planned_count = (
        len(active_motif_plan)
        + len(near_miss_plan)
        + len(memory_plan)
        + len(skill_plan)
        + len(chatgpt_plan)
        + len(knowledge_plan)
        + len(live_plan)
        + len(structural_plan)
        + len(seed_plan)
    )
    if not strict_skill_mode and planned_count < first_budget:
        memory_plan.extend(
            build_memory_mutation_plan(
                memory,
                seen=seen,
                limit=first_budget - planned_count,
                hypothesis=goal,
                min_sharpe=min_sharpe,
                min_fitness=min_fitness,
            )
        )
    first_plan = [
        item
        for item in (
            active_motif_plan
            + near_miss_plan
            + memory_plan
            + skill_plan
            + chatgpt_plan
            + knowledge_plan
            + live_plan
            + structural_plan
            + seed_plan
        )
        if not _violates_prevention_rules(item, memory.get("prevention_rules"))
    ]
    generation_results: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    all_failed: list[dict[str, Any]] = []
    all_invalid: list[dict[str, Any]] = []
    selected_families = select_research_families(memory, family_count)
    try:
        base_validation_cap = max(1, min(8, int(os.environ.get("WQ_RESEARCH_VALIDATION_CAP", "2"))))
        replenishment_cap = max(base_validation_cap, min(8, int(os.environ.get("WQ_REPLENISHMENT_VALIDATION_CAP", "4"))))
    except (TypeError, ValueError):
        base_validation_cap, replenishment_cap = 2, 4
    # The formal Submission Gate already retains the hard safety blockers and
    # BRAIN's official SC check. Before today's two ACTIVE outcomes are reached,
    # local robustness is advisory rather than a pre-submit veto; otherwise a
    # small unstable cross-setting sample can exhaust the day without ever
    # letting a primary-qualified Alpha reach the authoritative platform check.
    validation_cap = 0 if active_target_mode else (replenishment_cap if inventory_mode == "REPLENISHMENT" else base_validation_cap)
    validation_candidates_used = 0
    validation_simulations = 0
    robustness_feedback_failures = 0
    # Strict Skill-first means every repair/diversification child must return to
    # DevSpace for wq-alpha-repair / wq-alpha-diversify + wq-alpha-review.
    # Native parameter rescue is a deterministic server-side repair path, so it
    # must stay behind the explicit fallback opt-in just like template mutations.
    native_decay_parent: dict[str, Any] | None = None
    native_decay_values: list[int] = []
    if not strict_skill_mode:
        native_decay_parent, native_decay_values = _select_native_decay_rescue_parent(
            memory,
            min_sharpe=min_sharpe,
            min_fitness=min_fitness,
            current_decay=decay,
        )
        if active_target_mode:
            platform_rescue_parent, platform_rescue_values = _select_active_target_native_decay_rescue_parent(
                recent_alphas,
                min_sharpe=min_sharpe,
                min_fitness=min_fitness,
                current_decay=decay,
            )
            if platform_rescue_parent is not None and (
                native_decay_parent is None
                or _safe_metric(platform_rescue_parent.get("fitness")) >= _safe_metric(native_decay_parent.get("fitness"))
            ):
                native_decay_parent, native_decay_values = platform_rescue_parent, platform_rescue_values
    native_decay_results: list[dict[str, Any]] = []
    remaining = max_simulations
    plan = first_plan
    for item in first_plan:
        seen.add(normalize_wq_expression(item["expression"]))

    for generation_index in range(1, generations + 1):
        if not plan or remaining <= 0 or (check_cancelled and check_cancelled()):
            break
        plan = plan[:remaining]
        expressions = [item["expression"] for item in plan]
        if on_progress:
            planned_families = sorted({str(item.get("family") or "unknown") for item in plan})
            on_progress(0, len(expressions), f"planned:{','.join(planned_families)}")
        batch = run_research_batch(
            client,
            expressions,
            goal=goal,
            tag=f"{tag}-g{generation_index}",
            region=region,
            universe=universe,
            delay=delay,
            decay=decay,
            neutralization=neutralization,
            truncation=truncation,
            max_candidates=len(expressions),
            min_sharpe=min_sharpe,
            min_fitness=min_fitness,
            skip_existing=True,
            sweep_top_n=0,
            on_progress=(
                (lambda current, total, stage, gi=generation_index: on_progress(current, total, f"g{gi}:{stage}"))
                if on_progress
                else None
            ),
            check_cancelled=check_cancelled,
        )
        _annotate_batch(batch, plan)
        for result_item in batch.get("results") or []:
            result_item["active_prior_score"] = _active_reference_score(result_item, active_reference_profile)
        for candidate_item in batch.get("candidates") or []:
            candidate_item["active_prior_score"] = _active_reference_score(candidate_item, active_reference_profile)
        batch["generation"] = generation_index
        generation_results.append(batch)
        all_results.extend(batch.get("results") or [])
        all_candidates.extend(batch.get("candidates") or [])
        all_failed.extend(batch.get("failed") or [])
        all_invalid.extend(batch.get("invalid") or [])
        simulated = int((batch.get("summary") or {}).get("simulated") or 0)
        remaining = max(0, remaining - max(simulated, len(expressions)))

        # Feed cross-setting evidence back before planning the next generation.
        # This lets a TOP3000 primary pass that collapses on TOP1000 immediately
        # route to stable-data / exposure-control mutations in the same research run.
        batch_results_by_alpha = {
            str(item.get("alpha_id")): item for item in (batch.get("results") or []) if item.get("alpha_id")
        }
        for candidate in sorted(batch.get("candidates") or [], key=_rank_result, reverse=True):
            if validation_candidates_used < validation_cap and not (check_cancelled and check_cancelled()):
                robustness_validation = validate_candidate_robustness(
                    client,
                    candidate,
                    region=region,
                    universe=universe,
                    delay=delay,
                    decay=decay,
                    neutralization=neutralization,
                    truncation=truncation,
                    check_cancelled=check_cancelled,
                )
                validation_candidates_used += 1
            else:
                robustness_validation = {
                    "status": "validation_pending",
                    "robustness_score": None,
                    "validation_simulations": 0,
                    "reason": f"per-round validation cap:{validation_cap}",
                }
            validation = _merge_candidate_validation(
                candidate,
                robustness_validation,
                mirrored=batch_results_by_alpha.get(str(candidate.get("alpha_id"))),
            )
            validation_simulations += int(validation.get("validation_simulations") or 0)
            if validation.get("status") == "robustness_fail" and generation_index < generations and remaining > 0:
                robustness_feedback_failures += 1

        if generation_index >= generations or remaining <= 0:
            break

        ranked = sorted(batch.get("results") or [], key=_rank_result, reverse=True)
        next_plan = _build_next_generation_plan(
            ranked,
            seen=seen,
            limit=remaining,
            hypothesis=goal,
        )

        # SELF_CORRELATION needs orthogonal information, not another nearby lookback.
        self_corr_present = any("reduce_self_correlation" in (item.get("mutation_targets") or []) for item in ranked)
        if self_corr_present and len(next_plan) < remaining:
            alternate_memory = dict(memory)
            alternate_counts = Counter(alternate_memory.get("family_counts") or {})
            for family in selected_families:
                alternate_counts[family] += 1000
            alternate_memory["family_counts"] = dict(alternate_counts)
            alternate_memory["normalized_expressions"] = sorted(seen)
            orthogonal = build_seed_plan(
                alternate_memory,
                limit=remaining - len(next_plan),
                family_count=min(2, len(FAMILY_SEEDS)),
                generation=generation_index + 1,
                hypothesis=f"orthogonalize after self-correlation: {goal}",
            )
            next_plan.extend(orthogonal)
            for item in orthogonal:
                seen.add(normalize_wq_expression(item["expression"]))
        plan = [item for item in next_plan if not _violates_prevention_rules(item, memory.get("prevention_rules"))]

    # A near-threshold literature-backed Alpha gets one tiny settings sweep after
    # the primary expression budget. This reuses the proven hypothesis and avoids
    # consuming more FASTEXPR generations merely to fix implementation turnover.
    if native_decay_parent and native_decay_values and not (check_cancelled and check_cancelled()):
        native_decay_results = _run_native_decay_rescue(
            client,
            native_decay_parent,
            native_decay_values,
            goal=goal,
            tag=tag,
            region=region,
            universe=universe,
            delay=delay,
            neutralization=neutralization,
            truncation=truncation,
            min_sharpe=min_sharpe,
            min_fitness=min_fitness,
            check_cancelled=check_cancelled,
        )
        for result_item in native_decay_results:
            result_item["active_prior_score"] = _active_reference_score(result_item, active_reference_profile)
        all_results.extend(native_decay_results)
        rescue_candidates = [item for item in native_decay_results if item.get("passes_primary_thresholds")]
        for candidate in rescue_candidates:
            if validation_candidates_used < validation_cap and not (check_cancelled and check_cancelled()):
                validation = validate_candidate_robustness(
                    client,
                    candidate,
                    region=region,
                    universe=universe,
                    delay=delay,
                    decay=int(((candidate.get("settings") or {}).get("decay")) or 0),
                    neutralization=neutralization,
                    truncation=truncation,
                    check_cancelled=check_cancelled,
                )
                validation_candidates_used += 1
            else:
                validation = {
                    "status": "validation_pending",
                    "robustness_score": None,
                    "validation_simulations": 0,
                    "reason": f"per-round validation cap:{validation_cap}",
                }
            _merge_candidate_validation(candidate, validation)
            validation_simulations += int(validation.get("validation_simulations") or 0)
        all_candidates.extend(rescue_candidates)

    # All primary candidates were validated (or explicitly marked pending) as
    # they emerged, so no terminal-only robustness pass is needed here.
    ranked_primary_candidates = sorted(all_candidates, key=_rank_result, reverse=True)
    ready_candidates = [
        item
        for item in ranked_primary_candidates
        if str((item.get("validation") or {}).get("status") or "").lower() in {"ready", "evidence_collected"}
        and ((item.get("validation") or {}).get("submission_gate") or {}).get("ready", True)
    ]
    all_results.sort(key=_rank_result, reverse=True)
    ranked_primary_candidates.sort(key=_rank_result, reverse=True)
    best = all_results[0] if all_results else None
    settings = {
        "region": region,
        "universe": universe,
        "delay": delay,
        "decay": decay,
        "neutralization": neutralization,
        "truncation": truncation,
        "min_sharpe": min_sharpe,
        "min_fitness": min_fitness,
    }
    return {
        "ok": True,
        "mode": "skill_first" if strict_skill_mode else "autonomous",
        "goal": goal,
        "tag": tag,
        "selected_families": selected_families,
        "adaptive_allocation": adaptive_allocation,
        "learning_maturity": memory.get("learning_maturity") or {},
        "research_memory_guidance": memory.get("research_memory_guidance") or {},
        "knowledge_guidance": memory.get("knowledge_guidance") or {},
        "inventory_mode": inventory_mode,
        "research_strategy": research_strategy,
        "active_target_mode": active_target_mode,
        "active_reference_profile": active_reference_profile,
        "live_catalog": live_catalog,
        "memory_before": {
            "trials": int(memory.get("trials") or 0),
            "family_counts": dict(memory.get("family_counts") or {}),
        },
        "settings": settings,
        "summary": {
            "generations_requested": generations,
            "generations_completed": len(generation_results),
            "remaining_active_target": remaining_active_target,
            "research_strategy": research_strategy,
            "skill_candidate_batch_supplied": len(skill_candidates),
            "skill_candidate_batch_accepted": len(skill_plan),
            "skill_candidate_batch_rejected": len(skill_plan_rejections),
            "skill_candidate_rejection_reasons": dict(Counter(item.get("reason") for item in skill_plan_rejections)),
            "skill_batch_diversity": skill_batch_diversity,
            "skill_batch_execution_status": (
                "needs_replacement" if strict_skill_mode and skill_candidates and not skill_plan else "ready"
            ),
            "skill_candidate_batch_target": min(max_simulations, 12) if active_target_mode and strict_skill_mode else None,
            "skill_candidate_batch_underfilled": bool(
                active_target_mode and strict_skill_mode and len(skill_candidates) < min(max_simulations, 8)
            ),
            "primary_simulation_budget": max_simulations,
            "primary_simulation_budget_scope": "research_generations_only",
            "native_decay_rescue_parent": (native_decay_parent or {}).get("alpha_id") if native_decay_parent else None,
            "native_decay_rescue_values": list(native_decay_values),
            "native_decay_rescue_simulations": len(native_decay_results),
            "active_reference_alphas": len(active_alphas),
            "active_motif_expressions": len(active_motif_plan),
            "platform_near_miss_expressions": len(near_miss_plan),
            "memory_parent_mutations": len(memory_plan),
            "skill_expressions": len(skill_plan),
            "skill_first_required": strict_skill_mode,
            "deterministic_fallback_enabled": bool(allow_deterministic_fallback),
            "knowledge_seed_expressions": len(knowledge_plan),
            "live_field_expressions": len(live_plan),
            "structural_live_expressions": len(structural_plan),
            "chatgpt_expressions": len(chatgpt_plan),
            "seed_expressions": len(seed_plan),
            "simulated": sum(int((item.get("summary") or {}).get("simulated") or 0) for item in generation_results),
            "primary_candidates": len(ranked_primary_candidates),
            "submission_eligible_candidates": len(ranked_primary_candidates),
            "research_high_confidence_candidates": len(ready_candidates),
            "candidates": len(ready_candidates),
            "validation_pending": sum(1 for item in ranked_primary_candidates if (item.get("validation") or {}).get("status") == "validation_pending"),
            "robustness_failed": sum(1 for item in ranked_primary_candidates if (item.get("validation") or {}).get("status") == "robustness_fail"),
            "same_round_robustness_feedback": robustness_feedback_failures,
            "validation_simulations": validation_simulations,
            "validation_cap": validation_cap,
            "validation_simulation_budget_upper_bound": validation_cap * 2,
            "native_decay_rescue_budget_upper_bound": 0 if strict_skill_mode else 2,
            "total_simulation_budget_upper_bound": max_simulations + validation_cap * 2 + (0 if strict_skill_mode else 2),
            "total_simulations": sum(int((item.get("summary") or {}).get("simulated") or 0) for item in generation_results) + validation_simulations + len(native_decay_results),
            "simulation_budget_note": (
                "max_simulations is the primary-generation budget; strict Skill-first disables server-side native-decay rescue"
                if strict_skill_mode
                else "max_simulations is the primary-generation budget; robustness validation and a max-2 knowledge native-decay rescue are separately bounded and explicit"
            ),
            "directed_mutation_routes": sum(1 for item in all_results if item.get("directed_mutation_routed")),
            "simulation_failed": len(all_failed),
            "invalid": len(all_invalid),
            "formally_submitted": 0,
            "cancelled": bool(check_cancelled and check_cancelled()),
        },
        "generations": generation_results,
        "results": all_results,
        "candidates": ranked_primary_candidates,
        "submission_candidates": ranked_primary_candidates,
        "ready_candidates": ready_candidates,
        "research_high_confidence_candidates": ready_candidates,
        "failed": all_failed,
        "invalid": all_invalid,
        "skill_plan_rejections": skill_plan_rejections,
        "best": best,
    }
