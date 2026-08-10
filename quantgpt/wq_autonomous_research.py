"""Bounded autonomous WorldQuant Alpha planning and evolutionary research."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any, Callable

from .wq_brain_service import run_batch_simulation, run_list_alphas
from .wq_lineage import extract_expression_metadata
from .wq_mutation_policy import preferred_mutation_classes
from .wq_operator_registry import validate_wq_expression
from .wq_research_agent import run_research_batch
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
    if not fields or limit <= 0:
        return [], catalog, fields
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
                return plan, catalog, fields
    return plan, catalog, fields


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


def build_llm_live_plan(
    client,
    fields: list[dict[str, Any]],
    *,
    seen: set[str],
    limit: int,
    hypothesis: str,
    memory: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Use the existing DeepSeek/OpenAI-compatible provider as a bounded WQ idea generator."""
    if limit <= 0 or not os.environ.get("DEEPSEEK_API_KEY") or not fields:
        return []
    try:
        supported = client.list_operator_names()
    except Exception:
        return []
    usable_fields = [_live_field_id(field) for field in fields if _live_field_id(field)][:24]
    if not usable_fields:
        return []

    from .iteration import _call_llm

    operators = sorted(supported)
    operator_text = ", ".join(operators[:120])
    field_text = ", ".join(usable_fields)
    system_prompt = (
        "You design WorldQuant BRAIN FASTEXPR research hypotheses. Output exactly one FASTEXPR expression, no prose. "
        "Use only the supplied data fields/operators. Prefer simple economically interpretable structures, 2-5 operator calls, "
        "and avoid pure lookback parameter tuning. Do not use assignments, semicolons, local-only indicators, or invented fields."
    )
    guidance = (memory or {}).get("research_memory_guidance") or {}
    positive = [
        f"{item.get('family')}|{item.get('dataset_id') or 'unresolved'}|{item.get('operator_pattern')}"
        for item in (guidance.get("positive") or [])[:6]
    ]
    negative = [
        f"{item.get('family')}|{item.get('dataset_id') or 'unresolved'}|{item.get('operator_pattern')}|{item.get('failure_reason')}"
        for item in (guidance.get("negative") or [])[:8]
    ]
    overused = [str(item.get("operator_pattern") or "") for item in (guidance.get("overused_structures") or [])[:6]]
    out: list[dict[str, Any]] = []
    for index in range(limit):
        avoid = [item for item in list(seen)[-8:]]
        user_prompt = (
            f"Research objective: {hypothesis}\n"
            f"Allowed MATRIX fields: {field_text}\n"
            f"Allowed operators: {operator_text}\n"
            f"Positive structural memory to learn from (do not copy literally): {positive}\n"
            f"Negative structural memory to avoid repeating: {negative}\n"
            f"Overused operator structures to diversify away from softly: {overused}\n"
            f"Already researched normalized expressions to avoid: {avoid}\n"
            f"Generate idea {index + 1} with a distinct structural hypothesis."
        )
        try:
            expression = _call_llm(
                system_prompt,
                user_prompt,
                temperature=min(1.0 + index * 0.15, 1.4),
                max_tokens=4096,
            ).strip()
        except Exception:
            break
        validation = validate_wq_expression(expression, supported)
        if not validation.ok:
            continue
        expression = validation.expression
        normalized = normalize_wq_expression(expression)
        if not normalized or normalized in seen:
            continue
        if not _expression_uses_only_catalog_fields(expression, supported, set(usable_fields)):
            continue
        seen.add(normalized)
        used = [field for field in usable_fields if re.search(rf"(?<![a-z0-9_]){re.escape(field.lower())}(?![a-z0-9_])", normalized)]
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
                "family": _classify_live_field(used_items[0] if used_items else {}),
                "generation": 1,
                "hypothesis": hypothesis,
                "parent_expression": None,
                "mutation_type": "llm_live_field_seed",
                "planner_strategy": "positive_negative_memory_llm",
                "data_fields": used,
                "dataset_id": dataset_id,
                "dataset_category": dataset_category,
                "provenance_state": "resolved" if dataset_id else "unresolved",
                "provenance_reason": "live_field_catalog" if dataset_id else "llm_uses_multiple_or_unresolved_datasets",
            }
        )
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
        max_generation = max(1, min(5, int(os.environ.get("WQ_MUTATION_MAX_GENERATION", "3"))))
    except (TypeError, ValueError):
        max_generation = 3
    limit = min(int(limit), max_children)

    meta = result.get("research_meta") or {}
    family = str(meta.get("family") or classify_wq_family(expression))
    generation = int(meta.get("generation") or 1) + 1
    if generation > max_generation:
        result["mutation_route_terminal_reason"] = "mutation_generation_budget_exhausted"
        return []
    targets = list(result.get("mutation_targets") or [])
    metrics = result.get("is_metrics") or {}
    sharpe = _safe_metric(metrics.get("sharpe"))
    directives = preferred_mutation_classes(result)
    variants: list[tuple[str, str, str, str | None]] = []
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
        elif mutation_class in {"smooth_signal", "cost_reduction"}:
            variants.append((f"rank(ts_mean(({expression}), 10))", mutation_class, rationale, None))
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
    recent.sort(
        key=lambda item: _trial_promise_score(item, min_sharpe=min_sharpe, min_fitness=min_fitness),
        reverse=True,
    )

    plan: list[dict[str, Any]] = []
    for trial in recent:
        generation = int(trial.get("generation") or 0)
        if generation >= 3:
            continue
        targets = list(trial.get("mutation_targets") or [])
        if not targets or targets == ["candidate_passes_primary_thresholds"]:
            continue
        promise = _trial_promise_score(trial, min_sharpe=min_sharpe, min_fitness=min_fitness)
        if promise < min_parent_promise:
            continue
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
    """Run a small alternate-setting funnel before a primary-pass Alpha becomes READY."""
    expression = str(candidate.get("expression") or "").strip()
    if not expression:
        return {"status": "robustness_unavailable", "robustness_score": 0.0, "reason": "missing_expression"}
    alternate_universe = "TOP1000" if str(universe).upper() == "TOP3000" else "TOP3000"
    alternate_neutralization = "INDUSTRY" if str(neutralization).upper() != "INDUSTRY" else "SUBINDUSTRY"
    try:
        sweep = run_batch_simulation(
            client,
            expression,
            regions=[region],
            delays=[delay],
            universes=[alternate_universe],
            neutralizations=[neutralization, alternate_neutralization],
            decay=decay,
            truncation=truncation,
            auto_submit=False,
            check_cancelled=check_cancelled,
        )
    except Exception as exc:
        return {
            "status": "robustness_unavailable",
            "robustness_score": 0.0,
            "reason": str(exc)[:300],
            "validation_simulations": 0,
        }

    sub_results = list((sweep.get("sub_results") or {}).values()) if isinstance(sweep, dict) else []
    completed = [item for item in sub_results if item.get("status") == "completed"]
    passes = []
    for item in completed:
        sharpe = _safe_metric(item.get("sharpe"), -999.0)
        fitness = _safe_metric(item.get("fitness"), -999.0)
        turnover = _safe_metric(item.get("turnover"), 0.0)
        returns = _safe_metric(item.get("returns"), -999.0)
        if sharpe >= 1.0 and fitness >= 0.7 and 0.01 <= turnover <= 0.7 and returns > 0:
            passes.append(item)
    score = len(passes) / max(1, len(completed))
    ready = len(completed) >= 2 and score >= 0.5
    return {
        "status": "ready" if ready else "robustness_fail",
        "robustness_score": round(score, 4),
        "validation_simulations": len(sub_results),
        "completed": len(completed),
        "passed": len(passes),
        "alternate_universe": alternate_universe,
        "neutralizations": [neutralization, alternate_neutralization],
        "details": sub_results,
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


def _rank_result(item: dict[str, Any]) -> tuple[float, float, float]:
    metrics = item.get("is_metrics") or {}
    def num(key: str) -> float:
        try:
            return float(metrics.get(key) or -999.0)
        except (TypeError, ValueError):
            return -999.0
    return num("fitness"), num("sharpe"), num("returns")


def run_autonomous_research(
    client,
    *,
    memory: dict[str, Any] | None = None,
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
    memory = dict(memory or {})
    family_counts = Counter(memory.get("family_counts") or {})
    normalized = set(memory.get("normalized_expressions") or [])

    # Seed the persistent memory with recent platform research so the first autonomous
    # run does not simply rediscover the last externally-created expressions.
    recent = run_list_alphas(client, limit=100)
    if recent.get("ok"):
        for alpha in recent.get("alphas") or []:
            expression = str(alpha.get("expression") or "")
            norm = normalize_wq_expression(expression)
            if norm:
                normalized.add(norm)
                family_counts[classify_wq_family(expression)] += 1
    memory["family_counts"] = dict(family_counts)
    memory["normalized_expressions"] = sorted(normalized)

    first_budget = max_simulations if generations == 1 else max(4, math.ceil(max_simulations * 0.65))
    inventory = memory.get("inventory") or {}
    explicit_inventory_mode = str(inventory.get("mode") or memory.get("inventory_mode") or "").upper()
    high_confidence_inventory = sum(
        int(cell.get("high_confidence_candidates") or 0) for cell in (memory.get("research_cells") or [])
    )
    inventory_mode = explicit_inventory_mode or (
        "REPLENISHMENT" if high_confidence_inventory < 30 else "EXPLORATION" if high_confidence_inventory > 50 else "NORMAL"
    )
    adaptive_allocation = allocate_research_cells(
        memory.get("research_cells") or [],
        budget=first_budget,
        inventory_mode=inventory_mode,
        learning_maturity=memory.get("learning_maturity") or {},
    )
    memory["adaptive_allocation"] = adaptive_allocation
    seen = set(normalized)

    # Reserve most first-generation budget for genuinely new hypotheses. In
    # replenishment mode, repeated failed parents are capped aggressively so the
    # search does not spend another round in the same low-yield local optimum.
    memory_parent_budget = min(
        first_budget,
        max(0, math.ceil(first_budget * (0.25 if inventory_mode == "REPLENISHMENT" else 0.30))),
    )
    memory_plan = build_memory_mutation_plan(
        memory,
        seen=seen,
        limit=memory_parent_budget,
        hypothesis=goal,
        min_sharpe=min_sharpe,
        min_fitness=min_fitness,
    )
    exploration_budget = max(0, first_budget - len(memory_plan))
    live_budget = min(exploration_budget, max(0, math.ceil(exploration_budget * 0.35)))
    live_plan, live_catalog, live_fields = build_live_field_plan(
        client,
        memory,
        seen=seen,
        limit=live_budget,
        region=region,
        universe=universe,
        delay=delay,
        hypothesis=goal,
    )
    structural_budget = min(
        max(0, exploration_budget - len(live_plan)),
        max(0, math.ceil(exploration_budget * 0.35)),
    )
    structural_plan = build_structural_live_plan(
        client,
        live_fields,
        seen=seen,
        limit=structural_budget,
        hypothesis=goal,
    )
    llm_budget = min(4, max(0, exploration_budget - len(live_plan) - len(structural_plan)))
    llm_plan = build_llm_live_plan(
        client,
        live_fields,
        seen=seen,
        limit=llm_budget,
        hypothesis=goal,
        memory=memory,
    )
    seed_memory = dict(memory)
    seed_memory["normalized_expressions"] = sorted(seen)
    seed_plan = build_seed_plan(
        seed_memory,
        limit=max(0, exploration_budget - len(live_plan) - len(structural_plan) - len(llm_plan)),
        family_count=family_count,
        generation=1,
        hypothesis=goal,
    )
    for item in seed_plan:
        seen.add(normalize_wq_expression(item["expression"]))

    planned_count = len(memory_plan) + len(live_plan) + len(structural_plan) + len(llm_plan) + len(seed_plan)
    if planned_count < first_budget:
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
        item for item in (memory_plan + live_plan + structural_plan + llm_plan + seed_plan)
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
    validation_cap = replenishment_cap if inventory_mode == "REPLENISHMENT" else base_validation_cap
    validation_candidates_used = 0
    validation_simulations = 0
    robustness_feedback_failures = 0
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

    # All primary candidates were validated (or explicitly marked pending) as
    # they emerged, so no terminal-only robustness pass is needed here.
    ranked_primary_candidates = sorted(all_candidates, key=_rank_result, reverse=True)
    ready_candidates = [
        item
        for item in ranked_primary_candidates
        if (item.get("validation") or {}).get("status") == "ready"
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
        "mode": "autonomous",
        "goal": goal,
        "tag": tag,
        "selected_families": selected_families,
        "adaptive_allocation": adaptive_allocation,
        "learning_maturity": memory.get("learning_maturity") or {},
        "research_memory_guidance": memory.get("research_memory_guidance") or {},
        "inventory_mode": inventory_mode,
        "live_catalog": live_catalog,
        "memory_before": {
            "trials": int(memory.get("trials") or 0),
            "family_counts": dict(memory.get("family_counts") or {}),
        },
        "settings": settings,
        "summary": {
            "generations_requested": generations,
            "generations_completed": len(generation_results),
            "primary_simulation_budget": max_simulations,
            "primary_simulation_budget_scope": "research_generations_only",
            "memory_parent_mutations": len(memory_plan),
            "live_field_expressions": len(live_plan),
            "structural_live_expressions": len(structural_plan),
            "llm_expressions": len(llm_plan),
            "seed_expressions": len(seed_plan),
            "simulated": sum(int((item.get("summary") or {}).get("simulated") or 0) for item in generation_results),
            "primary_candidates": len(ranked_primary_candidates),
            "candidates": len(ready_candidates),
            "validation_pending": sum(1 for item in ranked_primary_candidates if (item.get("validation") or {}).get("status") == "validation_pending"),
            "robustness_failed": sum(1 for item in ranked_primary_candidates if (item.get("validation") or {}).get("status") == "robustness_fail"),
            "same_round_robustness_feedback": robustness_feedback_failures,
            "validation_simulations": validation_simulations,
            "validation_cap": validation_cap,
            "validation_simulation_budget_upper_bound": validation_cap * 2,
            "total_simulation_budget_upper_bound": max_simulations + validation_cap * 2,
            "total_simulations": sum(int((item.get("summary") or {}).get("simulated") or 0) for item in generation_results) + validation_simulations,
            "simulation_budget_note": "max_simulations is the primary-generation budget; robustness validation is separately bounded and both budgets are explicit",
            "directed_mutation_routes": sum(1 for item in all_results if item.get("directed_mutation_routed")),
            "simulation_failed": len(all_failed),
            "invalid": len(all_invalid),
            "formally_submitted": 0,
            "cancelled": bool(check_cancelled and check_cancelled()),
        },
        "generations": generation_results,
        "results": all_results,
        "candidates": ranked_primary_candidates,
        "ready_candidates": ready_candidates,
        "failed": all_failed,
        "invalid": all_invalid,
        "best": best,
    }
