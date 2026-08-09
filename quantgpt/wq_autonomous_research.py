"""Bounded autonomous WorldQuant Alpha planning and evolutionary research."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable

from .wq_brain_service import run_list_alphas
from .wq_research_agent import run_research_batch
from .wq_research_memory import classify_wq_family, normalize_wq_expression

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
    # Lower is better: retain exploration pressure while exploiting families that
    # have produced candidates or several near-threshold trials.
    score = math.log1p(trials) - candidate_rate * 8.0 - promise * 1.5 + self_corr_rate * 4.0
    return score, family


def select_research_families(memory: dict[str, Any] | None = None, count: int = 3) -> list[str]:
    memory = memory or {}
    ordered = sorted(FAMILY_SEEDS, key=lambda family: _family_priority(memory, family))
    return ordered[: max(1, min(len(ordered), int(count)))]


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
    plan: list[dict[str, Any]] = []
    offsets = Counter()
    pools = {family: _family_seed_pool(family, generation, hypothesis) for family in families}

    while len(plan) < max(1, int(limit)):
        added = False
        for family in families:
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

    meta = result.get("research_meta") or {}
    family = str(meta.get("family") or classify_wq_family(expression))
    generation = int(meta.get("generation") or 1) + 1
    targets = list(result.get("mutation_targets") or [])
    metrics = result.get("is_metrics") or {}
    sharpe = _safe_metric(metrics.get("sharpe"))
    variants: list[tuple[str, str]] = []

    if "reduce_turnover" in targets or "improve_sub_universe_robustness" in targets:
        variants.append((_widen_windows(expression), "widen_windows"))
    if "increase_turnover" in targets:
        variants.append((_narrow_windows(expression), "narrow_windows"))
    if "reduce_weight_concentration" in targets and not expression.lstrip().startswith("rank("):
        variants.append((f"rank({expression})", "cross_sectional_rank"))
    if "improve_signal_sharpe" in targets or "improve_fitness" in targets:
        # A positive-but-insufficient signal should not waste a simulation on a
        # deterministic sign flip that is likely to make Sharpe negative. Smooth
        # it first; only invert genuinely negative signals.
        if sharpe < 0:
            variants.append((f"-1 * ({expression})", "invert_signal"))
        variants.append((f"rank(ts_mean(({expression}), 10))", "smooth_signal"))

    out: list[dict[str, Any]] = []
    for variant, mutation_type in variants:
        normalized = normalize_wq_expression(variant)
        if not normalized or normalized in seen or normalized == normalize_wq_expression(expression):
            continue
        seen.add(normalized)
        out.append(
            {
                "expression": variant,
                "family": family,
                "generation": generation,
                "hypothesis": hypothesis or str(meta.get("hypothesis") or ""),
                "parent_expression": expression,
                "mutation_type": mutation_type,
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
            },
            "mutation_targets": targets,
            "research_meta": {
                "family": trial.get("family") or classify_wq_family(str(trial.get("expression") or "")),
                "generation": generation,
                "hypothesis": trial.get("hypothesis") or hypothesis,
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
    seen = set(normalized)

    # Spend roughly one third of the first-generation budget improving the best
    # recent failures. The rest explores under-covered signal families. If the
    # static seed catalog is exhausted, research memory is allowed to consume the
    # remaining budget so autonomous research keeps advancing across hourly runs.
    memory_parent_budget = min(first_budget, max(0, first_budget // 3))
    memory_plan = build_memory_mutation_plan(
        memory,
        seen=seen,
        limit=memory_parent_budget,
        hypothesis=goal,
        min_sharpe=min_sharpe,
        min_fitness=min_fitness,
    )
    seed_memory = dict(memory)
    seed_memory["normalized_expressions"] = sorted(seen)
    seed_plan = build_seed_plan(
        seed_memory,
        limit=max(0, first_budget - len(memory_plan)),
        family_count=family_count,
        generation=1,
        hypothesis=goal,
    )
    for item in seed_plan:
        seen.add(normalize_wq_expression(item["expression"]))

    if len(memory_plan) + len(seed_plan) < first_budget:
        memory_plan.extend(
            build_memory_mutation_plan(
                memory,
                seen=seen,
                limit=first_budget - len(memory_plan) - len(seed_plan),
                hypothesis=goal,
                min_sharpe=min_sharpe,
                min_fitness=min_fitness,
            )
        )
    first_plan = seed_plan + memory_plan
    generation_results: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    all_failed: list[dict[str, Any]] = []
    all_invalid: list[dict[str, Any]] = []
    selected_families = select_research_families(memory, family_count)
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
        plan = next_plan

    all_results.sort(key=_rank_result, reverse=True)
    all_candidates.sort(key=_rank_result, reverse=True)
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
        "memory_before": {
            "trials": int(memory.get("trials") or 0),
            "family_counts": dict(memory.get("family_counts") or {}),
        },
        "settings": settings,
        "summary": {
            "generations_requested": generations,
            "generations_completed": len(generation_results),
            "max_simulations": max_simulations,
            "memory_parent_mutations": len(memory_plan),
            "seed_expressions": len(seed_plan),
            "simulated": sum(int((item.get("summary") or {}).get("simulated") or 0) for item in generation_results),
            "candidates": len(all_candidates),
            "simulation_failed": len(all_failed),
            "invalid": len(all_invalid),
            "formally_submitted": 0,
            "cancelled": bool(check_cancelled and check_cancelled()),
        },
        "generations": generation_results,
        "results": all_results,
        "candidates": all_candidates,
        "failed": all_failed,
        "invalid": all_invalid,
        "best": best,
    }
