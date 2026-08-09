"""WorldQuant Alpha Research Agent orchestration.

ChatGPT/another caller provides candidate expressions. This module owns the
expensive deterministic work: canonicalization, deduplication, real BRAIN
simulation, threshold filtering, failure diagnosis, ranking, and optional
parameter sweeps. It never formally submits an alpha.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable

from .wq_brain_client import SUBMIT_THRESHOLDS
from .wq_brain_service import (
    _run_adaptive_parallel,
    run_batch_simulation,
    run_list_alphas,
    run_single_simulation,
)
from .wq_operator_registry import WQ_FALLBACK_OPERATORS, canonicalize_wq_expression, validate_wq_expression


def _normalize(expression: str) -> str:
    return re.sub(r"\s+", "", canonicalize_wq_expression(expression).lower())


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def diagnose_wq_result(result: dict, min_sharpe: float, min_fitness: float) -> list[str]:
    """Return concise mutation targets from a completed BRAIN simulation."""
    metrics = result.get("is_metrics", {})
    sharpe = _safe_number(metrics.get("sharpe"))
    fitness = _safe_number(metrics.get("fitness"))
    turnover = _safe_number(metrics.get("turnover"))
    failed_checks = [
        str(check.get("name")) for check in metrics.get("checks", []) if str(check.get("result", "")).upper() == "FAIL"
    ]

    actions: list[str] = []
    if sharpe < min_sharpe:
        actions.append("improve_signal_sharpe")
    if fitness < min_fitness:
        actions.append("improve_fitness")
    if turnover > SUBMIT_THRESHOLDS["turnover_max"]:
        actions.append("reduce_turnover")
    elif turnover and turnover < SUBMIT_THRESHOLDS["turnover_min"]:
        actions.append("increase_turnover")
    if "LOW_SUB_UNIVERSE_SHARPE" in failed_checks:
        actions.append("improve_sub_universe_robustness")
    if "CONCENTRATED_WEIGHT" in failed_checks:
        actions.append("reduce_weight_concentration")
    if "SELF_CORRELATION" in failed_checks:
        actions.append("reduce_self_correlation")
    return actions or ["candidate_passes_primary_thresholds"]


def _passes_primary_thresholds(result: dict, min_sharpe: float, min_fitness: float) -> bool:
    metrics = result.get("is_metrics", {})
    sharpe = _safe_number(metrics.get("sharpe"))
    fitness = _safe_number(metrics.get("fitness"))
    turnover = _safe_number(metrics.get("turnover"))
    failed_checks = {
        str(check.get("name")) for check in metrics.get("checks", []) if str(check.get("result", "")).upper() == "FAIL"
    }
    blocking_checks = failed_checks
    return (
        sharpe >= min_sharpe
        and fitness >= min_fitness
        and SUBMIT_THRESHOLDS["turnover_min"] <= turnover <= SUBMIT_THRESHOLDS["turnover_max"]
        and not blocking_checks
    )


def _rank_key(item: dict) -> tuple[float, float, float]:
    metrics = item.get("is_metrics", {})
    return (
        _safe_number(metrics.get("fitness"), -999.0),
        _safe_number(metrics.get("sharpe"), -999.0),
        _safe_number(metrics.get("returns"), -999.0),
    )


def run_research_batch(
    client,
    expressions: Iterable[str],
    *,
    goal: str = "",
    tag: str = "wq-research",
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: int = 0,
    neutralization: str = "SUBINDUSTRY",
    truncation: float = 0.08,
    max_candidates: int = 20,
    min_sharpe: float = 1.25,
    min_fitness: float = 1.0,
    skip_existing: bool = True,
    sweep_top_n: int = 0,
    sweep_universes: list[str] | None = None,
    sweep_neutralizations: list[str] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    check_cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Research a bounded candidate batch using real BRAIN simulations.

    Formal submission is deliberately excluded. ``sweep_top_n`` may run extra
    simulations for the best N candidates but also keeps ``auto_submit=False``.
    """
    requested = [expr.strip() for expr in expressions if expr and expr.strip()]
    requested = requested[: max(1, max_candidates)]

    try:
        supported = client.list_operator_names()
        operator_catalog_source = "live"
        if not supported:
            raise RuntimeError("empty operator catalog")
    except Exception:
        supported = WQ_FALLBACK_OPERATORS
        operator_catalog_source = "fallback"

    unique: list[str] = []
    seen: set[str] = set()
    invalid: list[dict] = []
    duplicates_in_batch: list[str] = []

    for raw in requested:
        validation = validate_wq_expression(raw, supported)
        if not validation.ok:
            invalid.append({"expression": raw, "error": validation.error})
            continue
        normalized = _normalize(validation.expression)
        if normalized in seen:
            duplicates_in_batch.append(raw)
            continue
        seen.add(normalized)
        unique.append(validation.expression)

    existing_norms: set[str] = set()
    existing_skipped: list[str] = []
    if skip_existing:
        existing_result = run_list_alphas(client, limit=100)
        if existing_result.get("ok"):
            existing_norms = {
                _normalize(alpha.get("expression", ""))
                for alpha in existing_result.get("alphas", [])
                if alpha.get("expression")
            }

    to_simulate: list[str] = []
    for expression in unique:
        if _normalize(expression) in existing_norms:
            existing_skipped.append(expression)
        else:
            to_simulate.append(expression)

    results: list[dict] = []
    failed: list[dict] = []

    def simulate_expression(worker_client, expression: str, mark_throttled):
        def on_sim_progress(_pct: int, message: str) -> None:
            if "并发限制" in message or "速率限制" in message or "CONCURRENT_SIMULATION_LIMIT" in message:
                mark_throttled()

        return run_single_simulation(
            worker_client,
            expression,
            region=region,
            universe=universe,
            delay=delay,
            decay=decay,
            neutralization=neutralization,
            truncation=truncation,
            auto_submit=False,
            tag=tag,
            progress_callback=on_sim_progress,
            check_cancelled=check_cancelled,
        )

    def report_sim_progress(current: int, total: int, expression: str) -> None:
        if on_progress:
            on_progress(current, total, f"simulate:{expression}")

    completed, concurrency, cancelled = _run_adaptive_parallel(
        client,
        to_simulate,
        simulate_expression,
        on_progress=report_sim_progress,
        check_cancelled=check_cancelled,
    )

    for expression, result in completed:
        if result.get("cancelled"):
            cancelled = True
            continue
        if not result.get("ok"):
            failed.append({"expression": expression, "error": result.get("error", "simulation failed")})
            continue
        result["passes_primary_thresholds"] = _passes_primary_thresholds(result, min_sharpe, min_fitness)
        result["mutation_targets"] = diagnose_wq_result(result, min_sharpe, min_fitness)
        results.append(result)

    results.sort(key=_rank_key, reverse=True)
    candidates = [item for item in results if item.get("passes_primary_thresholds")]

    sweeps: list[dict] = []
    if sweep_top_n > 0 and not cancelled:
        universes = sweep_universes or [universe]
        neutralizations = sweep_neutralizations or [neutralization]
        for sweep_index, item in enumerate(results[:sweep_top_n], start=1):
            if check_cancelled and check_cancelled():
                cancelled = True
                break
            if on_progress:
                on_progress(sweep_index, min(sweep_top_n, len(results)), f"sweep:{item['expression']}")
            sweep = run_batch_simulation(
                client,
                item["expression"],
                regions=[region],
                delays=[delay],
                universes=universes,
                neutralizations=neutralizations,
                decay=decay,
                truncation=truncation,
                auto_submit=False,
                tag=tag,
                check_cancelled=check_cancelled,
            )
            sweeps.append(sweep)

    return {
        "ok": True,
        "goal": goal,
        "tag": tag,
        "operator_catalog_source": operator_catalog_source,
        "settings": {
            "region": region,
            "universe": universe,
            "delay": delay,
            "decay": decay,
            "neutralization": neutralization,
            "truncation": truncation,
            "min_sharpe": min_sharpe,
            "min_fitness": min_fitness,
        },
        "summary": {
            "requested": len(requested),
            "valid_unique": len(unique),
            "duplicates_in_batch": len(duplicates_in_batch),
            "existing_skipped": len(existing_skipped),
            "simulated": len(results),
            "simulation_failed": len(failed),
            "candidates": len(candidates),
            "sweeps": len(sweeps),
            "formally_submitted": 0,
            "cancelled": cancelled,
            "concurrency": concurrency,
        },
        "invalid": invalid,
        "duplicates": duplicates_in_batch,
        "existing_skipped_expressions": existing_skipped,
        "failed": failed,
        "results": results,
        "candidates": candidates,
        "sweeps": sweeps,
        "best": results[0] if results else None,
    }
