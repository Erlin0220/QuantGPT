"""Persistent research memory for WorldQuant autonomous Alpha exploration."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from sqlalchemy import select

from .db import _get_session_factory
from .models import WQResearchTrial
from .wq_operator_registry import canonicalize_wq_expression


def normalize_wq_expression(expression: str) -> str:
    """Canonical normalized expression used for local research-memory dedupe."""
    value = str(expression or "").strip()
    if not value:
        return ""
    try:
        value = canonicalize_wq_expression(value)
    except Exception:
        pass
    return re.sub(r"\s+", "", value).lower()


def classify_wq_family(expression: str) -> str:
    """Classify an expression into a coarse signal family for budget allocation."""
    expr = normalize_wq_expression(expression)
    if not expr:
        return "unknown"
    if "analyst_" in expr or "revision" in expr:
        return "analyst_revision"
    if any(
        token in expr
        for token in (
            "implied_volatility",
            "historical_volatility",
            "pcr_",
            "open_interest",
            "option",
        )
    ):
        return "options_volatility"
    if "snt_" in expr or "social" in expr or "sentiment" in expr:
        return "sentiment"
    if any(
        token in expr
        for token in (
            "mdf_",
            "cashflow",
            "free_cash_flow",
            "net_income",
            "dividends",
            "enterprise_value",
            "market_cap",
            "net_debt",
            "assets",
            "leverage",
            "quality",
        )
    ):
        return "fundamental_quality"
    if any(token in expr for token in ("vwap", "volume", "adv20", "adv60", "turnover")):
        return "price_volume"
    if any(token in expr for token in ("ts_std", "std_dev", "volatility", "ts_arg_max", "ts_arg_min")):
        return "volatility_structure"
    if any(token in expr for token in ("returns", "close", "open", "high", "low")):
        return "momentum_reversal"
    return "other"


def _self_correlation_failed(item: dict[str, Any]) -> bool:
    metrics = item.get("is_metrics") or {}
    for check in metrics.get("checks") or []:
        if str(check.get("name") or "").upper() != "SELF_CORRELATION":
            continue
        if str(check.get("result") or "").upper() == "FAIL":
            return True
    return False


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trial_from_item(
    account: str,
    item: dict[str, Any],
    *,
    status: str,
    settings: dict[str, Any],
    tag: str | None,
    default_family: str,
    default_hypothesis: str,
) -> WQResearchTrial | None:
    expression = str(item.get("expression") or "").strip()
    normalized = normalize_wq_expression(expression)
    if not expression or not normalized:
        return None

    metrics = item.get("is_metrics") or {}
    meta = item.get("research_meta") or {}
    mutation_targets = item.get("mutation_targets")
    if mutation_targets is None and status in {"invalid", "simulation_failed"}:
        error = str(item.get("error") or "").strip()
        mutation_targets = [f"{status}:{error}"] if error else [status]

    return WQResearchTrial(
        account=account,
        alpha_id=str(item.get("alpha_id") or "").strip() or None,
        expression=expression,
        expression_normalized=normalized,
        family=str(meta.get("family") or default_family or classify_wq_family(expression)),
        hypothesis=str(meta.get("hypothesis") or default_hypothesis or "") or None,
        parent_expression=str(meta.get("parent_expression") or "") or None,
        generation=int(meta.get("generation") or 0),
        mutation_type=str(meta.get("mutation_type") or "") or None,
        status=status,
        sharpe=_safe_float(metrics.get("sharpe")),
        fitness=_safe_float(metrics.get("fitness")),
        returns=_safe_float(metrics.get("returns")),
        turnover=_safe_float(metrics.get("turnover")),
        self_correlation_failed=_self_correlation_failed(item),
        mutation_targets=list(mutation_targets or []),
        settings=dict(item.get("settings") or settings or {}),
        data_fields=list(meta.get("data_fields") or []),
        dataset_id=str(meta.get("dataset_id") or "") or None,
        tag=tag,
    )


async def record_research_trials(
    account: str,
    result: dict[str, Any],
    *,
    default_family: str = "external",
    hypothesis: str = "",
    tag: str | None = None,
) -> int:
    """Persist simulated, invalid, and failed expressions from a research batch."""
    if not result:
        return 0

    settings = dict(result.get("settings") or {})
    effective_tag = tag or result.get("tag")
    factory = _get_session_factory()
    rows: list[WQResearchTrial] = []

    candidate_ids = {
        str(item.get("alpha_id"))
        for item in (result.get("candidates") or [])
        if item.get("alpha_id")
        and (
            "validation" not in item
            or str((item.get("validation") or {}).get("status") or "").lower() == "ready"
        )
    }
    for item in result.get("results") or []:
        status = "candidate" if str(item.get("alpha_id")) in candidate_ids else "rejected"
        row = _trial_from_item(
            account,
            item,
            status=status,
            settings=settings,
            tag=effective_tag,
            default_family=default_family,
            default_hypothesis=hypothesis,
        )
        if row is not None:
            rows.append(row)

    for status_key, status in (("failed", "simulation_failed"), ("invalid", "invalid")):
        for item in result.get(status_key) or []:
            row = _trial_from_item(
                account,
                item,
                status=status,
                settings=settings,
                tag=effective_tag,
                default_family=default_family,
                default_hypothesis=hypothesis,
            )
            if row is not None:
                rows.append(row)

    if not rows:
        return 0

    async with factory() as session:
        session.add_all(rows)
        await session.commit()
    return len(rows)


async def load_research_memory(account: str = "primary", limit: int = 2000) -> dict[str, Any]:
    """Load a bounded summary used by the autonomous planner."""
    factory = _get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(WQResearchTrial)
            .where(WQResearchTrial.account == account)
            .order_by(WQResearchTrial.created_at.desc())
            .limit(max(1, min(5000, int(limit))))
        )
        rows = list(result.scalars().all())

    family_counts = Counter(str(row.family or "unknown") for row in rows)
    status_counts = Counter(str(row.status or "unknown") for row in rows)
    candidate_family_counts = Counter(
        str(row.family or "unknown") for row in rows if str(row.status or "") == "candidate"
    )
    self_corr_family_counts = Counter(
        str(row.family or "unknown") for row in rows if bool(row.self_correlation_failed)
    )
    normalized = {str(row.expression_normalized or "") for row in rows if row.expression_normalized}
    recent_trials = [
        {
            "expression": row.expression,
            "family": row.family,
            "hypothesis": row.hypothesis,
            "parent_expression": row.parent_expression,
            "generation": int(row.generation or 0),
            "mutation_type": row.mutation_type,
            "status": row.status,
            "sharpe": row.sharpe,
            "fitness": row.fitness,
            "returns": row.returns,
            "turnover": row.turnover,
            "self_correlation_failed": bool(row.self_correlation_failed),
            "mutation_targets": list(row.mutation_targets or []),
            "data_fields": list(row.data_fields or []),
            "dataset_id": row.dataset_id,
        }
        for row in rows[:200]
        if row.expression
    ]

    return {
        "trials": len(rows),
        "family_counts": dict(family_counts),
        "candidate_family_counts": dict(candidate_family_counts),
        "self_correlation_family_counts": dict(self_corr_family_counts),
        "status_counts": dict(status_counts),
        "normalized_expressions": sorted(normalized),
        "recent_trials": recent_trials,
    }


def record_research_trials_sync(*args, **kwargs) -> int:
    from .wq_submission_policy import _run_coro_sync

    return int(_run_coro_sync(record_research_trials(*args, **kwargs)))


def load_research_memory_sync(account: str = "primary", limit: int = 2000) -> dict[str, Any]:
    from .wq_submission_policy import _run_coro_sync

    return dict(_run_coro_sync(load_research_memory(account, limit=limit)))
