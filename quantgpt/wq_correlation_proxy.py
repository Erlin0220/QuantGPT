"""Local candidate-vs-ACTIVE daily PnL correlation evidence.

This is a pre-submission diversity proxy only. It never represents BRAIN's
formal SELF_CORRELATION check and missing evidence remains non-blocking.
"""
from __future__ import annotations

from datetime import datetime, timezone
from math import sqrt
from typing import Any, Iterable

_MIN_CORRELATION_SAMPLES = 20
_HIGH_CORRELATION_THRESHOLD = 0.70


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def daily_changes_from_pnl(records: Any) -> dict[str, float]:
    if isinstance(records, dict):
        records = records.get("records") or records.get("data") or records.get("results") or []
    if not isinstance(records, list):
        return {}
    points: list[tuple[str, float]] = []
    for index, item in enumerate(records):
        if isinstance(item, dict):
            date = str(item.get("date") or item.get("day") or item.get("timestamp") or "") or None
            value = _safe_float(item.get("pnl") if "pnl" in item else item.get("value"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            date, value = str(item[0]), _safe_float(item[-1])
        else:
            date, value = str(index), _safe_float(item)
        if date is not None and value is not None:
            points.append((date, value))
    if len(points) < 2:
        return {}
    changes: dict[str, float] = {}
    previous = points[0][1]
    for date, value in points[1:]:
        changes[date] = value - previous
        previous = value
    return changes


def _pearson(left: Iterable[float], right: Iterable[float]) -> float | None:
    xs, ys = list(left), list(right)
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    dx, dy = [x - mean_x for x in xs], [y - mean_y for y in ys]
    denom = sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom <= 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def portfolio_correlation_proxy(candidate_changes: dict[str, float], active_changes: dict[str, dict[str, float]], *, min_samples: int = _MIN_CORRELATION_SAMPLES) -> dict[str, Any]:
    best_value: float | None = None
    best_alpha_id: str | None = None
    best_samples = 0
    for alpha_id, series in active_changes.items():
        overlap = sorted(set(candidate_changes) & set(series))
        if len(overlap) < max(2, int(min_samples)):
            continue
        value = _pearson((candidate_changes[d] for d in overlap), (series[d] for d in overlap))
        if value is None:
            continue
        absolute = abs(value)
        if best_value is None or absolute > best_value:
            best_value, best_alpha_id, best_samples = absolute, str(alpha_id), len(overlap)
    calculated_at = datetime.now(timezone.utc).isoformat()
    if best_value is None:
        return {"status": "unavailable", "max_correlation": None, "matching_alpha_id": None, "sample_length": 0, "calculated_at": calculated_at, "high_correlation": False, "official_sc": False}
    return {"status": "available", "max_correlation": round(best_value, 6), "matching_alpha_id": best_alpha_id, "sample_length": best_samples, "calculated_at": calculated_at, "high_correlation": best_value >= _HIGH_CORRELATION_THRESHOLD, "official_sc": False}


def correlation_evidence_is_fresh(evidence: dict[str, Any] | None, *, max_age_hours: float = 24.0, now: datetime | None = None) -> bool:
    evidence = evidence or {}
    if evidence.get("status") != "available" or not evidence.get("calculated_at"):
        return False
    try:
        calculated = datetime.fromisoformat(str(evidence["calculated_at"]).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    if calculated.tzinfo is None:
        calculated = calculated.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return max(0.0, (current.astimezone(timezone.utc) - calculated.astimezone(timezone.utc)).total_seconds() / 3600.0) <= max(0.0, float(max_age_hours))


def correlation_priority_multiplier(evidence: dict[str, Any] | None) -> float:
    evidence = evidence or {}
    value = _safe_float(evidence.get("max_correlation"))
    if evidence.get("status") != "available" or value is None or not correlation_evidence_is_fresh(evidence):
        return 1.0
    if value < 0.50:
        return 1.0
    if value >= 0.85:
        return 0.35
    return max(0.35, 1.0 - (value - 0.50) * 1.5)
