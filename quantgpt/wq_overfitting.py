"""Multiple-testing-aware soft evidence for WorldQuant research candidates.

These diagnostics are local research evidence, not WorldQuant platform checks.
Missing evidence is intentionally neutral so sparse candidate inventory is not
starved when PnL history is unavailable.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Mapping, Sequence

_TRADING_DAYS = 252.0


def _finite(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _moments(values: Sequence[float]) -> tuple[float, float, float, float]:
    data = _finite(values)
    n = len(data)
    if n < 3:
        return 0.0, 0.0, 0.0, 3.0
    mean = sum(data) / n
    centered = [value - mean for value in data]
    variance = sum(value * value for value in centered) / max(1, n - 1)
    std = math.sqrt(max(0.0, variance))
    if std <= 0.0:
        return mean, std, 0.0, 3.0
    skewness = sum((value / std) ** 3 for value in centered) / n
    kurtosis = sum((value / std) ** 4 for value in centered) / n
    return mean, std, skewness, kurtosis


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def probabilistic_sharpe_ratio(
    annualized_sharpe: float,
    *,
    benchmark_annualized_sharpe: float = 0.0,
    sample_count: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float | None:
    """Return PSR using the Bailey/López de Prado non-normality correction.

    Annualized Sharpe values are converted to daily scale before applying the
    finite-sample formula. ``None`` means the evidence is unavailable.
    """
    n = int(sample_count)
    if n < 3:
        return None
    sr = float(annualized_sharpe) / math.sqrt(_TRADING_DAYS)
    benchmark = float(benchmark_annualized_sharpe) / math.sqrt(_TRADING_DAYS)
    denominator_term = 1.0 - float(skewness) * sr + ((float(kurtosis) - 1.0) / 4.0) * sr * sr
    if denominator_term <= 0.0:
        return None
    z_score = (sr - benchmark) * math.sqrt(n - 1) / math.sqrt(denominator_term)
    return max(0.0, min(1.0, _normal_cdf(z_score)))


def deflated_sharpe_evidence(
    daily_returns: Sequence[float],
    *,
    annualized_sharpe: float,
    related_trials: int = 1,
) -> dict[str, Any]:
    """Build bounded DSR-style evidence for one coherent variant family.

    Without a full cross-trial Sharpe distribution we use the expected maximum
    noise Sharpe approximation ``sqrt(2 log N / T)`` as the hurdle. This is a
    conservative soft ranking feature, never a formal submission gate.
    """
    data = _finite(daily_returns)
    n = len(data)
    trials = max(1, int(related_trials))
    calculated_at = datetime.now(timezone.utc).isoformat()
    if n < 20:
        return {
            "status": "unavailable",
            "reason": "insufficient_return_history",
            "sample_count": n,
            "related_trials": trials,
            "calculated_at": calculated_at,
            "official_platform_check": False,
        }
    _, std, skewness, kurtosis = _moments(data)
    if std <= 0.0:
        return {
            "status": "unavailable",
            "reason": "zero_return_variance",
            "sample_count": n,
            "related_trials": trials,
            "calculated_at": calculated_at,
            "official_platform_check": False,
        }
    benchmark_daily = math.sqrt(max(0.0, 2.0 * math.log(trials)) / n) if trials > 1 else 0.0
    benchmark_annualized = benchmark_daily * math.sqrt(_TRADING_DAYS)
    psr = probabilistic_sharpe_ratio(
        annualized_sharpe,
        benchmark_annualized_sharpe=benchmark_annualized,
        sample_count=n,
        skewness=skewness,
        kurtosis=kurtosis,
    )
    if psr is None:
        return {
            "status": "unavailable",
            "reason": "invalid_psr_denominator",
            "sample_count": n,
            "related_trials": trials,
            "calculated_at": calculated_at,
            "official_platform_check": False,
        }
    return {
        "status": "available",
        "score": round(psr, 6),
        "psr": round(psr, 6),
        "benchmark_sharpe": round(benchmark_annualized, 6),
        "observed_sharpe": round(float(annualized_sharpe), 6),
        "sample_count": n,
        "related_trials": trials,
        "skewness": round(skewness, 6),
        "kurtosis": round(kurtosis, 6),
        "assumption": "expected_max_noise_sharpe_from_related_trials",
        "calculated_at": calculated_at,
        "official_platform_check": False,
    }


def pbo_eligibility(
    return_series_by_variant: Mapping[str, Sequence[float]],
    *,
    min_variants: int = 4,
    min_observations: int = 120,
    max_variants: int = 12,
) -> dict[str, Any]:
    """Describe whether bounded CSCV/PBO is appropriate for a coherent family."""
    usable = {
        str(key): _finite(values)
        for key, values in return_series_by_variant.items()
        if len(_finite(values)) >= min_observations
    }
    if len(usable) < min_variants:
        return {
            "eligible": False,
            "reason": "insufficient_comparable_variants",
            "variant_count": len(usable),
            "min_variants": min_variants,
            "min_observations": min_observations,
        }
    common_observations = min(len(values) for values in usable.values())
    if common_observations < min_observations:
        return {
            "eligible": False,
            "reason": "insufficient_common_history",
            "variant_count": len(usable),
            "common_observations": common_observations,
        }
    selected_keys = sorted(usable)[: max(1, max_variants)]
    return {
        "eligible": True,
        "reason": None,
        "variant_count": len(selected_keys),
        "common_observations": common_observations,
        "selected_variants": selected_keys,
        "bounded": len(usable) > len(selected_keys),
    }


def estimate_pbo_cscv(
    return_series_by_variant: Mapping[str, Sequence[float]],
    *,
    blocks: int = 4,
    max_combinations: int = 64,
    max_variants: int = 12,
) -> dict[str, Any]:
    """Estimate PBO with a bounded CSCV split set.

    The input must already represent one coherent variant/parameter family.
    Arbitrary unrelated alphas should never be mixed into this calculation.
    """
    eligibility = pbo_eligibility(return_series_by_variant, max_variants=max_variants)
    if not eligibility["eligible"]:
        return {"status": "skipped", **eligibility, "official_platform_check": False}
    keys = list(eligibility["selected_variants"])
    common_n = int(eligibility["common_observations"])
    block_count = max(4, int(blocks))
    if block_count % 2:
        block_count += 1
    block_size = common_n // block_count
    if block_size < 10:
        return {
            "status": "skipped",
            "eligible": False,
            "reason": "blocks_too_short",
            "block_count": block_count,
            "official_platform_check": False,
        }
    trimmed_n = block_size * block_count
    series = {key: _finite(return_series_by_variant[key])[-trimmed_n:] for key in keys}
    split_candidates = list(combinations(range(block_count), block_count // 2))
    split_candidates = split_candidates[: max(1, int(max_combinations))]
    overfit = 0
    evaluated = 0

    def sharpe(values: Sequence[float]) -> float:
        mean, std, _, _ = _moments(values)
        return mean / std if std > 0.0 else -math.inf

    for train_blocks in split_candidates:
        train_set = set(train_blocks)
        train_indices = [
            index
            for block in range(block_count)
            if block in train_set
            for index in range(block * block_size, (block + 1) * block_size)
        ]
        test_indices = [
            index
            for block in range(block_count)
            if block not in train_set
            for index in range(block * block_size, (block + 1) * block_size)
        ]
        train_scores = {
            key: sharpe([series[key][index] for index in train_indices])
            for key in keys
        }
        winner = max(keys, key=lambda key: (train_scores[key], key))
        test_scores = {
            key: sharpe([series[key][index] for index in test_indices])
            for key in keys
        }
        ranked = sorted(keys, key=lambda key: (test_scores[key], key))
        winner_rank = ranked.index(winner) + 1
        if winner_rank <= len(keys) / 2:
            overfit += 1
        evaluated += 1
    return {
        "status": "available",
        "eligible": True,
        "pbo": round(overfit / max(1, evaluated), 6),
        "evaluated_splits": evaluated,
        "max_combinations": max_combinations,
        "variant_count": len(keys),
        "sample_count": trimmed_n,
        "block_count": block_count,
        "assumption": "coherent_variant_family_required",
        "official_platform_check": False,
    }


def overfitting_priority_multiplier(evidence: Any) -> float:
    """Small soft ranking adjustment; unavailable evidence stays neutral."""
    if not isinstance(evidence, dict) or evidence.get("status") != "available":
        return 1.0
    try:
        score = max(0.0, min(1.0, float(evidence.get("score"))))
    except (TypeError, ValueError):
        return 1.0
    return 0.9 + 0.2 * score
