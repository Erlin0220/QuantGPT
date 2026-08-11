"""Tests for multiple-testing-aware WQ research evidence."""

from quantgpt.wq_candidate_calibration import quality_baseline
from quantgpt.wq_overfitting import (
    deflated_sharpe_evidence,
    estimate_pbo_cscv,
    overfitting_priority_multiplier,
    pbo_eligibility,
)


def _returns(n: int = 252) -> list[float]:
    return [0.001 + (0.004 if index % 3 == 0 else -0.0015) for index in range(n)]


def test_deflated_sharpe_penalty_grows_with_related_trial_count():
    single = deflated_sharpe_evidence(_returns(), annualized_sharpe=1.8, related_trials=1)
    mined = deflated_sharpe_evidence(_returns(), annualized_sharpe=1.8, related_trials=50)

    assert single["status"] == "available"
    assert mined["status"] == "available"
    assert mined["benchmark_sharpe"] > single["benchmark_sharpe"]
    assert mined["score"] < single["score"]


def test_missing_overfitting_evidence_is_neutral_and_does_not_starve_inventory():
    unavailable = deflated_sharpe_evidence([0.01] * 10, annualized_sharpe=2.0, related_trials=100)

    assert unavailable["status"] == "unavailable"
    assert overfitting_priority_multiplier(unavailable) == 1.0
    assert overfitting_priority_multiplier(None) == 1.0


def test_available_dsr_is_preserved_as_named_evidence_not_folded_into_magic_score():
    base = {
        "expression": "rank(close)",
        "is_metrics": {"sharpe": 1.8, "fitness": 1.3, "returns": 0.1, "turnover": 0.2},
        "validation": {"status": "ready", "robustness_score": 0.8},
        "novelty_score": 0.8,
    }
    neutral = quality_baseline(base)
    strong = quality_baseline({
        **base,
        "validation": {
            **base["validation"],
            "overfitting_evidence": {"status": "available", "score": 1.0},
        },
    })
    weak = quality_baseline({
        **base,
        "validation": {
            **base["validation"],
            "overfitting_evidence": {"status": "available", "score": 0.0},
        },
    })

    assert neutral["score"] is None
    assert strong["score"] is None
    assert weak["score"] is None
    assert strong["components"]["overfitting_evidence"]["score"] == 1.0
    assert weak["components"]["overfitting_evidence"]["score"] == 0.0


def test_pbo_eligibility_requires_comparable_history_and_bounds_variants():
    too_small = pbo_eligibility({"a": _returns(), "b": _returns(), "c": _returns()})
    assert too_small["eligible"] is False

    many = {f"variant-{index:02d}": _returns() for index in range(20)}
    eligible = pbo_eligibility(many, max_variants=8)
    assert eligible["eligible"] is True
    assert eligible["variant_count"] == 8
    assert eligible["bounded"] is True


def test_cscv_pbo_is_bounded_and_marks_local_diagnostic_only():
    variants = {
        f"variant-{index}": [
            0.0005 * (index + 1) + (0.003 if day % (index + 2) == 0 else -0.001)
            for day in range(160)
        ]
        for index in range(6)
    }
    result = estimate_pbo_cscv(variants, blocks=4, max_combinations=3)

    assert result["status"] == "available"
    assert 0.0 <= result["pbo"] <= 1.0
    assert result["evaluated_splits"] <= 3
    assert result["official_platform_check"] is False
    assert result["assumption"] == "coherent_variant_family_required"
