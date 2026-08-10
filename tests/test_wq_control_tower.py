from quantgpt.wq_control_tower import build_research_control_tower


def _memory():
    return {
        "learning_funnel": {
            "trial_to_candidate": {"count": 4, "denominator": 100, "rate": 0.04},
            "candidate_to_high_confidence": {"count": 2, "denominator": 4, "rate": 0.5},
            "candidate_to_submit": {"count": 2, "denominator": 4, "rate": 0.5},
            "submit_to_active": {"count": 1, "denominator": 2, "rate": 0.5},
            "active_to_points_settled": {"count": 1, "denominator": 1, "rate": 1.0},
        },
        "failure_stage_counts": {"simulation": 20, "robustness": 3},
        "failure_reason_counts": {"low_sharpe": 18, "sub_universe": 3},
        "candidate_funnel": {"dominant_bottleneck_stage": "simulation"},
        "metadata_completeness": {"family": {"present": 98, "missing": 2, "rate": 0.98}},
        "research_cells": [
            {
                "cell_key": "fundamental_quality|fundamental6|rank>ts_rank",
                "family": "fundamental_quality",
                "dataset_id": "fundamental6",
                "operator_pattern": "rank>ts_rank",
                "trials": 20,
                "candidates": 5,
                "active": 2,
                "terminal_failures": 1,
                "recent_failures": 0,
            },
            {
                "cell_key": "price_volume|pv1|rank>ts_delta",
                "family": "price_volume",
                "dataset_id": "pv1",
                "operator_pattern": "rank>ts_delta",
                "trials": 60,
                "candidates": 0,
                "active": 0,
                "terminal_failures": 3,
                "recent_failures": 10,
                "dominant_failure_reason": "low_sharpe",
            },
        ],
        "local_correlation_risk": {
            "available": 8,
            "high_risk": 2,
            "high_risk_threshold": 0.70,
            "official_platform_check": False,
        },
        "active_conversion": {
            "global": {"rate": 0.5, "samples": 10},
            "calibration": {"samples": 10, "reliable": False, "brier_score": 0.21},
        },
        "family_active_conversion": {"fundamental_quality": {"rate": 0.6, "samples": 5}},
        "dataset_active_conversion": {"fundamental6": {"rate": 0.6, "samples": 5}},
        "points_feedback_coverage": {
            "settled_attempts": 3,
            "usable_attempts": 2,
            "usable_family_groups": 1,
            "usable_dataset_groups": 1,
            "usable_operator_groups": 1,
            "mean_confidence": 0.8,
        },
        "points_feedback_gate": {"min_confidence": 0.6, "min_samples_per_group": 2},
        "family_points_feedback_usable": {"fundamental_quality": 1200.0},
        "dataset_points_feedback_usable": {"fundamental6": 1200.0},
        "operator_points_feedback_usable": {"rank>ts_rank": 1200.0},
        "overfitting_evidence": {
            "available": 4,
            "unavailable": 6,
            "mean_dsr_score": 0.72,
            "official_platform_check": False,
        },
    }


def test_control_tower_exposes_learning_loop_and_safeguards():
    snapshot = build_research_control_tower(
        _memory(),
        {
            "daily_submission_budget": 2,
            "remaining_submission_slots": 1,
            "submission_frozen": False,
            "inventory": {
                "floor": 30,
                "target_low": 40,
                "target_high": 50,
                "high_confidence_count": 12,
                "deficit": 18,
                "tier_counts": {"S": 2, "A": 10, "B": 5},
                "freshness_hours": 24.0,
                "mode": "REPLENISHMENT",
            },
        },
        research_budget=20,
    )

    assert snapshot["funnel"]["trial_to_candidate"]["rate"] == 0.04
    assert snapshot["failures"]["dominant_bottleneck_stage"] == "simulation"
    assert snapshot["metadata"]["completeness"]["family"]["rate"] == 0.98
    assert snapshot["inventory"]["floor"] == 30
    assert snapshot["inventory"]["target_low"] == 40
    assert snapshot["inventory"]["target_high"] == 50
    assert snapshot["inventory"]["tier_counts"]["A"] == 10
    assert snapshot["scheduler"]["inventory_mode"] == "REPLENISHMENT"
    assert snapshot["scheduler"]["next_focus"] is not None
    assert snapshot["points_feedback"]["coverage"]["usable_attempts"] == 2
    assert snapshot["calibration"]["samples"] == 10
    assert snapshot["overfitting"]["official_platform_check"] is False
    assert snapshot["correlation"]["official_platform_check"] is False
    assert snapshot["safeguards"]["daily_submission_budget"] == 2
    assert snapshot["safeguards"]["local_correlation_is_predictive_only"] is True
    assert snapshot["safeguards"]["fresh_high_local_correlation_blocks_submission"] is False
    assert snapshot["safeguards"]["weak_available_overfitting_evidence_blocks_submission"] is False
    assert snapshot["safeguards"]["local_robustness_failure_blocks_target_fill"] is False
    assert snapshot["safeguards"]["local_risk_signals_deprioritize_candidates"] is True
    assert snapshot["safeguards"]["official_brain_eligibility_blocks_submission"] is True
    assert snapshot["safeguards"]["official_sc_pending_blocks_first_submission"] is False
    assert snapshot["safeguards"]["research_gate_mode"] == "singleflight"
    assert snapshot["safeguards"]["async_mcp_tasks_required"] is True


def test_control_tower_preserves_inventory_defaults_and_sparse_evidence_neutrality():
    snapshot = build_research_control_tower({}, {"daily_submission_budget": 2}, research_budget=20)

    assert snapshot["inventory"] == {
        "floor": 30,
        "target_low": 40,
        "target_high": 50,
        "high_confidence_count": 0,
        "research_readiness_eligible_count": 0,
        "eligibility_mode": "research_readiness",
        "deficit": 0,
        "tier_counts": {},
        "freshness_hours": 24.0,
        "mode": "NORMAL",
    }
    assert snapshot["funnel"] == {}
    assert snapshot["correlation"] == {}
    assert snapshot["overfitting"] == {}
    assert snapshot["scheduler"]["selected_cells"] == []
    assert snapshot["safeguards"]["daily_submission_budget"] == 2
