"""Machine-readable WorldQuant research learning/control-tower summary."""
from __future__ import annotations

from typing import Any

from .wq_research_scheduler import allocate_research_cells


def _inventory_view(policy: dict[str, Any]) -> dict[str, Any]:
    inventory = policy.get("inventory") or {}
    return {
        "floor": inventory.get("floor", 30),
        "target_low": inventory.get("target_low", 40),
        "target_high": inventory.get("target_high", 50),
        "high_confidence_count": inventory.get("high_confidence_count", policy.get("high_confidence_candidate_count", 0)),
        "research_readiness_eligible_count": inventory.get("research_readiness_eligible_count", policy.get("research_readiness_eligible_count", 0)),
        "eligibility_mode": inventory.get("eligibility_mode", policy.get("candidate_eligibility_mode", "research_readiness")),
        "deficit": inventory.get("deficit", policy.get("inventory_deficit", 0)),
        "tier_counts": inventory.get("tier_counts") or policy.get("candidate_confidence_counts") or {},
        "freshness_hours": inventory.get("freshness_hours", 24.0),
        "mode": str(inventory.get("mode") or policy.get("research_mode") or "NORMAL").upper(),
    }


def build_research_control_tower(
    memory: dict[str, Any],
    submission_policy: dict[str, Any] | None = None,
    *,
    research_budget: int = 20,
) -> dict[str, Any]:
    """Build one structured snapshot for status output and autonomous planning.

    All local correlation/overfitting fields are explicitly predictive research
    evidence, never aliases for official WorldQuant checks.
    """
    policy = submission_policy or {}
    inventory = _inventory_view(policy)
    allocation = allocate_research_cells(
        memory.get("research_cells") or [],
        budget=max(1, int(research_budget)),
        inventory_mode=inventory["mode"],
        learning_maturity=memory.get("learning_maturity") or {},
    )
    selected = list(allocation.get("selected_cells") or [])

    return {
        "funnel": memory.get("learning_funnel") or {},
        "failures": {
            "stage_counts": memory.get("failure_stage_counts") or {},
            "reason_counts": memory.get("failure_reason_counts") or {},
            "dominant_bottleneck_stage": (memory.get("candidate_funnel") or {}).get("dominant_bottleneck_stage"),
        },
        "metadata": {
            "completeness": memory.get("metadata_completeness") or {},
            "provenance": memory.get("provenance") or {},
        },
        "credit_assignment": memory.get("credit_assignment") or {},
        "round_audits": list(memory.get("research_round_audits") or [])[:8],
        "learning_maturity": memory.get("learning_maturity") or {},
        "research_memory": memory.get("research_memory_guidance") or {},
        "scheduler": {
            "inventory_mode": inventory["mode"],
            "policy": allocation.get("policy"),
            "learning_status": allocation.get("learning_status"),
            "cooldown_enabled": allocation.get("cooldown_enabled"),
            "exploration_share": allocation.get("exploration_share"),
            "exploration_slots": allocation.get("exploration_slots"),
            "exploitation_slots": allocation.get("exploitation_slots"),
            "selected_cells": selected,
            "next_focus": selected[0] if selected else None,
            "cooldown_cells": [item for item in allocation.get("cell_summaries") or [] if item.get("cooldown")],
            "rationale": "coverage_first_until_learning_maturity" if allocation.get("policy") == "coverage_first" else "smoothed_adaptive_allocation",
        },
        "inventory": inventory,
        "correlation": memory.get("local_correlation_risk") or {},
        "calibration": (memory.get("active_conversion") or {}).get("calibration") or {},
        "active_conversion": {
            "global": (memory.get("active_conversion") or {}).get("global") or {},
            "family": memory.get("family_active_conversion") or {},
            "dataset": memory.get("dataset_active_conversion") or {},
        },
        "points_feedback": {
            "coverage": memory.get("points_feedback_coverage") or {},
            "gate": memory.get("points_feedback_gate") or {},
            "family_usable": memory.get("family_points_feedback_usable") or {},
            "dataset_usable": memory.get("dataset_points_feedback_usable") or {},
            "operator_usable": memory.get("operator_points_feedback_usable") or {},
        },
        "overfitting": memory.get("overfitting_evidence") or {},
        "safeguards": {
            "daily_submission_budget": policy.get("daily_submission_budget"),
            "remaining_submission_slots": policy.get("remaining_submission_slots"),
            "submission_frozen": policy.get("submission_frozen"),
            "local_correlation_is_predictive_only": True,
            "fresh_high_local_correlation_blocks_submission": False,
            "weak_available_overfitting_evidence_blocks_submission": False,
            "local_robustness_failure_blocks_target_fill": False,
            "local_risk_signals_deprioritize_candidates": True,
            "official_brain_eligibility_blocks_submission": True,
            "official_sc_pending_blocks_first_submission": False,
            "research_gate_mode": "singleflight",
            "async_mcp_tasks_required": True,
        },
    }
