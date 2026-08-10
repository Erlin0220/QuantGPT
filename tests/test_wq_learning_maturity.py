from quantgpt.wq_learning_maturity import (
    active_feedback_gate,
    build_learning_maturity,
    points_feedback_gate,
)
from quantgpt.wq_research_scheduler import allocate_research_cells


def test_sparse_active_outcomes_are_observable_but_not_decision_ready(monkeypatch):
    monkeypatch.setenv("WQ_ACTIVE_DECISION_MIN_SAMPLES", "20")
    gate = active_feedback_gate({"global": {"samples": 7, "rate": 0.2}})

    assert gate["status"] == "cold_start"
    assert gate["decision_weight_enabled"] is False
    assert gate["reason"] == "insufficient_terminal_active_outcomes"


def test_zero_settled_points_have_zero_planner_eligibility():
    gate = points_feedback_gate({"settled_attempts": 0, "usable_attempts": 0})

    assert gate["status"] == "cold_start"
    assert gate["decision_weight_enabled"] is False
    assert gate["reason"] == "no_settled_points"


def test_scheduler_adapts_from_research_evidence_while_active_learning_is_cold(monkeypatch):
    monkeypatch.setenv("WQ_ACTIVE_DECISION_MIN_SAMPLES", "20")
    maturity = build_learning_maturity(
        trials=60,
        provenance_resolved=55,
        active_feedback={"global": {"samples": 7, "rate": 0.2}},
        points_coverage={"settled_attempts": 0, "usable_attempts": 0},
    )
    cells = [
        {"cell_key": "a|d1|p1", "family": "a", "dataset": "d1", "operator_pattern": "p1", "trials": 40, "candidates": 5},
        {"cell_key": "b|d2|p2", "family": "b", "dataset": "d2", "operator_pattern": "p2", "trials": 12, "candidates": 0, "recent_failure_reasons": {"low_fitness": 12}},
    ]

    allocation = allocate_research_cells(cells, budget=6, inventory_mode="REPLENISHMENT", learning_maturity=maturity)

    assert maturity["active_outcome_weighting"]["ready"] is False
    assert maturity["scheduler_adaptation"]["ready"] is True
    assert maturity["scheduler_adaptation"]["active_outcome_dependency"] == "independent"
    assert allocation["policy"] == "adaptive"
    assert allocation["cooldown_enabled"] is True
    weak = next(item for item in allocation["cell_summaries"] if item["cell_key"] == "b|d2|p2")
    assert weak["cooldown"] is True
    assert allocation["exploitation_slots"] > 0


def test_scheduler_keeps_coverage_first_until_local_research_is_mature(monkeypatch):
    monkeypatch.setenv("WQ_SCHEDULER_ADAPTIVE_MIN_TRIALS", "50")
    maturity = build_learning_maturity(
        trials=20,
        provenance_resolved=20,
        active_feedback={"global": {"samples": 25, "rate": 0.4}},
        points_coverage={"settled_attempts": 3, "usable_attempts": 2},
    )
    cells = [
        {"cell_key": "a|d1|p1", "family": "a", "dataset": "d1", "operator_pattern": "p1", "trials": 10, "candidates": 1},
        {"cell_key": "b|d2|p2", "family": "b", "dataset": "d2", "operator_pattern": "p2", "trials": 10, "candidates": 0},
    ]

    allocation = allocate_research_cells(cells, budget=4, learning_maturity=maturity)

    assert maturity["scheduler_adaptation"]["ready"] is False
    assert "insufficient_research_trials" in maturity["scheduler_adaptation"]["reasons"]
    assert allocation["policy"] == "coverage_first"
    assert allocation["cooldown_enabled"] is False


def test_scheduler_can_return_to_adaptive_after_all_gates_are_ready(monkeypatch):
    monkeypatch.setenv("WQ_ACTIVE_DECISION_MIN_SAMPLES", "20")
    monkeypatch.setenv("WQ_SCHEDULER_ADAPTIVE_MIN_TRIALS", "50")
    monkeypatch.setenv("WQ_SCHEDULER_MIN_PROVENANCE_RATE", "0.80")
    maturity = build_learning_maturity(
        trials=100,
        provenance_resolved=60,
        provenance_classified=90,
        active_feedback={"global": {"samples": 25, "rate": 0.4}},
        points_coverage={"settled_attempts": 3, "usable_attempts": 2},
    )
    cells = [
        {"cell_key": "a|d1|p1", "family": "a", "dataset": "d1", "operator_pattern": "p1", "trials": 30, "candidates": 8},
        {"cell_key": "b|d2|p2", "family": "b", "dataset": "d2", "operator_pattern": "p2", "trials": 30, "candidates": 0},
    ]

    allocation = allocate_research_cells(cells, budget=10, learning_maturity=maturity)

    assert maturity["scheduler_adaptation"]["ready"] is True
    assert maturity["scheduler_adaptation"]["provenance_resolved"] == 60
    assert maturity["scheduler_adaptation"]["provenance_classified"] == 90
    assert maturity["scheduler_adaptation"]["provenance_rate"] == 0.9
    assert maturity["scheduler_adaptation"]["provenance_rate_basis"] == "classified_resolved_or_partial"
    assert allocation["policy"] == "adaptive"
    assert allocation["cooldown_enabled"] is True
    assert allocation["exploitation_slots"] > 0
