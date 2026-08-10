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


def test_scheduler_uses_coverage_first_while_learning_is_cold(monkeypatch):
    monkeypatch.setenv("WQ_ACTIVE_DECISION_MIN_SAMPLES", "20")
    maturity = build_learning_maturity(
        trials=60,
        provenance_resolved=55,
        active_feedback={"global": {"samples": 7, "rate": 0.2}},
        points_coverage={"settled_attempts": 0, "usable_attempts": 0},
    )
    cells = [
        {"cell_key": "a|d1|p1", "family": "a", "dataset": "d1", "operator_pattern": "p1", "trials": 40, "candidates": 5},
        {"cell_key": "b|d2|p2", "family": "b", "dataset": "d2", "operator_pattern": "p2", "trials": 2, "candidates": 0},
    ]

    allocation = allocate_research_cells(cells, budget=6, inventory_mode="REPLENISHMENT", learning_maturity=maturity)

    assert allocation["policy"] == "coverage_first"
    assert allocation["exploration_slots"] == 6
    assert allocation["exploitation_slots"] == 0
    assert allocation["cooldown_enabled"] is False
    assert all(item["rationale"] == "cold_start_coverage_first" for item in allocation["selected_cells"])
    assert {item["cell_key"] for item in allocation["selected_cells"]} == {"a|d1|p1", "b|d2|p2"}


def test_scheduler_can_return_to_adaptive_after_all_gates_are_ready(monkeypatch):
    monkeypatch.setenv("WQ_ACTIVE_DECISION_MIN_SAMPLES", "20")
    monkeypatch.setenv("WQ_SCHEDULER_ADAPTIVE_MIN_TRIALS", "50")
    monkeypatch.setenv("WQ_SCHEDULER_MIN_PROVENANCE_RATE", "0.80")
    maturity = build_learning_maturity(
        trials=100,
        provenance_resolved=90,
        active_feedback={"global": {"samples": 25, "rate": 0.4}},
        points_coverage={"settled_attempts": 3, "usable_attempts": 2},
    )
    cells = [
        {"cell_key": "a|d1|p1", "family": "a", "dataset": "d1", "operator_pattern": "p1", "trials": 30, "candidates": 8},
        {"cell_key": "b|d2|p2", "family": "b", "dataset": "d2", "operator_pattern": "p2", "trials": 30, "candidates": 0},
    ]

    allocation = allocate_research_cells(cells, budget=10, learning_maturity=maturity)

    assert maturity["scheduler_adaptation"]["ready"] is True
    assert allocation["policy"] == "adaptive"
    assert allocation["cooldown_enabled"] is True
    assert allocation["exploitation_slots"] > 0
