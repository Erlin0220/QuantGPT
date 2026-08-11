from datetime import datetime, timedelta, timezone

from quantgpt.wq_research_scheduler import (
    allocate_research_cells,
    research_cell_key,
    summarize_research_cells,
)


def _cell(key: str, *, trials: int, candidates: int, reason: str | None = None, last_hours_ago: int = 1):
    family, dataset, pattern = key.split("|", 2)
    return {
        "cell_key": key,
        "family": family,
        "dataset": dataset,
        "operator_pattern": pattern,
        "trials": trials,
        "candidates": candidates,
        "high_confidence_candidates": 0,
        "formal_submissions": 0,
        "active": 0,
        "terminal_failures": 0,
        "recent_failure_reasons": {reason: trials} if reason else {},
        "last_sampled_at": (datetime.now(timezone.utc) - timedelta(hours=last_hours_ago)).isoformat(),
    }


def test_research_cell_key_has_stable_fallbacks():
    assert research_cell_key({"family": "price_volume", "dataset_id": "pv1", "operator_pattern": "rank(ts_mean(*,#))"}) == "price_volume|pv1|rank(ts_mean(*,#))"
    assert research_cell_key({}) == "unknown|unknown|unknown"


def test_cell_summary_tracks_research_candidate_and_formal_outcomes():
    trials = [
        {"family": "price_volume", "dataset_id": "pv1", "operator_pattern": "rank(*)", "status": "candidate", "failure_reason": None, "created_at": datetime.now(timezone.utc)},
        {"family": "price_volume", "dataset_id": "pv1", "operator_pattern": "rank(*)", "status": "rejected", "failure_reason": "low_fitness", "created_at": datetime.now(timezone.utc)},
    ]
    candidate = {"alpha_id": "a1", "family": "price_volume", "dataset_id": "pv1", "operator_pattern": "rank(*)", "validation_status": "ready", "robustness_score": None, "sharpe": 1.5, "fitness": 1.2, "turnover": 0.2}
    attempts = [({"alpha_id": "a1", "status": "ACTIVE"}, candidate)]
    rows = summarize_research_cells(trials, [candidate], attempts)
    assert len(rows) == 1
    row = rows[0]
    assert row["trials"] == 2
    assert row["candidates"] == 1
    assert row["high_confidence_candidates"] == 1
    assert row["formal_submissions"] == 1
    assert row["active"] == 1
    assert row["recent_failure_reasons"] == {"low_fitness": 1}


def test_sparse_cells_shrink_to_global_prior_instead_of_raw_rate():
    allocation = allocate_research_cells([
        _cell("a|d|p", trials=1, candidates=1),
        _cell("b|d|p", trials=20, candidates=4),
    ], budget=4)
    summaries = {row["cell_key"]: row for row in allocation["cell_summaries"]}
    assert summaries["a|d|p"]["posterior_yield"] < 1.0
    assert summaries["a|d|p"]["posterior_alpha"] == 3.0
    assert summaries["a|d|p"]["posterior_beta"] == 18.0


def test_allocator_exploits_higher_posterior_but_forces_exploration():
    allocation = allocate_research_cells([
        _cell("good|d|p", trials=20, candidates=6),
        _cell("weak|d|p", trials=20, candidates=0),
        _cell("novel|d|p", trials=0, candidates=0),
    ], budget=8, exploration_share=0.25)
    selected = {row["cell_key"]: row for row in allocation["selected_cells"]}
    assert selected["good|d|p"]["slots"] > selected.get("weak|d|p", {}).get("slots", 0)
    assert selected["novel|d|p"]["slots"] >= 1
    assert allocation["exploration_slots"] == 2


def test_poor_repeated_failure_cell_enters_cooldown_and_recovers_weight_over_time():
    recent = allocate_research_cells([_cell("bad|d|p", trials=12, candidates=0, reason="low_fitness", last_hours_ago=1)], budget=1)
    old = allocate_research_cells([_cell("bad|d|p", trials=12, candidates=0, reason="low_fitness", last_hours_ago=48)], budget=1)
    recent_cell = recent["cell_summaries"][0]
    old_cell = old["cell_summaries"][0]
    assert recent_cell["cooldown"] is True
    assert recent_cell["cooldown_penalty"] == 0.35
    assert old_cell["cooldown_penalty"] == 0.7


def test_inventory_mode_changes_exploration_share_without_stopping_research():
    cells = [_cell("a|d|p", trials=5, candidates=1), _cell("b|d|p", trials=1, candidates=0)]
    deficit = allocate_research_cells(cells, budget=10, inventory_mode="REPLENISHMENT")
    healthy = allocate_research_cells(cells, budget=10, inventory_mode="EXPLORATION")
    assert deficit["exploration_share"] == 0.2
    assert healthy["exploration_share"] >= 0.4
    assert sum(row["slots"] for row in deficit["selected_cells"]) == 10
    assert sum(row["slots"] for row in healthy["selected_cells"]) == 10


def test_exploitation_slots_are_not_round_robin_equal_when_posterior_differs():
    allocation = allocate_research_cells([
        _cell("good|d|p", trials=30, candidates=9),
        _cell("weak|d|p", trials=30, candidates=0),
    ], budget=10, exploration_share=0.2)
    slots = {row["cell_key"]: row["slots"] for row in allocation["selected_cells"]}
    assert slots["good|d|p"] > slots.get("weak|d|p", 0)


def test_allocator_is_deterministic_for_same_input():
    cells = [_cell("a|d|p", trials=3, candidates=1), _cell("b|d|p", trials=3, candidates=0)]
    first = allocate_research_cells(cells, budget=7)
    second = allocate_research_cells(cells, budget=7)
    assert [(row["cell_key"], row["slots"]) for row in first["selected_cells"]] == [(row["cell_key"], row["slots"]) for row in second["selected_cells"]]
