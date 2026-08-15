from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from quantgpt import wq_research_cycle_runner as runner


def _cycle(now: datetime | None = None):
    return runner.new_research_cycle_state(
        account="primary",
        cycle_id="20260815-1600",
        target_simulations=100,
        budget_minutes=50,
        now=now,
    )


def _skill_candidate():
    return {
        "expression": "rank(ts_mean(returns, 20))",
        "hypothesis": "recent return persistence should rank future returns",
        "family": "momentum_reversal",
        "data_fields": ["returns"],
        "knowledge_card_ids": [],
        "skill_chain": [
            "wq-economic-hypothesis",
            "wq-alpha-hypothesis",
            "wq-alpha-review",
            "wq-robustness-validation",
            "wq-candidate-evidence",
        ],
        "review_decision": "RUN",
        "robustness_plan": {
            "mode": "skill_defined",
            "checks": [{"universe": "TOP1000", "purpose": "test universe sensitivity"}],
        },
        "candidate_evidence_policy": {"mode": "calibrated_evidence_hierarchy"},
    }


def test_cycle_progress_enforces_target_before_budget():
    started = datetime(2026, 8, 15, 8, 0, tzinfo=timezone.utc)
    state = _cycle(started)
    state["counters"]["simulations"] = 100

    progress = runner.research_cycle_progress(state, now=started + timedelta(minutes=60))

    assert progress["should_stop"] is True
    assert progress["stop_reason"] == "target_reached"
    assert progress["remaining_simulations"] == 0


def test_cycle_progress_expires_at_budget_without_target():
    started = datetime(2026, 8, 15, 8, 0, tzinfo=timezone.utc)
    state = _cycle(started)
    state["counters"]["simulations"] = 37

    progress = runner.research_cycle_progress(state, now=started + timedelta(minutes=50))

    assert progress["should_stop"] is True
    assert progress["stop_reason"] == "budget_exhausted"
    assert progress["remaining_simulations"] == 63


def test_apply_batch_accumulates_primary_and_total_brain_simulations():
    started = datetime(2026, 8, 15, 8, 0, tzinfo=timezone.utc)
    state = _cycle(started)

    state = runner.apply_research_batch_to_cycle(
        state,
        {
            "tag": "batch-1",
            "summary": {"simulated": 8, "total_simulations": 10},
            "candidates": [],
            "failed": [{"expression": "a"}],
            "invalid": [],
        },
        source_run_id="run-1",
        started_at=started,
        ended_at=started + timedelta(minutes=8),
    )
    state = runner.apply_research_batch_to_cycle(
        state,
        {
            "tag": "batch-2",
            "summary": {"simulated": 9, "total_simulations": 9},
            "candidates": [{"alpha_id": "alpha-1"}],
            "failed": [],
            "invalid": [],
        },
        source_run_id="run-2",
        started_at=started + timedelta(minutes=9),
        ended_at=started + timedelta(minutes=18),
    )

    assert state["counters"]["iterations"] == 2
    assert state["counters"]["simulations"] == 17
    assert state["counters"]["total_brain_simulations"] == 19
    assert state["counters"]["candidates"] == 1
    assert state["next_action"] == "advance_candidate_to_submission_gate"
    assert [item["source_run_id"] for item in state["batches"]] == ["run-1", "run-2"]


def test_cycle_decision_prioritizes_submission_without_stopping_cycle():
    state = _cycle()
    policy = {
        "remaining_submission_slots": 2,
        "submission_candidate_top": [{"alpha_id": "ready"}],
        "submission_frozen": False,
        "submission_reconciliation_required": False,
    }

    assert runner._cycle_decision(state, policy) == "SUBMISSION_REQUIRED"
    assert runner.research_cycle_progress(state)["should_stop"] is False


def test_runner_contract_requires_economic_hypothesis_skill():
    candidate = _skill_candidate()
    candidate["skill_chain"] = [name for name in candidate["skill_chain"] if name != "wq-economic-hypothesis"]

    assert "wq-economic-hypothesis" in runner._runner_candidate_contract_error(candidate)


def test_cycle_decision_continues_inventory_after_daily_active_target():
    state = _cycle()
    policy = {
        "remaining_active_target": 0,
        "remaining_submission_slots": 0,
        "submission_candidate_top": [],
        "submission_frozen": False,
        "submission_reconciliation_required": False,
    }

    assert runner._cycle_decision(state, policy) == "NEEDS_SKILL_BATCH"


def test_run_skill_batch_uses_direct_service_and_persists_cycle(monkeypatch):
    state = _cycle()
    persisted = {}
    client = Mock()
    client.authenticate.return_value = True
    client.last_error = None

    async def policy(_account="primary"):
        return {
            "remaining_active_target": 2,
            "remaining_submission_slots": 2,
            "submission_candidate_top": [],
            "submission_frozen": False,
            "submission_reconciliation_required": False,
            "inventory": {},
        }

    async def ensure_session(_account, _policy):
        return {"session_id": "active-session"}

    async def record_transition(_session_id, _result, _params):
        return {"session_id": "active-session", "status": "RUNNING"}

    monkeypatch.setattr(runner, "_runner_process_lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(runner, "_active_mcp_research_task", lambda: None)
    monkeypatch.setattr(runner, "ensure_research_cycle_sync", lambda *args, **kwargs: state)
    monkeypatch.setattr(runner, "get_submission_policy_status", policy)
    monkeypatch.setattr(runner, "is_configured", lambda _account: True)
    monkeypatch.setattr(runner, "get_client", lambda *_args, **_kwargs: client)
    monkeypatch.setattr(runner, "ensure_active_first_session", ensure_session)
    monkeypatch.setattr(runner, "record_research_transition", record_transition)
    monkeypatch.setattr(runner, "load_research_memory_sync", lambda _account: {})
    monkeypatch.setattr(runner, "build_research_control_tower", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "record_research_trials_sync", lambda *_args, **_kwargs: 8)
    monkeypatch.setattr(runner, "record_research_candidates_sync", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        runner,
        "run_autonomous_research",
        lambda *_args, **kwargs: {
            "ok": True,
            "tag": kwargs["tag"],
            "summary": {"simulated": 8, "total_simulations": 8},
            "results": [],
            "candidates": [],
            "failed": [],
            "invalid": [],
        },
    )

    def update_cycle(_cycle_id, _account, updated):
        persisted.update(updated)
        return {"task_id": "cycle-task", **updated}

    monkeypatch.setattr(runner, "update_research_cycle_sync", update_cycle)
    monkeypatch.setattr(runner, "get_research_cycle_sync", lambda *_args, **_kwargs: persisted or state)
    monkeypatch.setattr(
        runner,
        "research_cycle_snapshot",
        lambda **_kwargs: {
            "ok": True,
            "status": "NEEDS_SKILL_BATCH",
            "cycle_id": "20260815-1600",
            "research_cycle": persisted,
        },
    )

    result = runner.run_skill_batch(
        account="primary",
        cycle_id="20260815-1600",
        skill_candidates=[_skill_candidate()],
        max_simulations=8,
    )

    assert result["ok"] is True
    assert result["batch_executed"] is True
    assert persisted["counters"]["simulations"] == 8
    assert persisted["counters"]["iterations"] == 1
    client.authenticate.assert_called_once_with()
    client.close.assert_called_once_with()


def test_run_skill_batch_refuses_when_mcp_research_is_active(monkeypatch):
    monkeypatch.setattr(runner, "_runner_process_lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        runner,
        "_active_mcp_research_task",
        lambda: {"task_id": "mcp-active", "task_type": "wq_research", "status": "researching"},
    )

    result = runner.run_skill_batch(
        account="primary",
        cycle_id="20260815-1600",
        skill_candidates=[_skill_candidate()],
    )

    assert result["ok"] is False
    assert result["status"] == "research_singleflight_busy"
    assert result["active_task"]["task_id"] == "mcp-active"
    assert result["batch_executed"] is False
