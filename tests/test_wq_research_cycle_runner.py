from __future__ import annotations

import json
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


def test_completed_budget_cycle_elapsed_time_stays_frozen_at_deadline():
    started = datetime(2026, 8, 15, 8, 0, tzinfo=timezone.utc)
    state = _cycle(started)
    state["status"] = "COMPLETED"
    state["stop_reason"] = "budget_exhausted"
    state["counters"]["simulations"] = 21

    progress = runner.research_cycle_progress(state, now=started + timedelta(hours=4))

    assert progress["elapsed_minutes"] == 50.0
    assert progress["remaining_minutes"] == 0.0
    assert progress["stop_reason"] == "budget_exhausted"


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
    assert state["batch_inflight"] is False
    assert state["batch_started_at"] is None


def test_recover_stale_batch_inflight_when_runner_owner_is_dead(monkeypatch):
    state = runner.set_research_cycle_inflight(
        _cycle(),
        True,
        now=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    persisted = {}
    monkeypatch.setattr(runner, "_runner_lock_owner_status", lambda _path: False)
    monkeypatch.setattr(runner, "_runner_lock_path", lambda _account: Mock())

    def update_cycle(_cycle_id, _account, updated):
        persisted.update(updated)
        return {"task_id": "cycle-task", **updated}

    monkeypatch.setattr(runner, "update_research_cycle_sync", update_cycle)

    recovered = runner._recover_stale_cycle_inflight_sync("primary", state)

    assert recovered["batch_inflight"] is False
    assert recovered["last_error"] == "stale_batch_inflight_recovered_after_runner_exit"
    assert recovered["next_action"] == "diagnose_execution_then_generate_replacement_batch"
    assert persisted["batch_inflight"] is False


def test_reconcile_cycle_recovers_primary_progress_from_matching_active_session(monkeypatch):
    state = _cycle()
    state["counters"].update({"simulations": 17, "iterations": 3, "candidates": 0, "total_brain_simulations": 17})
    state["deadline_at"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    persisted = {}

    def update_cycle(_cycle_id, _account, updated):
        persisted.update(updated)
        return {"task_id": "cycle-task", **updated}

    monkeypatch.setattr(runner, "update_research_cycle_sync", update_cycle)
    active_session = {
        "research_cycle": {
            "cycle_id": state["cycle_id"],
            "simulations": 21,
            "iterations": 4,
            "candidates": 1,
        }
    }

    recovered = runner._reconcile_cycle_from_active_session_sync("primary", state, active_session)

    assert recovered["counters"]["simulations"] == 21
    assert recovered["counters"]["iterations"] == 4
    assert recovered["counters"]["candidates"] == 1
    assert recovered["counters"]["total_brain_simulations"] == 21
    assert recovered["stop_reason"] == "budget_exhausted"
    assert recovered["next_action"] == "cycle_complete"
    assert recovered["recovery_events"][-1]["simulations_before"] == 17
    assert recovered["recovery_events"][-1]["simulations_after"] == 21
    assert persisted["counters"]["simulations"] == 21


def test_reconcile_cycle_ignores_other_active_session_cycle(monkeypatch):
    state = _cycle()
    monkeypatch.setattr(
        runner,
        "update_research_cycle_sync",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not persist unrelated cycle")),
    )

    recovered = runner._reconcile_cycle_from_active_session_sync(
        "primary",
        state,
        {"research_cycle": {"cycle_id": "other-cycle", "simulations": 99, "iterations": 9}},
    )

    assert recovered is state


def test_live_runner_owner_keeps_batch_inflight(monkeypatch):
    state = runner.set_research_cycle_inflight(_cycle(), True)
    monkeypatch.setattr(runner, "_runner_lock_owner_status", lambda _path: True)
    monkeypatch.setattr(runner, "_runner_lock_path", lambda _account: Mock())

    recovered = runner._recover_stale_cycle_inflight_sync("primary", state)

    assert recovered is state
    assert recovered["batch_inflight"] is True


def test_cycle_decision_prioritizes_submission_without_stopping_cycle():
    state = _cycle()
    policy = {
        "remaining_submission_slots": 2,
        "submission_candidate_top": [{"alpha_id": "ready"}],
        "submission_frozen": False,
        "submission_reconciliation_required": False,
    }

    assert runner._cycle_decision(state, policy) == "SUBMISSION_REQUIRED"
    assert runner._effective_cycle_decision(state, policy) == "SUBMISSION_REQUIRED"
    assert runner._effective_cycle_decision(state, policy, submission_deferred=True) == "NEEDS_SKILL_BATCH"
    assert runner.research_cycle_progress(state)["should_stop"] is False


def test_submission_deferred_does_not_override_cycle_stop_gate():
    started = datetime(2026, 8, 15, 8, 0, tzinfo=timezone.utc)
    state = _cycle(started)
    state["counters"]["simulations"] = 100
    policy = {
        "remaining_submission_slots": 2,
        "submission_candidate_top": [{"alpha_id": "ready"}],
        "submission_frozen": False,
        "submission_reconciliation_required": False,
    }

    assert runner._effective_cycle_decision(state, policy, submission_deferred=True, now=started) == "TARGET_REACHED"


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
            "submission_candidate_top": [{"alpha_id": "deferred-ready"}],
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
    research_kwargs = {}

    def run_research(*_args, **kwargs):
        research_kwargs.update(kwargs)
        return {
            "ok": True,
            "tag": kwargs["tag"],
            "summary": {"simulated": 8, "total_simulations": 8},
            "results": [],
            "candidates": [],
            "failed": [],
            "invalid": [],
        }

    monkeypatch.setattr(runner, "run_autonomous_research", run_research)

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
        submission_deferred=True,
    )

    assert result["ok"] is True
    assert result["batch_executed"] is True
    assert persisted["counters"]["simulations"] == 8
    assert persisted["counters"]["iterations"] == 1
    client.authenticate.assert_called_once_with()
    client.close.assert_called_once_with()
    assert callable(research_kwargs["check_cancelled"])
    assert research_kwargs["check_cancelled"]() is False


def test_enqueue_skill_batch_reserves_cycle_and_spawns_background_worker(monkeypatch, tmp_path):
    state = _cycle()
    persisted = {}
    batch_file = tmp_path / "batch.json"
    batch_file.write_text(json.dumps({"skill_candidates": [_skill_candidate()]}), encoding="utf-8")

    async def policy(_account="primary"):
        return {
            "remaining_active_target": 1,
            "remaining_submission_slots": 1,
            "submission_candidate_top": [],
            "submission_frozen": False,
            "submission_reconciliation_required": False,
        }

    monkeypatch.setattr(runner, "_runner_process_lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(runner, "_runner_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(runner, "_active_mcp_research_task", lambda: None)
    monkeypatch.setattr(runner, "ensure_research_cycle_sync", lambda *args, **kwargs: state)
    monkeypatch.setattr(runner, "_recover_stale_cycle_inflight_sync", lambda _account, cycle: cycle)
    monkeypatch.setattr(runner, "get_submission_policy_status", policy)
    monkeypatch.setattr(runner, "_background_python_executable", lambda: "python")

    def update_cycle(_cycle_id, _account, updated):
        persisted.clear()
        persisted.update(updated)
        return {"task_id": "cycle-task", **updated}

    monkeypatch.setattr(runner, "update_research_cycle_sync", update_cycle)
    process = Mock(pid=4321)
    popen = Mock(return_value=process)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)

    result = runner.enqueue_skill_batch(
        account="primary",
        cycle_id="20260815-1600",
        batch_file=str(batch_file),
        max_simulations=1,
    )

    assert result["ok"] is True
    assert result["status"] == "BATCH_INFLIGHT"
    assert result["worker_pid"] == 4321
    assert persisted["batch_inflight"] is True
    assert persisted["next_action"] == "wait_for_batch_completion"
    command = popen.call_args.args[0]
    assert "execute-batch" in command
    assert str(batch_file.resolve()) in command


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
