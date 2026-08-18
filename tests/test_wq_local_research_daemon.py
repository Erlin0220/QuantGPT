from __future__ import annotations

from quantgpt import wq_local_research_daemon as daemon


def test_classify_completed_cycle_starts_next_cycle_quickly():
    decision = daemon.classify_daemon_result(
        {"ok": True, "status": "TARGET_REACHED", "progress": {"should_stop": True, "stop_reason": "target_reached"}}
    )
    assert decision.completed_cycle is True
    assert decision.delay_seconds == 2.0


def test_classify_singleflight_uses_short_backoff():
    decision = daemon.classify_daemon_result(
        {"ok": True, "status": "research_singleflight_busy", "progress": {"should_stop": False}}
    )
    assert decision.completed_cycle is False
    assert decision.delay_seconds == 15.0


def test_classify_submission_attempt_rechecks_quickly():
    decision = daemon.classify_daemon_result(
        {"ok": True, "status": "SUBMISSION_ATTEMPTED", "progress": {"should_stop": False}}
    )
    assert decision.completed_cycle is False
    assert decision.delay_seconds == 2.0
    assert decision.reason == "SUBMISSION_ATTEMPTED"


def test_classify_generation_failure_retries_quickly():
    decision = daemon.classify_daemon_result(
        {"ok": True, "status": "local_llm_generation_failed", "progress": {"should_stop": False}}
    )
    assert decision.delay_seconds == 3.0


def test_classify_auth_failure_uses_long_backoff():
    decision = daemon.classify_daemon_result(
        {"ok": False, "status": "brain_authentication_failed", "progress": {"should_stop": False}}
    )
    assert decision.delay_seconds == 300.0


def test_daemon_consumes_submission_gate_before_continuing_research(monkeypatch):
    calls: list[dict] = []
    submission_calls: list[dict] = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "ok": True,
                "status": "SUBMISSION_REQUIRED",
                "cycle_id": "cycle-1",
                "progress": {"should_stop": False},
                "planner_context": {"submission_candidates": [{"alpha_id": "ready-alpha"}]},
                "research_cycle": {},
                "local_llm_rounds_this_run": [],
            }
        return {
            "ok": True,
            "status": "TARGET_REACHED",
            "cycle_id": "cycle-1",
            "model": "opencode-go/deepseek-v4-flash",
            "reasoning_policy": kwargs["reasoning_policy"],
            "progress": {"should_stop": True, "stop_reason": "target_reached"},
            "research_cycle": {},
            "local_llm_rounds_this_run": [],
        }

    def fake_submit_gate(*, account, trigger):
        submission_calls.append({"account": account, "trigger": trigger})
        return {"ok": True, "status": "SUBMISSION_ATTEMPTED", "progress": {"should_stop": False}}

    monkeypatch.setattr(daemon, "run_local_llm_research", fake_run)
    monkeypatch.setattr(daemon, "run_submission_gate", fake_submit_gate)
    monkeypatch.setattr(daemon.time, "sleep", lambda _seconds: None)

    daemon.run_local_research_daemon(max_cycles=1)

    assert len(calls) == 2
    assert len(submission_calls) == 1
    assert submission_calls[0]["account"] == "primary"
    assert submission_calls[0]["trigger"]["status"] == "SUBMISSION_REQUIRED"
    assert submission_calls[0]["trigger"]["planner_context"]["submission_candidates"][0]["alpha_id"] == "ready-alpha"
    assert all(call["submission_deferred"] is False for call in calls)
    assert all(call["reasoning_policy"] == "max" for call in calls)
    assert all(call["target_simulations"] == 100 for call in calls)
    assert all(call["budget_minutes"] == 50 for call in calls)
    assert all(call["batch_size"] == 8 for call in calls)
    assert all(call["max_simulations_per_batch"] == 8 for call in calls)
