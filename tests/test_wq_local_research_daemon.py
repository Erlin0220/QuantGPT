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


def test_classify_auth_failure_uses_long_backoff():
    decision = daemon.classify_daemon_result(
        {"ok": False, "status": "brain_authentication_failed", "progress": {"should_stop": False}}
    )
    assert decision.delay_seconds == 300.0


def test_daemon_runs_restart_safe_cycles_and_never_requests_submission(monkeypatch):
    calls: list[dict] = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {
            "ok": True,
            "status": "TARGET_REACHED",
            "cycle_id": f"cycle-{len(calls)}",
            "model": "deepseek-v4-flash",
            "reasoning_policy": kwargs["reasoning_policy"],
            "progress": {"should_stop": True, "stop_reason": "target_reached"},
            "research_cycle": {},
            "local_llm_rounds_this_run": [],
        }

    monkeypatch.setattr(daemon, "run_local_llm_research", fake_run)
    monkeypatch.setattr(daemon.time, "sleep", lambda _seconds: None)

    daemon.run_local_research_daemon(max_cycles=2)

    assert len(calls) == 2
    assert all(call["submission_deferred"] is True for call in calls)
    assert all(call["reasoning_policy"] == "adaptive" for call in calls)
    assert all(call["target_simulations"] == 100 for call in calls)
    assert all(call["budget_minutes"] == 50 for call in calls)
    assert all(call["batch_size"] == 4 for call in calls)
