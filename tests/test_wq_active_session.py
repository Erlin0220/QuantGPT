from quantgpt.wq_active_session import (
    apply_research_result,
    apply_submission_result,
    new_active_first_state,
)


def _policy(remaining=1):
    return {
        "daily_active_target": 2,
        "daily_active_count": 2 - remaining,
        "remaining_active_target": remaining,
    }


def _new_state():
    return new_active_first_state(account="primary", policy=_policy(1), max_iterations=4)


def _skill_candidate(*extra_skills):
    return {
        "skill_chain": [
            "wq-economic-hypothesis",
            "wq-alpha-hypothesis",
            "wq-alpha-review",
            "wq-robustness-validation",
            "wq-candidate-evidence",
            *extra_skills,
        ]
    }


def test_new_active_first_session_cannot_stop_while_target_remains():
    state = _new_state()

    assert state["status"] == "RUNNING"
    assert state["phase"] == "NEEDS_RESEARCH"
    assert state["session_stop_allowed"] is False
    assert state["remaining_active_target"] == 1


def test_zero_candidate_iteration_forces_continuation():
    state = apply_research_result(
        _new_state(),
        {"summary": {"simulated": 1}, "candidates": []},
        {"skill_candidates": [_skill_candidate()]},
    )

    assert state["counters"]["iterations"] == 1
    assert state["counters"]["simulations"] == 1
    assert state["counters"]["new_hypotheses"] == 1
    assert state["phase"] == "CONTINUE_REQUIRED"
    assert state["next_action"] == "failure_diagnosis_then_allocate_next_iteration"
    assert state["session_stop_allowed"] is False


def test_candidate_found_requires_submission_instead_of_stopping():
    state = apply_research_result(
        _new_state(),
        {"summary": {"simulated": 1}, "candidates": [{"alpha_id": "alpha-1"}]},
        {"skill_candidates": [_skill_candidate("wq-alpha-diversify")]},
    )

    assert state["counters"]["diversifications"] == 1
    assert state["counters"]["candidates"] == 1
    assert state["phase"] == "NEEDS_SUBMISSION"
    assert state["session_stop_allowed"] is False


def test_sc_fail_forces_next_iteration_when_budget_remains():
    state = apply_submission_result(
        _new_state(),
        {
            "sc_fail": 1,
            "results": {"alpha-1": {"final_status": "SC_FAIL"}},
        },
    )

    assert state["counters"]["formal_submissions"] == 1
    assert state["counters"]["terminal_failures"] == 1
    assert state["phase"] == "CONTINUE_REQUIRED"
    assert state["end_reason"] == "formal_submission_failed"
    assert state["session_stop_allowed"] is False


def test_local_submission_block_does_not_end_session():
    state = apply_submission_result(
        _new_state(),
        {
            "blocked": 1,
            "results": {"alpha-1": {"final_status": "POLICY_BLOCKED"}},
        },
    )

    assert state["counters"]["formal_submissions"] == 0
    assert state["phase"] == "CONTINUE_REQUIRED"
    assert state["end_reason"] == "submission_not_formally_attempted"
    assert state["session_stop_allowed"] is False


def test_platform_unknown_is_legitimate_stop_condition():
    state = apply_submission_result(
        _new_state(),
        {
            "timeout": 1,
            "results": {"alpha-1": {"final_status": "SUBMIT_UNKNOWN"}},
        },
    )

    assert state["phase"] == "PLATFORM_INFLIGHT_OR_UNKNOWN"
    assert state["session_stop_allowed"] is True
    assert state["end_reason"] == "platform_inflight_or_unknown"


def test_four_informative_iterations_exhaust_session_budget():
    state = _new_state()
    for _ in range(4):
        state = apply_research_result(
            state,
            {"summary": {"simulated": 1}, "candidates": []},
            {"skill_candidates": [_skill_candidate("wq-alpha-repair")]},
        )

    assert state["counters"]["iterations"] == 4
    assert state["counters"]["repairs"] == 4
    assert state["phase"] == "BUDGET_EXHAUSTED"
    assert state["session_stop_allowed"] is True
    assert state["end_reason"] == "session_iteration_budget_exhausted"


def test_active_completes_session_and_switches_to_replenishment():
    state = apply_submission_result(
        _new_state(),
        {
            "active": 1,
            "results": {"alpha-1": {"final_status": "ACTIVE"}},
        },
    )

    assert state["counters"]["active"] == 1
    assert state["remaining_active_target"] == 0
    assert state["phase"] == "DONE"
    assert state["session_stop_allowed"] is True
    assert state["next_action"] == "replenishment"
