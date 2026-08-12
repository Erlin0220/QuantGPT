from datetime import datetime, timezone

from quantgpt.wq_research_audit import build_research_round_audit


def _base_candidate(**overrides):
    value = {
        "expression": "rank(close)",
        "hypothesis": "test",
        "skill_chain": [
            "wq-alpha-hypothesis",
            "wq-alpha-review",
            "wq-robustness-validation",
            "wq-candidate-evidence",
        ],
        "review_decision": "RUN",
        "robustness_plan": {"mode": "skill_defined", "checks": [{"universe": "TOP1000"}]},
        "candidate_evidence_policy": {"mode": "calibrated_evidence_hierarchy"},
    }
    value.update(overrides)
    return value


def test_round_audit_measures_skill_compliance_and_simulation_cost():
    task = {
        "task_id": "round-1",
        "status": "completed",
        "created_at": datetime(2026, 8, 11, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 8, 11, 0, 5, tzinfo=timezone.utc),
        "params": {"tag": "audit-1", "skill_candidates": [_base_candidate()]},
        "result": {
            "mode": "skill_first",
            "summary": {"simulated": 3, "validation_simulations": 2, "total_simulations": 5},
        },
    }
    audit = build_research_round_audit(
        task,
        trials=[{"alpha_id": "a1"}, {"alpha_id": "a2"}, {"alpha_id": "a3"}],
        candidates=[{"alpha_id": "a1"}],
        attempts=[{"alpha_id": "a1", "status": "ACTIVE"}],
    )

    assert audit["source_run_id"] == "round-1"
    assert audit["skill_compliance"]["rate"] == 1.0
    assert audit["skill_compliance"]["routes"] == {"new_hypothesis": 1}
    assert audit["execution"]["simulation_to_candidate_rate"] == 0.3333
    assert audit["execution"]["simulations_per_candidate"] == 5.0
    assert audit["submission"]["candidate_to_active_rate"] == 1.0
    assert audit["iteration"]["new_hypotheses"] == 1
    assert audit["iteration"]["repairs"] == 0
    assert audit["iteration"]["candidates"] == 1
    assert audit["iteration"]["formal_submissions"] == 1
    assert audit["iteration"]["active"] == 1
    assert audit["iteration"]["end_reason"] == "candidate_found"
    assert audit["created_at"].startswith("2026-08-11T00:00:00")


def test_round_audit_requires_failure_signature_and_diversity_case_for_routed_children():
    repair = _base_candidate(
        skill_chain=[
            "wq-alpha-hypothesis",
            "wq-failure-diagnosis",
            "wq-experiment-allocation",
            "wq-alpha-repair",
            "wq-alpha-review",
            "wq-robustness-validation",
            "wq-candidate-evidence",
        ]
    )
    diversify = _base_candidate(
        skill_chain=[
            "wq-alpha-hypothesis",
            "wq-failure-diagnosis",
            "wq-experiment-allocation",
            "wq-alpha-diversify",
            "wq-alpha-review",
            "wq-robustness-validation",
            "wq-candidate-evidence",
        ],
        diversity_case={"changed_dimensions": ["information_source"], "state": "designed_diverse"},
    )
    task = {
        "task_id": "round-2",
        "status": "completed",
        "params": {"skill_candidates": [repair, diversify]},
        "result": {"summary": {"simulated": 2, "total_simulations": 2}},
    }

    audit = build_research_round_audit(task)
    assert audit["skill_compliance"]["valid"] == 1
    assert audit["skill_compliance"]["repair_required"] == 1
    assert audit["skill_compliance"]["repair_with_failure_signature"] == 0
    assert audit["skill_compliance"]["diversify_with_diversity_case"] == 1
    assert audit["skill_compliance"]["rate"] == 0.5
    assert audit["iteration"]["repairs"] == 1
    assert audit["iteration"]["diversifications"] == 1


def test_round_audit_marks_failed_active_target_iteration_for_continuation():
    task = {
        "task_id": "round-active-target",
        "status": "completed",
        "params": {"skill_candidates": [_base_candidate()]},
        "result": {
            "summary": {
                "simulated": 1,
                "total_simulations": 1,
                "remaining_active_target": 1,
                "candidates": 0,
            }
        },
    }

    audit = build_research_round_audit(task, trials=[{"alpha_id": "failed-1"}])

    assert audit["iteration"]["end_reason"] == "iteration_exhausted_no_candidate"
    assert audit["iteration"]["next_action"] == "failure_diagnosis_then_allocate_next_iteration"
    assert audit["iteration"]["session_stop_allowed"] is False
    assert audit["iteration"]["stop_repair_scope"] == "parent_only_never_session"


def test_round_audit_allows_hard_failed_task_to_stop_session():
    task = {
        "task_id": "round-hard-fail",
        "status": "failed",
        "params": {"skill_candidates": [_base_candidate()]},
        "result": {"summary": {"remaining_active_target": 1}},
    }

    audit = build_research_round_audit(task)

    assert audit["iteration"]["end_reason"] == "hard_task_failure"
    assert audit["iteration"]["next_action"] == "stop_for_hard_failure"
    assert audit["iteration"]["session_stop_allowed"] is True
