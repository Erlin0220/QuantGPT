from __future__ import annotations

import json

import pytest

from quantgpt import wq_local_candidate_generator as generator
from quantgpt import wq_research_cycle_runner as runner


def _raw_candidate(index: int = 1) -> dict:
    return {
        "expression": f"rank(ts_delta(close, {index + 4})) * rank(volume / adv20)",
        "hypothesis": "short-horizon price movement confirmed by unusual volume may contain delayed information diffusion",
        "family": f"price_volume_{index}",
        "data_fields": ["close", "volume", "adv20"],
        "knowledge_card_ids": [],
        "review_notes": "bounded mechanism test",
        "robustness_plan": {
            "checks": [{"universe": "TOP1000", "purpose": "test universe sensitivity"}],
        },
    }


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_generate_local_skill_batch_normalizes_to_runner_contract(monkeypatch):
    seen: dict = {}

    def fake_post(url, *, headers, json, timeout):
        seen.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        content = json_module.dumps({"skill_candidates": [_raw_candidate(1), _raw_candidate(2)]})
        return _FakeResponse({"choices": [{"message": {"content": content}}]})

    json_module = json
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(generator, "_load_skill_context", lambda _context: (["wq-alpha-hypothesis"], "skill text"))
    monkeypatch.setattr(generator.httpx, "post", fake_post)

    result = generator.generate_local_skill_batch({"current_cycle_trial_evidence": []}, batch_size=2)

    assert result["model"] == "deepseek-v4-flash"
    assert result["generated"] == 2
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["json"]["max_tokens"] >= 1024
    assert seen["json"]["thinking"] == {"type": "disabled"}
    assert seen["json"]["response_format"] == {"type": "json_object"}
    assert result["thinking"] == "disabled"
    assert all(runner._runner_candidate_contract_error(item) is None for item in result["skill_candidates"])


def test_generate_local_skill_batch_supports_thinking_max(monkeypatch):
    seen: dict = {}

    def fake_post(url, *, headers, json, timeout):
        seen.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        content = json_module.dumps({"skill_candidates": [_raw_candidate(1)]})
        return _FakeResponse({"choices": [{"message": {"content": content, "reasoning_content": "reasoning"}}]})

    json_module = json
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(generator, "_load_skill_context", lambda _context: (["wq-alpha-hypothesis"], "skill text"))
    monkeypatch.setattr(generator.httpx, "post", fake_post)

    result = generator.generate_local_skill_batch({}, batch_size=1, thinking="enabled", reasoning_effort="max")

    assert seen["json"]["thinking"] == {"type": "enabled"}
    assert seen["json"]["reasoning_effort"] == "max"
    assert "temperature" not in seen["json"]
    assert result["thinking"] == "enabled"
    assert result["reasoning_effort"] == "max"


def test_generate_local_skill_batch_hard_rejects_history_and_retries(monkeypatch):
    duplicate = _raw_candidate(1)
    replacement = _raw_candidate(2)
    payloads = [duplicate, replacement]
    calls = 0

    def fake_post(*_args, **_kwargs):
        nonlocal calls
        candidate = payloads[calls]
        calls += 1
        content = json.dumps({"skill_candidates": [candidate]})
        return _FakeResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("WQ_LOCAL_LLM_CHUNK_ATTEMPTS", "2")
    monkeypatch.setattr(generator, "_load_skill_context", lambda _context: (["wq-alpha-hypothesis"], "skill text"))
    monkeypatch.setattr(generator.httpx, "post", fake_post)

    result = generator.generate_local_skill_batch(
        {"local_llm_exclude_expressions": [duplicate["expression"]]},
        batch_size=1,
    )

    assert calls == 2
    assert result["skill_candidates"][0]["expression"] == replacement["expression"]


def test_generate_local_skill_batch_retries_unresolved_data_fields(monkeypatch):
    invalid = _raw_candidate(1)
    invalid["expression"] = "rank(ts_mean(return_invested_capital, 20))"
    invalid["data_fields"] = ["return_invested_capital"]
    replacement = _raw_candidate(2)
    payloads = [invalid, replacement]
    calls = 0

    def fake_post(*_args, **_kwargs):
        nonlocal calls
        candidate = payloads[calls]
        calls += 1
        content = json.dumps({"skill_candidates": [candidate]})
        return _FakeResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("WQ_LOCAL_LLM_CHUNK_ATTEMPTS", "2")
    monkeypatch.setattr(generator, "_load_skill_context", lambda _context: (["wq-alpha-hypothesis"], "skill text"))
    monkeypatch.setattr(generator.httpx, "post", fake_post)

    result = generator.generate_local_skill_batch({}, batch_size=1)

    assert calls == 2
    assert result["skill_candidates"][0]["expression"] == replacement["expression"]


def test_noncore_data_field_is_allowed_when_grounded_in_planner_context():
    context_text = json.dumps({"current_cycle_trial_evidence": [{"expression": "rank(return_assets)"}]})
    assert generator._data_field_is_grounded("return_assets", context_text) is True
    assert generator._data_field_is_grounded("return_invested_capital", context_text) is False


def test_generate_local_skill_batch_enforces_forced_diversify(monkeypatch):
    first = _raw_candidate(1)
    replacement = _raw_candidate(2)
    replacement["family"] = "independent_fundamental_quality"
    replacement["data_fields"] = ["returns"]
    replacement["expression"] = "rank(ts_mean(returns, 20))"
    payloads = [first, replacement]
    calls = 0

    def fake_post(*_args, **_kwargs):
        nonlocal calls
        candidate = payloads[calls]
        calls += 1
        content = json.dumps({"skill_candidates": [candidate]})
        return _FakeResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("WQ_LOCAL_LLM_CHUNK_ATTEMPTS", "2")
    monkeypatch.setattr(generator, "_load_skill_context", lambda _context: (["wq-alpha-diversify"], "skill text"))
    monkeypatch.setattr(generator.httpx, "post", fake_post)

    result = generator.generate_local_skill_batch(
        {
            "local_llm_force_diversify": True,
            "local_llm_recent_families": [first["family"]],
            "current_cycle_trial_evidence": [{"expression": first["expression"], "family": first["family"]}],
        },
        batch_size=1,
    )

    assert calls == 2
    candidate = result["skill_candidates"][0]
    assert candidate["local_llm_route"] == "DIVERSIFY"
    assert "information_source" in candidate["diversity_case"]["changed_dimensions"]


def test_candidate_payload_accepts_common_key_drift():
    candidate = _raw_candidate(1)
    assert generator._candidate_items_from_payload({"candidates": [candidate]}) == [candidate]
    assert generator._candidate_items_from_payload({"result": {"alphas": [candidate]}}) == [candidate]
    assert generator._candidate_items_from_payload(candidate) == [candidate]


def test_generate_local_skill_batch_rejects_reasoning_only_response(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(generator, "_load_skill_context", lambda _context: (["wq-alpha-hypothesis"], "skill text"))
    monkeypatch.setattr(
        generator.httpx,
        "post",
        lambda *args, **kwargs: _FakeResponse(
            {"choices": [{"message": {"content": "", "reasoning_content": "thinking"}}]}
        ),
    )

    with pytest.raises(generator.LocalCandidateGenerationError, match="reasoning_content"):
        generator.generate_local_skill_batch({}, batch_size=1)


def test_local_llm_poc_runs_bounded_research_only_loop(monkeypatch):
    snapshot = {
        "ok": True,
        "status": "NEEDS_SKILL_BATCH",
        "progress": {"should_stop": False, "simulations": 0},
        "planner_context": {"current_cycle_trial_evidence": []},
        "research_cycle": {"cycle_id": "poc-cycle"},
    }
    monkeypatch.setattr(runner, "research_cycle_snapshot", lambda **_kwargs: dict(snapshot))
    generation_calls: list[dict] = []

    def fake_generate(context, *, batch_size, model=None, thinking=None, reasoning_effort=None):
        generation_calls.append({
            "context": context,
            "batch_size": batch_size,
            "model": model,
            "thinking": thinking,
            "reasoning_effort": reasoning_effort,
        })
        candidate = runner_test_candidate()
        return {
            "provider": "opencode-go-compatible",
            "model": model or "deepseek-v4-flash",
            "thinking": thinking,
            "reasoning_effort": reasoning_effort if thinking == "enabled" else None,
            "requested": batch_size,
            "generated": 1,
            "skills_loaded": ["wq-alpha-hypothesis"],
            "skill_candidates": [candidate],
        }

    run_calls: list[dict] = []

    def fake_run_skill_batch(**kwargs):
        run_calls.append(kwargs)
        return {
            "ok": True,
            "status": "NEEDS_SKILL_BATCH",
            "batch_executed": True,
            "source_run_id": f"run-{len(run_calls)}",
            "progress": {"should_stop": False, "simulations": len(run_calls) * 4},
            "batch_result": {"summary": {"simulated": 4, "candidates": 0}},
        }

    monkeypatch.setattr(runner, "generate_local_skill_batch", fake_generate)
    monkeypatch.setattr(runner, "run_skill_batch", fake_run_skill_batch)
    monkeypatch.setattr(runner, "_record_local_llm_round_sync", lambda *_args, **_kwargs: None)

    result = runner.run_local_llm_poc(
        cycle_id="poc-cycle",
        max_batches=2,
        batch_size=3,
        max_simulations_per_batch=4,
    )

    assert result["ok"] is True
    assert result["formal_submission"] is False
    assert len(result["poc_batches"]) == 2
    assert len(generation_calls) == 2
    assert len(run_calls) == 2
    assert all(call["submission_deferred"] is True for call in run_calls)
    assert all(call["thinking"] == "disabled" for call in generation_calls)
    assert all(call["reasoning_effort"] == "max" for call in generation_calls)
    assert all(call["max_simulations"] == 4 for call in run_calls)


def test_normalize_repair_candidate_preserves_failure_contract():
    raw = _raw_candidate(1)
    raw.update({
        "route": "REPAIR",
        "parent_expression": "rank(ts_delta(close, 5))",
        "mutation_type": "horizon_repair",
        "mutation_reason": "test whether slower response improves weak robustness",
        "failure_signature": {
            "observed_symptoms": ["low_fitness", "sub_universe_instability"],
            "plausible_causes": [{"cause": "horizon_mismatch", "confidence": "medium"}],
        },
    })

    candidate = generator._normalize_candidate(
        raw,
        {"current_cycle_trial_evidence": [{"expression": "rank(ts_delta(close, 5))"}]},
    )

    assert candidate["local_llm_route"] == "REPAIR"
    assert "wq-failure-diagnosis" in candidate["skill_chain"]
    assert "wq-experiment-allocation" in candidate["skill_chain"]
    assert "wq-alpha-repair" in candidate["skill_chain"]
    assert candidate["parent_expression"] == "rank(ts_delta(close, 5))"
    assert runner._runner_candidate_contract_error(candidate) is None


def test_repair_without_current_cycle_parent_is_downgraded_to_new_hypothesis():
    raw = _raw_candidate(1)
    raw.update({
        "route": "REPAIR",
        "parent_expression": "rank(ts_delta(close, 5))",
        "failure_signature": {
            "observed_symptoms": ["low_fitness"],
            "plausible_causes": [{"cause": "weak_signal", "confidence": "medium"}],
        },
    })

    candidate = generator._normalize_candidate(raw, {"current_cycle_trial_evidence": []})

    assert candidate["local_llm_route"] == "NEW_HYPOTHESIS"
    assert "wq-alpha-repair" not in candidate["skill_chain"]
    assert "parent_expression" not in candidate


def test_normalize_diversify_candidate_preserves_diversity_contract():
    raw = _raw_candidate(1)
    raw.update({
        "route": "DIVERSIFY",
        "diversity_case": {
            "changed_dimensions": ["information_source", "economic_mechanism"],
            "why_independent": "switches from price-volume to a distinct information source",
        },
    })

    candidate = generator._normalize_candidate(raw)

    assert candidate["local_llm_route"] == "DIVERSIFY"
    assert "wq-alpha-diversify" in candidate["skill_chain"]
    assert runner._runner_candidate_contract_error(candidate) is None


def test_adaptive_reasoning_uses_fast_for_new_hypotheses():
    mode = runner._local_llm_reasoning_mode({"current_cycle_trial_evidence": []}, "adaptive")
    assert mode["thinking"] == "disabled"
    assert mode["reason"] == "new_hypothesis_throughput"


def test_adaptive_reasoning_uses_max_to_escape_duplicate_pressure():
    mode = runner._local_llm_reasoning_mode(
        {"current_cycle_trial_evidence": [], "local_llm_force_diversify": True},
        "adaptive",
    )
    assert mode["thinking"] == "enabled"
    assert mode["reasoning_effort"] == "max"
    assert mode["reason"] == "duplicate_pressure_escape"


def test_adaptive_reasoning_uses_max_for_near_miss_repair():
    mode = runner._local_llm_reasoning_mode(
        {
            "current_cycle_trial_evidence": [
                {
                    "expression": "rank(close)",
                    "sharpe": 1.05,
                    "fitness": 0.72,
                    "failure_reason": "low_fitness",
                    "failure_reasons": [],
                }
            ]
        },
        "adaptive",
    )
    assert mode["thinking"] == "enabled"
    assert mode["reasoning_effort"] == "max"
    assert mode["parent_expression"] == "rank(close)"


def test_local_llm_exclusions_include_cross_cycle_history():
    exclusions = runner._local_llm_exclusion_expressions(
        {"recent_trial_expressions": ["rank(close)", "rank(volume)"]},
        {
            "local_llm": {
                "rounds": [
                    {"generated_candidates": [{"expression": "rank(ts_mean(returns, 20))"}]},
                ]
            }
        },
    )

    assert exclusions == ["rank(close)", "rank(volume)", "rank(ts_mean(returns, 20))"]


def test_duplicate_pressure_forces_diversify_after_repeated_zero_sim_rounds():
    pressure = runner._local_llm_duplicate_pressure(
        {
            "local_llm": {
                "rounds": [
                    {
                        "status": "completed_no_simulation",
                        "summary": {"simulated": 0},
                        "skill_plan_rejections": [{"reason": "duplicate_expression_history"}],
                        "generated_candidates": [{"family": "family_a"}],
                    },
                    {
                        "status": "completed_no_simulation",
                        "summary": {"simulated": 0},
                        "skill_plan_rejections": [{"reason": "duplicate_expression_history"}],
                        "generated_candidates": [{"family": "family_b"}],
                    },
                ]
            }
        }
    )

    assert pressure["force_diversify"] is True
    assert pressure["consecutive_zero_sim_rounds"] == 2
    assert pressure["recent_families"] == ["family_a", "family_b"]


def test_generation_failures_force_diversify_and_max_then_recover(monkeypatch):
    candidate = runner_test_candidate()
    state = {
        "ok": True,
        "status": "NEEDS_SKILL_BATCH",
        "cycle_id": "generation-recovery-cycle",
        "progress": {"should_stop": False, "simulations": 0},
        "planner_context": {"current_cycle_trial_evidence": []},
        "research_cycle": {"cycle_id": "generation-recovery-cycle", "local_llm": {"rounds": []}},
    }

    def fake_snapshot(**_kwargs):
        return {
            **state,
            "research_cycle": {
                **state["research_cycle"],
                "local_llm": {"rounds": [dict(item) for item in state["research_cycle"]["local_llm"]["rounds"]]},
            },
        }

    generation_calls: list[str] = []

    def fake_generate(_context, *, batch_size, model=None, thinking=None, reasoning_effort=None):
        del batch_size, model, reasoning_effort
        generation_calls.append(str(thinking))
        if len(generation_calls) <= 2:
            raise generator.LocalCandidateGenerationError("duplicate-only generation")
        return {"model": "deepseek-v4-flash", "skill_candidates": [candidate]}

    def fake_record(_account, _cycle_id, event):
        rounds = state["research_cycle"]["local_llm"]["rounds"]
        for index, existing in enumerate(rounds):
            if existing.get("round_id") == event.get("round_id"):
                rounds[index] = {**existing, **event}
                return state["research_cycle"]
        rounds.append(dict(event))
        return state["research_cycle"]

    monkeypatch.setattr(runner, "research_cycle_snapshot", fake_snapshot)
    monkeypatch.setattr(runner, "generate_local_skill_batch", fake_generate)
    monkeypatch.setattr(runner, "_record_local_llm_round_sync", fake_record)
    monkeypatch.setattr(
        runner,
        "run_skill_batch",
        lambda **_kwargs: {
            "ok": True,
            "status": "NEEDS_SKILL_BATCH",
            "batch_executed": True,
            "source_run_id": "recovered-run",
            "progress": {"should_stop": False, "simulations": 1},
            "batch_result": {"summary": {"simulated": 1}},
        },
    )

    result = runner.run_local_llm_research(cycle_id="generation-recovery-cycle", max_batches=1)

    assert generation_calls == ["disabled", "disabled", "enabled"]
    assert result["productive_batches_this_run"] == 1
    assert result["generation_attempts_this_run"] == 3
    assert result["local_llm_rounds_this_run"][0]["status"] == "generation_failed"
    assert result["local_llm_rounds_this_run"][1]["status"] == "generation_failed"
    assert result["local_llm_rounds_this_run"][2]["status"] == "completed"


def test_zero_simulation_round_does_not_consume_productive_batch_budget(monkeypatch):
    candidate = runner_test_candidate()
    snapshot = {
        "ok": True,
        "status": "NEEDS_SKILL_BATCH",
        "cycle_id": "zero-sim-cycle",
        "progress": {"should_stop": False, "simulations": 0},
        "planner_context": {"current_cycle_trial_evidence": []},
        "research_cycle": {"cycle_id": "zero-sim-cycle", "local_llm": {"rounds": []}},
    }
    monkeypatch.setattr(runner, "research_cycle_snapshot", lambda **_kwargs: dict(snapshot))
    monkeypatch.setattr(
        runner,
        "generate_local_skill_batch",
        lambda *_args, **_kwargs: {"model": "deepseek-v4-flash", "skill_candidates": [candidate]},
    )
    monkeypatch.setattr(runner, "_record_local_llm_round_sync", lambda *_args, **_kwargs: None)
    run_calls = 0

    def fake_run_skill_batch(**_kwargs):
        nonlocal run_calls
        run_calls += 1
        simulated = 0 if run_calls == 1 else 1
        return {
            "ok": True,
            "status": "NEEDS_SKILL_BATCH",
            "batch_executed": True,
            "source_run_id": f"run-{run_calls}",
            "progress": {"should_stop": False, "simulations": simulated},
            "batch_result": {
                "summary": {"simulated": simulated},
                "skill_plan_rejections": [{"reason": "duplicate_expression_history"}] if simulated == 0 else [],
            },
        }

    monkeypatch.setattr(runner, "run_skill_batch", fake_run_skill_batch)

    result = runner.run_local_llm_research(cycle_id="zero-sim-cycle", max_batches=1)

    assert run_calls == 2
    assert result["productive_batches_this_run"] == 1
    assert result["generation_attempts_this_run"] == 2
    assert result["local_llm_rounds_this_run"][0]["status"] == "completed_no_simulation"


def test_pending_round_is_reused_without_regeneration(monkeypatch):
    candidate = runner_test_candidate()
    snapshot = {
        "ok": True,
        "status": "NEEDS_SKILL_BATCH",
        "cycle_id": "resume-cycle",
        "progress": {"should_stop": False, "simulations": 0},
        "planner_context": {"current_cycle_trial_evidence": []},
        "research_cycle": {
            "cycle_id": "resume-cycle",
            "local_llm": {
                "rounds": [
                    {
                        "round_id": "llm-pending",
                        "batch_index": 1,
                        "status": "pending_execution",
                        "candidate_payloads": [candidate],
                        "generated_candidates": [{"expression": candidate["expression"]}],
                        "reasoning_mode": {"thinking": "disabled"},
                    }
                ]
            },
        },
    }
    monkeypatch.setattr(runner, "research_cycle_snapshot", lambda **_kwargs: dict(snapshot))
    monkeypatch.setattr(runner, "generate_local_skill_batch", lambda *_args, **_kwargs: pytest.fail("must reuse pending"))
    monkeypatch.setattr(runner, "_record_local_llm_round_sync", lambda *_args, **_kwargs: None)
    run_calls: list[dict] = []

    def fake_run_skill_batch(**kwargs):
        run_calls.append(kwargs)
        return {
            "ok": True,
            "status": "NEEDS_SKILL_BATCH",
            "batch_executed": True,
            "source_run_id": "resume-run",
            "progress": {"should_stop": False, "simulations": 1},
            "batch_result": {"summary": {"simulated": 1}},
        }

    monkeypatch.setattr(runner, "run_skill_batch", fake_run_skill_batch)

    result = runner.run_local_llm_research(cycle_id="resume-cycle", max_batches=1)

    assert result["ok"] is True
    assert len(run_calls) == 1
    assert run_calls[0]["skill_candidates"] == [candidate]
    assert result["local_llm_rounds_this_run"][0]["round_id"] == "llm-pending"


def test_local_research_can_explicitly_defer_submission_and_continue(monkeypatch):
    candidate = runner_test_candidate()
    snapshots = [
        {
            "ok": True,
            "status": "SUBMISSION_REQUIRED",
            "cycle_id": "defer-cycle",
            "progress": {"should_stop": False, "simulations": 1},
            "planner_context": {"current_cycle_trial_evidence": []},
            "research_cycle": {"cycle_id": "defer-cycle"},
        },
        {
            "ok": True,
            "status": "SUBMISSION_REQUIRED",
            "cycle_id": "defer-cycle",
            "progress": {"should_stop": False, "simulations": 1},
            "planner_context": {"current_cycle_trial_evidence": []},
            "research_cycle": {"cycle_id": "defer-cycle"},
        },
        {
            "ok": True,
            "status": "TARGET_REACHED",
            "cycle_id": "defer-cycle",
            "progress": {"should_stop": True, "simulations": 4, "stop_reason": "target_reached"},
            "planner_context": {},
            "research_cycle": {"cycle_id": "defer-cycle"},
        },
    ]

    def fake_snapshot(**_kwargs):
        return dict(snapshots.pop(0) if len(snapshots) > 1 else snapshots[0])

    monkeypatch.setattr(runner, "research_cycle_snapshot", fake_snapshot)
    monkeypatch.setattr(
        runner,
        "generate_local_skill_batch",
        lambda *_args, **_kwargs: {
            "model": "deepseek-v4-flash",
            "skill_candidates": [candidate],
        },
    )
    monkeypatch.setattr(runner, "_record_local_llm_round_sync", lambda *_args, **_kwargs: None)
    run_calls: list[dict] = []

    def fake_run_skill_batch(**kwargs):
        run_calls.append(kwargs)
        return {
            "ok": True,
            "status": "TARGET_REACHED",
            "batch_executed": True,
            "source_run_id": "defer-run",
            "progress": {"should_stop": True, "simulations": 4, "stop_reason": "target_reached"},
            "batch_result": {"summary": {"simulated": 3}},
        }

    monkeypatch.setattr(runner, "run_skill_batch", fake_run_skill_batch)

    result = runner.run_local_llm_research(cycle_id="defer-cycle", max_batches=1, submission_deferred=True)

    assert result["ok"] is True
    assert len(run_calls) == 1
    assert run_calls[0]["submission_deferred"] is True


def test_local_research_stops_at_submission_gate_without_generation(monkeypatch):
    snapshot = {
        "ok": True,
        "status": "SUBMISSION_REQUIRED",
        "cycle_id": "formal-cycle",
        "progress": {"should_stop": False, "simulations": 3},
        "planner_context": {},
        "research_cycle": {"cycle_id": "formal-cycle"},
    }
    monkeypatch.setattr(runner, "research_cycle_snapshot", lambda **_kwargs: dict(snapshot))
    monkeypatch.setattr(runner, "generate_local_skill_batch", lambda *_args, **_kwargs: pytest.fail("must not generate"))

    result = runner.run_local_llm_research(cycle_id="formal-cycle", max_batches=2)

    assert result["ok"] is True
    assert result["status"] == "SUBMISSION_REQUIRED"
    assert result["formal_submission"] is False
    assert result["local_llm_rounds_this_run"] == []


def runner_test_candidate() -> dict:
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
