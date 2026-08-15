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
