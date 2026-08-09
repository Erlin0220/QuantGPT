"""Tests for autonomous WorldQuant research planning and bounded evolution."""

from quantgpt import wq_autonomous_research as autonomous
from quantgpt.wq_research_memory import normalize_wq_expression


def test_family_selector_prefers_undercovered_family():
    memory = {
        "family_counts": {
            family: (0 if family == "analyst_revision" else 20)
            for family in autonomous.FAMILY_SEEDS
        }
    }

    selected = autonomous.select_research_families(memory, count=2)

    assert selected[0] == "analyst_revision"


def test_seed_plan_skips_already_researched_expression():
    first = autonomous.FAMILY_SEEDS["price_volume"][0]
    memory = {
        "family_counts": {family: 50 for family in autonomous.FAMILY_SEEDS},
        "normalized_expressions": [normalize_wq_expression(first)],
    }
    memory["family_counts"]["price_volume"] = 0

    plan = autonomous.build_seed_plan(memory, limit=2, family_count=1)

    assert len(plan) == 2
    assert all(normalize_wq_expression(item["expression"]) != normalize_wq_expression(first) for item in plan)
    assert all(item["family"] == "price_volume" for item in plan)


def test_memory_mutation_plan_reuses_recent_failure_across_runs():
    parent = "rank(ts_mean(returns, 10))"
    memory = {
        "recent_trials": [
            {
                "expression": parent,
                "family": "momentum_reversal",
                "generation": 1,
                "fitness": 0.8,
                "sharpe": 1.0,
                "mutation_targets": ["reduce_turnover", "improve_signal_sharpe"],
            }
        ]
    }

    plan = autonomous.build_memory_mutation_plan(memory, seen=set(), limit=2, hypothesis="continue")

    assert len(plan) == 2
    assert all(item["parent_expression"] == parent for item in plan)
    assert all(item["generation"] == 2 for item in plan)
    assert {item["mutation_type"] for item in plan} == {"widen_windows", "invert_signal"}


def test_targeted_mutations_follow_diagnostics_and_preserve_lineage():
    expression = "rank(ts_mean(returns, 10))"
    result = {
        "expression": expression,
        "mutation_targets": ["reduce_turnover", "improve_signal_sharpe"],
        "research_meta": {
            "family": "momentum_reversal",
            "generation": 1,
            "hypothesis": "test hypothesis",
        },
    }

    mutations = autonomous.build_targeted_mutations(result, seen=set(), limit=3)

    assert any(item["mutation_type"] == "widen_windows" for item in mutations)
    assert any(item["mutation_type"] == "invert_signal" for item in mutations)
    assert all(item["generation"] == 2 for item in mutations)
    assert all(item["parent_expression"] == expression for item in mutations)


def test_autonomous_research_runs_bounded_second_generation(monkeypatch):
    monkeypatch.setattr(autonomous, "run_list_alphas", lambda *_args, **_kwargs: {"ok": True, "alphas": []})

    calls = []

    def fake_research_batch(_client, expressions, **kwargs):
        calls.append((list(expressions), kwargs["tag"]))
        generation = 2 if kwargs["tag"].endswith("g2") else 1
        results = []
        candidates = []
        for index, expression in enumerate(expressions):
            good = generation == 2 and index == 0
            item = {
                "ok": True,
                "alpha_id": f"g{generation}-{index}",
                "expression": expression,
                "is_metrics": {
                    "sharpe": 1.6 if good else 0.8,
                    "fitness": 1.2 if good else 0.5,
                    "returns": 0.1 if good else 0.02,
                    "turnover": 0.2,
                    "checks": [],
                },
                "passes_primary_thresholds": good,
                "mutation_targets": ["candidate_passes_primary_thresholds"] if good else ["improve_signal_sharpe"],
            }
            results.append(item)
            if good:
                candidates.append(item)
        return {
            "ok": True,
            "tag": kwargs["tag"],
            "settings": {
                "region": kwargs["region"],
                "universe": kwargs["universe"],
                "delay": kwargs["delay"],
                "decay": kwargs["decay"],
                "neutralization": kwargs["neutralization"],
                "truncation": kwargs["truncation"],
            },
            "summary": {"simulated": len(expressions)},
            "results": results,
            "candidates": candidates,
            "failed": [],
            "invalid": [],
        }

    monkeypatch.setattr(autonomous, "run_research_batch", fake_research_batch)

    result = autonomous.run_autonomous_research(
        object(),
        memory={"family_counts": {}, "normalized_expressions": []},
        max_simulations=8,
        generations=2,
        family_count=2,
        tag="test-auto",
    )

    assert len(calls) == 2
    assert result["summary"]["generations_completed"] == 2
    assert result["summary"]["formally_submitted"] == 0
    assert result["candidates"]
    assert result["candidates"][0]["research_meta"]["generation"] == 2
