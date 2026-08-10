"""Tests for autonomous WorldQuant research planning and bounded evolution."""

from quantgpt import wq_autonomous_research as autonomous
from quantgpt.wq_research_memory import normalize_wq_expression


def test_family_selector_uses_only_confidence_gated_points_feedback():
    family_counts = {family: 10 for family in autonomous.FAMILY_SEEDS}
    memory = {
        "family_counts": family_counts,
        "family_points_feedback": {"price_volume": 10000.0},
        "family_points_feedback_usable": {"analyst_revision": 1500.0},
    }

    selected = autonomous.select_research_families(memory, count=1)

    assert selected == ["analyst_revision"]


def test_family_selector_ignores_raw_points_feedback_when_gate_has_no_usable_evidence():
    family_counts = {family: 10 for family in autonomous.FAMILY_SEEDS}
    selected = autonomous.select_research_families(
        {
            "family_counts": family_counts,
            "family_points_feedback": {"price_volume": 100000.0},
            "family_points_feedback_usable": {},
        },
        count=1,
    )

    assert selected == [sorted(autonomous.FAMILY_SEEDS)[0]]


def test_family_selector_prefers_undercovered_family():
    memory = {
        "family_counts": {
            family: (0 if family == "analyst_revision" else 20)
            for family in autonomous.FAMILY_SEEDS
        }
    }

    selected = autonomous.select_research_families(memory, count=2)

    assert selected[0] == "analyst_revision"


def test_family_selector_exploits_family_with_candidate_and_near_threshold_trials():
    memory = {
        "family_counts": {family: 20 for family in autonomous.FAMILY_SEEDS},
        "candidate_family_counts": {"price_volume": 1},
        "recent_trials": [
            {
                "family": "price_volume",
                "sharpe": 1.2,
                "fitness": 0.95,
                "turnover": 0.2,
            },
            {
                "family": "fundamental_quality",
                "sharpe": 0.3,
                "fitness": 0.2,
                "turnover": 0.2,
            },
        ],
    }

    selected = autonomous.select_research_families(memory, count=1)

    assert selected == ["price_volume"]


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


def test_seed_plan_expands_curated_catalog_after_raw_seeds_are_exhausted():
    seeds = autonomous.FAMILY_SEEDS["price_volume"]
    memory = {
        "family_counts": {family: 50 for family in autonomous.FAMILY_SEEDS},
        "normalized_expressions": [normalize_wq_expression(item) for item in seeds],
    }
    memory["family_counts"]["price_volume"] = 0

    plan = autonomous.build_seed_plan(memory, limit=3, family_count=1)

    assert len(plan) == 3
    assert all(item["family"] == "price_volume" for item in plan)
    assert all(item["mutation_type"].startswith("seed_") for item in plan)
    assert all(item["parent_expression"] in seeds for item in plan)


def test_seed_plan_skips_fields_brain_reported_as_unknown():
    memory = {
        "family_counts": {family: 50 for family in autonomous.FAMILY_SEEDS},
        "recent_trials": [
            {
                "family": "fundamental_quality",
                "status": "simulation_failed",
                "mutation_targets": [
                    'simulation_failed:WQ simulation failed: Attempted to use unknown variable "mdf_quality".'
                ],
            },
            {
                "family": "fundamental_quality",
                "status": "simulation_failed",
                "mutation_targets": [
                    'simulation_failed:WQ simulation failed: Attempted to use unknown variable "mdf_bp".'
                ],
            },
        ],
    }
    memory["family_counts"]["fundamental_quality"] = 0

    plan = autonomous.build_seed_plan(memory, limit=4, family_count=1)

    assert plan
    assert all("mdf_quality" not in item["expression"] for item in plan)
    assert all("mdf_bp" not in item["expression"] for item in plan)


def test_active_motif_plan_transfers_proven_structure_to_different_live_field():
    class FakeClient:
        pass

    active = [
        {
            "alpha_id": "active-1",
            "status": "ACTIVE",
            "expression": "ts_decay_linear(rank(ts_mean(ts_backfill(old_signal, 60), 20)) * rank(-returns), 5)",
            "sharpe": 1.9,
            "fitness": 1.3,
            "returns": 0.12,
            "turnover": 0.2,
        }
    ]
    fields = [
        {"id": "fresh_signal", "type": "MATRIX", "dataset": {"id": "news12", "category": {"id": "news"}}}
    ]
    seen = {normalize_wq_expression(active[0]["expression"])}

    plan = autonomous.build_active_motif_plan(
        FakeClient(),
        active,
        fields,
        seen=seen,
        limit=2,
        hypothesis="replenish",
    )

    assert plan
    assert all(item["planner_strategy"] == "active_motif_transfer" for item in plan)
    assert all("fresh_signal" in item["expression"] for item in plan)
    assert all("old_signal" not in item["expression"] for item in plan)


def test_platform_near_miss_plan_prioritizes_repairable_alpha_and_skips_weak_trials():
    near = {
        "alpha_id": "near-1",
        "status": "UNSUBMITTED",
        "expression": "-1 * ts_mean(ts_delta(close, 1), 3) / ts_std_dev(close, 20)",
        "sharpe": 1.63,
        "fitness": 0.85,
        "returns": 0.14,
        "turnover": 0.72,
    }
    weak = {
        "alpha_id": "weak-1",
        "status": "UNSUBMITTED",
        "expression": "rank(ts_mean(volume, 20))",
        "sharpe": 0.5,
        "fitness": 0.2,
        "returns": 0.01,
        "turnover": 0.2,
    }
    seen = {normalize_wq_expression(near["expression"]), normalize_wq_expression(weak["expression"])}

    plan = autonomous.build_platform_near_miss_plan(
        [near, weak],
        seen=seen,
        limit=2,
        hypothesis="repair near misses",
    )

    assert plan
    assert all(item["planner_strategy"] == "platform_near_miss" for item in plan)
    assert all(item["parent_expression"] == near["expression"] for item in plan)
    assert all("near-1" in item["mutation_reason"] for item in plan)


def test_active_reference_score_prefers_candidate_near_real_active_quality():
    profile = autonomous._active_reference_profile(
        [
            {"sharpe": 1.9, "fitness": 1.3, "returns": 0.12, "turnover": 0.2},
            {"sharpe": 1.6, "fitness": 1.2, "returns": 0.10, "turnover": 0.15},
        ]
    )
    strong = {"is_metrics": {"sharpe": 1.8, "fitness": 1.25, "returns": 0.11, "turnover": 0.2}}
    weak = {"is_metrics": {"sharpe": 1.1, "fitness": 0.8, "returns": 0.03, "turnover": 0.9}}

    assert autonomous._active_reference_score(strong, profile) > autonomous._active_reference_score(weak, profile)


def test_live_dataset_selection_uses_points_feedback_as_weak_tiebreaker():
    datasets = [
        {"id": "news_a", "category": {"id": "news"}},
        {"id": "news_b", "category": {"id": "news"}},
        {"id": "pv1", "category": {"id": "pv"}},
    ]
    memory = {
        "dataset_points_feedback": {"news_b": 1500.0},
        "learning_maturity": {"points_planner_weighting": {"ready": True}},
    }

    selected = autonomous._select_live_datasets(datasets, memory, limit=2)

    ids = [item["id"] for item in selected]
    assert "news_b" in ids
    assert "news_a" not in ids
    assert "pv1" in ids


def test_seed_plan_softly_avoids_repeated_negative_structural_motif():
    memory = {
        "family_counts": {family: 100 for family in autonomous.FAMILY_SEEDS if family != "price_volume"},
        "research_memory_guidance": {
            "negative": [
                {
                    "family": "price_volume",
                    "operator_pattern": "rank>ts_decay_linear",
                    "failure_reason": "low_fitness",
                    "count": 40,
                }
            ],
            "overused_structures": [{"operator_pattern": "rank>ts_decay_linear", "trials": 80}],
        },
    }

    plan = autonomous.build_seed_plan(memory, limit=2, family_count=1)

    assert plan
    assert all("ts_decay_linear" not in item["expression"] for item in plan)


def test_live_dataset_selection_spreads_across_categories_and_avoids_overused_dataset():
    datasets = [
        {"id": "fundamental2", "category": {"id": "fundamental"}},
        {"id": "fundamental6", "category": {"id": "fundamental"}},
        {"id": "news12", "category": {"id": "news"}},
        {"id": "option8", "category": {"id": "option"}},
        {"id": "pv1", "category": {"id": "pv"}},
    ]
    memory = {"recent_trials": [{"dataset_id": "fundamental2"}] * 10}

    selected = autonomous._select_live_datasets(datasets, memory, limit=4)

    ids = [item["id"] for item in selected]
    categories = {autonomous._dataset_category(item) for item in selected}
    assert "fundamental2" not in ids
    assert "fundamental6" in ids
    assert len(categories) == 4


def test_live_field_candidates_round_robin_across_selected_datasets():
    class FakeClient:
        def list_datasets(self, **_kwargs):
            return [
                {"id": "analyst4", "category": {"id": "analyst"}},
                {"id": "news12", "category": {"id": "news"}},
                {"id": "pv1", "category": {"id": "pv"}},
            ]

        def list_data_fields(self, dataset_id=None, **_kwargs):
            prefix = {"analyst4": "analyst", "news12": "news", "pv1": "pv"}[dataset_id]
            return [
                {"id": f"{prefix}_field_a", "type": "MATRIX", "dataset": {"id": dataset_id}},
                {"id": f"{prefix}_field_b", "type": "MATRIX", "dataset": {"id": dataset_id}},
            ]

    fields, catalog = autonomous._live_field_candidates(
        FakeClient(), {}, region="USA", universe="TOP3000", delay=1, limit=3
    )

    assert catalog["selected_datasets"] == ["analyst4", "news12", "pv1"]
    assert [item["dataset"]["id"] for item in fields[:3]] == ["analyst4", "news12", "pv1"]


def test_live_field_plan_prefers_matrix_fields_and_skips_unknown_memory_fields():
    class FakeClient:
        def list_data_fields(self, **_kwargs):
            return [
                {"id": "fresh_quality", "type": "MATRIX", "description": "company quality score", "dataset": {"id": "fundamentalX"}},
                {"id": "bad_field", "type": "MATRIX", "description": "bad"},
                {"id": "vector_field", "type": "VECTOR", "description": "vector"},
            ]

        def list_operator_names(self):
            return {"rank", "ts_mean", "ts_backfill", "ts_delta", "ts_std_dev"}

    memory = {
        "recent_trials": [
            {
                "expression": "rank(bad_field)",
                "mutation_targets": ['simulation_failed:Attempted to use unknown variable "bad_field".'],
            }
        ]
    }

    plan, catalog, fields = autonomous.build_live_field_plan(
        FakeClient(),
        memory,
        seen=set(),
        limit=3,
        region="USA",
        universe="TOP3000",
        delay=1,
        hypothesis="discover live fields",
    )

    assert catalog["available"] is True
    assert len(fields) == 1
    assert len(plan) == 3
    assert all("fresh_quality" in item["expression"] for item in plan)
    assert all(item["mutation_type"] == "live_field_seed" for item in plan)
    assert all(item["family"] == "fundamental_quality" for item in plan)


def test_structural_live_plan_uses_cross_field_motifs_from_same_dataset():
    class FakeClient:
        def list_operator_names(self):
            return {"rank", "ts_corr", "ts_backfill", "group_rank", "ts_mean", "ts_delta"}

    fields = [
        {"id": "field_a", "type": "MATRIX", "description": "analyst estimate", "dataset": {"id": "analyst4", "category": {"id": "analyst"}}},
        {"id": "field_b", "type": "MATRIX", "description": "analyst revision", "dataset": {"id": "analyst4", "category": {"id": "analyst"}}},
    ]

    plan = autonomous.build_structural_live_plan(
        FakeClient(),
        fields,
        seen=set(),
        limit=3,
        hypothesis="cross-field diversity",
    )

    assert len(plan) == 3
    assert all(item["mutation_type"] == "structural_live_seed" for item in plan)
    assert all(item["dataset_id"] == "analyst4" for item in plan)
    assert all(item["data_fields"] == ["field_a", "field_b"] for item in plan)
    assert len({item["expression"] for item in plan}) == 3
    assert any("ts_corr" in item["expression"] for item in plan)


def test_robustness_failure_reduces_cross_run_parent_promise():
    base = {"sharpe": 1.27, "fitness": 1.14, "turnover": 0.2}
    robust_failure = {
        **base,
        "failure_reason": "robustness_instability",
        "mutation_targets": ["improve_cross_setting_robustness"],
    }

    assert autonomous._trial_promise_score(robust_failure) < autonomous._trial_promise_score(base)


def test_llm_live_plan_rejects_invented_fields_and_keeps_catalog_expression(monkeypatch):
    from quantgpt import iteration

    class FakeClient:
        def list_operator_names(self):
            return {"rank", "ts_mean", "ts_backfill"}

    fields = [{"id": "real_field", "type": "MATRIX", "description": "analyst revision signal"}]
    generated = iter(["rank(fake_field)", "rank(ts_mean(ts_backfill(real_field, 60), 20))"])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(iteration, "_call_llm", lambda *_args, **_kwargs: next(generated))

    plan = autonomous.build_llm_live_plan(
        FakeClient(),
        fields,
        seen=set(),
        limit=2,
        hypothesis="new hypothesis",
    )

    assert len(plan) == 1
    assert "real_field" in plan[0]["expression"]
    assert plan[0]["mutation_type"] == "llm_live_field_seed"
    assert plan[0]["data_fields"] == ["real_field"]


def test_llm_live_plan_receives_bounded_positive_negative_memory(monkeypatch):
    from quantgpt import iteration

    class FakeClient:
        def list_operator_names(self):
            return {"rank"}

    prompts = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(iteration, "_call_llm", lambda _system, user, **_kwargs: prompts.append(user) or "rank(real_field)")
    memory = {
        "research_memory_guidance": {
            "positive": [{"family": "analyst_revision", "dataset_id": "analyst4", "operator_pattern": "rank>ts_mean"}],
            "negative": [{"family": "price_volume", "dataset_id": "pv1", "operator_pattern": "rank>ts_decay_linear", "failure_reason": "low_fitness", "count": 20}],
            "overused_structures": [{"operator_pattern": "rank>ts_decay_linear", "trials": 80}],
        }
    }
    fields = [{"id": "real_field", "type": "MATRIX", "dataset": {"id": "analyst4", "category": {"id": "analyst"}}}]

    plan = autonomous.build_llm_live_plan(FakeClient(), fields, seen=set(), limit=1, hypothesis="memory-guided", memory=memory)

    assert len(plan) == 1
    assert "analyst_revision|analyst4|rank>ts_mean" in prompts[0]
    assert "price_volume|pv1|rank>ts_decay_linear|low_fitness" in prompts[0]
    assert plan[0]["dataset_id"] == "analyst4"
    assert plan[0]["provenance_state"] == "resolved"


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
                "turnover": 0.8,
                "mutation_targets": ["reduce_turnover", "improve_signal_sharpe"],
            }
        ]
    }

    plan = autonomous.build_memory_mutation_plan(memory, seen=set(), limit=2, hypothesis="continue")

    assert len(plan) == 2
    assert all(item["parent_expression"] == parent for item in plan)
    assert all(item["generation"] == 2 for item in plan)
    assert {item["mutation_type"] for item in plan} == {"widen_windows", "smooth_signal"}


def test_targeted_mutations_follow_diagnostics_and_preserve_lineage(monkeypatch):
    expression = "rank(ts_mean(returns, 10))"
    result = {
        "expression": expression,
        "is_metrics": {"sharpe": -0.4, "fitness": 0.3, "turnover": 0.8},
        "mutation_targets": ["reduce_turnover", "improve_signal_sharpe"],
        "research_meta": {
            "family": "momentum_reversal",
            "generation": 1,
            "hypothesis": "test hypothesis",
        },
    }

    monkeypatch.setenv("WQ_MUTATION_MAX_CHILDREN", "3")
    mutations = autonomous.build_targeted_mutations(result, seen=set(), limit=3)

    assert any(item["mutation_type"] == "widen_windows" for item in mutations)
    assert any(item["mutation_type"] == "invert_signal" for item in mutations)
    assert all(item["generation"] == 2 for item in mutations)
    assert all(item["parent_expression"] == expression for item in mutations)


def test_knowledge_backed_high_turnover_prioritizes_cost_rescue():
    card_id = "ab65eb3e-ec5e-4c94-ac8c-6aff83566112"
    expression = "-1 * ts_delta(close, 1) / ts_std_dev(close, 20)"
    result = {
        "expression": expression,
        "is_metrics": {"sharpe": 1.65, "fitness": 0.48, "turnover": 1.4575},
        "mutation_targets": ["improve_fitness", "reduce_turnover"],
        "research_meta": {
            "family": "momentum_reversal",
            "generation": 1,
            "knowledge_card_ids": [card_id],
            "knowledge_mutation_strategies": [
                "Add a liquidity filter using volume or market cap to focus on liquid stocks.",
                "Combine reversal signal with volatility scaling to normalize risk.",
            ],
        },
    }

    mutations = autonomous.build_targeted_mutations(result, seen=set(), limit=2)

    assert [item["mutation_type"] for item in mutations] == [
        "knowledge_decay_smoothing",
        "knowledge_short_smoothing",
    ]
    assert all(item["knowledge_card_ids"] == [card_id] for item in mutations)
    assert "ts_decay_linear" in mutations[0]["expression"]
    assert "ts_mean" in mutations[1]["expression"]


def test_memory_plan_rehydrates_knowledge_guidance_for_cost_rescue():
    card_id = "ab65eb3e-ec5e-4c94-ac8c-6aff83566112"
    expression = "-1 * ts_delta(close, 1) / ts_std_dev(close, 20)"
    memory = {
        "recent_trials": [
            {
                "expression": expression,
                "family": "momentum_reversal",
                "generation": 1,
                "fitness": 0.48,
                "sharpe": 1.65,
                "turnover": 1.4575,
                "mutation_targets": ["improve_fitness", "reduce_turnover"],
                "failure_reason": "low_fitness",
                "knowledge_card_ids": [card_id],
            }
        ],
        "knowledge_guidance": {
            "cards": [
                {
                    "id": card_id,
                    "failure_modes": ["High transaction costs can eliminate profits."],
                    "mutation_strategies": ["Reduce trading intensity while preserving the hypothesis."],
                }
            ]
        },
    }

    plan = autonomous.build_memory_mutation_plan(memory, seen=set(), limit=2, hypothesis="continue")

    assert [item["mutation_type"] for item in plan] == [
        "knowledge_decay_smoothing",
        "knowledge_short_smoothing",
    ]
    assert all(item["knowledge_card_ids"] == [card_id] for item in plan)
    assert all("knowledge guidance" in item["mutation_reason"] for item in plan)


def test_memory_plan_reserves_first_slot_for_viable_knowledge_followup():
    card_id = "ab65eb3e-ec5e-4c94-ac8c-6aff83566112"
    knowledge_parent = "rank(ts_decay_linear((-1 * ts_delta(close, 1) / ts_std_dev(close, 20)), 5))"
    generic_parent = "group_rank((-rank(ts_std_dev(ts_backfill(accumulated_amortization_customer_intangibles, 60), 20))), subindustry)"
    memory = {
        "recent_trials": [
            {
                "expression": generic_parent,
                "family": "fundamental_quality",
                "generation": 1,
                "fitness": 0.97,
                "sharpe": 1.46,
                "turnover": 0.2,
                "mutation_targets": ["improve_fitness"],
            },
            {
                "expression": knowledge_parent,
                "family": "momentum_reversal",
                "generation": 2,
                "fitness": 0.85,
                "sharpe": 1.84,
                "turnover": 0.7267,
                "mutation_targets": ["improve_fitness", "reduce_turnover"],
                "failure_reason": "low_fitness",
                "knowledge_card_ids": [card_id],
            },
        ],
        "knowledge_guidance": {
            "cards": [
                {
                    "id": card_id,
                    "failure_modes": ["High transaction costs can eliminate profits."],
                    "mutation_strategies": ["Reduce trading intensity while preserving the hypothesis."],
                }
            ]
        },
    }

    plan = autonomous.build_memory_mutation_plan(memory, seen=set(), limit=1, hypothesis="continue")

    assert len(plan) == 1
    assert plan[0]["parent_expression"] == knowledge_parent
    assert plan[0]["mutation_type"] == "knowledge_decay_step"
    assert plan[0]["knowledge_card_ids"] == [card_id]


def test_knowledge_decay_parent_steps_decay_without_changing_signal_lookbacks():
    card_id = "ab65eb3e-ec5e-4c94-ac8c-6aff83566112"
    expression = "rank(ts_decay_linear((-1 * ts_delta(close, 1) / ts_std_dev(close, 20)), 5))"
    result = {
        "expression": expression,
        "is_metrics": {"sharpe": 1.84, "fitness": 0.85, "turnover": 0.7267},
        "mutation_targets": ["improve_fitness", "reduce_turnover"],
        "research_meta": {
            "family": "momentum_reversal",
            "generation": 2,
            "knowledge_card_ids": [card_id],
            "knowledge_mutation_strategies": ["Reduce trading intensity while preserving the hypothesis."],
        },
    }

    mutations = autonomous.build_targeted_mutations(result, seen=set(), limit=2)

    assert mutations[0]["mutation_type"] == "knowledge_decay_step"
    assert mutations[0]["expression"] == "rank(ts_decay_linear((-1 * ts_delta(close, 1) / ts_std_dev(close, 20)), 10))"
    assert "ts_std_dev(close, 20)" in mutations[0]["expression"]
    assert mutations[1]["mutation_type"] == "knowledge_post_decay_smoothing"
    assert all(item["knowledge_card_ids"] == [card_id] for item in mutations)


def test_generation_three_knowledge_parent_is_capped_and_defers_to_native_parameter_rescue():
    card_id = "ab65eb3e-ec5e-4c94-ac8c-6aff83566112"
    expression = "rank(ts_decay_linear((-1 * ts_delta(close, 1) / ts_std_dev(close, 20)), 10))"
    result = {
        "expression": expression,
        "is_metrics": {"sharpe": 1.59, "fitness": 0.81, "turnover": 0.5295},
        "mutation_targets": ["improve_fitness"],
        "failure_reason": "low_fitness",
        "research_meta": {
            "family": "momentum_reversal",
            "generation": 3,
            "knowledge_card_ids": [card_id],
            "knowledge_mutation_strategies": ["Reduce trading intensity while preserving the hypothesis."],
        },
    }

    mutations = autonomous.build_targeted_mutations(result, seen=set(), limit=2)

    assert mutations == []
    assert result["mutation_route_terminal_reason"] == "mutation_generation_budget_exhausted"


def test_native_decay_rescue_selects_best_knowledge_parent_and_skips_tested_values():
    expression = "rank(ts_decay_linear((-1 * ts_delta(close, 1) / ts_std_dev(close, 20)), 5))"
    memory = {
        "recent_trials": [
            {
                "alpha_id": "base-0",
                "expression": expression,
                "settings": {"decay": 0},
                "family": "momentum_reversal",
                "sharpe": 1.84,
                "fitness": 0.85,
                "returns": 0.15,
                "turnover": 0.7267,
                "knowledge_card_ids": ["card-1"],
            },
            {
                "alpha_id": "base-2",
                "expression": expression,
                "settings": {"decay": 2},
                "family": "momentum_reversal",
                "sharpe": 1.74,
                "fitness": 0.88,
                "returns": 0.149,
                "turnover": 0.5821,
                "knowledge_card_ids": ["card-1"],
            },
            {
                "alpha_id": "generic",
                "expression": "rank(close)",
                "settings": {"decay": 0},
                "sharpe": 2.0,
                "fitness": 0.95,
                "returns": 0.2,
                "turnover": 0.2,
                "knowledge_card_ids": [],
            },
        ]
    }

    parent, pending = autonomous._select_native_decay_rescue_parent(
        memory,
        min_sharpe=1.25,
        min_fitness=1.0,
        current_decay=0,
    )

    assert parent is not None
    assert parent["alpha_id"] == "base-2"
    assert pending == [4]


def test_native_decay_rescue_preserves_knowledge_lineage_and_diagnoses_result():
    class FakeClient:
        def list_operator_names(self):
            return {"rank", "ts_decay_linear", "ts_delta", "ts_std_dev"}

        def simulate(self, expression, **kwargs):
            decay = int(kwargs["decay"])
            return {
                "ok": True,
                "alpha_id": f"alpha-d{decay}",
                "is": {
                    "sharpe": 1.7,
                    "fitness": 0.9 if decay == 2 else 0.8,
                    "returns": 0.14,
                    "turnover": 0.58 if decay == 2 else 0.46,
                    "checks": [{"name": "LOW_FITNESS", "result": "FAIL"}],
                },
                "oos": {},
                "settings": {"decay": decay},
                "simulation_id": f"sim-d{decay}",
            }

    parent = {
        "alpha_id": "parent-alpha",
        "expression": "rank(ts_decay_linear((-1 * ts_delta(close, 1) / ts_std_dev(close, 20)), 5))",
        "family": "momentum_reversal",
        "generation": 2,
        "knowledge_card_ids": ["card-1"],
    }

    results = autonomous._run_native_decay_rescue(
        FakeClient(),
        parent,
        [2, 4],
        goal="rescue",
        tag="test",
        region="USA",
        universe="TOP3000",
        delay=1,
        neutralization="SUBINDUSTRY",
        truncation=0.08,
        min_sharpe=1.25,
        min_fitness=1.0,
    )

    assert [item["settings"]["decay"] for item in results] == [2, 4]
    assert all(item["research_meta"]["knowledge_card_ids"] == ["card-1"] for item in results)
    assert all(item["research_meta"]["planner_strategy"] == "knowledge_native_parameter_rescue" for item in results)
    assert all(item["mutation_targets"] == ["improve_fitness"] for item in results)
    assert all(item["passes_primary_thresholds"] is False for item in results)


def test_generation_three_generic_parent_remains_capped():
    result = {
        "expression": "rank(ts_mean(returns, 10))",
        "is_metrics": {"sharpe": 1.4, "fitness": 0.85, "turnover": 0.3},
        "mutation_targets": ["improve_fitness"],
        "research_meta": {"family": "momentum_reversal", "generation": 3},
    }

    mutations = autonomous.build_targeted_mutations(result, seen=set(), limit=2)

    assert mutations == []
    assert result["mutation_route_terminal_reason"] == "mutation_generation_budget_exhausted"


def test_positive_weak_signal_is_smoothed_not_inverted():
    result = {
        "expression": "rank(ts_mean(returns, 10))",
        "is_metrics": {"sharpe": 0.9, "fitness": 0.6, "turnover": 0.2},
        "mutation_targets": ["improve_signal_sharpe", "improve_fitness"],
        "research_meta": {"family": "momentum_reversal", "generation": 1},
    }

    mutations = autonomous.build_targeted_mutations(result, seen=set(), limit=2)

    assert mutations
    assert mutations[0]["mutation_type"] == "smooth_signal"
    assert all(item["mutation_type"] != "invert_signal" for item in mutations)


def test_memory_plan_skips_very_weak_parent():
    memory = {
        "recent_trials": [
            {
                "expression": "rank(close)",
                "family": "momentum_reversal",
                "generation": 1,
                "fitness": 0.05,
                "sharpe": 0.1,
                "turnover": 0.2,
                "mutation_targets": ["improve_signal_sharpe", "improve_fitness"],
            }
        ]
    }

    assert autonomous.build_memory_mutation_plan(memory, seen=set(), limit=2) == []


def test_next_generation_diversifies_parents_before_second_child():
    ranked = [
        {
            "expression": "rank(ts_mean(returns, 10))",
            "is_metrics": {"sharpe": 0.9, "fitness": 0.7, "turnover": 0.2},
            "mutation_targets": ["improve_signal_sharpe"],
            "research_meta": {"family": "momentum_reversal", "generation": 1},
        },
        {
            "expression": "rank(ts_mean(volume, 10))",
            "is_metrics": {"sharpe": 0.8, "fitness": 0.7, "turnover": 0.2},
            "mutation_targets": ["improve_signal_sharpe"],
            "research_meta": {"family": "price_volume", "generation": 1},
        },
    ]

    plan = autonomous._build_next_generation_plan(ranked, seen=set(), limit=2, hypothesis="diverse")

    assert len(plan) == 2
    assert {item["parent_expression"] for item in plan} == {item["expression"] for item in ranked}


def test_candidate_robustness_requires_cross_setting_support(monkeypatch):
    monkeypatch.setattr(
        autonomous,
        "run_batch_simulation",
        lambda *_args, **_kwargs: {
            "ok": True,
            "sub_results": {
                "a": {"status": "completed", "sharpe": 1.2, "fitness": 0.9, "returns": 0.05, "turnover": 0.2},
                "b": {"status": "completed", "sharpe": 1.1, "fitness": 0.8, "returns": 0.03, "turnover": 0.25},
            },
        },
    )

    result = autonomous.validate_candidate_robustness(
        object(),
        {"expression": "rank(close)"},
        region="USA",
        universe="TOP3000",
        delay=1,
        decay=0,
        neutralization="SUBINDUSTRY",
        truncation=0.08,
    )

    assert result["status"] == "ready"
    assert result["robustness_score"] == 1.0
    assert result["validation_simulations"] == 2


def test_candidate_robustness_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(
        autonomous,
        "run_batch_simulation",
        lambda *_args, **_kwargs: {
            "ok": True,
            "sub_results": {
                "a": {"status": "completed", "sharpe": 0.4, "fitness": 0.3, "returns": -0.02, "turnover": 0.2},
                "b": {"status": "completed", "sharpe": 0.6, "fitness": 0.4, "returns": 0.01, "turnover": 0.2},
            },
        },
    )

    result = autonomous.validate_candidate_robustness(
        object(),
        {"expression": "rank(close)"},
        region="USA",
        universe="TOP3000",
        delay=1,
        decay=0,
        neutralization="SUBINDUSTRY",
        truncation=0.08,
    )

    assert result["status"] == "robustness_fail"
    assert result["robustness_score"] == 0.0


def test_autonomous_research_routes_robustness_failure_into_same_round_generation(monkeypatch):
    monkeypatch.setattr(autonomous, "run_list_alphas", lambda *_args, **_kwargs: {"ok": True, "alphas": []})
    monkeypatch.setattr(
        autonomous,
        "build_live_field_plan",
        lambda *_args, **_kwargs: ([], {"available": False, "count": 0}, []),
    )
    monkeypatch.setattr(autonomous, "build_structural_live_plan", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(autonomous, "build_llm_live_plan", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        autonomous,
        "validate_candidate_robustness",
        lambda *_args, **_kwargs: {"status": "robustness_fail", "robustness_score": 0.0, "validation_simulations": 2},
    )

    calls = []

    def fake_research_batch(_client, expressions, **kwargs):
        generation = 2 if kwargs["tag"].endswith("g2") else 1
        calls.append((generation, list(expressions)))
        results = []
        candidates = []
        for index, expression in enumerate(expressions):
            primary = generation == 1 and index == 0
            item = {
                "ok": True,
                "alpha_id": f"robust-g{generation}-{index}",
                "expression": expression,
                "is_metrics": {
                    "sharpe": 1.3 if primary else 0.8,
                    "fitness": 1.1 if primary else 0.5,
                    "returns": 0.08 if primary else 0.02,
                    "turnover": 0.2,
                    "checks": [],
                },
                "passes_primary_thresholds": primary,
                "mutation_targets": ["candidate_passes_primary_thresholds"] if primary else ["improve_signal_sharpe"],
            }
            results.append(item)
            if primary:
                candidates.append(item)
        return {
            "ok": True,
            "tag": kwargs["tag"],
            "settings": {},
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
        tag="same-round-robustness",
    )

    assert [generation for generation, _ in calls] == [1, 2]
    assert result["summary"]["same_round_robustness_feedback"] == 1
    generation_two = [item for item in result["results"] if (item.get("research_meta") or {}).get("generation") == 2]
    assert generation_two
    assert any((item.get("research_meta") or {}).get("mutation_type") == "stable_data_reseed" for item in generation_two)


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
