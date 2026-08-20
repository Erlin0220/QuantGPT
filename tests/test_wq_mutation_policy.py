from quantgpt import wq_autonomous_research as autonomous
from quantgpt.wq_mutation_policy import (
    compile_prevention_rules,
    preferred_mutation_classes,
    summarize_mutation_outcomes,
)
from quantgpt.wq_research_memory import normalize_wq_expression


def _classes(item):
    return [entry["mutation_class"] for entry in preferred_mutation_classes(item)]


def test_high_turnover_routes_to_smoothing_before_conditional_execution():
    classes = _classes({"is_metrics": {"sharpe": 1.4, "fitness": 0.8, "turnover": 0.9}})
    assert classes[:2] == ["widen_windows", "smooth_signal"]
    assert "conditional_execution" in classes


def test_low_fitness_with_acceptable_sharpe_preserves_hypothesis_and_reduces_cost():
    classes = _classes({"is_metrics": {"sharpe": 1.4, "fitness": 0.8, "turnover": 0.3}})
    assert classes[:2] == ["smooth_signal", "cost_reduction"]


def test_correlation_failure_prioritizes_economic_reseed_not_window_tweak():
    classes = _classes({
        "failure_reason": "official_self_correlation",
        "is_metrics": {"sharpe": 1.8, "fitness": 1.3, "turnover": 0.2},
    })
    assert classes[:3] == ["economic_reseed", "operator_family_switch", "neutralization_shift"]
    assert "smooth_signal" not in classes


def test_robustness_failure_prefers_stable_data_then_exposure_controls():
    classes = _classes({"failure_reason": "sub_universe_instability"})
    assert classes == ["stable_data_reseed", "neutralization_shift", "weight_control"]


def test_compile_failure_only_updates_prevention_route():
    assert _classes({"failure_reason": "incompatible_field_operator"}) == ["compile_prevention"]


def test_targeted_mutation_records_parent_failure_and_rationale():
    result = {
        "expression": "rank(ts_mean(returns, 20))",
        "failure_reason": "official_self_correlation",
        "failure_reasons": [{"stage": "diversity", "reason": "official_self_correlation", "source": "check"}],
        "is_metrics": {"sharpe": 1.6, "fitness": 1.2, "turnover": 0.2},
        "research_meta": {"family": "momentum_reversal", "generation": 1, "lineage_id": "parent-lineage"},
    }
    seen = {normalize_wq_expression(result["expression"])}
    children = autonomous.build_targeted_mutations(result, seen=seen, limit=2, hypothesis="diversify")
    assert children
    child = children[0]
    assert child["mutation_type"] == "economic_reseed"
    assert child["family"] != "momentum_reversal"
    assert child["parent_expression"] == result["expression"]
    assert child["parent_lineage_id"] == "parent-lineage"
    assert child["planner_strategy"] == "failure_directed"
    assert "correlation" in child["mutation_reason"]
    assert child["failure_trigger"] == ["official_self_correlation"]
    assert any(item["mutation_type"] == "neutralization_shift" for item in children + autonomous.build_targeted_mutations(result, seen=seen, limit=3, hypothesis="diversify"))


def test_targeted_mutation_dedupes_known_failed_expression():
    result = {
        "expression": "rank(close)",
        "failure_reason": "low_fitness",
        "is_metrics": {"sharpe": 1.4, "fitness": 0.7, "turnover": 0.3},
        "research_meta": {"family": "momentum_reversal", "generation": 1},
    }
    expected = normalize_wq_expression("rank(ts_mean((rank(close)), 10))")
    children = autonomous.build_targeted_mutations(result, seen={expected}, limit=2)
    assert all(normalize_wq_expression(child["expression"]) != expected for child in children)


def test_compile_failures_create_reusable_operator_and_field_prevention_rules():
    rules = compile_prevention_rules([
        {
            "expression": "bad_op(field_x)",
            "failure_reason": "unknown_operator",
            "operator_pattern": "bad_op(*)",
            "data_fields": ["field_x"],
        },
        {
            "expression": "ts_mean(event_field, 20)",
            "failure_reason": "incompatible_field_operator",
            "operator_pattern": "ts_mean(*,#)",
            "data_fields": ["event_field"],
        },
    ])
    assert normalize_wq_expression("bad_op(field_x)") in rules["blocked_exact_expressions"]
    assert "bad_op(*)" in rules["blocked_operator_patterns"]
    assert rules["blocked_operator_field_pairs"] == [
        {"operator_pattern": "ts_mean(*,#)", "data_fields": ["event_field"]}
    ]


def test_mutation_memory_summarizes_rescue_rate():
    summary = summarize_mutation_outcomes([
        {"mutation_type": "smooth_signal", "status": "candidate"},
        {"mutation_type": "smooth_signal", "status": "rejected"},
        {"mutation_type": "economic_reseed", "status": "candidate"},
    ])
    assert summary["smooth_signal"] == {"trials": 2, "rescued": 1, "failed": 1, "rescue_rate": 0.5}
    assert summary["economic_reseed"]["rescue_rate"] == 1.0


def test_prevention_rules_block_known_bad_plan_item():
    rules = {
        "blocked_exact_expressions": [normalize_wq_expression("bad_op(close)")],
        "blocked_operator_patterns": ["bad_op(*)"],
        "blocked_operator_field_pairs": [],
    }
    assert autonomous._violates_prevention_rules({"expression": "bad_op(close)"}, rules) is True
    assert autonomous._violates_prevention_rules({"expression": "rank(close)"}, rules) is False
