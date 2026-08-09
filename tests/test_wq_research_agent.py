"""Behavioral tests for the bounded WorldQuant research batch agent."""

from quantgpt.wq_research_agent import run_research_batch


class FakeClient:
    def list_operator_names(self):
        return {"rank", "ts_delta", "signed_power"}

    def simulate(self, expression, **_kwargs):
        good = "10" in expression
        return {
            "ok": True,
            "alpha_id": "good-alpha" if good else "weak-alpha",
            "is": {
                "sharpe": 1.5 if good else 0.8,
                "fitness": 1.2 if good else 0.4,
                "returns": 0.11 if good else 0.04,
                "turnover": 0.32,
                "checks": [
                    {"name": "LOW_SHARPE", "result": "PASS" if good else "FAIL"},
                    {"name": "LOW_FITNESS", "result": "PASS" if good else "FAIL"},
                ],
            },
            "oos": {},
            "settings": {},
            "simulation_id": "sim-1",
        }


def test_research_batch_canonicalizes_dedupes_ranks_and_never_submits():
    result = run_research_batch(
        FakeClient(),
        [
            "rank(-ts_delta(close, 5))",
            "rank(-ts_delta(close, 10))",
            "rank(-ts_delta(close, 10))",
            "rank(sign_power(close, 0.5))",
            "rank(tanh(close))",
        ],
        skip_existing=False,
    )

    assert result["summary"]["requested"] == 5
    assert result["summary"]["duplicates_in_batch"] == 1
    assert result["summary"]["simulated"] == 3
    assert result["summary"]["candidates"] == 1
    assert result["summary"]["formally_submitted"] == 0
    assert result["best"]["alpha_id"] == "good-alpha"
    assert result["candidates"][0]["passes_primary_thresholds"] is True
    assert result["invalid"][0]["expression"] == "rank(tanh(close))"
    assert any("signed_power" in item["expression"] for item in result["results"])


def test_self_correlation_failure_is_blocking():
    class SelfCorrelationClient(FakeClient):
        def simulate(self, expression, **_kwargs):
            result = super().simulate(expression, **_kwargs)
            result["is"]["sharpe"] = 1.8
            result["is"]["fitness"] = 1.4
            result["is"]["checks"].append({"name": "SELF_CORRELATION", "result": "FAIL"})
            return result

    result = run_research_batch(
        SelfCorrelationClient(),
        ["rank(-ts_delta(close, 10))"],
        skip_existing=False,
    )

    assert result["summary"]["candidates"] == 0
    assert result["results"][0]["passes_primary_thresholds"] is False
    assert "reduce_self_correlation" in result["results"][0]["mutation_targets"]
