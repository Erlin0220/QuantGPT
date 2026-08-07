"""Behavioral tests for the WorldQuant BRAIN operator registry."""

from quantgpt.wq_operator_registry import canonicalize_wq_expression, validate_wq_expression


def test_local_aliases_are_canonicalized_to_brain_operator_names():
    expression = "rank(sign_power(ts_std(close, 20), 0.5) + ts_cov(close, volume, 10))"

    assert canonicalize_wq_expression(expression) == (
        "rank(signed_power(ts_std_dev(close, 20), 0.5) + ts_covariance(close, volume, 10))"
    )


def test_validation_accepts_alias_when_canonical_operator_is_available():
    result = validate_wq_expression(
        "rank(sign_power(close, 0.5))",
        supported_operators={"rank", "signed_power"},
    )

    assert result.ok is True
    assert result.expression == "rank(signed_power(close, 0.5))"
    assert result.unsupported == ()


def test_validation_rejects_local_only_operator_with_brain_hint():
    result = validate_wq_expression(
        "rank(tanh(close))",
        supported_operators={"rank", "signed_power"},
    )

    assert result.ok is False
    assert result.unsupported == ("tanh",)
    assert "signed_power" in result.error


def test_validation_accepts_operator_from_live_brain_catalog():
    result = validate_wq_expression(
        "winsorize(rank(close), 4)",
        supported_operators={"rank", "winsorize"},
    )

    assert result.ok is True
