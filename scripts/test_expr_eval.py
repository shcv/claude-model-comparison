#!/usr/bin/env python3
"""Tests for expr_eval.py — safe expression evaluator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
import expr_eval
from expr_eval import ExprError, parse_expr, eval_expr, format_result, evaluate


# ── parse_expr ────────────────────────────────────────────────

def test_parse_simple_variable():
    expr, fmt = parse_expr("cost.avg_cost_a")
    assert expr == "cost.avg_cost_a"
    assert fmt == ""


def test_parse_with_format():
    expr, fmt = parse_expr("cost.avg_cost_a | $.2f")
    assert expr == "cost.avg_cost_a"
    assert fmt == "$.2f"


def test_parse_with_braces():
    expr, fmt = parse_expr("{{cost.avg_cost_a | $.2f}}")
    assert expr == "cost.avg_cost_a"
    assert fmt == "$.2f"


def test_parse_math_with_format():
    expr, fmt = parse_expr("cost.avg_output_b / cost.avg_output_a | .1f")
    assert expr == "cost.avg_output_b / cost.avg_output_a"
    assert fmt == ".1f"


def test_parse_no_format():
    expr, fmt = parse_expr("{{cost.avg_cost_a}}")
    assert expr == "cost.avg_cost_a"
    assert fmt == ""


def test_parse_pipe_in_parens():
    # A | inside parentheses should not be treated as format separator
    # (hypothetical, but tests robustness)
    expr, fmt = parse_expr("min(a, b) | .1f")
    assert expr == "min(a, b)"
    assert fmt == ".1f"


# ── eval_expr ─────────────────────────────────────────────────

NAMESPACE = {
    "cost.avg_cost_a": 0.15,
    "cost.avg_cost_b": 0.23,
    "cost.avg_output_a": 3500,
    "cost.avg_output_b": 7200,
    "dataset.total_tasks": 450,
    "dataset.tasks_a": 350,
    "dataset.tasks_b": 100,
    "edits.rewrite_rate_a": 0.045,
    "edits.rewrite_rate_b": 0.032,
}


def test_simple_lookup():
    assert eval_expr("cost.avg_cost_a", NAMESPACE) == 0.15


def test_addition():
    result = eval_expr("cost.avg_cost_a + cost.avg_cost_b", NAMESPACE)
    assert abs(result - 0.38) < 1e-10


def test_subtraction():
    result = eval_expr("cost.avg_cost_b - cost.avg_cost_a", NAMESPACE)
    assert abs(result - 0.08) < 1e-10


def test_multiplication():
    result = eval_expr("cost.avg_cost_a * 100", NAMESPACE)
    assert abs(result - 15.0) < 1e-10


def test_division():
    result = eval_expr("cost.avg_output_b / cost.avg_output_a", NAMESPACE)
    assert abs(result - 7200 / 3500) < 1e-10


def test_parentheses():
    result = eval_expr("(cost.avg_cost_b - cost.avg_cost_a) / cost.avg_cost_a", NAMESPACE)
    expected = (0.23 - 0.15) / 0.15
    assert abs(result - expected) < 1e-10


def test_unary_minus():
    result = eval_expr("-cost.avg_cost_a", NAMESPACE)
    assert result == -0.15


def test_abs_builtin():
    result = eval_expr("abs(cost.avg_cost_a - cost.avg_cost_b)", NAMESPACE)
    assert abs(result - 0.08) < 1e-10


def test_round_builtin():
    result = eval_expr("round(cost.avg_output_b / cost.avg_output_a, 1)", NAMESPACE)
    assert result == round(7200 / 3500, 1)


def test_min_builtin():
    result = eval_expr("min(cost.avg_cost_a, cost.avg_cost_b)", NAMESPACE)
    assert result == 0.15


def test_max_builtin():
    result = eval_expr("max(cost.avg_cost_a, cost.avg_cost_b)", NAMESPACE)
    assert result == 0.23


def test_integer_literal():
    result = eval_expr("dataset.total_tasks", NAMESPACE)
    assert result == 450


def test_numeric_literal():
    result = eval_expr("42", NAMESPACE)
    assert result == 42


def test_float_literal():
    result = eval_expr("3.14", NAMESPACE)
    assert abs(result - 3.14) < 1e-10


def test_division_by_zero():
    with pytest.raises(ExprError, match="Division by zero"):
        eval_expr("cost.avg_cost_a / 0", NAMESPACE)


def test_unknown_variable():
    with pytest.raises(ExprError, match="Unknown variable"):
        eval_expr("nonexistent.var", NAMESPACE)


def test_disallowed_import():
    with pytest.raises(ExprError):
        eval_expr("__import__('os')", NAMESPACE)


def test_disallowed_attribute_access():
    with pytest.raises(ExprError):
        eval_expr("''.__class__", {"": 0})


def test_disallowed_function():
    with pytest.raises(ExprError, match="Disallowed function"):
        eval_expr("eval('1+1')", NAMESPACE)


def test_syntax_error():
    with pytest.raises(ExprError, match="Syntax error"):
        eval_expr("1 +", NAMESPACE)


# ── format_result ─────────────────────────────────────────────

def test_format_default_int():
    assert format_result(42, "") == "42"


def test_format_default_float():
    assert format_result(3.14159, "") == "3.14159"


def test_format_decimal():
    assert format_result(3.14159, ".2f") == "3.14"


def test_format_zero_decimal():
    assert format_result(42.7, ".0f") == "43"


def test_format_thousands():
    assert format_result(1234567, ",.0f") == "1,234,567"


def test_format_currency():
    assert format_result(0.15, "$.2f") == "$0.15"


def test_format_signed():
    assert format_result(3.7, "+.1f") == "+3.7"
    assert format_result(-3.7, "+.1f") == "-3.7"


def test_format_pct():
    assert format_result(0.85, "pct") == "85%"


def test_format_pct1():
    assert format_result(0.856, "pct1") == "85.6%"


def test_format_ratio():
    assert format_result(2.5, "x.1f") == "2.5\u00d7"


def test_format_pp():
    result = format_result(0.05, "pp")
    assert "5.0pp" in result
    assert "+" in result


def test_format_pp_negative():
    result = format_result(-0.03, "pp")
    assert "3.0pp" in result


# ── evaluate (end-to-end) ────────────────────────────────────

def test_evaluate_simple():
    result = evaluate("{{cost.avg_cost_a}}", NAMESPACE,
                      default_formats={"cost.avg_cost_a": "$.2f"})
    assert result == "$0.15"


def test_evaluate_math():
    result = evaluate("{{cost.avg_output_b / cost.avg_output_a | .1f}}", NAMESPACE)
    expected = f"{7200 / 3500:.1f}"
    assert result == expected


def test_evaluate_pct():
    result = evaluate("{{edits.rewrite_rate_a | pct1}}", NAMESPACE)
    assert result == "4.5%"


def test_evaluate_no_braces():
    result = evaluate("cost.avg_cost_a | $.2f", NAMESPACE)
    assert result == "$0.15"


def test_evaluate_complex_expression():
    result = evaluate(
        "{{(cost.avg_cost_b - cost.avg_cost_a) / cost.avg_cost_a * 100 | .0f}}",
        NAMESPACE
    )
    expected = f"{(0.23 - 0.15) / 0.15 * 100:.0f}"
    assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
