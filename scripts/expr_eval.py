#!/usr/bin/env python3
"""Safe expression evaluator for report template variables.

Evaluates expressions like:
  {{cost.avg_cost_a}}                   — simple variable lookup
  {{cost.avg_output_b / cost.avg_output_a | .1f}}  — math + format
  {{cost.avg_cost_b * 100 | .0f}}       — arithmetic + format

Supports: +, -, *, /, %, parentheses, abs(), round(), min(), max()

Uses Python's ast module to walk the expression safely. Only whitelisted
node types are evaluated — no imports, attribute access, or function calls
beyond the safe builtins.
"""

import ast
import math
import re


# Format specs:
#   .0f, .1f, .2f   — decimal places
#   ,.0f             — thousands separator
#   $.2f             — currency prefix
#   +.1f             — signed
#   pct              — ×100 + % suffix
#   pct1             — ×100 + % suffix, 1 decimal
#   x.1f             — ratio with × suffix
SAFE_BUILTINS = {"abs": abs, "round": round, "min": min, "max": max}

# Whitelist of AST node types we'll evaluate
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
    ast.Name, ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
    ast.FloorDiv, ast.Pow, ast.USub, ast.UAdd, ast.Attribute,
    ast.Tuple, ast.List,
)


class ExprError(Exception):
    """Raised when an expression cannot be evaluated."""
    pass


def parse_expr(text: str) -> tuple[str, str]:
    """Split '{{expr | format}}' into (expr, format_spec).

    The format spec is optional. If absent, returns empty string.
    """
    text = text.strip()
    if text.startswith("{{") and text.endswith("}}"):
        text = text[2:-2].strip()

    # Split on last unparenthesised |
    depth = 0
    pipe_pos = -1
    for i, ch in enumerate(text):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '|' and depth == 0:
            pipe_pos = i

    if pipe_pos >= 0:
        expr = text[:pipe_pos].strip()
        fmt = text[pipe_pos + 1:].strip()
    else:
        expr = text
        fmt = ""

    return expr, fmt


def _reconstruct_dotted_name(node: ast.AST) -> str | None:
    """Reconstruct a dotted name like 'cost.avg_cost_a' from AST nodes.

    Variable references like cost.avg_cost_a parse as Attribute chains:
      Attribute(value=Name(id='cost'), attr='avg_cost_a')
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _reconstruct_dotted_name(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


def eval_expr(expr_str: str, namespace: dict) -> float | int:
    """Safely evaluate an expression string against a namespace.

    Args:
        expr_str: Expression like 'cost.avg_output_b / cost.avg_output_a'
        namespace: Flat dict mapping dotted names to numeric values.

    Returns:
        Numeric result.

    Raises:
        ExprError: If the expression is unsafe, has unknown variables, etc.
    """
    try:
        tree = ast.parse(expr_str, mode='eval')
    except SyntaxError as e:
        raise ExprError(f"Syntax error in expression: {e}")

    return _eval_node(tree.body, namespace)


def _eval_node(node: ast.AST, namespace: dict):
    """Recursively evaluate an AST node."""
    if not isinstance(node, ALLOWED_NODES):
        raise ExprError(f"Disallowed AST node: {type(node).__name__}")

    # Constants (Python 3.8+)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ExprError(f"Non-numeric constant: {node.value!r}")

    # Num (older Python, kept for compatibility)
    if isinstance(node, ast.Num):
        return node.n

    # Variable lookup — try dotted name first
    if isinstance(node, (ast.Name, ast.Attribute)):
        dotted = _reconstruct_dotted_name(node)
        if dotted is not None:
            if dotted in namespace:
                val = namespace[dotted]
                if not isinstance(val, (int, float)):
                    raise ExprError(f"Variable '{dotted}' is not numeric: {type(val).__name__}")
                return val
            # Check if it's a safe builtin name (for standalone use)
            if isinstance(node, ast.Name) and node.id in SAFE_BUILTINS:
                return SAFE_BUILTINS[node.id]
            raise ExprError(f"Unknown variable: '{dotted}'")
        raise ExprError(f"Cannot resolve attribute chain")

    # Binary operations
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, namespace)
        right = _eval_node(node.right, namespace)
        op = type(node.op)
        if op == ast.Add:
            return left + right
        elif op == ast.Sub:
            return left - right
        elif op == ast.Mult:
            return left * right
        elif op == ast.Div:
            if right == 0:
                raise ExprError("Division by zero")
            return left / right
        elif op == ast.FloorDiv:
            if right == 0:
                raise ExprError("Division by zero")
            return left // right
        elif op == ast.Mod:
            if right == 0:
                raise ExprError("Modulo by zero")
            return left % right
        elif op == ast.Pow:
            return left ** right
        else:
            raise ExprError(f"Unsupported operator: {op.__name__}")

    # Unary operations
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, namespace)
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return +operand
        else:
            raise ExprError(f"Unsupported unary operator: {type(node.op).__name__}")

    # Function calls (only safe builtins)
    if isinstance(node, ast.Call):
        func_name = _reconstruct_dotted_name(node.func)
        if func_name not in SAFE_BUILTINS:
            raise ExprError(f"Disallowed function: '{func_name}'")
        func = SAFE_BUILTINS[func_name]
        args = [_eval_node(a, namespace) for a in node.args]
        if node.keywords:
            raise ExprError("Keyword arguments not supported")
        return func(*args)

    # Tuple/List (for min/max arguments)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval_node(e, namespace) for e in node.elts]

    raise ExprError(f"Cannot evaluate node: {type(node).__name__}")


def format_result(value: float | int, fmt: str) -> str:
    """Format a numeric result according to a format spec.

    Format specs:
        ''       — default format from variables.json (caller handles)
        '.0f'    — 0 decimal places
        '.1f'    — 1 decimal place
        '.2f'    — 2 decimal places
        ',.0f'   — thousands separator, 0 decimals
        '$.2f'   — dollar prefix, 2 decimals
        '+.1f'   — signed, 1 decimal
        'pct'    — ×100, 0 decimal, % suffix
        'pct1'   — ×100, 1 decimal, % suffix
        'x.1f'   — ratio with × suffix
        'pp'     — ×100, 1 decimal, pp suffix, signed
    """
    if not fmt:
        # No format spec — return raw number as string
        if isinstance(value, int):
            return str(value)
        if value == int(value):
            return str(int(value))
        return f"{value:.6g}"

    # Special named formats
    if fmt == "pct":
        return f"{value * 100:.0f}%"
    if fmt == "pct1":
        return f"{value * 100:.1f}%"
    if fmt == "pp":
        pp_val = value * 100
        sign = "+" if pp_val > 0 else ("\u2212" if pp_val < 0 else "")
        return f"{sign}{abs(pp_val):.1f}pp"
    if fmt == "pp_raw":
        sign = "+" if value > 0 else ("\u2212" if value < 0 else "")
        return f"{sign}{abs(value):.1f}pp"

    # x prefix (ratio with ×)
    if fmt.startswith("x"):
        inner_fmt = fmt[1:]
        return f"{value:{inner_fmt}}\u00d7"

    # Dollar prefix
    if fmt.startswith("$"):
        inner_fmt = fmt[1:]
        return f"${value:{inner_fmt}}"

    # Standard Python format spec
    try:
        return f"{value:{fmt}}"
    except (ValueError, TypeError):
        return str(value)


def evaluate(text: str, namespace: dict, default_formats: dict | None = None) -> str:
    """Evaluate a {{expr | format}} expression and return formatted string.

    Args:
        text: Expression string, with or without {{ }} delimiters.
        namespace: Flat dict mapping dotted variable names to numeric values.
        default_formats: Optional dict mapping variable names to default
                        format specs (used when no explicit format given
                        and expression is a simple variable lookup).

    Returns:
        Formatted string result.

    Raises:
        ExprError: If evaluation fails.
    """
    expr_str, fmt = parse_expr(text)
    value = eval_expr(expr_str, namespace)

    # If no explicit format and expression is a simple variable name,
    # use the default format from variables.json
    if not fmt and default_formats and expr_str in default_formats:
        fmt = default_formats[expr_str]

    return format_result(value, fmt)
