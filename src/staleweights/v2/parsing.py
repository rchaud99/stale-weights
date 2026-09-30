"""Parse a model reply's JSON answer. Moved unchanged from the Mac pilot's `evaluate.py` (branch `old`), so the v2
study scores replies exactly as the pilot did."""

from __future__ import annotations

import ast
import json
import re

ARITHMETIC = re.compile(r"(:\s*)(\(?\s*-?\d[\d\s.+\-*/()]*?)(\s*[,}\]])")


def _arithmetic(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_arithmetic(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a, b = _arithmetic(node.left), _arithmetic(node.right)
        return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}.get(type(node.op)) if not isinstance(
            node.op, ast.Div) else a / b
    raise ValueError(node)


def _evaluate_value(match: re.Match) -> str:
    expr = match.group(2)
    if not re.search(r"\d\s*\)?\s*[-+*/]\s*\(?\s*\d", expr):
        return match.group(0)
    try:
        value = _arithmetic(ast.parse(expr.strip(), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return match.group(0)
    return f"{match.group(1)}{int(value) if float(value).is_integer() else value}{match.group(3)}"


def loads(text: str):
    """json.loads, but a value written as arithmetic ("refund": 1215 - (80 * 3)) is computed first: the models
    sometimes write the sum instead of its result, and that is an answer, not a parse failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(ARITHMETIC.sub(_evaluate_value, text))


def parse(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        answer = loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return answer if isinstance(answer, dict) else None


def number(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        return float(m.group(0)) if m else None
    return None
