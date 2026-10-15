"""Built-in tools available to every agent."""

from __future__ import annotations

import ast
import csv
import io
import math
import operator
import statistics

from agentic_os.tools.registry import Tool, tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str) -> float | int:
    """Evaluate an arithmetic expression without ``eval`` (AST whitelist)."""

    def ev(node: ast.AST):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"Unsupported expression element: {ast.dump(node)[:60]}")

    if len(expression) > 500:
        raise ValueError("expression too long")
    return ev(ast.parse(expression.replace("^", "**"), mode="eval"))


@tool(tags=["math", "core", "readonly"])
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Use this for any math instead of mental arithmetic.

    Args:
        expression: Arithmetic expression, e.g. "(17 * 23) / 4" or "sqrt(2) * pi".
    """
    result = safe_eval(expression)
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    return f"{expression} = {result}"


@tool(tags=["text", "core", "readonly"])
def text_stats(text: str) -> dict:
    """Count characters, words, sentences and lines in a piece of text.

    Args:
        text: The text to analyse.
    """
    words = text.split()
    sentences = [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "lines": text.count("\n") + 1 if text else 0,
    }


@tool(tags=["data", "core", "readonly"])
def csv_summary(csv_text: str) -> dict:
    """Summarize CSV data: row count, and per-column type plus statistics (min/max/mean for numbers, top values otherwise).

    Args:
        csv_text: Raw CSV content including a header row.
    """
    rows = list(csv.DictReader(io.StringIO(csv_text.strip())))
    summary: dict = {"rows": len(rows), "columns": {}}
    if not rows:
        return summary
    for col in rows[0]:
        values = [r[col] for r in rows if r.get(col) not in (None, "")]
        try:
            nums = [float(v) for v in values]
        except ValueError:
            nums = None
        if nums:
            summary["columns"][col] = {
                "type": "number",
                "count": len(nums),
                "min": min(nums),
                "max": max(nums),
                "mean": round(statistics.fmean(nums), 4),
                "sum": round(sum(nums), 4),
            }
        else:
            counts: dict[str, int] = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1
            top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            summary["columns"][col] = {"type": "text", "count": len(values), "unique": len(counts), "top": top}
    return summary


def builtin_tools() -> list[Tool]:
    return [calculator, text_stats, csv_summary]
