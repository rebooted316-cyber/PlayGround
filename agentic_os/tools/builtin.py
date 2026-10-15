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
    return [calculator, text_stats, csv_summary, portfolio_metrics, backtest_sma, python_check]


@tool(tags=["quant", "core", "readonly"])
def portfolio_metrics(returns: list[float]) -> dict:
    """Compute basic strategy metrics: total return, Sharpe ratio, max drawdown and win rate.

    Args:
        returns: Per-period returns as decimals, e.g. [0.01, -0.005, 0.02].
    """
    if not returns:
        raise ValueError("returns must be a non-empty list")
    if len(returns) > 10000:
        raise ValueError("returns must have at most 10000 entries")
    vals = [float(r) for r in returns]
    total = 1.0
    for r in vals:
        total *= 1.0 + r
    total_return = total - 1.0
    mean = statistics.fmean(vals)
    stdev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    sharpe = (mean / stdev * math.sqrt(252)) if stdev else 0.0
    peak, max_dd = 1.0, 0.0
    equity = 1.0
    for r in vals:
        equity *= 1.0 + r
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0) if peak else 0.0
    wins = sum(1 for r in vals if r > 0)
    return {
        "periods": len(vals),
        "total_return": round(total_return, 6),
        "mean_per_period": round(mean, 6),
        "sharpe_annualized": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "win_rate": round(wins / len(vals), 4),
    }


@tool(tags=["quant", "code", "core", "readonly"])
def backtest_sma(prices: list[float], fast: int = 10, slow: int = 30) -> dict:
    """Backtest a simple SMA-crossover strategy (long-only, no leverage).

    Buys when the fast SMA crosses above the slow SMA, sells on the opposite
    cross. Pure computation on the given price series; no market access.

    Args:
        prices: Ordered closing prices, oldest first (at least slow+2 points).
        fast: Fast SMA window (default 10).
        slow: Slow SMA window (default 30).
    """
    if not isinstance(fast, int) or not isinstance(slow, int):
        raise TypeError("fast and slow must be integers")
    if not 1 <= fast < slow:
        raise ValueError("require 1 <= fast < slow")
    if len(prices) < slow + 2:
        raise ValueError(f"need at least {slow + 2} prices, got {len(prices)}")
    if len(prices) > 10000:
        raise ValueError("prices must have at most 10000 entries")
    px = [float(p) for p in prices]
    if any(not math.isfinite(p) or p <= 0 for p in px):
        raise ValueError("prices must be positive finite numbers")

    def sma(i: int, window: int) -> float:
        return sum(px[i - window + 1:i + 1]) / window

    position, entry, trades, rets = False, 0.0, 0, []
    for i in range(slow - 1, len(px)):
        if sma(i, fast) > sma(i, slow) and not position:
            position, entry = True, px[i]
        elif sma(i, fast) < sma(i, slow) and position:
            position = False
            trades += 1
            rets.append(px[i] / entry - 1.0)
    if position:
        trades += 1
        rets.append(px[-1] / entry - 1.0)
    buy_hold = px[-1] / px[0] - 1.0
    strat = 1.0
    for r in rets:
        strat *= 1.0 + r
    return {
        "trades": trades,
        "strategy_return": round(strat - 1.0, 6),
        "buy_and_hold_return": round(buy_hold, 6),
        "avg_trade_return": round(sum(rets) / len(rets), 6) if rets else 0.0,
        "fast": fast,
        "slow": slow,
    }


@tool(tags=["code", "core", "readonly"])
def python_check(code: str) -> dict:
    """Syntax-check a Python snippet without executing it (AST parse only).

    Use this before proposing tests or code changes: it reports syntax errors
    with line numbers, plus counts of functions, classes and imports.

    Args:
        code: Python source code to check (max 20000 chars).
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty string")
    if len(code) > 20000:
        raise ValueError("code too long (max 20000 chars)")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"ok": False, "error": f"{exc.msg}", "line": exc.lineno, "column": exc.offset}
    funcs = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
    classes = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
    imports = sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
    return {"ok": True, "functions": funcs, "classes": classes, "imports": imports}
