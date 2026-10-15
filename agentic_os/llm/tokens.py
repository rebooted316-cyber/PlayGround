"""Token estimation and context-window budgeting.

Roadmap stage 4: models have a finite context window. This module gives a
cheap, dependency-free estimate (about four characters per token for English)
so memory and RAG can stay within budget. Use the provider's token-counting
endpoint when you need exact numbers.
"""

from __future__ import annotations

import math
import re

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Rough token count: the max of chars/4 and word-piece count * 0.75."""
    if not text:
        return 0
    by_chars = len(text) / 4
    by_words = len(_WORD.findall(text)) * 0.75
    return max(1, math.ceil(max(by_chars, by_words)))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Trim ``text`` so its estimate fits ``max_tokens`` (keeps the start)."""
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + " …"
