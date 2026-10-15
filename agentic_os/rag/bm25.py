"""BM25 keyword index: the lexical half of hybrid retrieval."""

from __future__ import annotations

import math
from collections import Counter

from agentic_os.memory.embeddings import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._docs: dict[str, Counter] = {}
        self._lengths: dict[str, int] = {}
        self._df: Counter = Counter()

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._docs:
            self.remove(doc_id)
        terms = Counter(tokenize(text))
        self._docs[doc_id] = terms
        self._lengths[doc_id] = sum(terms.values())
        self._df.update(terms.keys())

    def remove(self, doc_id: str) -> None:
        terms = self._docs.pop(doc_id, None)
        if terms is None:
            return
        self._lengths.pop(doc_id, None)
        self._df.subtract(terms.keys())

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if not self._docs:
            return []
        n = len(self._docs)
        avg_len = sum(self._lengths.values()) / n or 1.0
        scores: dict[str, float] = {}
        for term in set(tokenize(query)):
            df = self._df.get(term, 0)
            if df <= 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_id, terms in self._docs.items():
                tf = terms.get(term, 0)
                if not tf:
                    continue
                norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self._lengths[doc_id] / avg_len))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * norm
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
