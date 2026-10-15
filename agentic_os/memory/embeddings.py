"""Text embeddings for semantic search.

:class:`HashingEmbedder` is a dependency-free baseline (feature hashing of
words, word bigrams and character trigrams). It captures lexical overlap,
not deep meaning. Plug any model with the :class:`Embedder` protocol for
real semantic search.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how", "i", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your"]
)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder, L2-normalized."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = tokenize(text)
        features: list[tuple[str, float]] = [(f"w:{w}", 1.0) for w in words]
        features += [(f"b:{a}_{b}", 0.7) for a, b in itertools.pairwise(words)]
        for w in words:
            padded = f"#{w}#"
            features += [(f"c:{padded[i:i+3]}", 0.3) for i in range(len(padded) - 2)]
        for feature, weight in features:
            idx, sign = self._bucket(feature)
            vec[idx] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
