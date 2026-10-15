"""ML fundamentals inside the OS (roadmap stage 2).

A from-scratch multinomial logistic regression (softmax classifier) trained
with stochastic gradient descent on hashed bag-of-words features. It powers
:class:`IntentRouter`, which learns from labelled examples which agent should
handle a request: supervised learning, a loss function, gradients, and
regularization, in about 100 lines of plain Python.
"""

from __future__ import annotations

import math
import random

from agentic_os.memory.embeddings import HashingEmbedder


class SoftmaxClassifier:
    def __init__(self, dim: int = 256, lr: float = 0.5, epochs: int = 60, l2: float = 1e-4, seed: int = 0) -> None:
        self.embedder = HashingEmbedder(dim)
        self.dim, self.lr, self.epochs, self.l2, self.seed = dim, lr, epochs, l2, seed
        self.labels: list[str] = []
        self.weights: list[list[float]] = []
        self.bias: list[float] = []
        self.loss_history: list[float] = []

    def _features(self, text: str) -> list[float]:
        return self.embedder.embed(text)

    def _logits(self, x: list[float]) -> list[float]:
        return [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(self.weights, self.bias)]

    @staticmethod
    def _softmax(z: list[float]) -> list[float]:
        m = max(z)
        exps = [math.exp(v - m) for v in z]
        total = sum(exps)
        return [e / total for e in exps]

    def fit(self, texts: list[str], labels: list[str]) -> SoftmaxClassifier:
        if len(texts) != len(labels) or not texts:
            raise ValueError("texts and labels must be non-empty and the same length")
        self.labels = sorted(set(labels))
        k = len(self.labels)
        self.weights = [[0.0] * self.dim for _ in range(k)]
        self.bias = [0.0] * k
        data = [(self._features(t), self.labels.index(y)) for t, y in zip(texts, labels)]
        rng = random.Random(self.seed)
        self.loss_history = []
        for epoch in range(self.epochs):
            rng.shuffle(data)
            lr = self.lr / (1 + 0.05 * epoch)  # learning-rate decay
            total_loss = 0.0
            for x, y in data:
                probs = self._softmax(self._logits(x))
                total_loss += -math.log(max(probs[y], 1e-12))
                for c in range(k):
                    grad = probs[c] - (1.0 if c == y else 0.0)  # d(cross-entropy)/d(logit_c)
                    row = self.weights[c]
                    for i, xi in enumerate(x):
                        if xi:
                            row[i] -= lr * (grad * xi + self.l2 * row[i])
                    self.bias[c] -= lr * grad
            self.loss_history.append(total_loss / len(data))
        return self

    def predict_proba(self, text: str) -> dict[str, float]:
        if not self.labels:
            raise RuntimeError("Classifier is not trained")
        probs = self._softmax(self._logits(self._features(text)))
        return dict(zip(self.labels, probs))

    def predict(self, text: str) -> str:
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)


class IntentRouter:
    """Routes requests to agents, learned from example requests per agent."""

    def __init__(self, examples: dict[str, list[str]], min_confidence: float = 0.0, **clf_kwargs) -> None:
        self.min_confidence = min_confidence
        texts, labels = [], []
        for label, items in examples.items():
            texts += items
            labels += [label] * len(items)
        self.classifier = SoftmaxClassifier(**clf_kwargs).fit(texts, labels)

    def predict(self, text: str) -> str | None:
        probs = self.classifier.predict_proba(text)
        label = max(probs, key=probs.get)
        return label if probs[label] >= self.min_confidence else None
